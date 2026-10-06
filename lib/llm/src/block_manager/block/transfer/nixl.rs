// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use anyhow::Result;
use nixl_sys::{MemoryRegion, NixlDescriptor, XferDescList, XferStatus};
use std::future::Future;

fn append_xfer_request<Source, Destination>(
    src: &Source,
    dst: &mut Destination,
    src_dl: &mut XferDescList,
    dst_dl: &mut XferDescList,
) -> Result<()>
where
    Source: BlockDataProvider,
    Source::StorageType: NixlDescriptor,
    Destination: BlockDataProviderMut,
    Destination::StorageType: NixlDescriptor,
{
    let src_data = src.block_data();
    let dst_data = dst.block_data_mut();

    if src_data.is_fully_contiguous() && dst_data.is_fully_contiguous() {
        let src_desc = src_data.block_view()?.as_nixl_descriptor();
        let dst_desc = dst_data.block_view_mut()?.as_nixl_descriptor_mut();

        unsafe {
            src_dl.add_desc(
                src_desc.as_ptr() as usize,
                src_desc.size(),
                src_desc.device_id(),
            );

            dst_dl.add_desc(
                dst_desc.as_ptr() as usize,
                dst_desc.size(),
                dst_desc.device_id(),
            );
        }

        Ok(())
    } else {
        assert_eq!(src_data.num_layers(), dst_data.num_layers());
        for layer_idx in 0..src_data.num_layers() {
            for outer_idx in 0..src_data.num_outer_dims() {
                let src_view = src_data.layer_view(layer_idx, outer_idx)?;
                let mut dst_view = dst_data.layer_view_mut(layer_idx, outer_idx)?;

                debug_assert_eq!(src_view.size(), dst_view.size());

                let src_desc = src_view.as_nixl_descriptor();
                let dst_desc = dst_view.as_nixl_descriptor_mut();

                unsafe {
                    src_dl.add_desc(
                        src_desc.as_ptr() as usize,
                        src_desc.size(),
                        src_desc.device_id(),
                    );

                    dst_dl.add_desc(
                        dst_desc.as_ptr() as usize,
                        dst_desc.size(),
                        dst_desc.device_id(),
                    );
                }
            }
        }
        Ok(())
    }
}

/// Copy a block from a source to a destination using CUDA memcpy
pub fn write_blocks_to<Source, Destination>(
    src: &[Source],
    dst: &mut [Destination],
    ctx: &Arc<TransferContext>,
    transfer_type: NixlTransfer,
) -> Result<Box<dyn Future<Output = ()> + Send + Sync + Unpin>>
where
    Source: BlockDataProvider,
    Source::StorageType: NixlDescriptor,
    Destination: BlockDataProviderMut,
    Destination::StorageType: NixlDescriptor,
{
    if src.is_empty() || dst.is_empty() {
        return Ok(Box::new(std::future::ready(())));
    }
    assert_eq!(src.len(), dst.len());

    let nixl_agent_arc = ctx.as_ref().nixl_agent();
    let nixl_agent = nixl_agent_arc
        .as_ref()
        .as_ref()
        .expect("NIXL agent not found");

    let src_mem_type = src
        .first()
        .unwrap()
        .block_data()
        .storage_type()
        .nixl_mem_type();
    let dst_mem_type = dst
        .first()
        .unwrap()
        .block_data()
        .storage_type()
        .nixl_mem_type();

    let mut src_dl = XferDescList::new(src_mem_type)?;
    let mut dst_dl = XferDescList::new(dst_mem_type)?;

    for (src, dst) in src.iter().zip(dst.iter_mut()) {
        append_xfer_request(src, dst, &mut src_dl, &mut dst_dl)?;
    }

    let xfer_req = nixl_agent.create_xfer_req(
        transfer_type.as_xfer_op(),
        &src_dl,
        &dst_dl,
        &nixl_agent.name(),
        None,
    )?;

    let still_pending = nixl_agent.post_xfer_req(&xfer_req, None)?;

    if still_pending {
        Ok(Box::new(Box::pin(async move {
            let nixl_agent = nixl_agent_arc
                .as_ref()
                .as_ref()
                .expect("NIXL agent not found");

            loop {
                match nixl_agent.get_xfer_status(&xfer_req) {
                    Ok(XferStatus::Success) => break, // Transfer is complete.
                    Ok(XferStatus::InProgress) => {
                        tokio::time::sleep(std::time::Duration::from_millis(5)).await
                    } // Transfer is still in progress.
                    Err(e) => {
                        tracing::error!("Error getting transfer status: {}", e);
                        break;
                    }
                }
            }
        })))
    } else {
        Ok(Box::new(std::future::ready(())))
    }
}

/// Status poll interval of [`read_disk_blocks_into_host`]. Completion is noticed up to one
/// interval late, which is added to the measured disk-read time.
const DISK_READ_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(1);

/// Read disk blocks into host blocks with NIXL; the receiver yields the transfer's result.
///
/// The POSIX backend needs the host-memory list as the local descriptor list for reads as well
/// as writes (`kvbm-physical` calls this `NixlReadFlipped`), so the lists are swapped relative to
/// [`write_blocks_to`]. Unlike [`write_blocks_to`], a failed status poll is returned as an error
/// instead of being treated as completion. The poll runs as its own task, so the NIXL request
/// stays alive until the read has finished even if the caller stops waiting.
pub fn read_disk_blocks_into_host<Source, Destination>(
    src: &[Source],
    dst: &mut [Destination],
    ctx: &Arc<TransferContext>,
) -> Result<oneshot::Receiver<Result<()>>>
where
    Source: BlockDataProvider,
    Source::StorageType: NixlDescriptor,
    Destination: BlockDataProviderMut,
    Destination::StorageType: NixlDescriptor,
{
    let (tx, rx) = oneshot::channel();
    if src.is_empty() && dst.is_empty() {
        let _ = tx.send(Ok(()));
        return Ok(rx);
    }
    if src.len() != dst.len() {
        anyhow::bail!(
            "disk read has {} sources but {} destinations",
            src.len(),
            dst.len()
        );
    }

    let nixl_agent_arc = ctx.as_ref().nixl_agent();
    let nixl_agent = nixl_agent_arc
        .as_ref()
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("NIXL agent not found"))?;

    let src_mem_type = src[0].block_data().storage_type().nixl_mem_type();
    let dst_mem_type = dst[0].block_data().storage_type().nixl_mem_type();
    let mut src_dl = XferDescList::new(src_mem_type)?;
    let mut dst_dl = XferDescList::new(dst_mem_type)?;
    for (src, dst) in src.iter().zip(dst.iter_mut()) {
        append_xfer_request(src, dst, &mut src_dl, &mut dst_dl)?;
    }

    // Host memory is the local list and the file the remote list; READ moves remote -> local.
    let xfer_req = nixl_agent.create_xfer_req(
        nixl_sys::XferOp::Read,
        &dst_dl,
        &src_dl,
        &nixl_agent.name(),
        None,
    )?;
    if !nixl_agent.post_xfer_req(&xfer_req, None)? {
        let _ = tx.send(Ok(()));
        return Ok(rx);
    }

    ctx.async_rt_handle().spawn(async move {
        let result = match nixl_agent_arc.as_ref().as_ref() {
            None => Err(anyhow::anyhow!("NIXL agent not found")),
            Some(agent) => loop {
                match agent.get_xfer_status(&xfer_req) {
                    Ok(XferStatus::Success) => break Ok(()),
                    Ok(XferStatus::InProgress) => tokio::time::sleep(DISK_READ_POLL_INTERVAL).await,
                    Err(error) => break Err(anyhow::anyhow!("NIXL disk read failed: {error}")),
                }
            },
        };
        drop(xfer_req);
        let _ = tx.send(result);
    });
    Ok(rx)
}
