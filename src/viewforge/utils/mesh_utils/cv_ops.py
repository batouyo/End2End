from typing import Optional

import cv2
import numpy as np
import torch

try:
    import cvcuda
except ImportError:
    cvcuda = None


torch_to_cvc = lambda x, layout: cvcuda.as_tensor(x, layout)

cvc_to_torch = lambda x, device: torch.tensor(x.cuda(), device=device)


def _to_uint8_image(image: torch.Tensor) -> torch.Tensor:
    if image.dtype == torch.uint8:
        return image.detach()
    return (image.detach().clamp(0.0, 1.0) * 255).to(torch.uint8)


def _to_uint8_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype == torch.uint8:
        return mask.detach()
    return (mask.detach().float() * 255).to(torch.uint8)


def _from_numpy(data: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(data)).to(device=device)
    if dtype == torch.uint8:
        return tensor
    return tensor.to(dtype=dtype) / 255.0


def inpaint_cvc(
    image: torch.Tensor,
    mask: torch.Tensor,
    padding_size: int,
    return_dtype: Optional[torch.dtype] = None,
):
    input_dtype = image.dtype
    input_device = image.device

    image = _to_uint8_image(image)
    mask = _to_uint8_mask(mask)

    if cvcuda is not None:
        image_cvc = torch_to_cvc(image, "HWC")
        mask_cvc = torch_to_cvc(mask, "HW")
        output_cvc = cvcuda.inpaint(image_cvc, mask_cvc, padding_size)
        output = cvc_to_torch(output_cvc, device=input_device)
        if return_dtype == torch.uint8 or input_dtype == torch.uint8:
            return output
        return output.to(dtype=input_dtype) / 255.0

    image_np = image.cpu().numpy()
    mask_np = mask.cpu().numpy()
    output_np = cv2.inpaint(
        image_np,
        mask_np,
        float(max(padding_size, 1)),
        cv2.INPAINT_TELEA,
    )
    output_dtype = (
        torch.uint8
        if return_dtype == torch.uint8 or input_dtype == torch.uint8
        else input_dtype
    )
    return _from_numpy(output_np, input_device, output_dtype)


def batch_inpaint_cvc(
    images: torch.Tensor,
    masks: torch.Tensor,
    padding_size: int,
    return_dtype: Optional[torch.dtype] = None,
):
    output = torch.stack(
        [
            inpaint_cvc(image, mask, padding_size, return_dtype)
            for (image, mask) in zip(images, masks)
        ],
        axis=0,
    )
    return output


def _batch_morphology(
    masks: torch.Tensor,
    kernel_size: int,
    op: str,
    return_dtype: Optional[torch.dtype] = None,
):
    input_dtype = masks.dtype
    input_device = masks.device
    masks = _to_uint8_mask(masks)

    if cvcuda is not None:
        masks_cvc = torch_to_cvc(masks[..., None], "NHWC")
        morph_type = {
            "erode": cvcuda.MorphologyType.ERODE,
            "dilate": cvcuda.MorphologyType.DILATE,
        }[op]
        masks_out_cvc = cvcuda.morphology(
            masks_cvc,
            morph_type,
            maskSize=(kernel_size, kernel_size),
            anchor=(-1, -1),
        )
        masks_out = cvc_to_torch(masks_out_cvc, device=input_device)[..., 0]
    else:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        fn = cv2.erode if op == "erode" else cv2.dilate
        masks_np = masks.cpu().numpy()
        masks_out_np = np.stack(
            [fn(mask, kernel, iterations=1) for mask in masks_np],
            axis=0,
        )
        masks_out = torch.from_numpy(np.ascontiguousarray(masks_out_np)).to(
            device=input_device
        )

    if return_dtype == torch.uint8 or input_dtype == torch.uint8:
        return masks_out
    return (masks_out > 0).to(dtype=input_dtype)


def batch_erode(
    masks: torch.Tensor, kernel_size: int, return_dtype: Optional[torch.dtype] = None
):
    return _batch_morphology(masks, kernel_size, "erode", return_dtype)


def batch_dilate(
    masks: torch.Tensor, kernel_size: int, return_dtype: Optional[torch.dtype] = None
):
    return _batch_morphology(masks, kernel_size, "dilate", return_dtype)
