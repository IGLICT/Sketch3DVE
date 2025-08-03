import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.transforms.functional import pil_to_tensor, resize

def load_and_normalize_image_pil(image_path):
    img = Image.open(image_path)
    img_array = np.array(img).astype(np.float32)
    if len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, axis=-1)
    img_normalized = img_array # / 255.0
    return img_normalized

def mask_to_tensor(mask_path, threshold=127):
    mask = Image.open(mask_path).convert('L')
    mask_np = np.array(mask)
    mask_np = (mask_np > threshold).astype(np.uint8)
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(-1)
    return mask_tensor.float()

def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(W-1, 0, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def align_depth_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=True,
    max_resolution=None,
):
    ori_shape = pred_arr.shape  # input shape

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
            pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
            valid_mask = (
                downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float())
                .bool()
                .numpy()
            )

    assert (
        gt.shape == pred.shape == valid_mask.shape
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    # numpy solver
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred_arr * scale + shift
    
    print("scale:", scale)
    print("shift:", shift)

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)

    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred


# gt_arr: [H, W]（depth map）
# pred_arr: [H, W]（disparity map）
# valid_mask: A large value indicates the area of the mask
def align_disparity_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask,
    region_new_min = 0.1,
    region_new_max = 0.2,
    max_scale=1.4,
):
    if isinstance(valid_mask, Image.Image):
        valid_mask = valid_mask.convert("RGB")
        # convert to torch tensor [H, W, rgb] -> [rgb, H, W]
        valid_mask = pil_to_tensor(valid_mask)
        valid_mask = valid_mask[:1].squeeze(0)  # [1, H, W]
        valid_mask = valid_mask.cpu().numpy()
    if isinstance(valid_mask, torch.Tensor):
        valid_mask = valid_mask.cpu().numpy()
        
    gt = gt_arr.copy().squeeze()  # [H, W]
    pred = pred_arr.copy().squeeze()
    valid_mask = valid_mask.copy().squeeze().astype(np.float32)
    valid_mask = valid_mask/valid_mask.max()
    valid_mask = valid_mask < 0.5
    
    # Convert gt to a disparity map gt_inverse
    gt_inverse = 1.0/(gt+1e-6)
    # Align pred (disparity map)
    pred,_,_ = align_depth_least_square(gt_inverse, pred, valid_mask, return_scale_shift=True, max_resolution=None)
    # #Convert aligned pred to depth map
    pred = 1.0/(pred+1e-6)
    # For better alignment, clip the pred to prevent excessive 
    pred = pred.clip(gt.min()/max_scale, gt.max()*max_scale)
    
    # pred,_,_ = align_depth_least_square(gt, pred, valid_mask, return_scale_shift=True, max_resolution=None)
    
    valid_mask = ~valid_mask
    mask_region = pred[valid_mask]
    mask_region_max = np.max(mask_region)
    mask_region_min = np.min(mask_region)
    print('mask_region_min:', mask_region_min)
    print('mask_region_max:', mask_region_max)
    pred = (pred - mask_region_min) / (mask_region_max - mask_region_min) * (region_new_max - region_new_min) + region_new_min
    
    return pred


# gt_arr: [H, W]（depth map）
# pred_arr: [H, W]（disparity map）
# valid_mask: A large value indicates the area of the mask
def align_disparity_least_square_auto(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask,
):
    if isinstance(valid_mask, Image.Image):
        valid_mask = valid_mask.convert("RGB")
        # convert to torch tensor [H, W, rgb] -> [rgb, H, W]
        valid_mask = pil_to_tensor(valid_mask)
        valid_mask = valid_mask[:1].squeeze(0)  # [1, H, W]
        valid_mask = valid_mask.cpu().numpy()
    if isinstance(valid_mask, torch.Tensor):
        valid_mask = valid_mask.cpu().numpy()
        
    gt = gt_arr.copy().squeeze()  # [H, W]
    pred = pred_arr.copy().squeeze()
    valid_mask = valid_mask.copy().squeeze().astype(np.float32)
    valid_mask = valid_mask/valid_mask.max()
    valid_mask = valid_mask < 0.5
    
    # Align pred (depth map)
    pred,_,_ = align_depth_least_square(gt_arr, pred_arr, valid_mask, return_scale_shift=True, max_resolution=None)
    return pred, valid_mask

