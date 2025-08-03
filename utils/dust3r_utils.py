from extern.dust3r.inference import inference, load_model
from extern.dust3r.utils.image import ImgNorm
from extern.dust3r.image_pairs import make_pairs
from extern.dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from extern.dust3r.utils.device import to_numpy
import glob
import os

import numpy as np
import torch

import pytorch3d
from pytorch3d.structures import Pointclouds

from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
)

import torch.nn.functional as F

import scipy

from PIL import Image, ImageOps
import cv2
import math

from diffusers.utils import export_to_video
import trimesh
import imageio

import pandas as pd

from utils.pvd_utils import save_pointcloud_with_normals

def center_crop_pil_image(input_image, target_width=720, target_height=480):
    w, h = input_image.size
    h_ratio = h / target_height
    w_ratio = w / target_width

    if h_ratio > w_ratio:
        h = int(h / w_ratio)
        if h < target_height:
            h = target_height
        input_image = input_image.resize((target_width, h), Image.LANCZOS)
    else:
        w = int(w / h_ratio)
        if w < target_width:
            w = target_width
        input_image = input_image.resize((w, target_height), Image.LANCZOS)

    return ImageOps.fit(input_image, (target_width, target_height), Image.BICUBIC)

def load_images(pil_list, size, square_ok=False, force_1024=True):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    imgs = []
    for img in pil_list:
        img = img.convert('RGB')
        img_ori = img
        
        # get the resize version1
        S = max(img.size)
        long_edge_size = size
        if S > long_edge_size:
            interp = Image.LANCZOS
        elif S <= long_edge_size:
            interp = Image.BICUBIC
        
        W, H = img.size
        W_new = int(round(W*long_edge_size/S))
        H_new = int(round(H*long_edge_size/S))
        
        cx, cy = W_new//2, H_new//2
        halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
        if not (square_ok) and W == H:
            halfh = 3*halfw/4
        
        W_final = 2 * halfw
        H_final = 2 * halfh
        
        img = img.resize((W_final, H_final), interp)
            
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs)), img_ori=ImgNorm(img_ori)[None], ))

    return imgs

def video_to_image_list(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Unable to open video file: {video_path}")
        return None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        frames.append(image)
    cap.release()
    
    return frames

def run_dust3r(dust3r, device, input_images, clean_pc = False):
    batch_size = 4 # 1
    niter = 300
    schedule = 'linear'
    lr = 0.01
    
    pairs = make_pairs(input_images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, dust3r, device, batch_size=batch_size)

    mode = GlobalAlignerMode.PointCloudOptimizer #if len(self.images) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode)
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    if clean_pc:
        scene = scene.clean_pointcloud()
    else:
        scene = scene
    return scene

def setup_renderer(cameras, image_size):
    # Define the settings for rasterization and shading.
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius = 0.01,
        points_per_pixel = 10,
        bin_size = 0
    )

    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor()
    )

    render_setup =  {'cameras': cameras, 'raster_settings': raster_settings, 'renderer': renderer}

    return render_setup

def render_pcd(pts3d,imgs,masks,views,renderer,device,nbv=False):
    imgs = to_numpy(imgs)
    pts3d = to_numpy(pts3d)

    if masks == None:
        pts = torch.from_numpy(np.concatenate([p for p in pts3d])).view(-1, 3).to(device)
        col = torch.from_numpy(np.concatenate([p for p in imgs])).view(-1, 3).to(device)
    else:
        pts = torch.from_numpy(np.concatenate([p[m] for p, m in zip(pts3d, masks)])).to(device)
        col = torch.from_numpy(np.concatenate([p[m] for p, m in zip(imgs, masks)])).to(device)
        
    point_cloud = Pointclouds(points=[pts], features=[col]).extend(views)
    images = renderer(point_cloud)

    if nbv:
        color_mask = torch.ones(col.shape).to(device)
        point_cloud_mask = Pointclouds(points=[pts],features=[color_mask]).extend(views)
        view_masks = renderer(point_cloud_mask)
    else: 
        view_masks = None

    return images, view_masks

def run_render(pcd, imgs, masks, H, W, camera_traj, num_views, device, nbv=False):
    render_setup = setup_renderer(camera_traj, image_size=(H,W))
    renderer = render_setup['renderer']
    render_results, viewmask = render_pcd(pcd, imgs, masks, num_views, renderer, device, nbv=nbv)
    return render_results, viewmask

def change_camera_view(c2ws, H, W, fs, c, device):
    num_views = c2ws.shape[0] 
    R, T = c2ws[:,:3, :3], c2ws[:,:3, 3:]
    R = torch.stack([-R[:,:, 0], -R[:,:, 1], R[:,:, 2]], 2) # from RDF to LUF for Rotation
    new_c2w = torch.cat([R, T], 2)
    w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[[0,0,0,1]]]).to(device).repeat(new_c2w.shape[0],1,1)),1))
    R_new, T_new = w2c[:,:3, :3].permute(0,2,1), w2c[:,:3, 3] # convert R to row-major matrix
    image_size = ((H, W),)  # (h, w)
    
    cameras = PerspectiveCameras(focal_length=fs, principal_point=c, in_ndc=False, image_size=image_size, R=R_new, T=T_new, device=device)
    
    return cameras, num_views

def interpolate_poses_spline(poses, n_interp, spline_degree=5,
                               smoothness=.03, rot_weight=.1):
    """Creates a smooth spline path between input keyframe camera poses.

  Spline is calculated with poses in format (position, lookat-point, up-point).

  Args:
    poses: (n, 3, 4) array of input pose keyframes.
    n_interp: returned path will have n_interp * (n - 1) total poses.
    spline_degree: polynomial degree of B-spline.
    smoothness: parameter for spline smoothing, 0 forces exact interpolation.
    rot_weight: relative weighting of rotation/translation in spline solve.

  Returns:
    Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
  """

    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points
    
    def viewmatrix(lookdir, up, position):
        """Construct lookat view matrix."""
        vec2 = normalize(lookdir)
        vec0 = normalize(np.cross(up, vec2))
        vec1 = normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m

    def normalize(x):
        """Normalization helper function."""
        return x / np.linalg.norm(x)
    
    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(points,
                        n_interp * (points.shape[0] - 1),
                        k=spline_degree,
                        s=smoothness)
    new_poses = points_to_poses(new_points) 
    poses_tensor = torch.from_numpy(new_poses)
    extra_row = torch.tensor(np.repeat([[0, 0, 0, 1]], n_interp, axis=0), dtype=torch.float32).unsqueeze(1)
    poses_final = torch.cat([poses_tensor, extra_row], dim=1)

    return poses_final

def interp_traj(c2ws: torch.Tensor, n_inserts: int = 25, device='cuda') -> torch.Tensor:
    
    n_poses = c2ws.shape[0] 
    interpolated_poses = []

    for i in range(n_poses-1):
        start_pose = c2ws[i]
        end_pose = c2ws[(i + 1) % n_poses]
        interpolated_path = interpolate_poses_spline(torch.stack([start_pose, end_pose])[:, :3, :].cpu().numpy(), n_inserts).to(device)
        interpolated_path = interpolated_path[:-1]
        interpolated_poses.append(interpolated_path)

    interpolated_poses.append(c2ws[-1:])
    full_path = torch.cat(interpolated_poses, dim=0)

    return full_path

def interpolate_sequence(sequence, k,device):
    N, M = sequence.size()
    weights = torch.linspace(0, 1, k+1).view(1, -1, 1).to(device)
    left_values = sequence[:-1].unsqueeze(1).repeat(1, k+1, 1)
    right_values = sequence[1:].unsqueeze(1).repeat(1, k+1, 1)
    new_sequence = torch.einsum("ijk,ijl->ijl", (1 - weights), left_values)[:,:k,:] + torch.einsum("ijk,ijl->ijl", weights, right_values)[:,:k,:]
    new_sequence = new_sequence.reshape(-1, M)
    new_sequence = torch.cat([new_sequence, sequence[-1].view(1, -1)], dim=0)
    return new_sequence

