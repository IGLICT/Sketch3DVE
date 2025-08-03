import torch

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
)
from pytorch3d.structures import join_meshes_as_batch

import torch.nn.functional as F
from diffusers.utils import export_to_video

import imageio
import numpy as np
import cv2
import trimesh

def side_to_mesh(pcd_front, pcd_back, mask_path):
    img = cv2.imread(mask_path)
    img = cv2.resize(img, (512,336))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # change into gray maps
    ret, binary = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)  # change into binary maps
    contour, hierarchy = cv2.findContours(binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)  # Find the contour of mask
    
    #========================================================================================
    vectices_list = []
    f_all = None

    len_contours = len(contour)

    for contour_id in range(len_contours):
        num_points_before = len(vectices_list)

        # fit the mask contour
        epsilon = 1
        approx = cv2.approxPolyDP(contour[contour_id], epsilon, True)

        num_points = len(approx)
        # Build vertex set
        # for depth_pc in plane_list:
        for index in range(num_points):
            x, y = approx[index][0]
            vectices_list.append(pcd_front[y, x, :])
        
        for index in range(num_points):
            x, y = approx[index][0]
            vectices_list.append(pcd_back[y, x, :])

        # build side faces
        face_list = []

        # for depth_index in range(len(plane_list) - 1):
        face_depth_list = []
        for index in range(num_points):
            if index == num_points - 1:
                face_depth_list.append([index, 0, index + num_points])
                face_depth_list.append([0, 0 + num_points, index + num_points])
            else:
                face_depth_list.append([index, index+1, index + num_points])
                face_depth_list.append([index+1, index + 1 + num_points, index + num_points])
        face_list.append(np.array(face_depth_list))
        # face_list.append(np.array(face_depth_list) + depth_index*num_points)

        f = np.concatenate(face_list, axis=0)

        if f_all is None:
            f_all = f
        else:
            f_all = np.concatenate((f_all, f + num_points_before),axis=0)

    v = np.array(vectices_list)

    obj = trimesh.Trimesh(vertices = v, faces = f_all)
    return obj
    
def mask_to_mesh(point_cloud, mask):
    """
    :param point_cloud: [H, W, 3] 
    :param mask: [H, W]  bool mask
    :return: trimesh.Trimesh mesh objtect
    """
    H, W, _ = point_cloud.shape
    vertices = point_cloud[mask]
    index_map = -np.ones((H, W), dtype=int)
    index_map[mask] = np.arange(len(vertices))

    faces = []
    for i in range(H - 1):
        for j in range(W - 1):
            if mask[i, j] and mask[i, j + 1] and mask[i + 1, j]:
                v0 = index_map[i, j]
                v1 = index_map[i, j + 1]
                v2 = index_map[i + 1, j]
                faces.append([v0, v1, v2])
            if mask[i + 1, j] and mask[i, j + 1] and mask[i + 1, j + 1]:
                v0 = index_map[i + 1, j]
                v1 = index_map[i, j + 1]
                v2 = index_map[i + 1, j + 1]
                faces.append([v0, v1, v2])

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh


def render_mesh(mesh, device, camera_traj, H=336, W=512):
    # set lighting
    lights = PointLights(device=device, location=[[2.0, 2.0, -2.0]])

    # set Rasterization
    raster_settings = RasterizationSettings(
        image_size=(H, W), 
        blur_radius=0.0, 
        faces_per_pixel=1
    )

    # Make render
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera_traj, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=camera_traj,
            lights=lights
        )
    )

    # rener
    render_results = renderer(mesh)
    
    aplha = torch.where(render_results[:, :, :, 3:] > 0, 
                                         torch.tensor([1.0, 1.0, 1.0], device=device), 
                                         torch.tensor([0.0, 0.0, 0.0], device=device))
    return aplha

def merge_video(video_a_path, video_b_path, mask_video_path, output_video_path):
    # Read video A, video B and Mask
    video_a_reader = imageio.get_reader(video_a_path, format="ffmpeg")
    video_b_reader = imageio.get_reader(video_b_path, format="ffmpeg")
    mask_reader = imageio.get_reader(mask_video_path, format="ffmpeg")
    
    # Get video metadata (all the videos have the same resolution and frames)
    metadata = video_a_reader.get_meta_data()
    frames_per_second = metadata['fps']
    
    # make video writer
    writer = imageio.get_writer(output_video_path, fps=frames_per_second, format="ffmpeg")

    try:
        for frame_a, frame_b, mask_frame in zip(video_a_reader, video_b_reader, mask_reader):
            # Ensure frame and mask_frame have the same shape
            if frame_a.shape != frame_b.shape or frame_a.shape[:2] != mask_frame.shape[:2]:
                raise ValueError("The resolutions of video A, video B, and Mask are inconsistent!")

            # Normalize mask (Binary mask with single channel，vaule between 0 and 255)
            if len(mask_frame.shape) == 3: 
                mask_frame = mask_frame[:, :, 0]
            mask = mask_frame.astype(np.float32) / 255.0
            
            # Perform Gaussian blur on the mask (kernel size and standard deviation can be adjusted according to requirements)
            blurred_mask = cv2.GaussianBlur(mask, (21, 21), sigmaX=10, sigmaY=10)

            # Extend the mask to match the shape of the video frame
            mask = np.stack([blurred_mask] * 3, axis=-1)

            # merged video based on mask
            blended_frame = (frame_a * mask + frame_b * (1 - mask)).astype(np.uint8)
            writer.append_data(blended_frame)
    finally:
        # close video writer
        video_a_reader.close()
        video_b_reader.close()
        mask_reader.close()
        writer.close()

    print(f"Video have been saved into {output_video_path}")
