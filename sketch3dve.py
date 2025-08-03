import os
import tqdm
from PIL import Image
import matplotlib
from diffusers.utils import export_to_video
import copy

from utils.dust3r_utils import *
from utils.process_utils import resize_video
from utils.pvd_utils import save_pointcloud_with_normals
from utils.depth_align_utils import *
from utils.mask_box_utils import *

class Sketch3DVE:
    def __init__(self, opts):
        self.opts = opts
        # prepare the path of pretrained models
        self.dust3r_model_path = opts.dust3r_model_path
        self.depthanything_model_path = opts.depthanything_model_path
        self.Llava_model_path = opts.Llava_model_path
        self.controlnet_ckpt_path = opts.controlnet_ckpt_path
        self.basemodel_ckpt_path = opts.basemodel_ckpt_path
        self.device = 'cuda'

    def preprocess_video(self, video_path, crop_length=True):
        root_dir = os.path.dirname(video_path)
        frame_save_dir = os.path.join(root_dir, "frames")
    
        if not os.path.exists(frame_save_dir):
            os.makedirs(frame_save_dir)
    
        if crop_length:
            # If the input video does not have 49 frames
            tensor_frames = resize_video(video_path)
    
            pil_list = []
            for i in tqdm.tqdm(range(tensor_frames.shape[0])):
                img = tensor_frames[i]
                img = Image.fromarray(img)
                img = center_crop_pil_image(img)
                img.save(os.path.join(frame_save_dir, '%03d.png'%len(pil_list)))
                pil_list.append(img)
        else:
            # If the input video has 49 frames
            pil_list_ori = video_to_image_list(video_path)
            count = 0
            pil_list = []
            for img in tqdm.tqdm(pil_list_ori):
                count = count + 1
                img = center_crop_pil_image(img)
                img.save(os.path.join(frame_save_dir, '%03d.png'%len(pil_list)))
                pil_list.append(img)
    
        output_path = os.path.join(root_dir, 'original.mp4')
        export_to_video(pil_list, output_path, fps=8)

    def run_dust3r(self, root_dir):
        from extern.dust3r.inference import inference, load_model

        device = self.device
        dust3r = load_model(self.dust3r_model_path, device)

        # opt parameters
        dpt_trd = 1.0
        inter_param = 2
    
        video_dir = os.path.join(root_dir, 'frames')
        file_list = sorted(os.listdir(video_dir))
        pil_list = []
        for file_name in file_list:
            img = Image.open(os.path.join(video_dir, file_name))
            pil_list.append(img)
    
        video_length = len(pil_list)
        if video_length < 49:
            print("Video is too short. Please do not use.")
        video_length = math.floor((video_length-1) / 4) * 4 + 1
        inter_index = list(range(0, video_length, inter_param))
        pil_list_inter = [pil_list[i] for i in inter_index]
    
        dust3r_dir = os.path.join(root_dir, 'dust3r')
        if not os.path.exists(dust3r_dir):
            os.mkdir(dust3r_dir)

        images = load_images(pil_list_inter, size=512, force_1024 = True)
    
        # run dust3r
        scene = run_dust3r(dust3r, device, images)
        c2ws = scene.get_im_poses().detach()
        principal_points = scene.get_principal_points().detach()
        focals = scene.get_focals().detach()
        shape = images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
    
        depth = scene.get_depthmaps()
    
        # Get camera parameter
        c2ws_inter = interp_traj(c2ws, n_inserts=inter_param+1)
        focals_inter = interpolate_sequence(focals, inter_param, device=device)
        principal_points_inter = interpolate_sequence(principal_points, inter_param, device=device)
        camera_traj, num_views = change_camera_view(c2ws_inter, H, W, focals_inter, principal_points_inter, device)
    
        render_index_list = [0]
        for render_index in render_index_list:
            # Get render_index scene information
            pcd = [scene.get_pts3d(clip_thred=dpt_trd)[render_index]] # a list of points of size whc
            masks = None
            imgs = np.array([scene.imgs[render_index]])
            
            render_results, viewmask = run_render(pcd, imgs, masks, H, W, camera_traj, num_views, device)
            render_results = F.interpolate(render_results.permute(0,3,1,2), size=(480, 720), mode='bilinear', align_corners=False).permute(0,2,3,1)
            print("render_results:", render_results.shape)
            export_to_video(render_results.cpu().numpy(), os.path.join(dust3r_dir, file_name[:-4] + '_' + str(render_index)+'.mp4'), fps=8)
            
            ply_path = os.path.join(dust3r_dir, 'dust3r_points.ply')
            save_pointcloud_with_normals(imgs, pcd, msk=None, save_path=os.path.join(ply_path), mask_pc=False, reduce_pc=False)
    
        # save dust3r prediction results
        camera_dict = {
            'c2ws_inter': c2ws_inter,
            'principal_points_inter': principal_points_inter, 
            'focals_inter': focals_inter, 
            "depth": depth[0],
        }
        camera_dict_path = os.path.join(dust3r_dir, 'camera.pt')
        torch.save(camera_dict, camera_dict_path)
    
        depth = depth[0].detach().cpu().numpy()
        depth_path = os.path.join(dust3r_dir, 'depth.npy')
        np.save(depth_path, depth)
    
    def run_DepthAnythingv2(self, img_path, outdir):
        from extern.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2

        model_path = self.depthanything_model_path
        DEVICE = self.device

        encoder_type = 'vitl'
        input_size = 518
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
    
        depth_anything = DepthAnythingV2(**model_configs[encoder_type])
        depth_anything.load_state_dict(torch.load(model_path, map_location='cpu'))
        depth_anything = depth_anything.to(DEVICE).eval()
    
        if os.path.isfile(img_path):
            if img_path.endswith('txt'):
                with open(img_path, 'r') as f:
                    filenames = f.read().splitlines()
            else:
                filenames = [img_path]
        else:
            filenames = glob.glob(os.path.join(img_path, '**/*'), recursive=True)
        
        os.makedirs(outdir, exist_ok=True)
    
        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
        for k, filename in enumerate(filenames):
            print(f'Progress {k+1}/{len(filenames)}: {filename}')
            
            raw_image = cv2.imread(filename)
            
            depth = depth_anything.infer_image(raw_image, input_size)
            torch.save(depth, os.path.join(outdir, os.path.splitext(os.path.basename(filename))[0] + '.pt'))
            
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])
            cv2.imwrite(os.path.join(outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)
        
    def run_align_depth(self, root_dir, region_new_min, region_new_max):
        mask_path = os.path.join(root_dir, 'accurate_mask.png')
        mask = Image.open(mask_path).convert('L')
        mask_np = np.array(mask)
    
        # Load the dust3r depth
        dict_path = os.path.join(root_dir, 'dust3r/camera.pt')
        save_dict = torch.load(dict_path, map_location='cpu')
        H = 336
        W = 512
        dust3r_depth = save_dict['depth'].detach().numpy()
        
        # Load the depthanything-v2 depth
        depth_anything = torch.load(os.path.join(root_dir, 'depth_anything/editing.pt'))
        text_path = os.path.join(root_dir, 'depth_anything/mesh_coarse_box.text')

        with open(text_path, 'w') as file:
            print("depth_min:", region_new_min, file=file)
            print("depth_max:", region_new_max, file=file)
            print("type:", "new assign")
    
        align_pred = align_disparity_least_square(dust3r_depth, depth_anything, mask_np, region_new_min, region_new_max)
    
        np.save(os.path.join(root_dir, 'depth_align.npy'), align_pred)
    
        mask = mask_np > 0.5
        depth_merge = align_pred * mask + dust3r_depth * (1-mask)
        depth_merge = torch.from_numpy(depth_merge)
    
        # Obtain edited point cloud
        focal = save_dict['focals_inter'].detach()[0].item()
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
        K = torch.from_numpy(K)
        c2w = save_dict['c2ws_inter'][0]
        rays_o, rays_d = get_rays(H, W, K, c2w)
        pcd_new = rays_o - rays_d * depth_merge.unsqueeze(2)

        img_path = os.path.join(root_dir, 'editing.png')

        imgs = load_and_normalize_image_pil(img_path)
        imgs = imgs[:,:,0:3]
        imgs = torch.from_numpy(imgs)
        imgs = imgs.to(torch.uint8)

        save_pointcloud_with_normals([imgs], [pcd_new], msk=None, save_path=os.path.join(root_dir, 'edited_pc.ply'), mask_pc=False, reduce_pc=False)
    
    def run_align_depth_dust3r(self, root_dir):
        mask_path = os.path.join(root_dir, 'accurate_mask.png')
        mask = Image.open(mask_path).convert('L')
        mask_np = np.array(mask)
    
        # Load the dust3r depth
        dict_path = os.path.join(root_dir, 'dust3r/camera.pt')
        save_dict = torch.load(dict_path, map_location='cpu')
    
        # Load the dust3r model
        device = self.device
        dust3r = load_model(self.dust3r_model_path, device)
        
        # predict the edited image depth, using dust3r
        ref_image_path = os.path.join(root_dir, 'editing_ori.png')
        ref_image = Image.open(ref_image_path)
        images = load_images([ref_image], size=512,force_1024 = True)
        img_ori = (images[0]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2. # [576,1024,3] [0,1]

        if len(images) == 1:
            images = [images[0], copy.deepcopy(images[0])]
            images[1]['idx'] = 1
    
        scene = run_dust3r(dust3r, device, images)
    
        # depth alignment
        depth = scene.get_depthmaps()
        depth = depth[-1].detach().cpu().numpy()
        H = 336
        W = 512
        dust3r_depth = save_dict['depth'].detach().numpy()
    
        depth_align, valid_mask = align_disparity_least_square_auto(dust3r_depth, depth, mask)
    
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_np = mask_np / 255.0
        mask_np = cv2.dilate(mask_np, kernel)
        mask_np = cv2.GaussianBlur(mask_np, (5, 5), sigmaX=10, sigmaY=10)
    
        depth_align = depth_align * mask_np + dust3r_depth * (1.0 - mask_np)
        np.save(os.path.join(root_dir, 'depth_align.npy'), depth_align)
        
        # Obtain edited point cloud
        depth_align = torch.from_numpy(depth_align)
        focal = save_dict['focals_inter'].detach()[0].item()
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
        K = torch.from_numpy(K)
        c2w = save_dict['c2ws_inter'][0]
        rays_o, rays_d = get_rays(H, W, K, c2w)
        pcd_new = rays_o - rays_d * depth_align.unsqueeze(2)
    
        img_path = os.path.join(root_dir, 'editing.png')
    
        imgs = load_and_normalize_image_pil(img_path)
        imgs = imgs[:,:,0:3]
        imgs = torch.from_numpy(imgs)
        imgs = imgs.to(torch.uint8)
    
        save_pointcloud_with_normals([imgs], [pcd_new], msk=None, save_path=os.path.join(root_dir, 'edited_pc.ply'), mask_pc=False, reduce_pc=False)

    def render_edited_pc(self, root_dir):
        mesh_path = os.path.join(root_dir, "edited_pc.ply")
        dict_path = os.path.join(root_dir, 'dust3r/camera.pt')
        render_path = os.path.join(root_dir, 'edited_render.mp4')
        render_mask_path = os.path.join(root_dir, 'edited_render_mask.mp4')
        save_dict = torch.load(dict_path, map_location='cpu')
    
        H = 336
        W = 512
        device = self.device
    
        c2ws_inter = save_dict['c2ws_inter'].to(device)
        focals_inter = save_dict['focals_inter'].to(device)
        principal_points_inter = save_dict['principal_points_inter'].to(device)

        camera_traj, num_views = change_camera_view(c2ws_inter, H, W, focals_inter, principal_points_inter, device)
        num_views = 49
    
        mesh = trimesh.load(mesh_path)
        points = np.asarray(mesh.vertices).astype(np.float32)
    
        colors = np.asarray(mesh.visual.vertex_colors)[:, :3]
        colors = colors / 255.0 
        colors = colors.astype(np.float32)
    
        render_results, viewmask = run_render([points], [colors], None, H, W, camera_traj, num_views, device)
    
        render_results = F.interpolate(render_results.permute(0,3,1,2), size=(480, 720), mode='bilinear', align_corners=False).permute(0,2,3,1)
        print("render_results:", render_results.shape)
        export_to_video(render_results.cpu().numpy(), render_path, fps=8)
    
        render_results, viewmask = run_render([points], [colors], None, H, W, camera_traj, num_views, device, nbv=True)
    
        viewmask = F.interpolate(viewmask.permute(0,3,1,2), size=(480, 720), mode='bilinear', align_corners=False).permute(0,2,3,1)
        print("render_results:", viewmask.shape)
        export_to_video(viewmask.cpu().numpy(), render_mask_path, fps=8)
    
    def construct_3D_mesh(self, root_dir, front_offset, depth_max):
        # Make directory to save mesh models
        mask_box_dir = os.path.join(root_dir, 'mask_box')
        if not os.path.exists(mask_box_dir):
            os.mkdir(mask_box_dir)
    
        # save 3D mesh parameters
        text_path = os.path.join(root_dir, 'mask_box/mesh_coarse_box.text')
        with open(text_path, 'w') as file:
            print("front_offset:", front_offset, file=file)
            print("depth_max:", depth_max, file=file)
        
        # Prepare front faces
        # Load the original depth maps
        dict_path = os.path.join(root_dir, 'dust3r/camera.pt')
        save_dict = torch.load(dict_path, map_location='cpu')
        # Load the edited depth maps
        depth_edited = np.load(os.path.join(root_dir, 'depth_align.npy'))
        depth_ori = save_dict['depth'].detach().numpy()
        # Merge depth maps
        depth_merge = np.minimum(depth_edited, depth_ori)
        depth_merge = np.clip(depth_merge, 0, depth_max)
        # add an offset to the geometry
        depth_merge = depth_merge - front_offset

        # Prepare back faces
        depth_back = np.ones(depth_merge.shape) * depth_max
    
        # Load rendering rays parameters
        W = 512
        H = 336
        focal = save_dict['focals_inter'].detach()[0].item()
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
        K = torch.from_numpy(K)
        c2w = save_dict['c2ws_inter'][0]
        print("c2w:", c2w.shape)
        rays_o, rays_d = get_rays(H, W, K, c2w)
        rays_o = rays_o.numpy()
        rays_d = rays_d.numpy()
        
        # Construct 3D faces
        pcd_front = rays_o - rays_d * depth_merge[:,:,np.newaxis]
        pcd_back = rays_o - rays_d * depth_back[:,:, np.newaxis]
    
        mask_path = os.path.join(root_dir, 'accurate_mask.png')
        mask = Image.open(mask_path).convert('L') 
        mask_np = np.array(mask)
        valid_mask = mask_np > 125
    
        front_mesh = mask_to_mesh(pcd_front, valid_mask)
        # front_mesh.export(os.path.join(mask_box_dir, 'front_mask_mesh.obj'))
    
        back_mesh = mask_to_mesh(pcd_back, valid_mask)
        # back_mesh.export(os.path.join(mask_box_dir, 'back_mask_mesh.obj'))
    
        side_mesh = side_to_mesh(pcd_front, pcd_back, mask_path)
        # side_mesh.export(os.path.join(mask_box_dir, 'side_mask_mesh.obj'))
    
        merged_mesh = trimesh.util.concatenate([front_mesh, back_mesh, side_mesh])
        merged_mesh.export(os.path.join(mask_box_dir, 'mesh_coarse_box.obj'))
    
        # =========================================================
        # Read 3D mask mesh and rendering into mask video

        device = self.device
        c2ws_inter = save_dict['c2ws_inter'].to(device)
        focals_inter = save_dict['focals_inter'].to(device)
        principal_points_inter = save_dict['principal_points_inter'].to(device)
        camera_traj, num_views = change_camera_view(c2ws_inter, H, W, focals_inter, principal_points_inter, device)
    
        # Load mesh model
        device = self.device
        mesh = load_objs_as_meshes([os.path.join(root_dir, "mask_box/mesh_coarse_box.obj")], device=device)

        # Add white texture 
        white_color = torch.ones_like(mesh.verts_packed(), device=device)  # (V, 3)
        white_color = white_color.unsqueeze(0)
        mesh.textures = TexturesVertex(verts_features=white_color)
        
        # Render 3D Mesh model
        mesh = join_meshes_as_batch([mesh] * num_views) 
        aplha = render_mesh(mesh, device, camera_traj, H, W)
    
        # print("aplha:", aplha.shape)
        render_results = F.interpolate(aplha.permute(0,3,1,2), size=(480, 720), mode='bilinear', align_corners=False).permute(0,2,3,1)
        # print("render_results:", render_results.shape)
        export_to_video(render_results.cpu().numpy(), os.path.join(root_dir, 'mask_box/box_render.mp4'), fps=8)
    
        # Prepare output file path
        video_a_path = os.path.join(root_dir, "edited_render.mp4")
        video_b_path = os.path.join(root_dir, "original.mp4")
        mask_video_path = os.path.join(root_dir, "mask_box/box_render.mp4")
        output_video_path = os.path.join(root_dir, "mask_box/output_video.mp4")
        # Visualize the merge results
        merge_video(video_a_path, video_b_path, mask_video_path, output_video_path)
    
    def run_Llava(self, root_dir):
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

        model_path = self.Llava_model_path
        processor = LlavaNextProcessor.from_pretrained(model_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)

        model.to("cuda")
        text_path = os.path.join(root_dir, "editing.txt")
        image_path = os.path.join(root_dir, "editing_ori.png")

        image = Image.open(image_path)

        # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
        # Each value in "content" has to be a list of dicts with types ("text", "image") 
        conversation = [
            {
              "role": "user",
              "content": [
                  {"type": "text", "text": "What is shown in this image?"},
                  {"type": "image"},
                ],
            },]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

        # autoregressively complete prompt
        output = model.generate(**inputs, max_new_tokens=225)
        result_text = processor.decode(output[0], skip_special_tokens=True)
        print(result_text.split("[/INST] ")[1])
        result_text = result_text.split("[/INST] ")[1]
        with open(text_path, 'w', encoding='utf-8') as file:
            file.write(result_text)
    
    def run_diffusion(self, root_dir, seed=40, guidance_scale=10.0):
        from video_diffusion.pipeline_control_cogvideo import CogVideoXControlNetPipeline
        from video_diffusion.controlnet.controlnet_self_attn import CogVideoControlNetModel
        import video_diffusion.vae_tile
        import video_diffusion.controlnet_transformer_3d

        from diffusers import (
            AutoencoderKLCogVideoX,
            CogVideoXDDIMScheduler,
        )
        from decord import VideoReader
        
        # Load video diffusion models
        basemodel_path = self.basemodel_ckpt_path
        controlnet_path = self.controlnet_ckpt_path

        controlnet = CogVideoControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16, use_safetensors=True)
        pipeline = CogVideoXControlNetPipeline.from_pretrained(
            basemodel_path, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
        )

        device = self.device
        pipeline.scheduler = CogVideoXDDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(device)
        pipeline.vae.enable_tiling()
        
        # prepare input file paths
        validation_prompts_path = os.path.join(root_dir, "editing.txt")
        validation_pointcloud_video_path = os.path.join(root_dir, "edited_render.mp4")
        validation_ref_video_path = os.path.join(root_dir, "editing_ori.png")
        input_video_path = os.path.join(root_dir, "original.mp4")
        input_video_mask = os.path.join(root_dir, "mask_box/box_render.mp4")

        output_dir = os.path.join(root_dir, "result")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # 1. Read the pointcloud video
        vr = VideoReader(uri=validation_pointcloud_video_path, height=-1, width=-1)
        ori_vlen = len(vr)
        temp_frms = vr.get_batch(np.arange(0, ori_vlen))
        tensor_frms = torch.from_numpy(temp_frms.asnumpy()) if type(temp_frms) is not torch.Tensor else temp_frms
        tensor_frms = tensor_frms.permute(3, 0, 1, 2)  # [T, H, W, C] -> [C, T, H, W]
        condition_pc_input = (tensor_frms - 127.5) / 127.5
        condition_pc_input = condition_pc_input.unsqueeze(0)
        
        # 2. Read the original video
        temp_frms = Image.open(validation_ref_video_path)
        temp_frms = torch.from_numpy(np.array(temp_frms)).unsqueeze(0)
        temp_frms = temp_frms[:,:,:,0:3]
        temp_frms = temp_frms.permute(3, 0, 1, 2)  # [T, H, W, C] -> [C, T, H, W]
        condition_ref_image_input = (temp_frms - 127.5) / 127.5
        condition_ref_image_input = condition_ref_image_input.unsqueeze(0)
        
        # 3. Read the input video
        vr = VideoReader(uri=input_video_path, height=-1, width=-1)
        ori_vlen = len(vr)
        temp_frms = vr.get_batch(np.arange(0, ori_vlen))
        tensor_frms = torch.from_numpy(temp_frms.asnumpy()) if type(temp_frms) is not torch.Tensor else temp_frms
        tensor_frms = tensor_frms.permute(3, 0, 1, 2)  # [T, H, W, C] -> [C, T, H, W]
        input_image_input = (tensor_frms - 127.5) / 127.5
        input_image_input = input_image_input.unsqueeze(0)
            
        # 4. Read the input mask
        vr = VideoReader(uri=input_video_mask, height=-1, width=-1)
        ori_vlen = len(vr)
        temp_frms = vr.get_batch(np.arange(0, ori_vlen))
        tensor_frms = torch.from_numpy(temp_frms.asnumpy()) if type(temp_frms) is not torch.Tensor else temp_frms
        tensor_frms = tensor_frms.permute(3, 0, 1, 2)  # [T, H, W, C] -> [C, T, H, W]
        input_mask_input = tensor_frms / 255
        input_mask_input = input_mask_input.unsqueeze(0)
        
        # 5. Read the caption
        with open(validation_prompts_path, "r") as f:  # 打开文件
            validation_prompt = f.read()  # 读取文件

        control_scale = 1.0

        front_path = os.path.join(output_dir, "60000_test_video_")
        back_path = str(seed) + "_g" + str(guidance_scale) + "_c" + str(control_scale) + ".mp4"
        output_path = front_path + back_path
        generator = torch.Generator().manual_seed(seed)
    
        # 2. Inference the video results
        video = pipeline(
            prompt=validation_prompt, # Text prompt 
            pc_image=condition_pc_input, # Control point cloud video
            ref_image=condition_ref_image_input, # Control ref images
            
            input_image=input_image_input, # input video
            input_mask=input_mask_input, # input mask video
            
            num_videos_per_prompt=1,  # Number of videos to generate per prompt
            num_inference_steps=50,  # Number of inference steps
            num_frames=49,  # Number of frames to generate，changed to 49 for diffusers version `0.31.0` and after.
            use_dynamic_cfg=True,  ## This id used for DPM Sechduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,  # Guidance scale for classifier-free guidance, can set to 7 for DPM scheduler
            generator=generator,  # Set the seed for reproducibility
            
            controlnet_conditioning_scale=control_scale,
        ).frames[0]
    
        export_to_video(video, output_path, fps=8)

