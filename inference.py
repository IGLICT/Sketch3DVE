import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'extern'))

from sketch3dve import Sketch3DVE

import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    
    ## general 
    parser.add_argument('--root_dir', type=str, default='./output', help='Output directory')

    ## preprocess
    parser.add_argument(
        "--mode", type=str, choices=["full", "preprocess", "dust3r", "edit_pc", "mask_3D", "pred_text", "gen_video"],
        required=True, 
        help="choose the mode the inference scripts"
    )
    parser.add_argument('--crop_length', action='store_true', help='Change the length of the input video if frames are not 49')
    
    ## dustsr
    parser.add_argument('--dust3r_model_path', type=str, default='./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth', help='The path of the dust3r model')
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--schedule', type=str, default='linear')

    ## depth prediction, edited align
    parser.add_argument('--depthanything_model_path', type=str, default='./checkpoints/depth_anything_v2_vitl.pth', help='The path of the DepthAnything v2 model')
    parser.add_argument('--region_new_min', type=float, default=0.15, help='the minimum value of depth maps for edited regions')
    parser.add_argument('--region_new_max', type=float, default=0.6, help='the maxmum value of depth maps for edited regions')
    parser.add_argument('--dust3r_auto_align', action='store_true', help='utilize dust3r to predict depth map, then automatically align the edited regions')

    ## construct mask box
    parser.add_argument('--front_offset', type=float, default=0.01, help='the offset of depth maps for front faces of 3D mesh box')
    parser.add_argument('--mask_depth_max', type=float, default=0.6, help='the values of depth maps for back faces of 3D mesh box')
    
    ## Llava text detect
    parser.add_argument('--Llava_model_path', type=str, default='./checkpoints/llava-v1.6-mistral-7b-hf', help='The path of the Llava model')
    
    ## video diffusion
    parser.add_argument("--controlnet_ckpt_path", type=str, default='./checkpoints/controlnet', help="checkpoint path for controlnet")
    parser.add_argument("--basemodel_ckpt_path", type=str, default='./checkpoints/CogVideoX-2b', help="checkpoint path for based video diffusion")
    parser.add_argument("--cfg", type=float, default=10.0, help="prompt classifier-free guidance")
    parser.add_argument("--seed", type=int, default=123, help="seed for seed_everything")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt")

    return parser

if __name__=="__main__":
    parser = get_parser()
    opts = parser.parse_args()

    root_dir = opts.root_dir
    video_path = os.path.join(root_dir, "original.mp4")

    sketch3dve = Sketch3DVE(opts)

    if opts.mode == "preprocess":
        sketch3dve.preprocess_video(video_path)
    
    if opts.mode == "dust3r":
        sketch3dve.run_dust3r(root_dir)
    
    if opts.mode == "edit_pc":
        if opts.dust3r_auto_align:
            sketch3dve.run_align_depth_dust3r(root_dir)
        else:
            edit_img_path = os.path.join(root_dir, 'editing.png')
            depth_dir = os.path.join(root_dir, 'depth_anything')
            sketch3dve.run_DepthAnythingv2(edit_img_path, depth_dir)
            sketch3dve.run_align_depth(root_dir, region_new_min=opts.region_new_min, region_new_max=opts.region_new_max)
        sketch3dve.render_edited_pc(root_dir)

    if opts.mode == "mask_3D":
        sketch3dve.construct_3D_mesh(root_dir, front_offset=opts.front_offset, depth_max=opts.mask_depth_max)
    
    if opts.mode == "pred_text":
        sketch3dve.run_Llava(root_dir)
    
    if opts.mode == "gen_video":
        sketch3dve.run_diffusion(root_dir, opts.seed, opts.cfg)
    
    if opts.mode == "full":
        sketch3dve.preprocess_video(video_path)
        
        # We use existing dust3t results to ensure reproduction of examples
        # sketch3dve.run_dust3r(root_dir)
        
        # Hyperparameters should be tuned for "region_new_min" and "region_new_max"
        if opts.dust3r_auto_align:
            sketch3dve.run_align_depth_dust3r(root_dir)
        else:
            edit_img_path = os.path.join(root_dir, 'editing.png')
            depth_dir = os.path.join(root_dir, 'depth_anything')
            sketch3dve.run_DepthAnythingv2(edit_img_path, depth_dir)
            sketch3dve.run_align_depth(root_dir, region_new_min=opts.region_new_min, region_new_max=opts.region_new_max)
        
        sketch3dve.render_edited_pc(root_dir)

        # Hyperparameters should be tuned for "front_offset" and "mask_depth_max"
        sketch3dve.construct_3D_mesh(root_dir, front_offset=opts.front_offset, depth_max=opts.mask_depth_max)
    
        sketch3dve.run_Llava(root_dir)

        sketch3dve.run_diffusion(root_dir, opts.seed, opts.cfg)
    

