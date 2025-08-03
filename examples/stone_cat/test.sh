cd ../../

python inference.py --root_dir="./examples/stone_cat/" \
--mode "full" \
--dust3r_model_path "/home/jovyan/old/liufenglin/code/ViewCrafter_58/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \
--depthanything_model_path "/home/jovyan/old/liufenglin/code/CogVideo/controlnet_3_2/extern/Depth-Anything-V2-main/checkpoints/depth_anything_v2_vitl.pth" \
--Llava_model_path "/home/jovyan/old/liufenglin/Diffusion_models/llava-v1.6-mistral-7b-hf" \
--basemodel_ckpt_path '/home/jovyan/data/liufenglin/Diffusion_models/CogVideoX-2b' \
--controlnet_ckpt_path '/home/jovyan/old/liufenglin/code/CogVideo/viewcrafter_editing/control-ini-new/viewcrafter_editing_10_blocks/checkpoint-15000/controlnet' \
--region_new_min 0.14 \
--region_new_max 0.65 \
--front_offset 0.02 \
--mask_depth_max 0.5 \
--seed 44 \
