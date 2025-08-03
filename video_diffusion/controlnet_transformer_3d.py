from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from diffusers.utils import is_torch_version
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel

from einops import rearrange, repeat

def add_method(cls):
    def decorator(func):
        setattr(cls, func.__name__, func)
        return func
    return decorator


@add_method(CogVideoXTransformer3DModel)
def get_embedding(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
    
    batch_size, num_frames, channels, height, width = hidden_states.shape

    # 1. Time embedding
    timesteps = timestep
    t_emb = self.time_proj(timesteps)

    # timesteps does not contain any weights and will always return f32 tensors
    # but time_embedding might actually be running in fp16. so we need to cast here.
    # there might be better ways to encapsulate this.
    t_emb = t_emb.to(dtype=hidden_states.dtype)
    emb = self.time_embedding(t_emb, timestep_cond)

    # 2. Patch embedding
    hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)

    # 3. Position embedding
    text_seq_length = encoder_hidden_states.shape[1]
    if not self.config.use_rotary_positional_embeddings:
        seq_length = height * width * num_frames // (self.config.patch_size**2)

        pos_embeds = self.pos_embedding[:, : text_seq_length + seq_length]
        hidden_states = hidden_states + pos_embeds
        hidden_states = self.embedding_dropout(hidden_states)

    encoder_hidden_states = hidden_states[:, :text_seq_length]
    hidden_states = hidden_states[:, text_seq_length:]
    
    return emb, encoder_hidden_states, hidden_states


@add_method(CogVideoXTransformer3DModel)
def get_embedding_frame(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        control_frame_index: Tuple[int],
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
    
    max_num_frames = 13
    batch_size, num_frames, channels, height, width = hidden_states.shape

    # 1. Time embedding
    timesteps = timestep
    t_emb = self.time_proj(timesteps)

    # timesteps does not contain any weights and will always return f32 tensors
    # but time_embedding might actually be running in fp16. so we need to cast here.
    # there might be better ways to encapsulate this.
    t_emb = t_emb.to(dtype=hidden_states.dtype)
    emb = self.time_embedding(t_emb, timestep_cond)

    # 2. Patch embedding
    hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)

    # 3. Position embedding
    text_seq_length = encoder_hidden_states.shape[1]
    if not self.config.use_rotary_positional_embeddings:
        seq_length = height * width * max_num_frames // (self.config.patch_size**2)
        # Get Text and image embedding
        pos_embeds = self.pos_embedding[:, : text_seq_length + seq_length]
        # Split text embedding
        encoder_pos_embeds = pos_embeds[:, :text_seq_length]
        # Split image embedding, and index the embedding features
        image_pos_embeds = pos_embeds[:, text_seq_length:]
        image_pos_embeds = rearrange(image_pos_embeds, 'B (T L) C -> B T L C', T=max_num_frames)
        # For batch > 1, each batch should be independently process
        image_pos_embeds_list = []
        for i in range(batch_size):
            batch_frame_index = control_frame_index[i]
            image_pos_embeds_list.append(image_pos_embeds[0, batch_frame_index])
        image_pos_embeds = torch.stack(image_pos_embeds_list)
        image_pos_embeds = rearrange(image_pos_embeds, 'B T L C -> B (T L) C')
        # Add text and image embedding
        encoder_hidden_states = hidden_states[:, :text_seq_length] + encoder_pos_embeds
        # print("image_pos_embeds:", image_pos_embeds.shape)
        # print("hidden_states[:, text_seq_length:]:", hidden_states[:, text_seq_length:].shape)
        hidden_states = hidden_states[:, text_seq_length:] + image_pos_embeds
        # Add Dropout
        encoder_hidden_states = self.embedding_dropout(encoder_hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)
    else:
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

    return emb, encoder_hidden_states, hidden_states


@add_method(CogVideoXTransformer3DModel)
def forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor, 
    timestep: Union[int, float, torch.LongTensor],
    controlNet: torch.nn.Module,
    control_hidden_states: Optional[Tuple[torch.Tensor]] = None,
    timestep_cond: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    return_dict: bool = True,
    conditioning_scale: float = 1.0,
    height: int = 60,
    width: int = 90,
    channels: int = 16, 
):
    batch_size, num_frames, _ = hidden_states.shape
    num_frames = int(num_frames / (height * width / self.config.patch_size / self.config.patch_size ))
    text_seq_length = encoder_hidden_states.shape[1]

    # 1. Time embedding
    timesteps = timestep
    t_emb = self.time_proj(timesteps)

    # timesteps does not contain any weights and will always return f32 tensors
    # but time_embedding might actually be running in fp16. so we need to cast here.
    # there might be better ways to encapsulate this.
    t_emb = t_emb.to(dtype=hidden_states.dtype)
    emb = self.time_embedding(t_emb, timestep_cond)
    
    # Get ControlNet block index
    if isinstance(controlNet, torch.nn.parallel.DistributedDataParallel):
        control_block_index = controlNet.module.control_block_index
    else:
        control_block_index = controlNet.control_block_index
    
    # 2. Transformer blocks
    for i, block in enumerate(self.transformer_blocks):
        # Add Control residual features
        if i in control_block_index:
            residual_features, control_hidden_states = controlNet(
                hidden_states=hidden_states, 
                encoder_hidden_states=encoder_hidden_states, 
                sketch_hidden_states=control_hidden_states, 
                num_frames=num_frames,
                temb=emb,
                image_rotary_emb=image_rotary_emb,
                block_index=i,
            )
        
        if self.training and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                encoder_hidden_states,
                emb,
                image_rotary_emb,
                **ckpt_kwargs,
            )
        else:
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=emb,
                image_rotary_emb=image_rotary_emb,
            )
        
        # Add Control residual features
        if i in control_block_index:
            hidden_states += residual_features * conditioning_scale
    
    if not self.config.use_rotary_positional_embeddings:
        # CogVideoX-2B
        hidden_states = self.norm_final(hidden_states)
    else:
        # CogVideoX-5B
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        hidden_states = self.norm_final(hidden_states)
        hidden_states = hidden_states[:, text_seq_length:]

    # 3. Final block
    hidden_states = self.norm_out(hidden_states, temb=emb)
    hidden_states = self.proj_out(hidden_states)

    # 4. Unpatchify
    p = self.config.patch_size
    output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, channels, p, p)
    output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)

