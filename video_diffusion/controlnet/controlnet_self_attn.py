from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.utils import BaseOutput, logging, is_torch_version
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock, CogVideoXTransformer3DModel

from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import AttentionProcessor, CogVideoXAttnProcessor2_0, FusedCogVideoXAttnProcessor2_0
from diffusers.utils.torch_utils import maybe_allow_in_graph

from einops import rearrange, repeat
from safetensors.torch import load_file

class InputMappingBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hint_count: int = 3,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
    ):
        super().__init__()

        self.ff = FeedForward(
            dim * hint_count,
            dim_out = dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        
        self.out = zero_module(nn.Linear(dim, dim))

    def forward(
        self,
        control_hidden_states: torch.Tensor,
    ): # -> tuple(torch.Tensor, torch.Tensor):
        
        # feed-forward
        control_hidden_states = self.ff(control_hidden_states)
        output_hidden_states = self.out(control_hidden_states)

        return output_hidden_states, control_hidden_states

@dataclass
class ControlNetOutput(BaseOutput):
    """
    The output of [`ControlNetModel`].

    Args:
        control_block_res_samples (`tuple[torch.Tensor]`):
            A tuple of downsample activations at different resolutions for each downsampling block. Each tensor should
            be of shape `(batch_size, channel * resolution, height //resolution, width // resolution)`. Output can be
            used to condition the original UNet's downsampling activations.
    """

    control_block_res_samples: Tuple[torch.Tensor]

class CogVideoControlNetModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    @register_to_config
    def __init__(
        self,
        hint_count = 3, 
        control_block_index: list[int] = [0,6,12,18,24],
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        time_embed_dim: int = 512,
        dropout: float = 0.0,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        self.sketch_transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(len(control_block_index))
            ]
        )
        
        self.input_mapping_block = InputMappingBlock(inner_dim, hint_count=hint_count, ff_inner_dim=inner_dim*8)
        
        self.controlnet_blocks = nn.ModuleList(
            [
                zero_module(nn.Linear(inner_dim, inner_dim))
                for _ in range(len(control_block_index))
            ]
        )
        
        self.control_block_index = control_block_index
        # mapping the Cogvideo block index into Controlnet block index
        self.control_block_dict = {}
        for i in range(len(control_block_index)):
            self.control_block_dict[control_block_index[i]] = i
        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, value=False):
        self.gradient_checkpointing = value
    
    @classmethod
    def from_transformer(
        cls,
        transformer3D: CogVideoXTransformer3DModel,
        hint_count: int = 3,
        load_weights_from_transformer: bool = True,
        # control_block_index: list[int] = [0,5,10,15,20,25],
        control_block_index: list[int] = [0,6,12,18,24],
        load_ckpt_path: str=None,
    ):
        r"""
        Instantiate a [`ControlNetModel`] from [`CogVideoXTransformer3DModel`].

        Parameters:
            unet (`CogVideoXTransformer3DModel`):
                The Transformer model weights to copy to the [`ControlNetModel`]. All configuration options are also copied
                where applicable.
        """
        
        print("control_block_index:", control_block_index)
        
        controlnet = cls(
            hint_count = hint_count,
            control_block_index = control_block_index,
            num_attention_heads = transformer3D.config.num_attention_heads,
            attention_head_dim = transformer3D.config.attention_head_dim,
            time_embed_dim = transformer3D.config.time_embed_dim,
            dropout = transformer3D.config.dropout,
            norm_elementwise_affine = transformer3D.config.norm_elementwise_affine,
            norm_eps = transformer3D.config.norm_eps,
            attention_bias = transformer3D.config.attention_bias,
            activation_fn = transformer3D.config.activation_fn,
        )

        if load_weights_from_transformer:
            print("load controlnet from transformer")
            # controlnet.patch_embed.load_state_dict(transformer3D.patch_embed.state_dict(), strict=False)
            
            for i in range(len(controlnet.sketch_transformer_blocks)):
                controlnet.sketch_transformer_blocks[i].load_state_dict(transformer3D.transformer_blocks[control_block_index[i]].state_dict())
        
        # Load the checkpoints from the pretrained sketch-based generation model
        if load_ckpt_path is not None:
            print("load controlnet load_ckpt_path")
            weights_dict = load_file(load_ckpt_path)
            for k,v in weights_dict.items():
                if k.startswith("input_mapping_block") and k.endswith("ff.net.0.proj.weight"):
                    weights_dict[k] = torch.concat([v, torch.zeros(15360, 1920)], dim=1)
        controlnet.load_state_dict(weights_dict, strict=False)
        
        return controlnet
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        sketch_hidden_states: torch.Tensor,
        num_frames: int,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_dict: bool = True,
        block_index: int = None,
    ):
        """
        The input should be processed by CogVideoXTransformer3DModel
        hidden_states: CogVideo Transformer block output, after "Patch embedding, Position embedding"
        control_hidden_states: sketch features, after "Patch embedding, Position embedding"
        encoder_hidden_states: encoder output
        emb:
            After "Time embedding"
        """
        
        # Get the block index
        control_block_index = self.control_block_dict[block_index]
        
        if control_block_index == 0:
            # 1st block, Mapping the sketch condition, zero initization
            sketch_out_hidden_states, _ = self.input_mapping_block(sketch_hidden_states)
            # add residual features
            sketch_out_hidden_states += hidden_states
        else:
            sketch_out_hidden_states = sketch_hidden_states
        
        if self.training and self.gradient_checkpointing:
            sketch_block = self.transformer_blocks[control_block_index]
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            sketch_out_hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(sketch_block),
                sketch_out_hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                **ckpt_kwargs,
            )
            control_out_block = self.controlnet_blocks[control_block_index]
            residual_features = torch.utils.checkpoint.checkpoint(
                create_custom_forward(control_out_block),
                sketch_out_hidden_states
            )
        else:
            # Sketch mapping 
            sketch_block = self.sketch_transformer_blocks[control_block_index]
            sketch_out_hidden_states, encoder_hidden_states = sketch_block(
                hidden_states=sketch_out_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
            # control zero Linear
            residual_features = self.controlnet_blocks[control_block_index](sketch_out_hidden_states)
            
        return residual_features, sketch_out_hidden_states # , image_hidden_states

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

def identity_module(module):
    module.apply(weights_init)
    return module

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.eye_(m.weight)
        nn.init.constant_(m.bias, 0.0)

