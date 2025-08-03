"""
This script is designed to demonstrate how to use the CogVideoX-2b VAE model for video encoding and decoding.
It allows you to encode a video into a latent representation, decode it back into a video, or perform both operations sequentially.
Before running the script, make sure to clone the CogVideoX Hugging Face model repository and set the `{your local diffusers path}` argument to the path of the cloned repository.

Command 1: Encoding Video
Encodes the video located at ../resources/videos/1.mp4 using the CogVideoX-2b VAE model.
Memory Usage: ~34GB of GPU memory for encoding.
If you do not have enough GPU memory, we provide a pre-encoded tensor file (encoded.pt) in the resources folder and you can still run the decoding command.
$ python cli_vae_demo.py --model_path {your local diffusers path}/CogVideoX-2b/vae/ --video_path ../resources/videos/1.mp4 --mode encode

Command 2: Decoding Video

Decodes the latent representation stored in encoded.pt back into a video.
Memory Usage: ~19GB of GPU memory for decoding.
$ python cli_vae_demo.py --model_path {your local diffusers path}/CogVideoX-2b/vae/ --encoded_path ./encoded.pt --mode decode

Command 3: Encoding and Decoding Video
Encodes the video located at ../resources/videos/1.mp4 and then immediately decodes it.
Memory Usage: 34GB for encoding + 19GB for decoding (sequentially).
$ python cli_vae_demo.py --model_path {your local diffusers path}/CogVideoX-2b/vae/ --video_path ../resources/videos/1.mp4 --mode both
"""

import argparse
import torch
import imageio
import numpy as np
from diffusers import AutoencoderKLCogVideoX
from torchvision import transforms

from typing import Optional, Tuple, Union
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution, DecoderOutput

from diffusers.utils.accelerate_utils import apply_forward_hook

def add_method(cls):
    def decorator(func):
        setattr(cls, func.__name__, func)
        return func
    return decorator

@add_method(AutoencoderKLCogVideoX)
def tiled_encode(self, x: torch.Tensor, return_dict: bool = True) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
    r"""
    Encode a batch of images using a tiled encoder.

    Args:
        x (`torch.Tensor`): Input batch of images.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

    Returns:
            The latent representations of the encoded images. If `return_dict` is True, a
            [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
    """
    # Rough memory assessment:
    #   - In CogVideoX-2B, there are a total of 24 CausalConv3d layers.
    #   - The biggest intermediate dimensions are: [1, 128, 9, 480, 720].
    #   - Assume fp16 (2 bytes per value).
    # Memory required: 1 * 128 * 9 * 480 * 720 * 24 * 2 / 1024**3 = 17.8 GB
    #
    # Memory assessment when using tiling:
    #   - Assume everything as above but now HxW is 240x360 by tiling in half
    # Memory required: 1 * 128 * 9 * 240 * 360 * 24 * 2 / 1024**3 = 4.5 GB

    batch_size, num_channels, num_frames, height, width = x.shape

    overlap_height = int(self.tile_sample_min_height * (1 - self.tile_overlap_factor_height))
    overlap_width = int(self.tile_sample_min_width * (1 - self.tile_overlap_factor_width))
    blend_extent_height = int(self.tile_latent_min_height * self.tile_overlap_factor_height)
    blend_extent_width = int(self.tile_latent_min_width * self.tile_overlap_factor_width)
    row_limit_height = self.tile_latent_min_height - blend_extent_height
    row_limit_width = self.tile_latent_min_width - blend_extent_width
    frame_batch_size = self.num_latent_frames_batch_size * 4

    # Split z into overlapping tiles and decode them separately.
    # The tiles have an overlap to avoid seams between tiles.
    rows = []
    for i in range(0, height, overlap_height):
        row = []
        for j in range(0, width, overlap_width):
            # if num_frames == 1:
            if num_frames < frame_batch_size:
                # specifically process image
                tile = x[
                    :,
                    :,
                    :,
                    i : i + self.tile_sample_min_height,
                    j : j + self.tile_sample_min_width,
                ]
                tile = self.encoder(tile)
                if self.quant_conv is not None:
                    tile = self.quant_conv(tile)
                self._clear_fake_context_parallel_cache()
                row.append(tile)
                continue
        
            time = []
            for k in range(num_frames // frame_batch_size):
                remaining_frames = num_frames % frame_batch_size
                start_frame = frame_batch_size * k + (0 if k == 0 else remaining_frames)
                end_frame = frame_batch_size * (k + 1) + remaining_frames
                tile = x[
                    :,
                    :,
                    start_frame:end_frame,
                    i : i + self.tile_sample_min_height,
                    j : j + self.tile_sample_min_width,
                ]
                tile = self.encoder(tile)
                if self.quant_conv is not None:
                    tile = self.quant_conv(tile)
                time.append(tile)
            self._clear_fake_context_parallel_cache()
            row.append(torch.cat(time, dim=2))
        rows.append(row)

    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            # blend the above tile and the left tile
            # to the current tile and add the current tile to the result row
            if i > 0:
                tile = self.blend_v(rows[i - 1][j], tile, blend_extent_height)
            if j > 0:
                tile = self.blend_h(row[j - 1], tile, blend_extent_width)
            result_row.append(tile[:, :, :, :row_limit_height, :row_limit_width])
        result_rows.append(torch.cat(result_row, dim=4))

    moments = torch.cat(result_rows, dim=3)
    posterior = DiagonalGaussianDistribution(moments)
    
    if not return_dict:
        return (posterior,)
    return AutoencoderKLOutput(latent_dist=posterior)

@add_method(AutoencoderKLCogVideoX)
def tiled_decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Decode a batch of images using a tiled decoder.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        # Rough memory assessment:
        #   - In CogVideoX-2B, there are a total of 24 CausalConv3d layers.
        #   - The biggest intermediate dimensions are: [1, 128, 9, 480, 720].
        #   - Assume fp16 (2 bytes per value).
        # Memory required: 1 * 128 * 9 * 480 * 720 * 24 * 2 / 1024**3 = 17.8 GB
        #
        # Memory assessment when using tiling:
        #   - Assume everything as above but now HxW is 240x360 by tiling in half
        # Memory required: 1 * 128 * 9 * 240 * 360 * 24 * 2 / 1024**3 = 4.5 GB

        batch_size, num_channels, num_frames, height, width = z.shape

        overlap_height = int(self.tile_latent_min_height * (1 - self.tile_overlap_factor_height))
        overlap_width = int(self.tile_latent_min_width * (1 - self.tile_overlap_factor_width))
        blend_extent_height = int(self.tile_sample_min_height * self.tile_overlap_factor_height)
        blend_extent_width = int(self.tile_sample_min_width * self.tile_overlap_factor_width)
        row_limit_height = self.tile_sample_min_height - blend_extent_height
        row_limit_width = self.tile_sample_min_width - blend_extent_width
        frame_batch_size = self.num_latent_frames_batch_size

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, overlap_height):
            row = []
            for j in range(0, width, overlap_width):
                if num_frames < frame_batch_size:
                    tile = z[
                        :,
                        :,
                        :,
                        i : i + self.tile_latent_min_height,
                        j : j + self.tile_latent_min_width,
                    ]
                    if self.post_quant_conv is not None:
                        tile = self.post_quant_conv(tile)
                    tile = self.decoder(tile)
                    self._clear_fake_context_parallel_cache()
                    row.append(tile)
                    continue
                
                time = []
                for k in range(num_frames // frame_batch_size):
                    remaining_frames = num_frames % frame_batch_size
                    start_frame = frame_batch_size * k + (0 if k == 0 else remaining_frames)
                    end_frame = frame_batch_size * (k + 1) + remaining_frames
                    tile = z[
                        :,
                        :,
                        start_frame:end_frame,
                        i : i + self.tile_latent_min_height,
                        j : j + self.tile_latent_min_width,
                    ]
                    if self.post_quant_conv is not None:
                        tile = self.post_quant_conv(tile)
                    tile = self.decoder(tile)
                    time.append(tile)
                self._clear_fake_context_parallel_cache()
                row.append(torch.cat(time, dim=2))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent_width)
                result_row.append(tile[:, :, :, :row_limit_height, :row_limit_width])
            result_rows.append(torch.cat(result_row, dim=4))

        dec = torch.cat(result_rows, dim=3)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

@apply_forward_hook
@add_method(AutoencoderKLCogVideoX)
def encode(
    self, x: torch.Tensor, return_dict: bool = True
) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
    """
    Encode a batch of images into latents.

    Args:
        x (`torch.Tensor`): Input batch of images.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

    Returns:
            The latent representations of the encoded images. If `return_dict` is True, a
            [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
    """
    
    # batch_size, num_channels, num_frames, height, width = x.shape
    height, width = x.shape[-2], x.shape[-1]
    if self.use_tiling and (width > self.tile_latent_min_width or height > self.tile_latent_min_height):
        return self.tiled_encode(x, return_dict=return_dict)
    
    h = self.encoder(x)
    if self.quant_conv is not None:
        h = self.quant_conv(h)
    posterior = DiagonalGaussianDistribution(h)
    if not return_dict:
        return (posterior,)
    return AutoencoderKLOutput(latent_dist=posterior)

def encode_video(model_path, video_path, dtype, device):
    """
    Loads a pre-trained AutoencoderKLCogVideoX model and encodes the video frames.

    Parameters:
    - model_path (str): The path to the pre-trained model.
    - video_path (str): The path to the video file.
    - dtype (torch.dtype): The data type for computation.
    - device (str): The device to use for computation (e.g., "cuda" or "cpu").

    Returns:
    - torch.Tensor: The encoded video frames.
    """
    model = AutoencoderKLCogVideoX.from_pretrained(model_path, torch_dtype=dtype).to(device)
    model.enable_slicing()
    model.enable_tiling()
    video_reader = imageio.get_reader(video_path, "ffmpeg")

    frames = [transforms.ToTensor()(frame) for frame in video_reader]
    video_reader.close()

    frames_tensor = torch.stack(frames).to(device).permute(1, 0, 2, 3).unsqueeze(0).to(dtype)
    print(frames_tensor.shape)

    with torch.no_grad():
        encoded_frames = model.encode(frames_tensor)[0].sample()
        # encoded_frames = model.tiled_encode(frames_tensor)[0].sample()
    return encoded_frames


def decode_video(model_path, encoded_tensor_path, dtype, device):
    """
    Loads a pre-trained AutoencoderKLCogVideoX model and decodes the encoded video frames.

    Parameters:
    - model_path (str): The path to the pre-trained model.
    - encoded_tensor_path (str): The path to the encoded tensor file.
    - dtype (torch.dtype): The data type for computation.
    - device (str): The device to use for computation (e.g., "cuda" or "cpu").

    Returns:
    - torch.Tensor: The decoded video frames.
    """
    model = AutoencoderKLCogVideoX.from_pretrained(model_path, torch_dtype=dtype).to(device)
    model.enable_tiling()
    encoded_frames = torch.load(encoded_tensor_path, weights_only=True).to(device).to(dtype)
    with torch.no_grad():
        decoded_frames = []
        # for i in range(6):  # 6 seconds
        #     start_frame, end_frame = (0, 3) if i == 0 else (2 * i + 1, 2 * i + 3)
        #     current_frames = model.decode(encoded_frames[:, :, start_frame:end_frame]).sample
        #     decoded_frames.append(current_frames)
        # model.clear_fake_context_parallel_cache()

        decoded_frames = model.decode(encoded_frames).sample
    return decoded_frames


def save_video(tensor, output_path):
    """
    Saves the video frames to a video file.

    Parameters:
    - tensor (torch.Tensor): The video frames tensor.
    - output_path (str): The path to save the output video.
    """
    frames = tensor[0].squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    frames = np.clip(frames, 0, 1) * 255
    frames = frames.astype(np.uint8)

    writer = imageio.get_writer(output_path + "/output.mp4", fps=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CogVideoX encode/decode demo")
    parser.add_argument("--model_path", type=str, required=True, help="The path to the CogVideoX model")
    parser.add_argument("--video_path", type=str, help="The path to the video file (for encoding)")
    parser.add_argument("--encoded_path", type=str, help="The path to the encoded tensor file (for decoding)")
    parser.add_argument("--output_path", type=str, default=".", help="The path to save the output file")
    parser.add_argument(
        "--mode", type=str, choices=["encode", "decode", "both"], required=True, help="Mode: encode, decode, or both"
    )
    parser.add_argument(
        "--dtype", type=str, default="float16", help="The data type for computation (e.g., 'float16' or 'float32')"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="The device to use for computation (e.g., 'cuda' or 'cpu')"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    if args.mode == "encode":
        assert args.video_path, "Video path must be provided for encoding."
        encoded_output = encode_video(args.model_path, args.video_path, dtype, device)
        torch.save(encoded_output, args.output_path + "/encoded.pt")
        print(f"Finished encoding the video to a tensor, save it to a file at {encoded_output}/encoded.pt")
    elif args.mode == "decode":
        assert args.encoded_path, "Encoded tensor path must be provided for decoding."
        decoded_output = decode_video(args.model_path, args.encoded_path, dtype, device)
        save_video(decoded_output, args.output_path)
        print(f"Finished decoding the video and saved it to a file at {args.output_path}/output.mp4")
    elif args.mode == "both":
        assert args.video_path, "Video path must be provided for encoding."
        encoded_output = encode_video(args.model_path, args.video_path, dtype, device)
        torch.save(encoded_output, args.output_path + "/encoded.pt")
        decoded_output = decode_video(args.model_path, args.output_path + "/encoded.pt", dtype, device)
        save_video(decoded_output, args.output_path)

