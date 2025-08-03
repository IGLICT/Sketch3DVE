import cv2
import numpy as np
from decord import VideoReader

def pad_last_frame(tensor, num_frames):
    # T, H, W, C
    if tensor.shape[0] < num_frames:
        last_frame = tensor[-int(num_frames - tensor.shape[1]) :]
        padded_tensor = np.concatenate([tensor, last_frame], dim=0)
        return padded_tensor
    else:
        return tensor[:num_frames]

def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    if arr.shape[2] / arr.shape[1] > image_size[1] / image_size[0]:
        width_size = int(arr.shape[2] * image_size[0] / arr.shape[1])
        resized_images = np.zeros((arr.shape[0], image_size[0], width_size, 3), dtype=np.uint8)
        for i in range(arr.shape[0]):
            resized_images[i] = cv2.resize(arr[i], (width_size, image_size[0]))
    else:
        width_size = int(arr.shape[1] * image_size[1] / arr.shape[2])
        resized_images = np.zeros((arr.shape[0], width_size, image_size[1], 3), dtype=np.uint8)
        for i in range(arr.shape[0]):
            resized_images[i] = cv2.resize(arr[i], (image_size[1], width_size))

    h, w = resized_images.shape[1], resized_images.shape[2]

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = resized_images[:, top:top+image_size[0], left:left+image_size[1], :]
    return arr

def resize_video(video_path, fps=8, max_num_frames=49, skip_frms_num=0.):
    vr = VideoReader(uri=video_path, height=-1, width=-1)
    actual_fps = vr.get_avg_fps()
    ori_vlen = len(vr)
    
    if (ori_vlen - skip_frms_num) / actual_fps * fps > max_num_frames:
        print("ori_vlen:", ori_vlen)
        num_frames = max_num_frames
        start = int(skip_frms_num)
        interval = (ori_vlen - skip_frms_num) // num_frames
        end = int(start + interval * num_frames)
        end_safty = ori_vlen
        indices = np.arange(start, end, (end - start) // num_frames).astype(int)

        temp_frms = vr.get_batch(np.arange(start, end_safty))
        assert temp_frms is not None
        temp_frms = temp_frms.asnumpy()
        temp_frms = temp_frms[(indices - start).tolist()]
    else:
        if (ori_vlen - 2 * skip_frms_num) > max_num_frames:
            num_frames = max_num_frames
            start = int(skip_frms_num)
            end = int(ori_vlen - skip_frms_num)
            indices = np.arange(start, end, (end - start) // num_frames).astype(int)
            temp_frms = vr.get_batch(np.arange(start, end))
            assert temp_frms is not None
            temp_frms = temp_frms.asnumpy()
            temp_frms = temp_frms[(indices - start).tolist()]
        else:

            def nearest_smaller_4k_plus_1(n):
                remainder = n % 4
                if remainder == 0:
                    return n - 3
                else:
                    return n - remainder + 1

            start = int(skip_frms_num)
            end = int(ori_vlen - skip_frms_num)
            num_frames = nearest_smaller_4k_plus_1(
                end - start
            )  # 3D VAE requires the number of frames to be 4k+1
            end = int(start + num_frames)
            temp_frms = vr.get_batch(np.arange(start, end))
            assert temp_frms is not None
            indices = np.arange(start, end, (end - start) // num_frames).astype(int)
            temp_frms = temp_frms.asnumpy()
            temp_frms = temp_frms[(indices - start).tolist()]
            
    temp_frms = pad_last_frame(temp_frms, num_frames)
    temp_frms = resize_for_rectangle_crop(temp_frms, image_size=[480, 720], reshape_mode="center")
    return temp_frms
