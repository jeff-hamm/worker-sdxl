from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence, Tuple
import torch
from PIL import Image
import numpy as np
import cv2
SDXL_SUPPORTED_RESOLUTIONS = [
    (1024, 1024, 1.0),
    (1152, 896, 1.2857142857142858),
    (896, 1152, 0.7777777777777778),
    (1216, 832, 1.4615384615384615),
    (832, 1216, 0.6842105263157895),
    (1344, 768, 1.75),
    (768, 1344, 0.5714285714285714),
    (1536, 640, 2.4),
    (640, 1536, 0.4166666666666667),
]

SDXL_EXTENDED_RESOLUTIONS = [
    (512, 2048, 0.25),
    (512, 1984, 0.26),
    (512, 1920, 0.27),
    (512, 1856, 0.28),
    (576, 1792, 0.32),
    (576, 1728, 0.33),
    (576, 1664, 0.35),
    (640, 1600, 0.4),
    (640, 1536, 0.42),
    (704, 1472, 0.48),
    (704, 1408, 0.5),
    (704, 1344, 0.52),
    (768, 1344, 0.57),
    (768, 1280, 0.6),
    (832, 1216, 0.68),
    (832, 1152, 0.72),
    (896, 1152, 0.78),
    (896, 1088, 0.82),
    (960, 1088, 0.88),
    (960, 1024, 0.94),
    (1024, 1024, 1.0),
    (1024, 960, 1.8),
    (1088, 960, 1.14),
    (1088, 896, 1.22),
    (1152, 896, 1.30),
    (1152, 832, 1.39),
    (1216, 832, 1.47),
    (1280, 768, 1.68),
    (1344, 768, 1.76),
    (1408, 704, 2.0),
    (1472, 704, 2.10),
    (1536, 640, 2.4),
    (1600, 640, 2.5),
    (1664, 576, 2.90),
    (1728, 576, 3.0),
    (1792, 576, 3.12),
    (1856, 512, 3.63),
    (1920, 512, 3.76),
    (1984, 512, 3.89),
    (2048, 512, 4.0),
]


class Resolution(ABC):
    @classmethod
    @abstractmethod
    def resolutions(cls) -> Sequence[Tuple[int, int, float]]: ...

    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {
            "required": {
                "resolution": ([f"{res[0]}x{res[1]}" for res in cls.resolutions()],)
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "op"
    CATEGORY = "math/graphics"

    def op(self, resolution: str) -> tuple[int, int]:
        width, height = resolution.split("x")
        return (int(width), int(height))


class NearestResolution(ABC):
    @classmethod
    @abstractmethod
    def resolutions(cls) -> Sequence[Tuple[int, int, float]]: ...

    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "op"
    CATEGORY = "math/graphics"

    def op(self, image) -> tuple[int, int]:
        return self.for_size(image.size()[2], image.size()[1])
        
    def for_size(self, image_width, image_height):
        print(f"Input image resolution: {image_width}x{image_height}")
        image_ratio = image_width / image_height
        differences = [
            (abs(image_ratio - resolution[2]), resolution)
            for resolution in self.resolutions()
        ]
        smallest = None
        for difference in differences:
            if smallest is None:
                smallest = difference
            else:
                if difference[0] < smallest[0]:
                    smallest = difference
        if smallest is not None:
            width = smallest[1][0]
            height = smallest[1][1]
        else:
            width = 1024
            height = 1024
        print(f"Selected resolution: {width}x{height}")
        return (width, height)
        

class SDXLResolution(Resolution):
    @classmethod
    def resolutions(cls):
        return SDXL_SUPPORTED_RESOLUTIONS


class SDXLExtendedResolution(Resolution):
    @classmethod
    def resolutions(cls):
        return SDXL_EXTENDED_RESOLUTIONS


class NearestSDXLResolution(NearestResolution):
    @classmethod
    def resolutions(cls):
        return SDXL_SUPPORTED_RESOLUTIONS


class NearestSDXLExtendedResolution(NearestResolution):
    @classmethod
    def resolutions(cls):
        return SDXL_EXTENDED_RESOLUTIONS


MAX_RESOLUTION=16384
def HWC3(x):
    assert x.dtype == np.uint8
    print(x.ndim)
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()

UPSCALE_METHODS = ["INTER_NEAREST", "INTER_LINEAR", "INTER_AREA", "INTER_CUBIC", "INTER_LANCZOS4"]
def get_upscale_method(method_str):
    assert method_str in UPSCALE_METHODS, f"Method {method_str} not found in {UPSCALE_METHODS}"
    return getattr(cv2, method_str)

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

def resize_image_with_pad(input_image, resolution, upscale_method = "", skip_hwc3=False, mode='edge'):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    if resolution == 0:
        return img, lambda x: x
    k = float(resolution) / float(min(H_raw, W_raw))
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=get_upscale_method(upscale_method) if k > 1 else cv2.INTER_AREA)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode=mode)

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target, ...])

    return safer_memory(img_padded), remove_pad
    
def bislerp(samples, width, height):
    def slerp(b1, b2, r):
        '''slerps batches b1, b2 according to ratio r, batches should be flat e.g. NxC'''
        
        c = b1.shape[-1]

        #norms
        b1_norms = torch.norm(b1, dim=-1, keepdim=True)
        b2_norms = torch.norm(b2, dim=-1, keepdim=True)

        #normalize
        b1_normalized = b1 / b1_norms
        b2_normalized = b2 / b2_norms

        #zero when norms are zero
        b1_normalized[b1_norms.expand(-1,c) == 0.0] = 0.0
        b2_normalized[b2_norms.expand(-1,c) == 0.0] = 0.0

        #slerp
        dot = (b1_normalized*b2_normalized).sum(1)
        omega = torch.acos(dot)
        so = torch.sin(omega)

        #technically not mathematically correct, but more pleasing?
        res = (torch.sin((1.0-r.squeeze(1))*omega)/so).unsqueeze(1)*b1_normalized + (torch.sin(r.squeeze(1)*omega)/so).unsqueeze(1) * b2_normalized
        res *= (b1_norms * (1.0-r) + b2_norms * r).expand(-1,c)

        #edge cases for same or polar opposites
        res[dot > 1 - 1e-5] = b1[dot > 1 - 1e-5] 
        res[dot < 1e-5 - 1] = (b1 * (1.0-r) + b2 * r)[dot < 1e-5 - 1]
        return res
def lanczos(samples, width, height):
    images = [Image.fromarray(np.clip(255. * image.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8)) for image in samples]
    images = [image.resize((width, height), resample=Image.Resampling.LANCZOS) for image in images]
    images = [torch.from_numpy(np.array(image).astype(np.float32) / 255.0).movedim(-1, 0) for image in images]
    result = torch.stack(images)
    return result.to(samples.device, samples.dtype)
    
def common_upscale(samples, width, height, upscale_method, crop):
        if crop == "center":
            old_width = samples.shape[3]
            old_height = samples.shape[2]
            old_aspect = old_width / old_height
            new_aspect = width / height
            x = 0
            y = 0
            if old_aspect > new_aspect:
                x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
            elif old_aspect < new_aspect:
                y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
            s = samples[:,:,y:old_height-y,x:old_width-x]
        else:
            s = samples

        if upscale_method == "bislerp":
            return bislerp(s, width, height)
        elif upscale_method == "lanczos":
            return lanczos(s, width, height)
        else:
            return torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)
        
class ImageScale:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",), "upscale_method": (s.upscale_methods,),
                              "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                              "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                              "crop": (s.crop_methods,)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"

    def upscale(self, image, upscale_method, width, height, crop):
        if width == 0 and height == 0:
            s = image
        else:
            samples = image.movedim(-1,1)

            if width == 0:
                width = max(1, round(samples.shape[3] * height / samples.shape[2]))
            elif height == 0:
                height = max(1, round(samples.shape[2] * width / samples.shape[3]))

            s = common_upscale(samples, width, height, upscale_method, crop)
            s = s.movedim(1,-1)
        return (s,)