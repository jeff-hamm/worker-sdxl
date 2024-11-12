import os
from subprocess import call

model_dl_urls = {
    "canny": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth",
    "depth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_depth.pth",
    "hed": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_hed.pth",
    "normal": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_normal.pth",
    "mlsd": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_mlsd.pth",
    "openpose": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_openpose.pth",
    "scribble": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_scribble.pth",
    "seg": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_seg.pth",
}

annotator_dl_urls = {
    "body_pose_model.pth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/body_pose_model.pth",
    "dpt_hybrid-midas-501f0c75.pt": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt",
    "hand_pose_model.pth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/hand_pose_model.pth",
    "mlsd_large_512_fp32.pth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_large_512_fp32.pth",
    "mlsd_tiny_512_fp32.pth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_tiny_512_fp32.pth",
    "network-bsds500.pth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/network-bsds500.pth",
    "upernet_global_small.pth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/upernet_global_small.pth",
}


def download_model(model_name, urls_map):
    """
    Download model from huggingface with wget and save to models directory
    """
    model_url = urls_map[model_name]
    relative_path_to_model = model_url.replace(
        "https://huggingface.co/lllyasviel/ControlNet/resolve/main/", "")
    if not os.path.exists(relative_path_to_model):
        print(f"Downloading {model_name}...")
        call(["wget", "-O", relative_path_to_model, model_url])


def get_state_dict_path(model_name):
    """
    Get path to model state dict
    """
    return f"./models/control_sd15_{model_name}.pth"


import numpy as np
import cv2
import os
from diffusers.utils import load_image
import base64
from io import BytesIO
import PIL
def get_image_from_url(image):
    '''
    Get the image from the provided URL or base64 string.
    Returns a PIL image.
    '''
    if image.startswith("http://") or image.startswith("https://") or os.path.isfile(image):
        image = load_image(image).convert("RGB")
        return np.array(image)
    elif image.startswith("data:"): 
        print ("Found dataUrlString")
        image = image.split(",")[1]
    else:
        print("No dataUrl prefix found. Attempting to decode")
    image_bytes = base64.b64decode(image)
    image = BytesIO(image_bytes)
    input_image_pil = PIL.Image.open(image).convert("RGB")
    input_image = np.array(input_image_pil)
    return (input_image,input_image_pil)


annotator_ckpts_path = os.path.join(os.path.dirname(__file__), 'ckpts')


def HWC3(x):
    assert x.dtype == np.uint8
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


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img