from controlnet import CANNY_SCHEMA
CONTROLNET_SCHEMA = CANNY_SCHEMA.copy()
CONTROLNET_SCHEMA['conditioning_scale'] = {
    'type': float,
    'required': False,
    'default': 0.5
}
CONTROLNET_SCHEMA['image_resolution']['default'] = 512
CONTROLNET_SCHEMA['model'] = {
    'type': str,
    'required': False,
    'default': 'canny'
}

INPUT_SCHEMA = {
    'image_id': {
        'type': str,
        'required':False,
        'default': None
    },
    'use_refiner': {
        'type': bool,
        'required': False,
        'default': None
    },
    'model_type': {
        'type': str,
        'required':False,
        'default': None,
    },
    'prompt': {
        'type': str,
        'required': False,
    },
    'negative_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'height': {
        'type': int,
        'required': False,
        'default': 0
    },
    'width': {
        'type': int,
        'required': False,
        'default': 0
    },
    'seed': {
        'type': int,
        'required': False,
        'default': None
    },
    'scheduler': {
        'type': str,
        'required': False,
        'default': None
    },
    'num_inference_steps': {
        'type': int,
        'required': False,
        'default': 12
    },
    'refiner_inference_steps': {
        'type': int,
        'required': False,
        'default': None
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 4.0
    },
    'strength': {
        'type': float,
        'required': False,
        'default': 0.3
    },
    'image_url': {
        'type': str,
        'required': False,
        'default': None
    },
    'num_images': {
        'type': int,
        'required': False,
        'default': 1,
        'constraints': lambda img_count: 3 > img_count > 0
    },
    'high_noise_frac': {
        'type': float,
        'required': False,
        'default': 0.8
    },
    'controlnet_type': {
        'type': str,
        'required': False,
        'default': None
    },
    'controlnet_image_resolution': {
        'type': int,
        'required': False,
        'default': None
    },
    "controlnet_conditioning_scale": {
        'type': float,
        'required': False,
        'default': 0.7
    },
    'controlnet_low_threshold': {'type': int, 'required': False, 'default': 100, 'constraints': lambda threshold: 1 < threshold < 255},
    'controlnet_high_threshold': {'type': int, 'required': False, 'default': 200, 'constraints': lambda threshold: 1 < threshold < 255},
    'controlnet_image_url': {'type': str, 'required': False, 'default': None},
}
