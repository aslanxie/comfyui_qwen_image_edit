from inspect import cleandoc
import base64
import os
import numpy as np
import torch
from PIL import Image
import io
import httpx
from openai import OpenAI
import random

# ANSI escape codes for colors
RED = "\033[91m"
RESET = "\033[0m"

class QWenImageEdit:
    """
    A node for generating or editing images, compatible with OpenAI Image API style, supporting Qwen-Image-Edit-2509 model.
    """

    def __init__(self):
        pass

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return random.random()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful image"
                }),
                 "base_url": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "model": (["Qwen-Image-Edit-2509"],),
                "size": (["1024x1024", "1536x1024", "1024x1536"],),
                "quality": (["low", "medium", "high"],),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "image/OpenAI"

    def generate_image(self, prompt, base_url, api_key, model, size, quality, image=None):
        if api_key == "":
            raise RuntimeError("API key is empty")

        # === Proxy handling from environment ===
        proxy_url = os.getenv("http_proxy") or os.getenv("https_proxy")  # Standard env vars
        http_client = None

        if proxy_url:
            # httpx supports proxy for both http and https when passed as proxy=
            http_client = httpx.Client(
                proxy=proxy_url,
                timeout=600.0,
                follow_redirects=True,
            )
            print(f"{RED}Using proxy: {proxy_url}{RESET}")
        else:
            print(f"{RED}No proxy configured (not set in HTTPS_PROXY/HTTP_PROXY){RESET}")

        # Initialize OpenAI client pointing to Intel internal endpoint
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client
        )

        image_base64 = None

        try:
            if image is None or (isinstance(image, torch.Tensor) and image.numel() == 0):
                # Generation mode (no input image)
                result = client.images.generate(
                    model=model,
                    prompt=prompt,
                    size=size,
                    quality=quality
                )
                image_base64 = result.data[0].b64_json
            else:
                # Edit mode (one or more input images)
                batch_size = image.shape[0]
                files = []

                for b in range(batch_size):
                    single_img = image[b]  # (H, W, C)
                    img_np = (single_img.detach().cpu().numpy() * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_np)

                    byte_io = io.BytesIO()
                    pil_img.save(byte_io, format='PNG')
                    files.append(('image', (f'ref_image_{b}.png', byte_io.getvalue(), 'image/png')))

                data = {
                    'prompt': prompt,
                    'model': model,
                    'size': size,
                    'quality': quality,
                }

                headers = {
                    "Authorization": f"Bearer {api_key}"
                }

                response = client._client.post(
                    url=f"{base_url}/v1/images/edits",
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=600,
                )
                response.raise_for_status()
                edited = response.json()

                try:
                    image_base64 = edited["data"][0]["b64_json"]
                except KeyError:
                    print("Unexpected response:", edited)
                    raise ValueError(f"Unexpected API response format: {edited}")

            # === Common part: decode base64 and convert to ComfyUI IMAGE tensor ===
            if not image_base64:
                raise RuntimeError("No image data received from API")

            image_bytes = base64.b64decode(image_base64)
            generated_image = Image.open(io.BytesIO(image_bytes))

            # Convert to float32 [0-1] tensor with batch dimension
            image_np = np.array(generated_image).astype(np.float32) / 255.0
            image_np = np.expand_dims(image_np, axis=0)  # (1, H, W, C)

            image_tensor = torch.from_numpy(image_np)

            return (image_tensor,)

        except Exception as e:
            error_message = f"{str(e)}"
            print(f"{RED}Error calling OpenAI Image API: {error_message}{RESET}")
            raise RuntimeError(error_message) from e


# Node registration
NODE_CLASS_MAPPINGS = {
    "QWen Image Edit": QWenImageEdit
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QWen Image Edit": "Qwen-Image-Edit-2509 by OpenAI Image API"
}
