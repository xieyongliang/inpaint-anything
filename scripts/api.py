from typing import List, Optional
from pydantic import BaseModel, Field
from torch.hub import download_url_to_file
import cv2
import json

from modules import shared
from modules import script_callbacks

import numpy as np
from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from PIL import Image, ImageOps
import gradio as gr
import os
import io
import base64
import traceback
import math
import gc
from torchvision import transforms
import random

from ia_file_manager import ia_file_manager
from ia_logging import ia_logging
import inpalib
from ia_webui_controlnet import (find_controlnet, 
                                 get_sd_img2img_processing, 
                                 backup_alwayson_scripts, 
                                 disable_alwayson_scripts_wo_cn,
                                 get_controlnet_args_to,
                                 clear_controlnet_cache,
                                 restore_alwayson_scripts
                                )
from ia_threading import (await_pre_reload_model_weights)
from ia_config import (IAConfig, get_webui_setting, set_ia_config)
from modules.processing import process_images
from modules import devices

class SamRequest(BaseModel):
    input_image: str = Field(title="Input Image", description="The input image in base64 format.")
    sam_model_id: str = Field(title="SAM Model Id")
    pad_mask: Optional[dict] = Field(title="Pad Mask")
    anime_style_chk: Optional[bool]

class SamResponse(BaseModel):
    seg_image: str = Field(title="Segment Image", description="The generated image segment in base64 format.")
    sam_masks: List[dict] = Field(title="Sam Image Masks", description="The generated Sam image masks in base64 format.")

class MaskRequest(BaseModel):
    sam_masks: List[dict] = Field(title="Sam Image Masks", description="The generated Sam image masks in base64 format.")
    input_image: str = Field(title="Input Image", description="The input image in base64 format.")
    sam_image: dict = Field(title="Sam Image", description="The sam image in base64 format.")
    invert_chk: bool
    ignore_black_chk: bool

class MaskResponse(BaseModel):
    mask_image: str = Field(title="Image Mask", description="The image mask in base64 format.")
    sel_mask: str = Field(title="Selected Mask", description="The selected mask in base64 format.")

class CNInpaintRequest(BaseModel):
    mask_image: str = Field(title="Image Mask", description="The image mask in base64 format.")
    input_image: str = Field(title="Input Image", description="The input image in base64 format.")
    cn_prompt: str = Field(title="ControlNet Prompt")
    cn_n_prompt: str = Field(title="ControlNet Negtive Prompt")
    cn_sampler_id: str = Field(title="ControlNet Sampler Id")
    cn_ddim_steps: int = Field(title="ControlNet DDIM Steps")
    cn_cfg_scale: float = Field(title="ControlNet CFG Scale")
    cn_strength: float = Field(title="ControlNet CFG Strength")
    cn_seed: int = Field(title="ControlNet CFG Seed")
    cn_module_id: str = Field(title="ControlNet Module Id")
    cn_model_id: str = Field(title="ControlNet Mode Id")
    cn_low_vram_chk: bool = Field(title="Flag for Controlnet Low VRam")
    cn_weight: float = Field(title="Controlnet Weight")
    cn_mode: str = Field(title="Controlnet Mode")
    cn_iteration_count: int = Field(default=1, title="Flag for Controlnet Iteration Count")
    cn_ref_module_id: Optional[str] = Field(default=None, title="Controlnet Module Id")
    cn_ref_image: Optional[str] = Field(default=None, title="Controlnet Reference Image")
    cn_ref_weight: Optional[float] = Field(default=1.0, title="Controlnet Reference Weight")
    cn_ref_mode: Optional[str] = Field(default="Balanced", title="Controlnet Reference Weight")
    cn_ref_resize_mode: Optional[str] = Field(default="resize", title="Controlnet Reference Weight")
    cn_ipa_or_ref: Optional[str] = Field(default=None, title="Controlnet IP Adapter or Reference Model Id") 
    cn_ipa_model_id: Optional[str] = Field(default=None, title="Controlnet IP Adapter Model Id")

class CNInpaintResponse(BaseModel):
    output_images: List[str] = Field(title="Out Image", description="The generated out masks in base64 format.")
    iteration_count: int = Field(title="Iteration Count")

class PaddingRequest(BaseModel):
    input_image: str = Field(title="Input Image", description="The input image in base64 format.")
    orig_image: str = Field(title="Origin Image", description="The origin image in base64 format.")
    pad_scale_width: float = Field(title="Scale Width")
    pad_scale_height: float = Field(title="Scale Height")
    pad_lr_barance: float = Field(title="Left/Right Balance")
    pad_tb_barance: float = Field(title="Top/Bottom Balance")
    padding_mode: str = Field(default="edge", title="Padding Mode")

class PaddingResponse(BaseModel):
    pad_image: str = Field(title="Pad Image", description="The pad image in base64 format.")
    pad_mask: dict = Field(title="Pad mask")

class ExpandMaskRequest(BaseModel):
    input_image: str = Field(title="Input Image", description="The input image in base64 format.")
    mask_image: str = Field(title="Mask Image", description="The mask image in base64 format.")
    expand_iteration : int = Field(default=1, title="Expand iteration")

class ExpandMaskResponse(BaseModel):
    mask_image: str = Field(title="Image Mask")
    sel_mask: str = Field(title="Selected Mask", description="The selected mask in base64 format.")

class ApplyMaskRequest(BaseModel):
    input_image: str = Field(title="Input Image", description="The input image in base64 format.")
    sel_mask: str = Field(title="Selected Mask", description="The selected mask in base64 format.")
    mask_image: str = Field(title="Image Mask")

class ApplyMaskResponse(BaseModel):
    sel_mask: str = Field(title="Selected Mask", description="The selected mask in base64 format.")
    mask_image: str = Field(title="Image Mask")

class AddMaskRequest(BaseModel):
    input_image: str = Field(title="Input Image", description="The input image in base64 format.")
    sel_mask: str = Field(title="Selected Mask", description="The selected mask in base64 format.")
    mask_image: str = Field(title="Image Mask")

class AddMaskResponse(BaseModel):
    sel_mask: str = Field(title="Selected Mask", description="The selected mask in base64 format.")
    mask_image: str = Field(title="Image Mask")

def pack(array, bitdepth, bitorder="little"):
    itemsize = array.dtype.itemsize
    bits = np.unpackbits(
        array.astype(f"<u{itemsize}", copy=False).view("u1").reshape(-1, itemsize),
        axis=1,
        bitorder="little",
        count=bitdepth,
    )
    if bitorder == "big":
        bits = bits[:, ::-1]
    return np.packbits(bits, bitorder=bitorder)

def unpack(buf, bitdepth, bitorder="little"):
    bits = np.unpackbits(buf, bitorder=bitorder).reshape(-1, bitdepth)
    if bitorder == "big":
        bits = bits[:, ::-1]
    bytes_ = np.packbits(bits, axis=1, bitorder="little")
    itemsize = np.min_scalar_type((1 << bitdepth) - 1).itemsize
    if bytes_.shape[1] < itemsize:
        bytes_ = np.pad(bytes_, ((0, 0), (0, itemsize - bytes_.shape[1])))
    return bytes_.reshape(-1).view(f"<u{itemsize}")

def encode_image_to_base64(image):
    with io.BytesIO() as output_bytes:
        if isinstance(image, dict):
            image = image['image']
        format = "PNG" if image.mode == 'RGBA' else "JPEG"
        image.save(output_bytes, format=format)
        bytes_data = output_bytes.getvalue()

    encoded_string = base64.b64encode(bytes_data)

    base64_str = str(encoded_string, "utf-8")
    mimetype = "image/jpeg" if format == 'JPEG' else 'image/png'
    image_encoded_in_base64 = (
        "data:" + (mimetype if mimetype is not None else "") + ";base64," + base64_str
    )
    return image_encoded_in_base64

def auto_resize_to_pil(input_image, mask_image):
    init_image = Image.fromarray(input_image).convert("RGB")
    mask_image = Image.fromarray(mask_image).convert("RGB")
    assert init_image.size == mask_image.size, "The sizes of the image and mask do not match"
    width, height = init_image.size

    new_height = (height // 8) * 8
    new_width = (width // 8) * 8
    if new_width < width or new_height < height:
        if (new_width / width) < (new_height / height):
            scale = new_height / height
        else:
            scale = new_width / width
        resize_height = int(height*scale+0.5)
        resize_width = int(width*scale+0.5)
        if height != resize_height or width != resize_width:
            ia_logging.info(f"resize: ({height}, {width}) -> ({resize_height}, {resize_width})")
            init_image = transforms.functional.resize(init_image, (resize_height, resize_width), transforms.InterpolationMode.LANCZOS)
            mask_image = transforms.functional.resize(mask_image, (resize_height, resize_width), transforms.InterpolationMode.LANCZOS)
        if resize_height != new_height or resize_width != new_width:
            ia_logging.info(f"center_crop: ({resize_height}, {resize_width}) -> ({new_height}, {new_width})")
            init_image = transforms.functional.center_crop(init_image, (new_height, new_width))
            mask_image = transforms.functional.center_crop(mask_image, (new_height, new_width))

    return init_image, mask_image

def inpaint_anything_api(_: gr.Blocks, app: FastAPI):
    app.add_api_route('/inpaint-anything/sam', sam_api, methods=["POST"], response_model=SamResponse)
    app.add_api_route('/inpaint-anything/mask', mask_api, methods=["POST"], response_model=MaskResponse)
    app.add_api_route('/inpaint-anything/cninpaint', cninpaint_api, methods=["POST"], response_model=CNInpaintResponse)
    app.add_api_route('/inpaint-anything/padding', padding_api, methods=["POST"], response_model=PaddingResponse)
    app.add_api_route('/inpaint-anything/expand-mask', expand_mask_api, methods=["POST"], response_model=ExpandMaskResponse)
    app.add_api_route('/inpaint-anything/apply-mask', apply_mask_api, methods=["POST"], response_model=ApplyMaskResponse)
    app.add_api_route('/inpaint-anything/add-mask', add_mask_api, methods=["POST"], response_model=AddMaskResponse)

def decode_base64_to_image(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    try:
        image = Image.open(io.BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as e:
        print(e)

try:
    script_callbacks.on_app_started(inpaint_anything_api)
except:
    pass

def download_model(sam_model_id):
    if "_hq_" in sam_model_id:
        url_sam = "https://huggingface.co/Uminosachi/sam-hq/resolve/main/" + sam_model_id
    elif "FastSAM" in sam_model_id:
        url_sam = "https://huggingface.co/Uminosachi/FastSAM/resolve/main/" + sam_model_id
    elif "mobile_sam" in sam_model_id:
        url_sam = "https://huggingface.co/Uminosachi/MobileSAM/resolve/main/" + sam_model_id
    else:
        # url_sam_vit_h_4b8939 = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        url_sam = "https://dl.fbaipublicfiles.com/segment_anything/" + sam_model_id

    sam_checkpoint = os.path.join(ia_file_manager.models_dir, sam_model_id)
    if not os.path.isfile(sam_checkpoint):
        try:
            download_url_to_file(url_sam, sam_checkpoint)
        except Exception as e:
            ia_logging.error(str(e))
            return str(e)

def sam_api(req: SamRequest):
    download_model(req.sam_model_id)

    if not inpalib.sam_file_exists(req.sam_model_id):
        raise HTTPException(status_code=422, detail=f"{req.sam_model_id} not found, please download")

    input_image = np.array(decode_base64_to_image(req.input_image))

    if input_image is None:
        return HTTPException(status_code=422, detail="Input image does not exist") 

    try:
        input_image = np.array(decode_base64_to_image(req.input_image))
        pad_mask = dict(segmentation=np.array(req.pad_mask))

        set_ia_config(IAConfig.KEYS.SAM_MODEL_ID, req.sam_model_id, IAConfig.SECTIONS.USER)

        ia_logging.info(f"input_image: {input_image.shape} {input_image.dtype}")

        sam_masks = inpalib.generate_sam_masks(input_image, req.sam_model_id, req.anime_style_chk)
        sam_masks = inpalib.sort_masks_by_area(sam_masks)
        sam_masks = inpalib.insert_mask_to_sam_masks(sam_masks, pad_mask)

        seg_image = inpalib.create_seg_color_image(input_image, sam_masks)
        res_seg_image = encode_image_to_base64(Image.fromarray(seg_image))
        res_sam_masks = []
        for sam_mask in sam_masks:
            res_sam_mask = {}
            for key in sam_mask:
                if key == 'segmentation':
                    compressed_data = {
                        'height': sam_mask[key].shape[0],
                        'width': sam_mask[key].shape[1],
                        'data' : pack(sam_mask[key].astype(np.uint8), 1).tolist()
                    }
                    res_sam_mask[key] = compressed_data
                else:
                    res_sam_mask[key] = sam_mask[key]
            res_sam_masks.append(res_sam_mask)

        return SamResponse(seg_image=res_seg_image, sam_masks=res_sam_masks)
    except Exception as e:
        print(traceback.format_exc())
        ia_logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

def mask_api(req: MaskRequest):
    if req.sam_masks is None or req.sam_image is None:
        raise HTTPException(status_code=422, detail="sam_masks or sam_image does not exist")

    req_sam_masks = req.sam_masks
    sam_masks = []
    for _, req_sam_mask in enumerate(req_sam_masks):
        sam_mask = {}
        for key in req_sam_mask:
            if key == 'segmentation':
                compressed_data = req_sam_mask[key]
                uncompressed_data = unpack(np.array(compressed_data['data'], dtype=np.uint8), 1).reshape(compressed_data['height'], compressed_data['width']).astype(bool)
                sam_mask[key] = uncompressed_data
            else:
                sam_mask[key] = req_sam_mask[key]
        sam_masks.append(sam_mask)

    sam_image = {
        "image": np.array(decode_base64_to_image(req.sam_image["image"])) if req.sam_image["image"] else None, 
        "mask": np.array(decode_base64_to_image(req.sam_image["mask"])) if req.sam_image["mask"] else None, 
    }

    input_image = np.array(decode_base64_to_image(req.input_image))
    ignore_black_chk = req.ignore_black_chk
    invert_chk = req.invert_chk

    try:
        mask = sam_image["mask"][:, :, 0:1]
        
        seg_image = inpalib.create_mask_image(mask, sam_masks, ignore_black_chk)
        if invert_chk:
            seg_image = inpalib.invert_mask(seg_image)

        if input_image is not None and input_image.shape == seg_image.shape:
            ret_image = cv2.addWeighted(input_image, 0.5, seg_image, 0.5, 0)
        else:
            ret_image = seg_image
        
        ret_mask_image = encode_image_to_base64(Image.fromarray(seg_image))
        ret_sel_mask = encode_image_to_base64(Image.fromarray(ret_image))

        return MaskResponse(mask_image=ret_mask_image, sel_mask=ret_sel_mask)
    except Exception as e:
        print(traceback.format_exc())
        ia_logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

def padding_api(req: PaddingRequest):
    if req.input_image is None or req.orig_image is None:
        raise HTTPException(status_code=422, detail="Input image or origin_image does not exist")

    try:
        orig_image = np.array(decode_base64_to_image(req.orig_image))
        pad_scale_width = req.pad_scale_width
        pad_scale_height = req.pad_scale_height
        pad_lr_barance = req.pad_lr_barance
        pad_tb_barance = req.pad_tb_barance
        padding_mode = req.padding_mode

        height, width = orig_image.shape[:2]
        pad_width, pad_height = (int(width * pad_scale_width), int(height * pad_scale_height))
        ia_logging.info(f"resize by padding: ({height}, {width}) -> ({pad_height}, {pad_width})")

        pad_size_w, pad_size_h = (pad_width - width, pad_height - height)
        pad_size_l = int(pad_size_w * pad_lr_barance)
        pad_size_r = pad_size_w - pad_size_l
        pad_size_t = int(pad_size_h * pad_tb_barance)
        pad_size_b = pad_size_h - pad_size_t

        pad_width = [(pad_size_t, pad_size_b), (pad_size_l, pad_size_r), (0, 0)]
        if padding_mode == "constant":
            fill_value = get_webui_setting("inpaint_anything_padding_fill", 127)
            pad_image = np.pad(orig_image, pad_width=pad_width, mode=padding_mode, constant_values=fill_value)
        else:
            pad_image = np.pad(orig_image, pad_width=pad_width, mode=padding_mode)

        mask_pad_width = [(pad_size_t, pad_size_b), (pad_size_l, pad_size_r)]
        pad_mask = np.zeros((height, width), dtype=np.uint8)
        pad_mask = np.pad(pad_mask, pad_width=mask_pad_width, mode="constant", constant_values=255)
        
        ret_pad_image = encode_image_to_base64(Image.fromarray(pad_image))
        ret_pad_mask = dict(segmentation=pad_mask.astype(bool).tolist())

        return PaddingResponse(pad_image=ret_pad_image, pad_mask=ret_pad_mask)
    except Exception as e:
        print(traceback.format_exc())
        ia_logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

def expand_mask_api(req: ExpandMaskRequest):
    if req.mask_image is None:
        return HTTPException(status_code=422, detail="mask_image does not exist")

    try:
        input_image = np.array(decode_base64_to_image(req.input_image))
        new_sel_mask = np.array(decode_base64_to_image(req.mask_image))
        expand_iteration = req.expand_iteration

        expand_iteration = int(np.clip(expand_iteration, 1, 100))

        new_sel_mask = cv2.dilate(new_sel_mask, np.ones((3, 3), dtype=np.uint8), iterations=expand_iteration)

        if input_image is not None and input_image.shape == new_sel_mask.shape:
            ret_image = cv2.addWeighted(input_image, 0.5, new_sel_mask, 0.5, 0)
        else:
            ret_image = new_sel_mask

        ret_mask_image = encode_image_to_base64(Image.fromarray(new_sel_mask))
        ret_sel_mask = encode_image_to_base64(Image.fromarray(ret_image))

        return ExpandMaskResponse(mask_image=ret_mask_image, sel_mask=ret_sel_mask)
    except Exception as e:
        print(traceback.format_exc())
        ia_logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

def apply_mask_api(req: ApplyMaskRequest):
    if req.mask_image is None or req.sel_mask is None:
        raise HTTPException(status_code=400, detail="mask_image or sel_mask does not exist")

    try:
        input_image = np.array(decode_base64_to_image(req.input_image))
        sel_mask = {
            "image": np.array(decode_base64_to_image(req.sel_mask["image"])) if req.sel_mask["image"] else None,
            "mask": np.array(decode_base64_to_image(req.sel_mask["mask"])) if req.sel_mask["mask"] else None,
        }
        mask_image = np.array(decode_base64_to_image(req.mask_image))

        sel_mask_image = mask_image
        sel_mask_mask = np.logical_not(sel_mask["mask"][:, :, 0:3].astype(bool)).astype(np.uint8)
        new_sel_mask = sel_mask_image * sel_mask_mask

        ret_mask_image = encode_image_to_base64(Image.fromarray(new_sel_mask))

        if input_image is not None and input_image.shape == new_sel_mask.shape:
            ret_image = cv2.addWeighted(input_image, 0.5, new_sel_mask, 0.5, 0)
        else:
            ret_image = new_sel_mask
        
        ret_sel_mask = encode_image_to_base64(Image.fromarray(ret_image))

        return ApplyMaskResponse(sel_mask=ret_sel_mask, mask_image=ret_mask_image)
    except Exception as e:
        print(traceback.format_exc())
        ia_logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

def add_mask_api(req: AddMaskRequest):
    if req.mask_image is None or req.sel_mask is None:
        raise HTTPException(status_code=422, detail="mask_image or sel_mask does not exist")

    try:
        input_image = np.array(decode_base64_to_image(req.input_image))
        mask_image = np.array(decode_base64_to_image(req.mask_image))
        sel_mask = {
            "image": np.array(decode_base64_to_image(req.sel_mask["image"])) if req.sel_mask["image"] else None,
            "mask": np.array(decode_base64_to_image(req.sel_mask["mask"])) if req.sel_mask["mask"] else None,
        }

        sel_mask_image = mask_image
        sel_mask_mask = sel_mask["mask"][:, :, 0:3].astype(bool).astype(np.uint8)
        new_sel_mask = sel_mask_image + (sel_mask_mask * np.invert(sel_mask_image, dtype=np.uint8))

        ret_mask_image = encode_image_to_base64(Image.fromarray(new_sel_mask))

        if input_image is not None and input_image.shape == new_sel_mask.shape:
            ret_image = cv2.addWeighted(input_image, 0.5, new_sel_mask, 0.5, 0)
        else:
            ret_image = new_sel_mask

        ret_sel_mask = encode_image_to_base64(Image.fromarray(ret_image))

        return AddMaskResponse(mask_image=ret_mask_image, sel_mask=ret_sel_mask)
    except Exception as e:
        print(traceback.format_exc())
        ia_logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))    

def cninpaint_api(req: CNInpaintRequest):
    input_image = np.array(decode_base64_to_image(req.input_image))
    mask_image = np.array(decode_base64_to_image(req.mask_image))
    cn_prompt = req.cn_prompt
    cn_n_prompt = req.cn_n_prompt
    cn_sampler_id = req.cn_sampler_id
    cn_ddim_steps = req.cn_ddim_steps
    cn_cfg_scale = req.cn_cfg_scale
    cn_strength = req.cn_strength
    cn_seed = req.cn_seed
    cn_module_id = req.cn_module_id
    cn_model_id = req.cn_model_id
    cn_low_vram_chk = req.cn_low_vram_chk
    cn_weight = req.cn_weight
    cn_mode = req.cn_mode
    cn_iteration_count = req.cn_iteration_count
    cn_ref_module_id = req.cn_ref_module_id
    cn_ref_image = req.cn_ref_image
    cn_ref_weight = req.cn_ref_weight
    cn_ref_mode = req.cn_ref_mode
    cn_ref_resize_mode = req.cn_ref_resize_mode
    cn_ipa_or_ref = req.cn_ipa_or_ref
    cn_ipa_model_id = req.cn_ipa_model_id

    if input_image is None or mask_image is None:
        raise HTTPException(status_code=422, detail="The image or mask does not exist")

    if input_image.shape != mask_image.shape:
        raise HTTPException(status_code=422, detail="The sizes of the image and mask do not match")

    await_pre_reload_model_weights()

    if (shared.sd_model.parameterization == "v" and "sd15" in cn_model_id):
        raise HTTPException(status_code=422, detail="The SDv2 model is not compatible with the ControlNet model")

    if (getattr(shared.sd_model, "is_sdxl", False) and "sd15" in cn_model_id):
        raise HTTPException(status_code=422, detail="The SDXL model is not compatible with the ControlNet model")

    cnet = find_controlnet()
    if cnet is None:
        ia_logging.warning("The ControlNet extension is not loaded")
        raise HTTPException(status_code=500, detail="The ControlNet extension is not loaded")

    try:
        init_image, mask_image = auto_resize_to_pil(input_image, mask_image)
        width, height = init_image.size

        input_mask = None if "inpaint_only" in cn_module_id else mask_image
        p = get_sd_img2img_processing(init_image, input_mask,
                                    cn_prompt, cn_n_prompt, cn_sampler_id, cn_ddim_steps, cn_cfg_scale, cn_strength, cn_seed)

        backup_alwayson_scripts(p.scripts)
        disable_alwayson_scripts_wo_cn(cnet, p.scripts)

        cn_units = [cnet.to_processing_unit(dict(
            enabled=True,
            module=cn_module_id,
            model=cn_model_id,
            weight=cn_weight,
            image={"image": np.array(init_image), "mask": np.array(mask_image)},
            resize_mode=cnet.ResizeMode.RESIZE,
            low_vram=cn_low_vram_chk,
            processor_res=min(width, height),
            guidance_start=0.0,
            guidance_end=1.0,
            pixel_perfect=True,
            control_mode=cn_mode,
        ))]

        if cn_ref_module_id is not None and cn_ref_image is not None:
            if cn_ref_resize_mode == "tile":
                ref_height, ref_width = cn_ref_image.shape[:2]
                num_h = math.ceil(height / ref_height) if height > ref_height else 1
                num_h = num_h + 1 if (num_h % 2) == 0 else num_h
                num_w = math.ceil(width / ref_width) if width > ref_width else 1
                num_w = num_w + 1 if (num_w % 2) == 0 else num_w
                cn_ref_image = np.tile(cn_ref_image, (num_h, num_w, 1))
                cn_ref_image = transforms.functional.center_crop(Image.fromarray(cn_ref_image), (height, width))
                ia_logging.info(f"Reference image is tiled ({num_h}, {num_w}) times and cropped to ({height}, {width})")
            else:
                cn_ref_image = ImageOps.fit(Image.fromarray(cn_ref_image), (width, height), method=Image.Resampling.LANCZOS)
                ia_logging.info(f"Reference image is resized and cropped to ({height}, {width})")
            assert cn_ref_image.size == init_image.size, "The sizes of the reference image and input image do not match"

            cn_ref_model_id = None
            if cn_ipa_or_ref is not None and cn_ipa_model_id is not None:
                cn_ipa_module_ids = [cn for cn in cnet.get_modules() if "ip-adapter" in cn and "sd15" in cn]
                if len(cn_ipa_module_ids) > 0 and cn_ipa_or_ref == "IP-Adapter":
                    cn_ref_module_id = cn_ipa_module_ids[0]
                    cn_ref_model_id = cn_ipa_model_id

            cn_units.append(cnet.to_processing_unit(dict(
                enabled=True,
                module=cn_ref_module_id,
                model=cn_ref_model_id,
                weight=cn_ref_weight,
                image={"image": np.array(cn_ref_image), "mask": None},
                resize_mode=cnet.ResizeMode.RESIZE,
                low_vram=cn_low_vram_chk,
                processor_res=min(width, height),
                guidance_start=0.0,
                guidance_end=1.0,
                pixel_perfect=True,
                control_mode=cn_ref_mode,
                threshold_a=0.5,
            )))

        p.script_args = np.zeros(get_controlnet_args_to(cnet, p.scripts)).tolist()
        cnet.update_cn_script_in_processing(p, cn_units)

        output_list = []
        cn_iteration_count = cn_iteration_count if cn_iteration_count is not None else 1
        for count in range(int(cn_iteration_count)):
            gc.collect()
            if cn_seed < 0 or count > 0:
                cn_seed = random.randint(0, 2147483647)

            p.init_images = [init_image]
            p.seed = cn_seed

            processed = process_images(p)

            if processed is not None and len(processed.images) > 0:
                output_image = processed.images[0]
                output_list.append(encode_image_to_base64(output_image))

        clear_controlnet_cache(cnet, p.scripts)
        restore_alwayson_scripts(p.scripts)

        return CNInpaintResponse(output_images=output_list, iteration_count=max([1, cn_iteration_count - (count + 1)]))
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))