import json
import math

import numpy as np
import skimage

import modules.scripts as scripts
import gradio as gr
from PIL import Image, ImageDraw, ImageOps, ImageFont

from modules import images, processing, devices
from modules.processing import Processed, process_images, StableDiffusionProcessingImg2Img
from modules.shared import opts, cmd_opts
import modules.shared as shared
import modules.extras as extras
import scrython
import asyncio
import requests
import os
from extensions.mtg_card_art import proxyshop

# this function is taken from https://github.com/parlance-zz/g-diffuser-bot
def get_matched_noise(_np_src_image, np_mask_rgb, noise_q=1, color_variation=0.05):
    # helper fft routines that keep ortho normalization and auto-shift before and after fft
    def _fft2(data):
        if data.ndim > 2:  # has channels
            out_fft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
            for c in range(data.shape[2]):
                c_data = data[:, :, c]
                out_fft[:, :, c] = np.fft.fft2(np.fft.fftshift(c_data), norm="ortho")
                out_fft[:, :, c] = np.fft.ifftshift(out_fft[:, :, c])
        else:  # one channel
            out_fft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
            out_fft[:, :] = np.fft.fft2(np.fft.fftshift(data), norm="ortho")
            out_fft[:, :] = np.fft.ifftshift(out_fft[:, :])

        return out_fft

    def _ifft2(data):
        if data.ndim > 2:  # has channels
            out_ifft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
            for c in range(data.shape[2]):
                c_data = data[:, :, c]
                out_ifft[:, :, c] = np.fft.ifft2(np.fft.fftshift(c_data), norm="ortho")
                out_ifft[:, :, c] = np.fft.ifftshift(out_ifft[:, :, c])
        else:  # one channel
            out_ifft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
            out_ifft[:, :] = np.fft.ifft2(np.fft.fftshift(data), norm="ortho")
            out_ifft[:, :] = np.fft.ifftshift(out_ifft[:, :])

        return out_ifft

    def _get_gaussian_window(width, height, std=3.14, mode=0):
        window_scale_x = float(width / min(width, height))
        window_scale_y = float(height / min(width, height))

        window = np.zeros((width, height))
        x = (np.arange(width) / width * 2. - 1.) * window_scale_x
        for y in range(height):
            fy = (y / height * 2. - 1.) * window_scale_y
            if mode == 0:
                window[:, y] = np.exp(-(x ** 2 + fy ** 2) * std)
            else:
                window[:, y] = (1 / ((x ** 2 + 1.) * (fy ** 2 + 1.))) ** (std / 3.14)  # hey wait a minute that's not gaussian

        return window

    def _get_masked_window_rgb(np_mask_grey, hardness=1.):
        np_mask_rgb = np.zeros((np_mask_grey.shape[0], np_mask_grey.shape[1], 3))
        if hardness != 1.:
            hardened = np_mask_grey[:] ** hardness
        else:
            hardened = np_mask_grey[:]
        for c in range(3):
            np_mask_rgb[:, :, c] = hardened[:]
        return np_mask_rgb

    width = _np_src_image.shape[0]
    height = _np_src_image.shape[1]
    num_channels = _np_src_image.shape[2]

    np_src_image = _np_src_image[:] * (1. - np_mask_rgb)
    np_mask_grey = (np.sum(np_mask_rgb, axis=2) / 3.)
    img_mask = np_mask_grey > 1e-6
    ref_mask = np_mask_grey < 1e-3

    windowed_image = _np_src_image * (1. - _get_masked_window_rgb(np_mask_grey))
    windowed_image /= np.max(windowed_image)
    windowed_image += np.average(_np_src_image) * np_mask_rgb  # / (1.-np.average(np_mask_rgb))  # rather than leave the masked area black, we get better results from fft by filling the average unmasked color

    src_fft = _fft2(windowed_image)  # get feature statistics from masked src img
    src_dist = np.absolute(src_fft)
    src_phase = src_fft / src_dist

    # create a generator with a static seed to make outpainting deterministic / only follow global seed
    rng = np.random.default_rng(0)

    noise_window = _get_gaussian_window(width, height, mode=1)  # start with simple gaussian noise
    noise_rgb = rng.random((width, height, num_channels))
    noise_grey = (np.sum(noise_rgb, axis=2) / 3.)
    noise_rgb *= color_variation  # the colorfulness of the starting noise is blended to greyscale with a parameter
    for c in range(num_channels):
        noise_rgb[:, :, c] += (1. - color_variation) * noise_grey

    noise_fft = _fft2(noise_rgb)
    for c in range(num_channels):
        noise_fft[:, :, c] *= noise_window
    noise_rgb = np.real(_ifft2(noise_fft))
    shaped_noise_fft = _fft2(noise_rgb)
    shaped_noise_fft[:, :, :] = np.absolute(shaped_noise_fft[:, :, :]) ** 2 * (src_dist ** noise_q) * src_phase  # perform the actual shaping

    brightness_variation = 0.  # color_variation # todo: temporarily tieing brightness variation to color variation for now
    contrast_adjusted_np_src = _np_src_image[:] * (brightness_variation + 1.) - brightness_variation * 2.

    # scikit-image is used for histogram matching, very convenient!
    shaped_noise = np.real(_ifft2(shaped_noise_fft))
    shaped_noise -= np.min(shaped_noise)
    shaped_noise /= np.max(shaped_noise)
    shaped_noise[img_mask, :] = skimage.exposure.match_histograms(shaped_noise[img_mask, :] ** 1., contrast_adjusted_np_src[ref_mask, :], channel_axis=1)
    shaped_noise = _np_src_image[:] * (1. - np_mask_rgb) + shaped_noise * np_mask_rgb

    matched_noise = shaped_noise[:]

    return np.clip(matched_noise, 0., 1.)

def outpaint(cardname, artist, prompt, image, preprocessor, preprocessor_scale, postprocessor, postprocessor_scale, seed, negativePrompt, zoom, steps):
    if cardname == '':
        return # TODO error

    proxyshop_enabled = False
    batch_count = 1
    batch_size = 1

    shared.state.textinfo = "Fetching Scryfall data..."

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    query = cardname
    if artist != '':
        query += f' artist:"{artist}"'
    cards = scrython.cards.Search(q=query,unique='art').data()
    card = cards[0]

    if artist == '':
        artist = card['artist']
    if prompt == '':
        prompt = card['type_line'] # TODO
    if image is None:
        artCropURL = card['image_uris']['art_crop']
        image = Image.open(requests.get(artCropURL, stream=True).raw)

    # scryfall completed
    shared.state.textinfo = "Pre-processing image..."

    mask_blur = 16 # 1 px to prevent off by one errors with clamping
    direction = ['left', 'right', 'up', 'down']
    noise_q = 1.0
    color_variation = 0.05

    initial_seed_and_info = [None, None]


    p = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=f'{card["name"]} {prompt} MtG Card Art by ({artist})',
        negative_prompt=negativePrompt,
        seed=seed,
        sampler_name='Euler a',
        batch_size=batch_size,
        n_iter=batch_count,
        steps=steps,
        cfg_scale=7,
        width=512,
        height=512,
        restore_faces=False,
        tiling=False,
        init_images=[image],
        mask_blur=mask_blur,
        inpainting_fill=1,
        denoising_strength=1,
        inpaint_full_res=False,
        do_not_save_samples=True,
        do_not_save_grid=True,
    )



    # 'outpath_samples': 'outputs/img2img-images', 'outpath_grids': 'outputs/img2img-grids', 'prompt': "path, cloudy sky, lattice, lantern, landscape, cliff, arm, emrakul's infestation, rock formation, mountain, innistrad, point of view, scryfall MtG Card Art by Jung Park\n", 'prompt_for_display': None, 'negative_prompt': '', 'styles': ['None', 'None'], 'seed': 4187943543.0, 'subseed': -1, 'subseed_strength': 0, 'seed_resize_from_h': 0, 'seed_resize_from_w': 0, 'sampler_name': 'Euler a', 'batch_size': 1, 'n_iter': 1, 'steps': 20, 'cfg_scale': 7, 'width': 512, 'height': 512, 'restore_faces': False, 'tiling': False, 'do_not_save_samples': False, 'do_not_save_grid': False, 'extra_generation_params': {'Mask blur': 4}, 'overlay_images': None, 'eta': None, 'do_not_reload_embeddings': False, 'paste_to': None, 'color_corrections': None, 'denoising_strength': 0.75, 'sampler_noise_scheduler_override': None, 'ddim_discretize': 'uniform', 's_churn': 0.0, 's_tmin': 0.0, 's_tmax': inf, 's_noise': 1.0, 'override_settings': {}, 'override_settings_restore_afterwards': True, 'is_using_inpainting_conditioning': False, 'scripts': <modules.scripts.ScriptRunner object at 0x29ae6a200>, 'script_args': (5, 1, '', 0, '', True, False, False, False, '<ul>\n<li><code>CFG Scale</code> should be 2 or lower.</li>\n</ul>\n', True, True, '', '', True, 50, True, 1, 0, False, 4, 1, '<p style="margin-bottom:0.75em">Recommended settings: Sampling Steps: 80-100, Sampler: Euler a, Denoising strength: 0.8</p>', 128, 8, ['left', 'right', 'up', 'down'], 1, 0.05, '\n            <p style="margin-bottom:0.75em; padding-left: 0.8em;">\n                Recommended settings: Sampling Steps: 80-100, Sampler: Euler a, Denoising strength: 0.8\n            </p>\n        ', 128, 4, 0, ['left', 'right', 'up', 'down'], False, False, False, False, '', '<p style="margin-bottom:0.75em">Will upscale the image by the selected scale factor; use width and height sliders to set tile size</p>', 64, 0, 2, 1, '', 0, '', True, False, False), 'all_prompts': None, 'all_negative_prompts': None, 'all_seeds': None, 'all_subseeds': None, 'iteration': 0, 'init_images': [<PIL.Image.Image image mode=RGB size=626x457 at 0x31F271900>], 'resize_mode': 0, 'init_latent': None, 'image_mask': <PIL.Image.Image image mode=L size=626x457 at 0x31F270040>, 'latent_mask': None, 'mask_for_overlay': None, 'mask_blur': 4, 'inpainting_fill': 1, 'inpaint_full_res': 0, 'inpaint_full_res_padding': 32, 'inpainting_mask_invert': 0, 'initial_noise_multiplier': 1.0, 'mask': None, 'nmask': None, 'image_conditioning': None}

    def runProcessor(image, processor, scale):
        if processor == 'None':
            return image
        upscalerIndex = -1
        for i, upscaler in enumerate(shared.sd_upscalers):
            if upscaler.name == processor:
                upscalerIndex = i
                break
        if upscalerIndex == -1:
            return # todo error

        # todo add this to Progress
        result, _, _ = extras.run_extras(None, None, image, None, None, None, False, 0, 0, 0, scale, None, None, None, upscalerIndex, 0, 1, False, False)
        return result[0]

    image = runProcessor(image, preprocessor, preprocessor_scale)

    # pre-processing completed
    shared.state.textinfo = "Preparing for outpainting..."

    process_width = p.width
    process_height = p.height

    # these values are selected so that default zoom of 1 fits the art about where it was on the original card
    scaledZoom = math.pow(zoom, 2.0)
    up = math.floor(image.height * 0.25 / scaledZoom) if "up" in direction else 0
    down1 = math.floor(image.height * 0.31 / scaledZoom) if "down" in direction else 0 # first pass
    down2 = math.floor(image.height * 0.35 / scaledZoom) if "down" in direction else 0 # second pass
    left = math.floor(image.width * 0.2 / scaledZoom) if "left" in direction else 0
    right = math.floor(image.width * 0.2 / scaledZoom) if "right" in direction else 0

    target_w = math.ceil((image.width + left + right) / 64) * 64
    target_h1 = math.ceil((image.height + up + down1) / 64) * 64
    target_h2 = math.ceil((image.height + up + down1 + down2) / 64) * 64
    print("target_w corrected to ", target_w)
    print("target_h corrected to ", target_h1, target_h2)

    if left > 0:
        left = left * (target_w - image.width) // (left + right)
        print("left corrected to ", left)

    if right > 0:
        right = target_w - image.width - left
        print("right corrected to ", right)

    if up > 0:
        print("up was", up)
        up = up * (target_h2 - image.height) // (up + down1 + down2)

    if down1 > 0:
        down1 = (target_h1 - image.height - up)
        print("down1 ", down1)

    if down2 > 0:
        down2 = (target_h2 - image.height - up - down1)
        print("down2 ", down2)

    def expand(init, count, expand_pixels, is_left=False, is_right=False, is_top=False, is_bottom=False):
        is_horiz = is_left or is_right
        is_vert = is_top or is_bottom
        pixels_horiz = expand_pixels if is_horiz else 0
        pixels_vert = expand_pixels if is_vert else 0

        images_to_process = []
        output_images = []
        for n in range(count):
            res_w = init[n].width + pixels_horiz
            res_h = init[n].height + pixels_vert
            process_res_w = math.ceil(res_w / 64) * 64
            process_res_h = math.ceil(res_h / 64) * 64

            print("running expand, process image (w,h) = ", (process_res_w, process_res_h))
            img = Image.new("RGB", (process_res_w, process_res_h), "cyan")
            print("pasting in image at ", (pixels_horiz if is_left else 0, pixels_vert if is_top else 0))
            img.paste(init[n], (pixels_horiz if is_left else 0, pixels_vert if is_top else 0))
            mask = Image.new("RGB", (process_res_w, process_res_h), "white")
            # scryfall art crop can have thin black borders or black gradients that break outpainting, so we mask n pixels on the edge we are processing

            draw = ImageDraw.Draw(mask)
            # draw the rectangle where we want to outpaint
            draw.rectangle((
                expand_pixels + mask_blur if is_left else 0,
                expand_pixels + mask_blur if is_top else 0,
                mask.width - expand_pixels - mask_blur if is_right else res_w,
                mask.height - expand_pixels - mask_blur if is_bottom else res_h,
            ), fill="black")
            print("masking", (
                expand_pixels + mask_blur if is_left else 0,
                expand_pixels + mask_blur if is_top else 0,
                mask.width - expand_pixels - mask_blur if is_right else res_w,
                mask.height - expand_pixels - mask_blur if is_bottom else res_h,
            ))

            np_image = (np.asarray(img) / 255.0).astype(np.float64)
            np_mask = (np.asarray(mask) / 255.0).astype(np.float64)
            noised = get_matched_noise(np_image, np_mask, noise_q, color_variation)
            output_images.append(Image.fromarray(np.clip(noised * 255., 0., 255.).astype(np.uint8), mode="RGB"))

            target_width = min(process_width, init[n].width + pixels_horiz) if is_horiz else img.width
            target_height = min(process_height, init[n].height + pixels_vert) if is_vert else img.height
            p.width = target_width if is_horiz else img.width
            p.height = target_height if is_vert else img.height

            crop_region = (
                0 if is_left else output_images[n].width - target_width,
                0 if is_top else output_images[n].height - target_height,
                target_width if is_left else output_images[n].width,
                target_height if is_top else output_images[n].height,
            )
            print("cropping back to ", crop_region)
            mask = mask.crop(crop_region)
            p.image_mask = mask
            print("")

            image_to_process = output_images[n].crop(crop_region)
            images_to_process.append(image_to_process)

        p.init_images = images_to_process

        latent_mask = Image.new("RGB", (p.width, p.height), "white")
        draw = ImageDraw.Draw(latent_mask)
        draw.rectangle((
            expand_pixels + mask_blur * 2 if is_left else 0,
            expand_pixels + mask_blur * 2 if is_top else 0,
            mask.width - expand_pixels - mask_blur * 2 if is_right else res_w,
            mask.height - expand_pixels - mask_blur * 2 if is_bottom else res_h,
        ), fill="black")
        p.latent_mask = latent_mask

        proc = process_images(p)

        if initial_seed_and_info[0] is None:
            initial_seed_and_info[0] = proc.seed
            initial_seed_and_info[1] = proc.info

        for n in range(count):
            output_images[n].paste(proc.images[n], (0 if is_left else output_images[n].width - proc.images[n].width, 0 if is_top else output_images[n].height - proc.images[n].height))
            output_images[n] = output_images[n].crop((0, 0, res_w, res_h))

        shared.state.current_image = output_images[0] # TODO
        return output_images

    all_processed_images = []

    shared.state.job_count = batch_count * 5
    for i in range(batch_count):
        imgs = [image] * batch_size
        shared.state.job = f"Batch {i + 1} out of {batch_count}"

        # with landscape MtG art crops it is better to start with up<>down and do left<>right last
        if up > 0:
            shared.state.job = "job: Outpainting top..."
            shared.state.textinfo = "Outpainting top..."
            imgs = expand(imgs, batch_size, up, is_top=True)
        if down1 > 0:
            # Two separate passes, because it's better to provide all the context (the model is 512, so between 384 to 321 is context, the rest is the outpainting)
            shared.state.textinfo = "Outpainting bottom (first pass)..."
            imgs = expand(imgs, batch_size, down1, is_bottom=True)
        if left > 0:
            shared.state.textinfo = "Outpainting left side..."
            imgs = expand(imgs, batch_size, left, is_left=True)
        if right > 0:
            shared.state.textinfo = "Outpainting right side..."
            imgs = expand(imgs, batch_size, right, is_right=True)

        if down2 > 0:
            # second pass (after we do left<>right)
            shared.state.textinfo = "Outpainting bottom (second pass)..."
            imgs = expand(imgs, batch_size, down2, is_bottom=True)

        all_processed_images += imgs

    all_images = all_processed_images

    # outpainting steps completes
    shared.state.textinfo = "Post-processing image..."

    combined_grid_image = images.image_grid(all_processed_images)
    unwanted_grid_because_of_img_count = len(all_processed_images) < 2 and opts.grid_only_if_multiple
    if opts.return_grid and not unwanted_grid_because_of_img_count:
        all_images = [combined_grid_image] + all_processed_images

    shared.state.begin()
    shared.state.job_count = len(all_images)
    shared.state.textinfo = "Running post-processor..."
    post_images = []
    for image in all_images:
        post_image = runProcessor(image, postprocessor, postprocessor_scale)
        post_images = post_images + [post_image]
        shared.state.nextjob()
    shared.state.end()
    all_images = post_images

    def overlay(image, overlay):
        owidth, oheight = image.size
        width = 2 * owidth
        height = 2 * oheight
        preview = Image.new("RGB", (width, height), 'black')

        # legal/border is bottom 10% of the card
        smallerImage = ImageOps.contain(image, (width, (height * 9) // 10))
        smallerMargin = (width - smallerImage.width) // 2
        preview.paste(smallerImage, (smallerMargin, 0))

        overlay = ImageOps.contain(overlay, (width, height))
        horizontal_margin = (width - overlay.width) // 2
        preview.paste(overlay, (horizontal_margin, 0), overlay)

        fontDir = os.path.dirname(os.path.realpath(__file__)) + "/fonts"
        fontGotham = ImageFont.truetype(f"{fontDir}/Gotham Medium Regular.ttf", math.floor(overlay.height * 0.018))
        fontJace = ImageFont.truetype(f"{fontDir}/JaceBeleren Bold.ttf", math.floor(overlay.height * 0.037))
        draw = ImageDraw.Draw(preview)
        # artist attribution
        draw.text((horizontal_margin + math.floor(overlay.width * 0.195), math.floor(overlay.height * 0.929)), artist, (255, 255, 255), font=fontGotham)
        # card name
        cardNamePos = (horizontal_margin + math.floor(overlay.width * 0.182), math.floor(overlay.height * 0.087))
        # TODO this offset has to be relative to overlay size
        draw.text((cardNamePos[0] + 4, cardNamePos[1] + 2), card['name'], (0, 0, 0), font=fontJace)
        draw.text(cardNamePos, card['name'], (255, 255, 255), font=fontJace)

        preview = preview.crop((smallerMargin, 0, width - smallerMargin, height))
        return preview

    # scryfall completed
    shared.state.textinfo = "Generating previews..."

    extra_images = []
    for image in all_images:
        if proxyshop_enabled:
            render = proxyshop.run(image, card['name'], card['artist'])
            extra_images = extra_images + [render]
        else:
            overlayAlignment = Image.open("extensions/mtg_card_art/overlay-alignment.png")
            previewAlignment = overlay(image, overlayAlignment)
            overlayCrop = Image.open("extensions/mtg_card_art/overlay-crop.png")
            previewCrop = overlay(image, overlayCrop)
            extra_images = extra_images + [previewAlignment, previewCrop]

    all_images = all_images + extra_images

    res = Processed(p, all_images, initial_seed_and_info[0], initial_seed_and_info[1])
    res.js()

    if opts.samples_save:
        for img in all_processed_images:
            images.save_image(img, p.outpath_samples, "", res.seed, p.prompt, opts.grid_format, info=res.info, p=p)

    if opts.grid_save and not unwanted_grid_because_of_img_count:
        images.save_image(combined_grid_image, p.outpath_grids, "grid", res.seed, p.prompt, opts.grid_format, info=res.info, short_filename=not opts.grid_extended_filename, grid=True, p=p)

    msg = 'Done'
    meta = json.loads(res.js())
    meta.update({'download_name': f'{card["name"]} ({card["artist"]})'})
    return res.images, json.dumps(meta), msg, msg
