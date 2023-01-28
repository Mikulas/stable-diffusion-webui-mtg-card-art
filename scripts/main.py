import gradio as gr

from extensions.mtg_card_art import outpainting_mtg, save

from modules import script_callbacks, shared
from modules.call_queue import wrap_gradio_gpu_call, wrap_gradio_call
from modules.ui import setup_progressbar, save_files
import modules.shared as shared

def on_ui_tabs():
    with gr.Blocks() as sp_interface:
        with gr.Row(equal_height=True):
            with gr.Column(variant="panel"):
                gr.Markdown("""
                    Outpaint MtG Cards from Scryfall Art Crops for Showcase/Full-Art/Borderless frame.
                    Requires inpainting Stable Diffusion checkpoint.
                """)
                # * Download the latest Magic Diffusion model (`dataset=v3-epoch=08-step=19665-pruned-inpainting.ckpt`) from [huggingface.co/rullaf/magic-diffusion](https://huggingface.co/rullaf/magic-diffusion/tree/main), and place it into `models/Stable-diffusion/` directory under your `stable-diffusion-webui` root directory.
                #  * You should also get a halftone/rosetta removal model (`1x_halftone_patch_120000_G.pth`) from the same repo, and place it into `models/ESRGAN/`.
                sp_cardname = gr.Textbox(label='Card (Scryfall search syntax)')
                sp_artist = gr.Textbox(label='Artist (optional)')
                sp_prompt = gr.Textbox(label='Prompt (optional)')
                sp_image = gr.Image(type="pil", label='Image (Scryfall Art Crop, optional)')

                upscalers=[x.name for x in shared.sd_upscalers]
                sp_preprocessor = gr.Dropdown(choices=upscalers, value='mtg_net_g_100000', type='value', label='Pre-processor')
                sp_preprocessor_scale = gr.Slider(label='Pre-processor scale', minimum=1, value=1.0, maximum=4.0, step=1)
                sp_postprocessor = gr.Dropdown(choices=upscalers, value='R-ESRGAN 4x+', type='value', label='Post-processor')
                sp_postprocessor_scale = gr.Slider(label='Post-processor scale', minimum=1, value=4.0, maximum=8.0, step=1)

                with gr.Accordion("Advanced", open=False):
                    sp_negative = gr.Textbox(label='Negative Prompt', value='(frame), (border), artist signature, logo watermark text')
                    sp_seed = gr.Number(label='Seed (-1: randomise every run)', value=-1)
                    sp_zoom = gr.Slider(label='Zoom (low value: more outpainting, smaller original art)', minimum=0.8, value=1.0, maximum=1.5, step=0.05)
                    sp_steps = gr.Slider(label='Sampling steps', minimum=1, value=40, maximum=150, step=1)

                with gr.Row():
                    sp_cancel = gr.Button(value="Cancel")
                    sp_run = gr.Button(value="Run", variant='primary')

            # Preview/progress
            with gr.Column(variant="panel"):
                sp_progressbar = gr.HTML(elem_id="sp_progressbar")
                sp_progress = gr.HTML(elem_id="sp_progress", value="")
                sp_interrupt = gr.HTML(elem_id="sp_interrupt", value="")
                sp_outcome = gr.HTML(elem_id="sp_error", value="")
                sp_preview = gr.Image(elem_id='sp_preview', visible=True, label='Preview')
                setup_progressbar(sp_progressbar, sp_preview, 'sp', textinfo=sp_progress)

                sp_gallery = gr.Gallery(label='Output', show_label=False, elem_id='sp_gallery').style(grid=4)
                sp_save = gr.Button('Save', elem_id='save_mtg')
                html_log = gr.HTML(elem_id=f'html_log_mtg')
                generation_info = gr.Textbox(visible=False, elem_id=f'generation_info_mtg')
                download_files = gr.File(None, file_count="multiple", interactive=False, show_label=False, visible=False, elem_id=f'download_files_mtg')

        sp_cancel.click(
            fn=lambda: shared.state.interrupt()
        )

        sp_save.click(
            fn=wrap_gradio_call(save.save_files),
            _js="(x, y, z, w) => [x, y, false, selected_gallery_index()]",
            inputs=[
                generation_info,
                sp_gallery,
                html_log,
                html_log,
            ],
            outputs=[
                download_files,
                html_log,
            ]
        )

        sp_run.click(
            fn=wrap_gradio_gpu_call(outpainting_mtg.outpaint, extra_outputs=[gr.update()]),
            _js="start_outpainting_mtg",
            inputs=[
                sp_cardname,
                sp_artist,
                sp_prompt,
                sp_image,
                sp_preprocessor,
                sp_preprocessor_scale,
                sp_postprocessor,
                sp_postprocessor_scale,
                sp_seed,
                sp_negative,
                sp_zoom,
                sp_steps,
            ],
            outputs=[
                sp_gallery,
                generation_info,
                sp_progress,
                sp_outcome,
            ],
        )

    return (sp_interface, "MtG Card Art", "mtgcardart_interface"),

script_callbacks.on_ui_tabs(on_ui_tabs)
