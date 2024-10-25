import shutil
import subprocess

import torch
import gradio as gr
from fastapi import FastAPI
import os
from PIL import Image
import tempfile
from decord import VideoReader, cpu
from transformers import TextStreamer
import argparse

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle, Conversation
from llava.serve.gradio_utils import Chat, tos_markdown, learn_more_markdown, title_markdown, block_css
from llava.constants import LOGDIR
from llava.utils import (build_logger, server_error_msg,
    violates_moderation, moderation_msg)
import hashlib

conv_mode = "llama_3"
model_path = 'weizhiwang/LLaVA-Video-Llama-3'
device = 'cuda'
load_8bit = False
load_4bit = False
dtype = torch.float16
handler = Chat(model_path, conv_mode=conv_mode, load_8bit=load_8bit, load_4bit=load_8bit, device=device)

logger = build_logger("gradio_web_server", "gradio_web_server.log")

SYSTEM_PROMPT = """The input sequence of images are frames extracted from a short video. The video is about a comprehensive guidance to perform a specific task.

The full task guideline is as follows:
"""

def save_image_to_local(image):
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.jpg')
    image = Image.open(image)
    image.save(filename)
    # print(filename)
    return filename


def save_video_to_local(video_path):
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.mp4')
    shutil.copyfile(video_path, filename)
    return filename


def generate(image1, video, textbox_in, manual_textbox_in, first_run, state, state_, images_tensor):
    flag = 1
    if not textbox_in:
        if len(state_.messages) > 0:
            textbox_in = state_.messages[-1][1]
            state_.messages.pop(-1)
            flag = 0
        else:
            return "Please enter instruction"

    image1 = image1 if image1 else "none"
    video = video if video else "none"
    # assert not (os.path.exists(image1) and os.path.exists(video))

    if type(state) is not Conversation:
        state = conv_templates[conv_mode].copy()
        state_ = conv_templates[conv_mode].copy()
        images_tensor = []

    first_run = False if len(state.messages) > 0 else True

    text_en_in = textbox_in.replace("picture", "image")

    # images_tensor = [[], []]
    image_processor = handler.image_processor
    if os.path.exists(image1) and not os.path.exists(video):
        tensor = image_processor.preprocess(Image.open(image1).convert("RGB"), return_tensors='pt')['pixel_values'][0]
        # print(tensor.shape)
        tensor = tensor.to(handler.model.device, dtype=dtype)
        images_tensor.append(tensor)
    # video_processor = handler.video_processor
    if not os.path.exists(image1) and os.path.exists(video):
        tensor = handler.process_video(video, return_tensors='pt')#['pixel_values'][0]
        # print(tensor.shape)
        tensor = [t.to(handler.model.device, dtype=dtype) for t in tensor]
        images_tensor += tensor

    if os.path.exists(image1) and os.path.exists(video):
        tensor = handler.process_video(video, return_tensors='pt')['pixel_values'][0]
        # print(tensor.shape)
        tensor = tensor.to(handler.model.device, dtype=dtype)
        images_tensor.append(tensor)

        tensor = image_processor.preprocess(image1, return_tensors='pt')['pixel_values'][0]
        # print(tensor.shape)
        tensor = tensor.to(handler.model.device, dtype=dtype)
        images_tensor.append(tensor)

    if os.path.exists(image1) and not os.path.exists(video):
        text_en_in = DEFAULT_IMAGE_TOKEN + '\n' + SYSTEM_PROMPT + manual_textbox_in + text_en_in if first_run else text_en_in
    if not os.path.exists(image1) and os.path.exists(video):
        text_en_in = '\n'.join([DEFAULT_IMAGE_TOKEN] * len(images_tensor)) + '\n' + SYSTEM_PROMPT + manual_textbox_in + text_en_in if first_run else text_en_in
    if os.path.exists(image1) and os.path.exists(video):
        text_en_in = '\n'.join([DEFAULT_IMAGE_TOKEN] * len(images_tensor)) + '\n' + SYSTEM_PROMPT + manual_textbox_in + text_en_in + '\n' + DEFAULT_IMAGE_TOKEN if first_run else text_en_in
    # print(text_en_in)
    text_en_out, state_ = handler.generate(images_tensor, text_en_in, first_run=first_run, state=state_)
    state_.messages[-1] = (state_.roles[1], text_en_out)

    text_en_out = text_en_out.split('#')[0]
    textbox_out = text_en_out

    show_images = ""
    if os.path.exists(image1):
        filename = save_image_to_local(image1)
        show_images += f'<img src="./file={filename}" style="display: inline-block;width: 250px;max-height: 400px;">'
    # if os.path.exists(video):
    #     filename = save_video_to_local(video)
    #     show_images += f'<video controls playsinline width="500" style="display: inline-block;"  src="./file={filename}"></video>'

    if flag:
        state.append_message(state.roles[0], textbox_in + "\n" + show_images)
    state.append_message(state.roles[1], textbox_out)

    return (state, state_, state.to_gradio_chatbot(), False, gr.update(value=None, interactive=True), images_tensor, gr.update(value=image1 if os.path.exists(image1) else None, interactive=True))#, gr.update(value=video if os.path.exists(video) else None, interactive=True))


def regenerate(state, state_):
    state.messages.pop(-1)
    state_.messages.pop(-1)
    if len(state.messages) > 0:
        return state, state_, state.to_gradio_chatbot(), False
    return (state, state_, state.to_gradio_chatbot(), True)


def clear_history(state, state_):
    state = conv_templates[conv_mode].copy()
    state_ = conv_templates[conv_mode].copy()
    return (gr.update(value=None, interactive=True),
            gr.update(value=None, interactive=True), \
            gr.update(value=None, interactive=True), \
            True, state, state_, state.to_gradio_chatbot(), [])

def build_demo(embed_mode, cur_dir=None, concurrency_count=10):

    # handler.model.to(dtype=dtype)
    if not os.path.exists("temp"):
        os.makedirs("temp")

    app = FastAPI()

    textbox = gr.Textbox(
        show_label=False, placeholder="Enter text and press ENTER", container=False
    )
    manual_textbox = gr.Textbox(
        show_label=False, placeholder="Insert the Task Manual", container=False
    )
    with gr.Blocks(title='Video-LLaVA🚀', theme=gr.themes.Default(), css=block_css) as demo:
        gr.Markdown(title_markdown)
        state = gr.State()
        state_ = gr.State()
        first_run = gr.State()
        images_tensor = gr.State()

        with gr.Row():
            with gr.Column(scale=3):
                image1 = gr.Image(label="Input Image", type="filepath")
                video = gr.Video(label="Input Video")
                with gr.Column(scale=8):
                    manual_textbox.render()
                cur_dir = os.path.dirname(os.path.abspath(__file__))
                gr.Examples(
                    examples=[
                        [
                            f"{cur_dir}/examples/extreme_ironing.jpg",
                            "What is unusual about this image?",
                        ],
                        [
                            f"{cur_dir}/examples/waterview.jpg",
                            "What are the things I should be cautious about when I visit here?",
                        ],
                        [
                            f"{cur_dir}/examples/desert.jpg",
                            "If there are factual errors in the questions, point it out; if not, proceed answering the question. What’s happening in the desert?",
                        ],
                    ],
                    inputs=[image1, textbox],
                )

            with gr.Column(scale=7):
                chatbot = gr.Chatbot(label="LLaVA-Video-Llama-3", bubble_full_width=True, height=750)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(
                            value="Send", variant="primary", interactive=True
                        )
                with gr.Row(elem_id="buttons") as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=True)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=True)
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=True)
                    # stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=True)
                    clear_btn = gr.Button(value="🗑️  Clear history", interactive=True)

        with gr.Row():
            # gr.Examples(
            #     examples=[
            #         [
            #             f"{cur_dir}/examples/sample_img_22.png",
            #             f"{cur_dir}/examples/sample_demo_22.mp4",
            #             "Are the instruments in the pictures used in the video?",
            #         ],
            #         [
            #             f"{cur_dir}/examples/sample_img_13.png",
            #             f"{cur_dir}/examples/sample_demo_13.mp4",
            #             "Does the flag in the image appear in the video?",
            #         ],
            #         [
            #             f"{cur_dir}/examples/sample_img_8.png",
            #             f"{cur_dir}/examples/sample_demo_8.mp4",
            #             "Are the image and the video depicting the same place?",
            #         ],
            #     ],
            #     inputs=[image1, video, textbox],
            # )
            gr.Examples(
                examples=[
                    [
                        f"{cur_dir}/examples/sample_demo_1.mp4",
                        "This next job is removal of the recoil starter. Again, it is just the removal, not the install. And that's if you need to take additional parts off, drilling down deeper into the engine. You would remove it, set it aside for other maintenance, and then return and do the install task as you're reassembling the engine. So first thing you want to do is, if you remove both these top two nuts first, as you start loosening up the bottom nut, it can kind of shift left or right on you. So it's best to loosen them all up and leave one of the top ones in to be the last thing removed. So we'll loosen these three up. Once they're loose... I don't know if you can see that bolt down there, bottom dead center. I'm going to go ahead and get a maintenance ball out here to hold some of these parts. Alright, three bolts and that's it. This thing comes off. I'll talk about it more when we install, but as you can see, there's holes all the way around, and there's three holes evenly spaced here. So depending on how this thing is mounted, whether it's on a snowmobile, a generator, this pull starter can be rotated into different positions to facilitate different mounting positions on different pieces of equipment. So you may see this mounted in different ways, and that's okay. It can be mounted in a full clock position. And that is removal of the recoil starter.\n",
                        "The input sequence of images follow time order and stop on a specific step of the whole task. Can you reason step by step to predict what is the next step to complete in this task?\n",
                    ],
[
                        f"{cur_dir}/examples/sample_demo_2.mp4",
                        "This next task is similar to the first task we did, which was inspecting the air filters. This time we're going to remove the entire air assembly to get to the carburetor or other aspects. We're doing it for other maintenance removal. So we'll remove the nut or finger knot again, the air cleaner cover, the wing nut, the air filter assembly with the foam filter on the outside, and the grommet. We'll just put that to the side. The noise silencer, these two little baffles right here help reduce high frequency noise. And then now we need to remove what they call the air cleaner elbow. It's actually held in place with one bolt. Two nuts. And to remove this, you need to take your choke and pull it about halfway, and then your fuel shut off about halfway too. That way this can come out. We're going to pull the breather tube, which sits inside the overhead valve. It just snugly fits in there to allow breathing of the overhead valve. We will just pull that out. It doesn't clamp in or anything. And then we will remove... Might have to adjust these knobs just a little bit. And we will remove the air filter elbow. This allows you additional visibility to the control base, carburetor, spark plug, overhead valve. And this is, again, removal. So this is a whole task. We've removed it for other maintenance. There will be a separate install job later after you've done whatever you needed to do, maybe replace the carburetor. Then we'll reinstall it at another point for another job.\n",
                        "The input sequence of images follow time order and stop on a specific step of the whole task. Can you reason step by step to predict what is the next step to complete in this task?\n",
                    ],
                    # [
                    #     f"{cur_dir}/examples/sample_demo_9.mp4",
                    #     "Describe the video.",
                    # ],
                    # [
                    #     f"{cur_dir}/examples/sample_demo_22.mp4",
                    #     "Describe the activity in the video.",
                    # ],
                ],
                inputs=[video, manual_textbox, textbox],
            )

        submit_btn.click(generate, [image1, video, textbox, manual_textbox, first_run, state, state_, images_tensor],
                        [state, state_, chatbot, first_run, textbox, images_tensor, image1])

        regenerate_btn.click(regenerate, [state, state_], [state, state_, chatbot, first_run]).then(
            generate, [image1, video, textbox, manual_textbox, first_run, state, state_, images_tensor], [state, state_, chatbot, first_run, textbox, images_tensor, image1])

        clear_btn.click(clear_history, [state, state_],
                        [image1, video, textbox, manual_textbox, first_run, state, state_, chatbot, images_tensor])

    return demo
# app = gr.mount_gradio_app(app, demo, path="/")
# demo.launch()

# uvicorn videollava.serve.gradio_web_server:app
# python -m  videollava.serve.gradio_web_server
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=16)
    parser.add_argument("--model-list-mode", type=str, default="once",
        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # models = get_model_list()

    logger.info(args)
    demo = build_demo(args.embed, concurrency_count=args.concurrency_count)
    demo.queue(
        api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )
