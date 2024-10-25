from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from PIL import Image
import requests
import cv2
import torch

# load model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer, model, image_processor, context_len = load_pretrained_model("weizhiwang/Video-Language-Model-Llama-3.1-8B", None, "Video-Language-Model-Llama-3.1-8B", False, False, device=device)

# prepare image input
url = "https://github.com/PKU-YuanGroup/Video-LLaVA/raw/main/videollava/serve/examples/sample_demo_1.mp4"

def read_video(video_url):
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to download video")
        exit()
    else:
        with open("tmp_video.mp4", 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
    
    video = cv2.VideoCapture("tmp_video.mp4")
    video_frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        video_frames.append(pil_image)

    video.release()
    print(len(video_frames), "frames read.")
    return video_frames

video_frames = read_video(video_url=url)
image_tensors = []
samplng_interval = int(len(video_frames) / 30)
for i in range(0, len(video_frames), samplng_interval):
    image_tensor = image_processor.preprocess(video_frames[i], return_tensors='pt')['pixel_values'][0].half().cuda()
    image_tensors.append(image_tensor)

# prepare inputs for the model
text = "\n".join(['<image>' for i in range(len(image_tensors))]) + '\n' + "Why is this video funny"
conv = conv_templates["llama_3"].copy()
conv.append_message(conv.roles[0], text)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).cuda()

# autoregressively generate text
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensors,
        do_sample=False,
        max_new_tokens=512,
        use_cache=True)

outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
print(outputs[0])