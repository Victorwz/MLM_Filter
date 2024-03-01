from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria

import requests
import argparse
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from typing import List, Tuple
import logging
import os, sys
import torch
import transformers

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("Image Scoring")

LLAVA_INSTRUCTION_TEMPLATE = """Text Caption: {caption}

{criteria} A higher score indicates higher level of {aspect}. Ensure that your scoring is nuanced and uses the entire range from 0 to 100, reflecting the subtle differences. The score should be given as an integer, with each number between 0 and 100 considered as a potential score, avoiding the tendency to round to multiples of 10.  Please first output a single line containing the value indicating the scores. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias."""

criteria_image_text_matching = "Please evaluate if the provided text caption accurately represents the main features and objects of the image. The caption doesn't need to detail every aspect of the image, but it should capture its primary theme. Rate the overall quality of the text caption's match to the image on a scale of 1-100, considering the criteria mentioned."

criteria_object_detail_fulfillment = "Please evaluate the text caption to determine if it provides detailed descriptions of objects that align with the image. Specifically, assess if the caption sufficiently describes the color, size, position, shape, material, etc., of the objects. Afterward, rate the caption's overall accuracy in capturing object details from the image on a scale of 1-100, based on the criteria provided."

criteria_cation_text_quality = "Please evaluate the text caption based on the following criteria: Grammatical Correctness, Diversity of Vocabulary (e.g., the range and uniqueness of words used), Fluency (e.g., smoothness and natural flow of sentences), Readability, Length, and Structure. Assign an overall quality score on a scale of 1-100."

criteria_semantic_understanding = """Evaluate the given text caption in relation to its corresponding image. Your goal is to determine if the text caption provides additional semantic information that isn't readily apparent just from the image itself.

For example:

1. If the image mentions "a man" but the caption elaborates he is a "homeless man" or a "businessman," then the caption is enriching the semantic context.
2. If the caption introduces concepts like the mathematical tangent function, which require in-depth knowledge to deduce, it is imparting external semantics.
3. Captions revealing specific location addresses, festival details, or other nuanced data not easy to infer from the image also provide external semantic information.
4. Directly identifying specific entities in the image such as buildings, people, bird species, animal breeds, car models, engines, etc., in the caption introduces additional insights.
5. Should the image act as a contextual backdrop and the caption describes elements not explicitly showcased in the image, it has semantic depth.
6. Lastly, if the caption depicts relationships between the subjects in the image, which need commonsense knowledge to understand, it should be considered semantically rich.

Please assess and determine the extent of semantic enrichment the caption provides over the image. Rate the text caption's semantic depth on a scale from 1 to 100."""

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def main(args):
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=device)
    # set padding side to `left` for batch text generation
    model.config.tokenizer_padding_side = tokenizer.padding_side = "left"

    logger.info(f"Model loading finished for CUDA {torch.cuda.current_device()}")

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        logger.info('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode
    
    ALL_METRICS = {
        "image_text_matching": criteria_image_text_matching,
        "object_detail_fulfillment": criteria_object_detail_fulfillment,
        "caption_text_quality": criteria_cation_text_quality,
        "semantic_understanding": criteria_semantic_understanding,
    }

    if args.metric == "all":
        evaluation_metrics = ALL_METRICS
    else:
        evaluation_metrics = {args.metric: ALL_METRICS[args.metric]}
    
    for metric, criteria in evaluation_metrics.items():
        logger.info(f"The {metric} score is:")
        image = load_image(args.image_path)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
        text = args.caption
        text = LLAVA_INSTRUCTION_TEMPLATE.format(caption=text, criteria=criteria, aspect=metric.replace("_", " "))

        conv = conv_templates[args.conv_mode].copy()
        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text
            else:
                text = DEFAULT_IMAGE_TOKEN + '\n' + text
            conv.append_message(conv.roles[0], text)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        
        with torch.no_grad():
            conv = conv_templates[args.conv_mode].copy()
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=3 if args.score_only else 512,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
            outputs = [output.strip().strip("</s>") for output in outputs]
            logger.info(outputs[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="wangweizhi98/mlm-filter-llava-13b-gpt4v")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--metric", type=str, default="image_text_matching")
    parser.add_argument("--image-path", type=str, default="./assets/sample.png")
    parser.add_argument("--caption", type=str, default="an image with two histograms")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--score_only", action="store_true")
    
    args = parser.parse_args()
    logger.info(args)
    main(args)