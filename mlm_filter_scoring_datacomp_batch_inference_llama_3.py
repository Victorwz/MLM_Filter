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
import pandas as pd

from tqdm import tqdm
from typing import List, Tuple
import logging
import os, sys
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
import webdataset as wds
from dataclasses import dataclass, field

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("evaluation test")

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

@dataclass
class DataCollatorForImagePreprocessing(object):
    def __init__(self, tokenizer, image_processor, mm_use_im_start_end, criteria, task, conv_mode, max_len): 
        self.image_processor = image_processor
        self.mm_use_im_start_end = mm_use_im_start_end
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.conv_mode = conv_mode
        self.criteria = criteria
        self.task = task
        self.max_len = max_len
        self.whitespace_id = 220

    def format_text(self, text: str):
        text = LLAVA_INSTRUCTION_TEMPLATE.format(caption=text, criteria=self.criteria, aspect=self.task.replace("_", " "))
        if self.mm_use_im_start_end:
            text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + text
        else:
            text = DEFAULT_IMAGE_TOKEN + "\n" + text

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        text = conv.get_prompt()

        return text

    # @staticmethod
    def pad_sequence(self, sequence, padding_value=0):
        """Pad a sequence to the desired max length."""
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in sequence] 
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self,
                 batch: Tuple[List, List, List]) -> Tuple[torch.Tensor, torch.Tensor, list]:
        images, txts, infos = batch
        prompt = [self.format_text(text) for text in txts]
        batch_input_ids = [
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")[:self.max_len] for prompt in prompt
        ]
        batch_input_ids = self.pad_sequence(batch_input_ids)
        # directly add 220 whitespace to decrease the decoding cost for it
        batch_input_ids = torch.cat((batch_input_ids, torch.tensor([[self.whitespace_id]]).repeat(batch_input_ids.shape[0], 1)), dim=1)
        batch_image_tensor = self.image_processor(images, return_tensors="pt")["pixel_values"]

        return (batch_image_tensor, batch_input_ids, infos)


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def main(args, gpu_id=0):
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=device)
    # set padding side to `left` for batch text generation
    model.config.tokenizer_padding_side = tokenizer.padding_side = "left"

    logger.info(f"Model loading finished for CUDA {torch.cuda.current_device()}")
    
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
    
    for task, criteria in evaluation_metrics.items():
        
        for tar_id in list(range(13000))[gpu_id * args.tars_per_gpu: (gpu_id + 1) * args.tars_per_gpu]:
            logger.info(f"Start processing tar {tar_id}")
            collator = DataCollatorForImagePreprocessing(tokenizer, image_processor, model.config.mm_use_im_start_end, criteria, task, args.conv_mode, args.max_len)
            if os.path.exists(os.path.join(args.tar_file_path, f"{str(tar_id).zfill(8)}_{task}.parquet")):
                logger.info(f"Tar {tar_id} already processed")
                continue
            shard_path = args.tar_file_path + "/{:08d}.tar".format(tar_id)
            pipeline = [
                wds.SimpleShardList(shard_path),
                wds.split_by_worker,
                wds.tarfile_to_samples(),
                wds.decode("pilrgb"),
                wds.rename(image="jpg;png;jpeg;webp", text="txt", json="json"),
                wds.to_tuple("image", "text", "json"),
                wds.batched(args.batch_size, partial=True),
            ]
            dataset = wds.DataPipeline(*pipeline)
            
            dataloader = wds.WebLoader(
                dataset,
                collate_fn=collator,
                batch_size=None,
                shuffle=False,
                num_workers=args.workers,
                persistent_workers=args.workers > 0,
            )
            
            final_data = []
            for batch_image_tensor, batch_input_ids, info in tqdm(dataloader):
                with torch.inference_mode():
                    output_ids = model.generate(
                        batch_input_ids.cuda(),
                        images=batch_image_tensor.half().cuda(),
                        do_sample=False,
                        max_new_tokens=1,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id)

                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                
                for i in range(batch_input_ids.shape[0]):
                    info[i][f"{task}_score"] = outputs[i]

                final_data += info
            
            df = pd.DataFrame(final_data)
            df.to_parquet(f"{args.tar_file_path}/{str(tar_id).zfill(8)}_{task}.parquet")
            logger.info(f"Tar {tar_id} finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/share/edc/home/weizhiwang/models/mlm-filter-llava-llama-3-8b-gpt4v/")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--metric", type=str, default="image_text_matching")
    parser.add_argument("--tar-file-path", type=str, default="/share/edc/home/weizhiwang/data/medium/medium_rule_shards/")
    parser.add_argument("--num-gpus", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tars-per-gpu", type=int, default=128)
    parser.add_argument("--conv-mode", type=str, default="llama_3")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-len", type=int, default=2040)
    args = parser.parse_args()
    logger.info(args)
    main(args, gpu_id=args.gpu_id)
