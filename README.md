# MLM Filter


Official implementation of our paper "[Fine-Tuned Multimodal Language Models are High-Quality Image-Text Data Filters](https://arxiv.org/abs/)". <br>
[Weizhi Wang](https://victorwz.github.io/), [Khalil Mirini](https://khalilmrini.github.io/), [Linjie Yang](https://sites.google.com/site/linjieyang89/), [Sateesh Kumar](https://sateeshkumar21.github.io/), [Yu Tian](https://scholar.google.com/citations?user=DxPjkDoAAAAJ&hl=en), [Xifeng Yan](https://sites.cs.ucsb.edu/~xyan/index.htm), [Heng Wang](https://hengcv.github.io/)

Please cite our paper if you find this repository interesting or helpful:
```bibtex
@article{
}
```

## Release
- [2/25] 🔥 We released **Fine-Tuned Multimodal Language Models are High-Quality Image-Text Data Filters**. We propose to adopt fine-tuned Multimodal Language Model as effective and efficient data filters to select high-quality image-text pairs from large-scale web-crawled iamge-text data.  Checkout the [paper](https://arxiv.org/abs/2402).

<!-- <a href="https://llava.hliu.cc/"><img src="assets/demo.gif" width="70%"></a> -->


## Project Structure
- [LLaVA_ft](LLaVA_ft): Fine-Tune MLM as Data Filter
- [llava_scoring_datacomp_batch_inference.py](llava_scoring_datacomp_batch_inference.py): Sample code for perform large-scale quality score generation on Webdataset format image-text data
- [run_inference.sh](run_inference.sh): Sample code for perform large-scale quality score generation on Webdataset format image-text data on machines with 8 GPUs

## Install

Install dependencies for quality score generation:
```
bash setup.sh
```

<!-- ### CLI Inference

Chat about images using LLaVA without the need of Gradio interface. It also supports multiple GPUs, 4-bit and 8-bit quantized inference. With 4-bit quantization, for our LLaVA-1.5-7B, it uses less than 8GB VRAM on a single GPU.

```Shell
python 
``` -->

<!-- <img src="images/demo_cli.gif" width="70%"> -->

## Fine-Tuning MLM as Data Filter

1. Prepare data

Please download the images from constituting datasets:

- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), **we save all files as `.jpg`**
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)
- CC12M: ```unzip images.zip -C data/images```

After downloading all of them, organize the data as follows in `./data/images`,

```
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
│   ├── VG_100K
│   └── VG_100K_2
└── cc12m
```

2. Start training!

You may download LLaVA's pretrained projectors in [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md).

Visual instruction tuning takes around 4 hours for LLaVA-v1.5-13B on 8x A100 (80G) with sampled 50k instruction dataset.

Training script with DeepSpeed ZeRO-3: [`LLaVA_ft/v_1_5/finetune.sh`](LLaVA_ft/v_1_5/finetune.sh).

## Quality Score Generation

```Shell
bash run_inference.sh ${GPU_START_ID} ${Metric} ${Model_Path} ${Data_Path} ${Tars_Per_GPU} ${NUM_GPU}
```
Parameters to note:

- `GPU_START_ID`: for large-scale score generation using multi-machines, specify the index of machines
- `Metric`: quality scoring metric for generation, select among `image_text_matching`, `object_detail_fulfillment`, `caption_text_quality`, `semantic_understanding`, `all`
- `Model_Path`: path to the mlm filter model checkpoint
- `Data_Path`: path to the webdataset image-text tars
- `Tars_Per_GPU`: the number of webdataset image-text tars for a single-gpu to inference on
- `NUM_GPU`: the number of GPUs for one machine, e.g. 1, 8, 16


## License
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)
<br>
**Usage and License Notices**: The data and checkpoint are intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA, Vicuna and GPT-4. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.

## Contacts
For any question or issue, please feel free to contact [weizhiwang@ucsb.edu]() or submit github issues.

## Credits

MLM-Filter is developed based on
- [Vicuna](https://github.com/lm-sys/FastChat): foudation language model for LLaVA
- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase for fine-tuning LLaVA as image-text data filters
- [DataComp](https://github.com/mlfoundations/datacomp): the codebase for data filtering and CLIP pre-training