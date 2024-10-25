# LLaVA-Video-LLaMA-3

This repo supports the video understanding based on Llama-3-8b LLM backbone following LLaVA multimodal LLM architecture.

## Models
🤝 [[LLaVA-Video-Llama-3.1-8B](https://huggingface.co/weizhiwang/LLaVA-Video-Llama-3.1-8B)]

🤝 [[LLaVA-Video-Llama-3](https://huggingface.co/weizhiwang/LLaVA-Video-Llama-3)]

## Updates
- [8/11/2024] A completely new video-based LLM [LLaVA-Video-Llama-3.1-8B](https://huggingface.co/weizhiwang/LLaVA-Video-Llama-3.1-8B) is released, with the SigLIP-g-384px as vision encoder and average pooling vision-language projector.
- [6/4/2024] The codebase supports the video data fine-tuning for video understanding tasks.
- [5/14/2024] The codebase has been upgraded to llava-next (llava-v1.6). Now it supports the latest llama-3, phi-3, mistral-v0.1-7b models.

## Install

If you are using Windows, do *NOT* proceed, see instructions [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/Windows.md).

1. Setup
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### Fine-Tune Your Own LLaVA-Video-Llama-3 Model
Please follow the updated fine-tuning script with DeepSpeed ZeRO-3: [`finetune.sh`](https://github.com/Victorwz/LLaVA-Llama-3/blob/main/scripts/finetune.sh). The following parameters are updated to accomodate Llama-3:
- `--version`: v3, which adopts the tokenization and preprocessing function with Llama-3 tokenizer.

Please download the pre-trained vision-language projector weights in [Projector_MODEL](https://huggingface.co/weizhiwang/llava-v1.5-llama-3-8b-pretrain-clip-large-336px).

In terms of the image data preparation, please follow [`DATA.md`](DATA.md). The mixed SFT data with video instructions is available at [`video_data`](https://huggingface.co/datasets/weizhiwang/llava_v15_instruction_images/resolve/main/llava_phi_3_video_mix.json?download=true).

## Demo with Gradio
```shell
CUDA_VISIBLE_DEVICES=0 python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload --share
```

## Evaluation

TODO


## Credits
This is a reproduction project, all research credits should be attributed to original authors for LLaVA. Please cite their papers listed below as well.

```bibtex
@misc{wang2024llavavideollama3,
  title={LLaVA-Video-Llama-3: A Video Understanding Multimodal LLM based on Llama-3-8B LLM backbone},
  author={Wang, Weizhi},
  year={2024}
}
```

```bibtex
@misc{wang2024llavallama3,
  title={LLaVA-Llama-3-8B: A reproduction towards LLaVA-v1.5 based on Llama-3-8B LLM backbone},
  author={Wang, Weizhi},
  year={2024}
}
```
