#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
T2I-Adapter for Stable Diffusion XL 训练脚本。

该脚本用于训练 T2I-Adapter 模型，这是一种轻量级的条件控制模型，
可以添加到预训练的 Stable Diffusion XL 模型中，实现基于条件图像
（如边缘图、深度图、姿态图等）的精确图像生成控制。

主要功能：
1. 支持从 Hugging Face Hub 或本地加载预训练模型
2. 支持分布式训练和混合精度训练
3. 包含完整的训练循环、验证和模型保存
4. 支持 TensorBoard 和 WandB 日志记录
5. 支持将训练好的模型上传到 Hugging Face Hub

使用示例：
    accelerate launch train_t2i_adapter_sdxl.py \
        --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
        --dataset_name="your_dataset" \
        --output_dir="t2iadapter-model" \
        --train_batch_size=4 \
        --num_train_epochs=100 \
        --checkpointing_steps=5000 \
        --validation_steps=1000

参考：
    - T2I-Adapter 论文：https://arxiv.org/abs/2302.08453
    - Stable Diffusion XL：https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
"""

import argparse
import functools
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module


MAX_SEQ_LENGTH = 77

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.37.0.dev0")

logger = get_logger(__name__)


def image_grid(imgs, rows, cols):
    """
    将多张图像排列成网格形式。

    Args:
        imgs (list of PIL.Image.Image): 要排列的图像列表
        rows (int): 网格的行数
        cols (int): 网格的列数

    Returns:
        PIL.Image.Image: 组合后的网格图像

    Raises:
        AssertionError: 如果图像数量不等于 rows * cols
    """
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def log_validation(vae, unet, adapter, args, accelerator, weight_dtype, step):
    """
    运行验证并记录生成的图像。

    该函数在训练过程中定期调用，用于评估当前模型在验证集上的表现，
    并将生成的图像记录到 TensorBoard 或 WandB 等跟踪器中。

    Args:
        vae (AutoencoderKL): VAE 模型，用于图像编码和解码
        unet (UNet2DConditionModel): UNet 模型
        adapter (T2IAdapter): 正在训练的 T2I-Adapter 模型
        args (argparse.Namespace): 命令行参数
        accelerator (Accelerator): Accelerator 实例，用于分布式训练
        weight_dtype (torch.dtype): 权重数据类型（如 torch.float16, torch.bfloat16）
        step (int): 当前训练步数，用于日志记录

    Returns:
        list: 包含验证结果的字典列表，每个字典包含：
            - validation_image: 条件图像
            - images: 生成的图像列表
            - validation_prompt: 对应的提示词
    """
    logger.info("Running validation... ")

    # 从分布式训练中解包适配器模型
    adapter = accelerator.unwrap_model(adapter)

    # 创建推理管道
    pipeline = StableDiffusionXLAdapterPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        unet=unet,
        adapter=adapter,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # 启用 xformers 内存高效注意力（如果可用）
    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    # 设置随机种子生成器
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # 处理验证图像和提示词的配对
    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []

    # 对每个验证图像-提示词对进行推理
    for validation_prompt, validation_image in zip(validation_prompts, validation_images):
        validation_image = Image.open(validation_image).convert("RGB")
        validation_image = validation_image.resize((args.resolution, args.resolution))

        images = []

        # 为每个条件生成多个图像（增加统计可靠性）
        for _ in range(args.num_validation_images):
            with torch.autocast("cuda"):
                image = pipeline(
                    prompt=validation_prompt, image=validation_image, num_inference_steps=20, generator=generator
                ).images[0]
            images.append(image)

        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    # 将图像记录到不同的跟踪器
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images = [np.asarray(validation_image)]

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images.append(wandb.Image(validation_image, caption="adapter conditioning"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({"validation": formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

        # 清理内存
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        return image_logs


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    """
    根据模型名称或路径导入对应的文本编码器类。

    SDXL 使用两个文本编码器：CLIPTextModel 和 CLIPTextModelWithProjection。
    该函数通过读取模型的配置文件来确定使用哪个类。

    Args:
        pretrained_model_name_or_path (str): 预训练模型的名称或路径
        revision (str): 模型版本（如 main、fp16 等）
        subfolder (str, optional): 子文件夹名称，默认为 "text_encoder"

    Returns:
        type: 文本编码器类（CLIPTextModel 或 CLIPTextModelWithProjection）

    Raises:
        ValueError: 如果模型类不被支持
    """
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs: dict = None, base_model: str = None, repo_folder: str = None):
    """
    创建并保存模型卡片（README.md）到输出目录。

    模型卡片包含模型描述、训练信息、示例图像和必要的标签，
    用于在 Hugging Face Hub 上展示模型。

    Args:
        repo_id (str): 仓库 ID，用于模型卡片标题
        image_logs (dict, optional): 验证图像日志，包含生成的示例图像
        base_model (str, optional): 基础模型名称（如 "stabilityai/stable-diffusion-xl-base-1.0"）
        repo_folder (str, optional): 仓库文件夹路径，用于保存模型卡片

    Returns:
        None: 函数将模型卡片保存到 repo_folder/README.md
    """
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            # 保存条件图像
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            # 将条件图像和生成图像组合成网格
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    # 构建模型描述
    model_description = f"""
# t2iadapter-{repo_id}

These are t2iadapter weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    # 加载或创建模型卡片
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    # 添加相关标签
    tags = [
        "stable-diffusion-xl",
        "stable-diffusion-xl-diffusers",
        "text-to-image",
        "diffusers",
        "t2iadapter",
        "diffusers-training",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    # 保存模型卡片
    model_card.save(os.path.join(repo_folder, "README.md"))


def parse_args(input_args=None):
    """
    解析命令行参数。

    该函数定义了 T2I-Adapter 训练脚本的所有可配置参数，
    包括模型路径、训练超参数、数据集配置、验证设置等。

    Args:
        input_args (list, optional): 命令行参数列表。如果为 None，则从 sys.argv 读取。

    Returns:
        argparse.Namespace: 包含所有解析后参数的命名空间对象

    Raises:
        ValueError: 如果参数组合无效（如缺少必需参数或参数值无效）
    """
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to an improved VAE to stabilize training. For more details check out: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--adapter_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained adapter model or model identifier from huggingface.co/models."
        " If not specified adapter weights are initialized w.r.t the configurations of SDXL.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="t2iadapter-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--detection_resolution",
        type=int,
        default=None,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=3,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help=("Number of subprocesses to use for data loading."),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the adapter conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the t2iadapter conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="sd_xl_train_t2iadapter",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the t2iadapter encoder."
        )

    return args


def get_train_dataset(args, accelerator):
    """
    加载和预处理训练数据集。

    支持从 Hugging Face Hub 下载数据集或从本地目录加载。
    数据集应包含三列：图像、条件图像和文本描述。

    Args:
        args (argparse.Namespace): 命令行参数
        accelerator (Accelerator): Accelerator 实例，用于分布式训练中的进程同步

    Returns:
        Dataset: 预处理后的训练数据集

    Raises:
        ValueError: 如果指定的列名在数据集中不存在
    """
    # 获取数据集：可以从 Hugging Face Hub 下载或使用本地文件
    # 在分布式训练中，load_dataset 函数确保只有一个本地进程同时下载数据集
    if args.dataset_name is not None:
        # 从 Hub 下载和加载数据集
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        if args.train_data_dir is not None:
            dataset = load_dataset(
                args.train_data_dir,
                cache_dir=args.cache_dir,
            )
        # 有关加载自定义图像的更多信息，请参阅：
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # 预处理数据集
    column_names = dataset["train"].column_names

    # 获取输入/目标的列名
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    # 在主进程中先进行数据预处理，然后广播到其他进程
    with accelerator.main_process_first():
        train_dataset = dataset["train"].shuffle(seed=args.seed)
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))
    return train_dataset


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True):
    """
    对文本提示进行编码，生成文本嵌入。

    SDXL 使用两个文本编码器，该函数将两个编码器的输出拼接起来。
    支持空提示替换（用于 classifier-free guidance）。

    Args:
        prompt_batch (list): 文本提示列表
        text_encoders (list): 文本编码器列表（两个）
        tokenizers (list): 分词器列表（两个）
        proportion_empty_prompts (float): 空提示的比例（0-1之间）
        is_train (bool, optional): 是否为训练模式。在训练时从多个描述中随机选择，
                                    在推理时选择第一个描述。默认为 True。

    Returns:
        tuple: 包含两个元素的元组：
            - prompt_embeds (torch.Tensor): 拼接后的文本嵌入，形状为 (batch_size, seq_len, hidden_size*2)
            - pooled_prompt_embeds (torch.Tensor): 池化后的文本嵌入，形状为 (batch_size, hidden_size)
    """
    prompt_embeds_list = []

    # 处理提示词：可能替换为空字符串（用于 classifier-free guidance）
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")  # 空提示
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # 如果有多个描述，训练时随机选择一个，推理时选择第一个
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            # 分词
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            
            # 文本编码
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # 我们只对最终文本编码器的池化输出感兴趣
            pooled_prompt_embeds = prompt_embeds[0]  # 池化输出
            prompt_embeds = prompt_embeds.hidden_states[-2]  # 倒数第二层的隐藏状态
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    # 拼接两个文本编码器的输出
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def prepare_train_dataset(dataset, accelerator):
    """
    对训练数据集进行预处理转换。

    包括图像大小调整、中心裁剪、归一化等操作。
    为图像和条件图像分别定义不同的转换管道。

    Args:
        dataset (Dataset): 原始数据集
        accelerator (Accelerator): Accelerator 实例，用于进程同步

    Returns:
        Dataset: 应用了转换的数据集
    """
    # 目标图像的转换管道
    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # 归一化到 [-1, 1] 范围
        ]
    )

    # 条件图像的转换管道（不进行归一化，因为 T2I-Adapter 需要原始像素值）
    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),  # 转换为张量，范围 [0, 1]
        ]
    )

    def preprocess_train(examples):
        """
        单个批次的预处理函数。

        Args:
            examples (dict): 包含图像、条件图像和文本的批次数据

        Returns:
            dict: 预处理后的批次，包含 pixel_values 和 conditioning_pixel_values
        """
        # 处理目标图像
        images = [image.convert("RGB") for image in examples[args.image_column]]
        images = [image_transforms(image) for image in images]

        # 处理条件图像
        conditioning_images = [image.convert("RGB") for image in examples[args.conditioning_image_column]]
        conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images

        return examples

    # 在主进程中先进行数据转换，然后广播到其他进程
    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess_train)

    return dataset


def collate_fn(examples):
    """
    数据加载器的批处理函数。

    将多个样本组合成一个批次，并进行必要的张量转换和内存格式优化。

    Args:
        examples (list of dict): 样本列表，每个样本包含：
            - pixel_values: 目标图像张量
            - conditioning_pixel_values: 条件图像张量
            - prompt_embeds: 文本嵌入
            - text_embeds: 附加文本嵌入（用于 SDXL）
            - time_ids: 时间 ID（用于 SDXL）

    Returns:
        dict: 批处理后的数据，包含：
            - pixel_values: 目标图像批次
            - conditioning_pixel_values: 条件图像批次
            - prompt_ids: 文本嵌入批次
            - unet_added_conditions: 包含 text_embeds 和 time_ids 的字典
    """
    # 堆叠目标图像
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    # 堆叠条件图像
    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    # 堆叠文本嵌入
    prompt_ids = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples])

    # 堆叠 SDXL 特有的附加条件
    add_text_embeds = torch.stack([torch.tensor(example["text_embeds"]) for example in examples])
    add_time_ids = torch.stack([torch.tensor(example["time_ids"]) for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "prompt_ids": prompt_ids,
        "unet_added_conditions": {"text_embeds": add_text_embeds, "time_ids": add_time_ids},
    }


def main(args):
    """
    T2I-Adapter 训练主函数。

    该函数负责：
    1. 初始化训练环境（Accelerator、日志记录等）
    2. 加载预训练模型（VAE、UNet、文本编码器、T2I-Adapter）
    3. 准备数据集和数据加载器
    4. 配置优化器和学习率调度器
    5. 执行训练循环
    6. 定期验证和保存检查点
    7. 保存最终模型并上传到 Hub（如果启用）

    Args:
        args (argparse.Namespace): 命令行参数

    Raises:
        ValueError: 如果参数组合无效（如同时使用 wandb 和 hub_token）
    """
    # 安全检查：不能同时使用 wandb 和 hub_token，因为 wandb 可能会暴露 token
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    # 设置日志目录
    logging_dir = Path(args.output_dir, args.logging_dir)

    # 配置accelerator项目，设置了项目输出目录和日志目录
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    # 初始化 Accelerator（用于分布式训练、混合精度等）
    # 混合精度就是传播的时候用fp16，更新权重的时候用fp32，fp16节省显存，计算速度更快，fp32更稳定
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,   # 梯度积累步数
        mixed_precision=args.mixed_precision,                           # 混合精度设置
        log_with=args.report_to,                                        # 日志记录平台
        project_config=accelerator_project_config,                      # 项目配置
    )

    # Disable AMP for MPS.
    # mps平台关闭自动混合精度，支持不全
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    # 设置日志格式，时间格式，只显示info以上级别的日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # 所有的进程都会打印日志，而不仅仅是主进程
    logger.info(accelerator.state, main_process_only=False)

    # 主进程的日志更详细
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    # 主进程的专属任务：创建输出目录和设置hugging face hub仓库
    if accelerator.is_main_process:
        if args.output_dir is not None:
            # 创建输出目录
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            # 创建hugging face hub仓库
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
                private=True,
            ).repo_id

    # 加载分词器（SDXL 使用两个分词器）
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # 导入正确的文本编码器类
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # 加载调度器和模型组件
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )

    # 确定 VAE 路径：使用指定的 VAE 或默认 VAE
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    # 加载或初始化 T2I-Adapter
    if args.adapter_model_name_or_path:
        # 加载t2i adapter
        logger.info("Loading existing adapter weights.")
        t2iadapter = T2IAdapter.from_pretrained(args.adapter_model_name_or_path)
    else:
        # 初始化t2i adapter
        logger.info("Initializing t2iadapter weights.")
        t2iadapter = T2IAdapter(
            in_channels=3,  # 输入通道数（RGB 图像）
            channels=(320, 640, 1280, 1280),  # 各层通道数
            num_res_blocks=2,  # 每个分辨率块的残差块数量
            downscale_factor=16,  # 下采样因子
            adapter_type="full_adapter_xl",  # 适配器类型（针对 SDXL 的完整适配器）
        )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        # 保存和加载模型的钩子函数
        def save_model_hook(models, weights, output_dir):
            i = len(weights) - 1

            while len(weights) > 0:
                weights.pop()
                model = models[i]

                sub_dir = "t2iadapter"
                model.save_pretrained(os.path.join(output_dir, sub_dir))

                i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = T2IAdapter.from_pretrained(os.path.join(input_dir, "t2iadapter"))

                if args.control_type != "style":
                    model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # 需要训练adapter和unet
    t2iadapter.train()
    unet.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            # 开启xformers高效注意力机制
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    def unwrap_model(model):
        # 解包模型
        model = accelerator.unwrap_model(model) 
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if args.gradient_checkpointing:
        # 开启梯度检查点以节省内存
        unet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(t2iadapter).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(t2iadapter).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        # 开启tf32加速
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        # 如果需要缩放学习率，将原始学习率乘以一个缩放因子，这个因子考虑了梯度累积步数、批量大小和进程数
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        # 8位Adam优化器类，显著减少深度学习训练时的内存占用
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    # 创建优化器，只优化t2i adapter的参数
    params_to_optimize = t2iadapter.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    # 如果开启了混合精度，将仅用于推理的模型（文本编码器和VAE）转换为半精度
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    else:
        vae.to(accelerator.device, dtype=torch.float32)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    # Here, we compute not just the text embeddings but also the additional embeddings
    # needed for the SD XL UNet to operate.
    def compute_embeddings(batch, proportion_empty_prompts, text_encoders, tokenizers, is_train=True):
        """
        计算文本嵌入和 SDXL UNet 所需的附加条件。

        SDXL 需要额外的条件信息，如图像原始尺寸、裁剪坐标和目标尺寸。
        这些信息被编码为 time_ids 并与文本嵌入一起传递给 UNet。

        Args:
            batch (dict): 数据批次，包含文本描述列
            proportion_empty_prompts (float): 空提示的比例
            text_encoders (list): 文本编码器列表
            tokenizers (list): 分词器列表
            is_train (bool, optional): 是否为训练模式。默认为 True。

        Returns:
            dict: 包含以下键的字典：
                - prompt_embeds: 文本嵌入
                - text_embeds: 池化文本嵌入（用于 SDXL 的附加条件）
                - time_ids: 时间 ID（包含尺寸和裁剪信息）
        """
        original_size = (args.resolution, args.resolution)
        target_size = (args.resolution, args.resolution)
        crops_coords_top_left = (args.crops_coords_top_left_h, args.crops_coords_top_left_w)
        prompt_batch = batch[args.caption_column]

        # 编码文本提示
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train
        )
        add_text_embeds = pooled_prompt_embeds

        # 从 pipeline.StableDiffusionXLPipeline._get_add_time_ids 改编
        # 构建 time_ids：包含原始尺寸、裁剪坐标和目标尺寸
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])

        # 移动到正确的设备和数据类型
        prompt_embeds = prompt_embeds.to(accelerator.device)
        add_text_embeds = add_text_embeds.to(accelerator.device)
        add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)
        add_time_ids = add_time_ids.to(accelerator.device, dtype=prompt_embeds.dtype)
        unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        return {"prompt_embeds": prompt_embeds, **unet_added_cond_kwargs}

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        """
        根据时间步获取对应的噪声调度器 sigma 值。

        sigma 值用于缩放噪声，在 EDM 或 Karras 调度器中常用。
        该函数将时间步映射到调度器中的对应 sigma 值。

        Args:
            timesteps (torch.Tensor): 时间步张量
            n_dim (int, optional): 输出张量的目标维度。默认为 4。
            dtype (torch.dtype, optional): 输出数据类型。默认为 torch.float32。

        Returns:
            torch.Tensor: sigma 值，形状为 (batch_size, 1, 1, 1)（当 n_dim=4 时）
        """
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)

        # 找到每个时间步在调度器时间步列表中的索引
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        
        # 扩展维度以匹配输入张量的维度
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # Let's first compute all the embeddings so that we can free up the text encoders
    # from memory.
    text_encoders = [text_encoder_one, text_encoder_two]
    tokenizers = [tokenizer_one, tokenizer_two]
    train_dataset = get_train_dataset(args, accelerator)
    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        proportion_empty_prompts=args.proportion_empty_prompts,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
    )
    with accelerator.main_process_first():
        from datasets.fingerprint import Hasher

        # fingerprint used by the cache for the other processes to load the result
        # details: https://github.com/huggingface/diffusers/pull/4038#discussion_r1266078401
        new_fingerprint = Hasher.hash(args)
        train_dataset = train_dataset.map(compute_embeddings_fn, batched=True, new_fingerprint=new_fingerprint)

    # Then get the training dataset ready to be passed to the dataloader.
    train_dataset = prepare_train_dataset(train_dataset, accelerator)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    t2iadapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        t2iadapter, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(t2iadapter):
                if args.pretrained_vae_model_name_or_path is not None:
                    pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                else:
                    pixel_values = batch["pixel_values"]

                # encode pixel values with batch size of at most 8 to avoid OOM
                latents = []
                for i in range(0, pixel_values.shape[0], 8):
                    latents.append(vae.encode(pixel_values[i : i + 8]).latent_dist.sample())
                latents = torch.cat(latents, dim=0)
                latents = latents * vae.config.scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    latents = latents.to(weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # 立方采样：为每个图像采样一个随机时间步。
                # 使用立方采样（cubic sampling）而不是均匀采样，以便更频繁地采样较小的时间步（更接近干净图像）。
                # 这有助于模型更好地学习去噪过程。更多细节请参考 https://huggingface.co/papers/2302.08453 的第3.4节
                timesteps = torch.rand((bsz,), device=latents.device)
                timesteps = (1 - timesteps**3) * noise_scheduler.config.num_train_timesteps
                timesteps = timesteps.long().to(noise_scheduler.timesteps.dtype)
                timesteps = timesteps.clamp(0, noise_scheduler.config.num_train_timesteps - 1)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Scale the noisy latents for the UNet
                sigmas = get_sigmas(timesteps, len(noisy_latents.shape), noisy_latents.dtype)
                inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)

                # Adapter conditioning.
                t2iadapter_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
                down_block_additional_residuals = t2iadapter(t2iadapter_image)
                down_block_additional_residuals = [
                    sample.to(dtype=weight_dtype) for sample in down_block_additional_residuals
                ]

                # Predict the noise residual
                model_pred = unet(
                    inp_noisy_latents,
                    timesteps,
                    encoder_hidden_states=batch["prompt_ids"],
                    added_cond_kwargs=batch["unet_added_conditions"],
                    down_block_additional_residuals=down_block_additional_residuals,
                    return_dict=False,
                )[0]

                # Denoise the latents
                denoised_latents = model_pred * (-sigmas) + noisy_latents
                weighing = sigmas**-2.0

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = latents  # we are computing loss against denoise latents
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # MSE loss
                loss = torch.mean(
                    (weighing.float() * (denoised_latents.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    dim=1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = t2iadapter.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            vae,
                            unet,
                            t2iadapter,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        t2iadapter = unwrap_model(t2iadapter)
        t2iadapter.save_pretrained(args.output_dir)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
