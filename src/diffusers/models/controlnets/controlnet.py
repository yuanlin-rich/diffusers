# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...loaders.single_file_model import FromOriginalModelMixin
from ...utils import BaseOutput, logging
from ..attention import AttentionMixin
from ..attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from ..embeddings import TextImageProjection, TextImageTimeEmbedding, TextTimeEmbedding, TimestepEmbedding, Timesteps
from ..modeling_utils import ModelMixin
from ..unets.unet_2d_blocks import (
    UNetMidBlock2D,
    UNetMidBlock2DCrossAttn,
    get_down_block,
)
from ..unets.unet_2d_condition import UNet2DConditionModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class ControlNetOutput(BaseOutput):
    """
    ControlNet 模型的输出数据结构。

    ControlNet 通过提取输入条件图像的特征，生成一系列不同分辨率的特征图，
    这些特征图用于条件化原始 UNet 的对应层。输出包含两部分：
    1. 下采样块的特征样本（down_block_res_samples）
    2. 中间块的特征样本（mid_block_res_sample）

    Args:
        down_block_res_samples (`tuple[torch.Tensor]`):
            一个元组，包含每个下采样块在不同分辨率下的激活特征。
            每个张量的形状为 `(batch_size, channel * resolution, height // resolution, width // resolution)`。
            这些特征可以用于条件化原始 UNet 的下采样激活。
            具体来说：
            - 元组的长度等于下采样块的数量（通常为4个）
            - 每个张量的通道数对应相应块的输出通道数
            - 空间分辨率逐级减半（如 64x64, 32x32, 16x16, 8x8）
        mid_block_res_sample (`torch.Tensor`):
            中间块（最低采样分辨率）的激活特征。
            张量形状为 `(batch_size, channel * lowest_resolution, height // lowest_resolution, width // lowest_resolution)`。
            用于条件化原始 UNet 的中间块激活。
            通常这是分辨率最低的特征图（如 8x8），包含最高级别的语义信息。

    Example:
        ```python
        # 使用 ControlNet 进行前向传播
        output = controlnet(
            sample=noisy_latents,
            timestep=timestep,
            encoder_hidden_states=text_embeddings,
            controlnet_cond=condition_image
        )
        
        # 获取下采样块特征
        down_features = output.down_block_res_samples  # 元组，包含4个特征图
        
        # 获取中间块特征
        mid_feature = output.mid_block_res_sample  # 单个特征图
        ```
    """

    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor


class ControlNetConditioningEmbedding(nn.Module):
    """
    条件图像编码器，将输入的条件图像编码到特征空间。

    根据 ControlNet 论文 (https://huggingface.co/papers/2302.05543)：
    "Stable Diffusion 使用类似于 VQ-GAN 的预处理方法，将整个数据集的 512×512 图像转换为更小的 64×64 '潜在图像'以稳定训练。
    这要求 ControlNet 将基于图像的条件转换为 64×64 特征空间以匹配卷积大小。我们使用一个由四个卷积层组成的微型网络 E(·)，
    具有 4×4 核和 2×2 步长（使用 ReLU 激活，通道数为 16, 32, 64, 128，使用高斯权重初始化，与完整模型联合训练）
    将图像空间条件编码为特征图..."

    该网络的作用是将任意尺寸的条件图像（如边缘图、深度图、姿态图等）编码为与 UNet 潜在空间相匹配的特征表示。

    Args:
        conditioning_embedding_channels (int):
            输出特征图的通道数，通常与 UNet 第一层的通道数相同（如 320）。
        conditioning_channels (int, optional):
            输入条件图像的通道数。默认为 3（RGB 图像）。
        block_out_channels (Tuple[int, ...], optional):
            每个编码块的输出通道数。默认为 (16, 32, 96, 256)。

    Architecture:
        1. 输入卷积层 (conv_in): 3x3 卷积，将输入通道映射到第一个块的通道数
        2. 编码块序列 (blocks): 每个块包含：
           - 一个 3x3 卷积（保持通道数不变）
           - 一个 3x3 卷积（步长为 2，进行下采样，增加通道数）
        3. 输出卷积层 (conv_out): 3x3 卷积，使用零初始化，将最终通道数映射到目标通道数

    网络通过逐步下采样将输入图像转换为较低分辨率的特征图，同时增加通道数以捕获更丰富的语义信息。
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,                       # 输出通道数
        conditioning_channels: int = 3,                             # 输入通道数
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),    # 每个块的输出通道数
    ):
        super().__init__()

        # 输入卷积层：将条件图像映射到第一个特征块
        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        # 编码块序列：逐步下采样并增加通道数
        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            # 第一个卷积：保持空间分辨率和通道数不变
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            # 第二个卷积：步长为2进行下采样，同时增加通道数
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        # 输出卷积层：使用零初始化，确保训练开始时 ControlNet 不影响 UNet
        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        """
        前向传播：将条件图像编码为特征图。

        Args:
            conditioning (torch.Tensor):
                输入条件图像，形状为 `(batch_size, conditioning_channels, height, width)`。
                通常为 RGB 图像或其他条件图像（如边缘图、深度图等）。

        Returns:
            torch.Tensor:
                编码后的特征图，形状为 `(batch_size, conditioning_embedding_channels, height//scale, width//scale)`。
                其中 scale 取决于网络的下采样次数（通常为 8 倍下采样）。
        """
        # 初始卷积 + SiLU 激活
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        # 通过所有编码块
        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        # 最终输出卷积
        embedding = self.conv_out(embedding)

        return embedding


class ControlNetModel(ModelMixin, AttentionMixin, ConfigMixin, FromOriginalModelMixin, PeftAdapterMixin):
    """
    ControlNet 模型，用于在扩散过程中添加空间条件控制。

    ControlNet 是一种神经网络架构，用于向预训练的扩散模型（如 Stable Diffusion）添加额外的条件控制。
    它通过提取条件图像（如边缘图、深度图、姿态图等）的特征，并将这些特征注入到 UNet 的各个层中，
    从而实现对生成过程的精确空间控制。

    核心思想：
        1. 复制原始 UNet 的编码器部分（下采样块和中间块）
        2. 添加条件图像编码器（ControlNetConditioningEmbedding）将条件图像转换为特征
        3. 为每个 UNet 层添加可训练的零初始化卷积层（controlnet_down_blocks 和 controlnet_mid_block）
        4. 在推理时，将条件特征与 UNet 特征相加，然后通过零初始化卷积层进行变换

    零初始化技巧：
        所有 ControlNet 特定的卷积层都以零权重初始化，确保在训练开始时 ControlNet 不会影响原始 UNet 的行为，
        从而稳定训练过程。

    Args:
        in_channels (`int`, defaults to 4):
            输入样本的通道数。对于潜在扩散模型，通常是 4（潜在空间维度）。
        flip_sin_to_cos (`bool`, defaults to `True`):
            是否在时间嵌入中将 sin 翻转为 cos。影响时间步的正弦余弦编码方式。
        freq_shift (`int`, defaults to 0):
            应用于时间嵌入的频率偏移。
        down_block_types (`tuple[str]`, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            使用的下采样块类型的元组。必须与原始 UNet 的配置匹配。
        only_cross_attention (`Union[bool, Tuple[bool]]`, defaults to `False`):
            是否只使用交叉注意力。如果为布尔值，应用于所有块；如果为元组，长度必须与 down_block_types 相同。
        block_out_channels (`tuple[int]`, defaults to `(320, 640, 1280, 1280)`):
            每个块的输出通道数。通常与原始 UNet 的通道配置相同。
        layers_per_block (`int`, defaults to 2):
            每个块中的层数。
        downsample_padding (`int`, defaults to 1):
            下采样卷积使用的填充。
        mid_block_scale_factor (`float`, defaults to 1):
            中间块使用的缩放因子。
        act_fn (`str`, defaults to "silu"):
            使用的激活函数。
        norm_num_groups (`int`, *optional*, defaults to 32):
            归一化使用的组数。如果为 None，则在后处理中跳过归一化和激活层。
        norm_eps (`float`, defaults to 1e-5):
            归一化使用的 epsilon 值。
        cross_attention_dim (`int`, defaults to 1280):
            交叉注意力特征的维度。通常与文本编码器的输出维度相同。
        transformer_layers_per_block (`int` or `Tuple[int]`, *optional*, defaults to 1):
            [`~models.attention.BasicTransformerBlock`] 类型的 transformer 块数量。
            仅与 [`~models.unet_2d_blocks.CrossAttnDownBlock2D`]、[`~models.unet_2d_blocks.CrossAttnUpBlock2D`]、
            [`~models.unet_2d_blocks.UNetMidBlock2DCrossAttn`] 相关。
        encoder_hid_dim (`int`, *optional*, defaults to None):
            如果定义了 `encoder_hid_dim_type`，`encoder_hidden_states` 将从 `encoder_hid_dim` 维度投影到 `cross_attention_dim`。
        encoder_hid_dim_type (`str`, *optional*, defaults to `None`):
            如果给定，`encoder_hidden_states` 和其他可能的嵌入将根据 `encoder_hid_dim_type` 下投影到 `cross_attention` 维度的文本嵌入。
        attention_head_dim (`Union[int, Tuple[int]]`, defaults to 8):
            注意力头的维度。
        use_linear_projection (`bool`, defaults to `False`):
            是否在线性投影中使用注意力。
        class_embed_type (`str`, *optional*, defaults to `None`):
            使用的类别嵌入类型，最终与时间嵌入相加。可选 None、`"timestep"`、`"identity"`、`"projection"` 或 `"simple_projection"`。
        addition_embed_type (`str`, *optional*, defaults to `None`):
            配置将与时间嵌入相加的可选嵌入。可选 `None` 或 "text"。"text" 将使用 `TextTimeEmbedding` 层。
        num_class_embeds (`int`, *optional*, defaults to 0):
            当 `class_embed_type` 等于 `None` 时，可学习嵌入矩阵的输入维度，将被投影到 `time_embed_dim`。
        upcast_attention (`bool`, defaults to `False`):
            是否上投射注意力。
        resnet_time_scale_shift (`str`, defaults to `"default"`):
            ResNet 块的时间尺度偏移配置（参见 `ResnetBlock2D`）。可选 `default` 或 `scale_shift`。
        projection_class_embeddings_input_dim (`int`, *optional*, defaults to `None`):
            当 `class_embed_type="projection"` 时，`class_labels` 输入的维度。当 `class_embed_type="projection"` 时必须提供。
        controlnet_conditioning_channel_order (`str`, defaults to `"rgb"`):
            条件图像的通道顺序。如果是 `bgr` 将转换为 `rgb`。
        conditioning_embedding_out_channels (`tuple[int]`, *optional*, defaults to `(16, 32, 96, 256)`):
            `conditioning_embedding` 层中每个块的输出通道元组。
        global_pool_conditions (`bool`, defaults to `False`):
            TODO(Patrick) - 未使用的参数。
        addition_embed_type_num_heads (`int`, defaults to 64):
            `TextTimeEmbedding` 层使用的头数。
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        conditioning_channels: int = 3,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int, ...]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        projection_class_embeddings_input_dim: Optional[int] = None,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
        global_pool_conditions: bool = False,
        addition_embed_type_num_heads: int = 64,
    ):
        super().__init__()

        # 如果 `num_attention_heads` 未定义（大多数模型都是这种情况）
        # 它将默认为 `attention_head_dim`。这看起来很奇怪，但确实如此。
        # 这种行为的原因是为了纠正库创建时引入的错误命名变量。
        # 这个错误命名直到后来才在 https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131 中被发现
        # 为 40,000+ 配置将 `attention_head_dim` 更改为 `num_attention_heads` 会破坏向后兼容性
        # 因此我们在这里纠正命名。
        num_attention_heads = num_attention_heads or attention_head_dim

        # Check inputs
        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        # only_cross_attention是布尔值，或者者是一个布尔值元组，其长度必须与down_block_types相同
        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
            )

        # attention_head_dim是整数，或者者是一个整数元组，其长度必须与down_block_types相同
        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        # input
        conv_in_kernel = 3

        # 这样设置padding，保证输入和输出的空间尺度不变
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )

        # time
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]
        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
        )

        # 设置了encoder_hid_dim，如果没有设置encoder_hid_dim_type，则默认为'text_proj'
        if encoder_hid_dim_type is None and encoder_hid_dim is not None:
            encoder_hid_dim_type = "text_proj"
            self.register_to_config(encoder_hid_dim_type=encoder_hid_dim_type)
            logger.info("encoder_hid_dim_type defaults to 'text_proj' as `encoder_hid_dim` is defined.")

        # 设置了encoder_hid_dim_type，则必须设置encoder_hid_dim，否则抛出异常
        if encoder_hid_dim is None and encoder_hid_dim_type is not None:
            raise ValueError(
                f"`encoder_hid_dim` has to be defined when `encoder_hid_dim_type` is set to {encoder_hid_dim_type}."
            )

        # 根据encoder_hid_dim_type设置encoder_hidden_states的投影层
        if encoder_hid_dim_type == "text_proj":
            # 简单的线性投影
            self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
        elif encoder_hid_dim_type == "text_image_proj":
            # 如果使用文本-图像投影，则需要TextImageProjection层
            # image_embed_dim DOESN'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image_proj"` (Kandinsky 2.1)`
            self.encoder_hid_proj = TextImageProjection(
                text_embed_dim=encoder_hid_dim,
                image_embed_dim=cross_attention_dim,
                cross_attention_dim=cross_attention_dim,
            )

        elif encoder_hid_dim_type is not None:
            # encoder_hid_dim_type 设置错误，只能是 None, 'text_proj' or 'text_image_proj'
            raise ValueError(
                f"encoder_hid_dim_type: {encoder_hid_dim_type} must be None, 'text_proj' or 'text_image_proj'."
            )
        else:
            self.encoder_hid_proj = None

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            # 没有指定class_embed_type，但指定了num_class_embeds，默认为simple_projection
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            # class_embed_type为timestep，使用TimestepEmbedding
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            # class_embed_type为identity，使用恒等映射
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
            # class_embed_type为projection，使用投影层，需要指定projection_class_embeddings_input_dim
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                )
            # The projection `class_embed_type` is the same as the timestep `class_embed_type` except
            # 1. the `class_labels` inputs are not first converted to sinusoidal embeddings
            # 2. it projects from an arbitrary input dimension.
            #
            # Note that `TimestepEmbedding` is quite general, being mainly linear layers and activations.
            # When used for embedding actual timesteps, the timesteps are first converted to sinusoidal embeddings.
            # As a result, `TimestepEmbedding` can be passed arbitrary vectors.
            self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        else:
            self.class_embedding = None

        # additional embedding
        if addition_embed_type == "text":
            if encoder_hid_dim is not None:
                text_time_embedding_from_dim = encoder_hid_dim
            else:
                text_time_embedding_from_dim = cross_attention_dim

            self.add_embedding = TextTimeEmbedding(
                text_time_embedding_from_dim, time_embed_dim, num_heads=addition_embed_type_num_heads
            )
        elif addition_embed_type == "text_image":
            # text_embed_dim and image_embed_dim DON'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image"` (Kandinsky 2.1)`
            self.add_embedding = TextImageTimeEmbedding(
                text_embed_dim=cross_attention_dim, image_embed_dim=cross_attention_dim, time_embed_dim=time_embed_dim
            )
        elif addition_embed_type == "text_time":
            self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

        elif addition_embed_type is not None:
            raise ValueError(f"addition_embed_type: {addition_embed_type} must be None, 'text' or 'text_image'.")

        # control net conditioning embedding
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            block_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels,
        )

        self.down_blocks = nn.ModuleList([])
        self.controlnet_down_blocks = nn.ModuleList([])

        # 只做交叉注意力
        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        # 交叉注意力头数维度
        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # 交叉注意力头数
        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]

        # 零卷积块，输入通道和输出通道都是output_channel
        controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=num_attention_heads[i],
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                downsample_padding=downsample_padding,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.down_blocks.append(down_block)

            # 每个down_block对应的controlnet_down_blocks
            for _ in range(layers_per_block):
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

            if not is_final_block:
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

        # mid
        mid_block_channel = block_out_channels[-1]

        controlnet_block = nn.Conv2d(mid_block_channel, mid_block_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_mid_block = controlnet_block

        if mid_block_type == "UNetMidBlock2DCrossAttn":
            self.mid_block = UNetMidBlock2DCrossAttn(
                transformer_layers_per_block=transformer_layers_per_block[-1],
                in_channels=mid_block_channel,
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=num_attention_heads[-1],
                resnet_groups=norm_num_groups,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
            )
        elif mid_block_type == "UNetMidBlock2D":
            self.mid_block = UNetMidBlock2D(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                num_layers=0,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift=resnet_time_scale_shift,
                add_attention=False,
            )
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")

    @classmethod
    def from_unet(
        cls,
        unet: UNet2DConditionModel,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
        load_weights_from_unet: bool = True,
        conditioning_channels: int = 3,
    ):
        r"""
        从 [`UNet2DConditionModel`] 实例化 [`ControlNetModel`]。

        这是创建 ControlNet 的便捷方法，它会复制 UNet 的配置和权重（如果指定）。
        ControlNet 架构与 UNet 的编码器部分相同，因此可以直接复制权重以加速训练。

        Parameters:
            unet (`UNet2DConditionModel`):
                要复制到 [`ControlNetModel`] 的 UNet 模型权重。所有适用的配置选项也会被复制。
            controlnet_conditioning_channel_order (`str`, defaults to `"rgb"`):
                条件图像的通道顺序。
            conditioning_embedding_out_channels (`tuple[int]`, *optional*, defaults to `(16, 32, 96, 256)`):
                `conditioning_embedding` 层中每个块的输出通道元组。
            load_weights_from_unet (`bool`, defaults to `True`):
                是否从 UNet 加载权重。如果为 True，将复制 UNet 的卷积层、时间嵌入、下采样块和中间块的权重。
                这可以加速 ControlNet 的训练，因为编码器部分已经预训练过。
            conditioning_channels (`int`, defaults to 3):
                条件图像的通道数。

        Returns:
            [`ControlNetModel`]:
                新创建的 ControlNet 模型，配置与 UNet 相同，并可选地加载了 UNet 的权重。

        Example:
            ```python
            # 从预训练的 UNet 创建 ControlNet
            unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
            controlnet = ControlNetModel.from_unet(unet)
            
            # 现在 controlnet 具有与 unet 相同的架构，并加载了权重
            # 可以用于训练条件控制
            ```
        """
        transformer_layers_per_block = (
            unet.config.transformer_layers_per_block if "transformer_layers_per_block" in unet.config else 1
        )
        encoder_hid_dim = unet.config.encoder_hid_dim if "encoder_hid_dim" in unet.config else None
        encoder_hid_dim_type = unet.config.encoder_hid_dim_type if "encoder_hid_dim_type" in unet.config else None
        addition_embed_type = unet.config.addition_embed_type if "addition_embed_type" in unet.config else None
        addition_time_embed_dim = (
            unet.config.addition_time_embed_dim if "addition_time_embed_dim" in unet.config else None
        )

        controlnet = cls(
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=unet.config.in_channels,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            freq_shift=unet.config.freq_shift,
            down_block_types=unet.config.down_block_types,
            only_cross_attention=unet.config.only_cross_attention,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            downsample_padding=unet.config.downsample_padding,
            mid_block_scale_factor=unet.config.mid_block_scale_factor,
            act_fn=unet.config.act_fn,
            norm_num_groups=unet.config.norm_num_groups,
            norm_eps=unet.config.norm_eps,
            cross_attention_dim=unet.config.cross_attention_dim,
            attention_head_dim=unet.config.attention_head_dim,
            num_attention_heads=unet.config.num_attention_heads,
            use_linear_projection=unet.config.use_linear_projection,
            class_embed_type=unet.config.class_embed_type,
            num_class_embeds=unet.config.num_class_embeds,
            upcast_attention=unet.config.upcast_attention,
            resnet_time_scale_shift=unet.config.resnet_time_scale_shift,
            projection_class_embeddings_input_dim=unet.config.projection_class_embeddings_input_dim,
            mid_block_type=unet.config.mid_block_type,
            controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels,
        )

        # 从unet加载权重
        if load_weights_from_unet:
            controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
            controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
            controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())

            if controlnet.class_embedding:
                controlnet.class_embedding.load_state_dict(unet.class_embedding.state_dict())

            if hasattr(controlnet, "add_embedding"):
                controlnet.add_embedding.load_state_dict(unet.add_embedding.state_dict())

            controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())
            controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())

        return controlnet

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attention_slice
    def set_attention_slice(self, slice_size: Union[str, int, List[int]]) -> None:
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module splits the input tensor in slices to compute attention in
        several steps. This is useful for saving some memory in exchange for a small decrease in speed.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, input to the attention heads is halved, so attention is computed in two steps. If
                `"max"`, maximum amount of memory is saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_sliceable_layers * [1]

        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple[Tuple[torch.Tensor, ...], torch.Tensor]]:
        """
        ControlNet 模型的前向传播方法。

        该方法执行以下主要步骤：
        1. 预处理条件图像（通道顺序调整、注意力掩码处理）
        2. 时间步编码和条件嵌入（时间嵌入、类别嵌入、附加嵌入）
        3. 条件图像编码（通过 ControlNetConditioningEmbedding）
        4. 通过 UNet 编码器进行特征提取（下采样块和中间块）
        5. 应用 ControlNet 特定的零初始化卷积层
        6. 特征缩放（根据 conditioning_scale 和 guess_mode）
        7. 返回条件特征

        Args:
            sample (`torch.Tensor`):
                噪声输入张量，形状为 `(batch_size, in_channels, height, width)`。
                通常是扩散过程中的噪声潜在表示。
            timestep (`Union[torch.Tensor, float, int]`):
                去噪输入的时间步数。可以是标量、张量或整数/浮点数。
            encoder_hidden_states (`torch.Tensor`):
                编码器隐藏状态，通常是文本嵌入，形状为 `(batch_size, sequence_length, cross_attention_dim)`。
            controlnet_cond (`torch.Tensor`):
                条件输入张量，形状为 `(batch_size, conditioning_channels, height, width)`。
                通常是条件图像（如边缘图、深度图、姿态图等）。
            conditioning_scale (`float`, defaults to `1.0`):
                ControlNet 输出的缩放因子。控制条件影响的强度。
                值越大，条件对生成的影响越强。
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                可选的类别标签用于条件化。它们的嵌入将与时间步嵌入相加。
            timestep_cond (`torch.Tensor`, *optional*, defaults to `None`):
                时间步的附加条件嵌入。如果提供，这些嵌入将与通过 `self.time_embedding` 层的时间步嵌入相加，
                以获得最终的时间步嵌入。
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                应用于 `encoder_hidden_states` 的注意力掩码，形状为 `(batch, key_tokens)`。
                如果为 `1` 则保留掩码，如果为 `0` 则丢弃。掩码将转换为偏置，为 "丢弃" 标记的注意力分数添加大的负值。
            added_cond_kwargs (`dict`):
                Stable Diffusion XL UNet 的附加条件。
            cross_attention_kwargs (`dict[str]`, *optional*, defaults to `None`):
                如果指定，将传递给 `AttnProcessor` 的关键字参数字典。
            guess_mode (`bool`, defaults to `False`):
                在此模式下，即使移除所有提示，ControlNet 编码器也会尽力识别输入内容。
                建议使用 3.0 到 5.0 之间的 `guidance_scale`。
                当 guess_mode=True 时，不同层的特征会使用不同的缩放因子（从 0.1 到 1.0 的对数空间）。
            return_dict (`bool`, defaults to `True`):
                是否返回 [`~models.controlnets.controlnet.ControlNetOutput`] 而不是普通元组。

        Returns:
            [`~models.controlnets.controlnet.ControlNetOutput`] **or** `tuple`:
                如果 `return_dict` 为 `True`，则返回 [`~models.controlnets.controlnet.ControlNetOutput`]，
                否则返回一个元组，其中第一个元素是下采样块特征样本的元组，第二个元素是中间块特征样本。

        Example:
            ```python
            # 初始化 ControlNet
            controlnet = ControlNetModel.from_unet(unet)
            
            # 准备输入
            noisy_latents = torch.randn(1, 4, 64, 64)
            timestep = torch.tensor([50])
            text_embeddings = torch.randn(1, 77, 1280)
            condition_image = torch.randn(1, 3, 512, 512)  # 边缘图或深度图
            
            # 前向传播
            with torch.no_grad():
                output = controlnet(
                    sample=noisy_latents,
                    timestep=timestep,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=condition_image,
                    conditioning_scale=1.0,
                    guess_mode=False
                )
            
            # 使用输出条件化 UNet
            down_block_res_samples = output.down_block_res_samples
            mid_block_res_sample = output.mid_block_res_sample
            ```
        """
        # 检查通道顺序
        channel_order = self.config.controlnet_conditioning_channel_order

        if channel_order == "rgb":
            # 默认是rgb顺序，无需处理
            ...
        elif channel_order == "bgr":
            # 如果是bgr顺序，翻转通道维度以转换为rgb
            controlnet_cond = torch.flip(controlnet_cond, dims=[1])
        else:
            raise ValueError(f"未知的 `controlnet_conditioning_channel_order`: {channel_order}")

        # 准备注意力掩码
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. 时间步处理
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: 这需要在 CPU 和 GPU 之间同步。所以尽可能传递张量形式的时间步
            # 这是使用 `match` 语句（Python 3.10+）的好案例
            is_mps = sample.device.type == "mps"
            is_npu = sample.device.type == "npu"
            if isinstance(timestep, float):
                dtype = torch.float32 if (is_mps or is_npu) else torch.float64
            else:
                dtype = torch.int32 if (is_mps or is_npu) else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # 以与 ONNX/Core ML 兼容的方式广播到批次维度
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps 不包含任何权重，总是返回 f32 张量
        # 但 time_embedding 可能实际上在 fp16 中运行。所以我们需要在这里进行类型转换。
        # 可能有更好的封装方式。
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        if self.config.addition_embed_type is not None:
            if self.config.addition_embed_type == "text":
                aug_emb = self.add_embedding(encoder_hidden_states)

            elif self.config.addition_embed_type == "text_time":
                if "text_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                    )
                text_embeds = added_cond_kwargs.get("text_embeds")
                if "time_ids" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                    )
                time_ids = added_cond_kwargs.get("time_ids")
                time_embeds = self.add_time_proj(time_ids.flatten())
                time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

                add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                add_embeds = add_embeds.to(emb.dtype)
                aug_emb = self.add_embedding(add_embeds)

        emb = emb + aug_emb if aug_emb is not None else emb

        # 2. pre-process
        sample = self.conv_in(sample)

        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        sample = sample + controlnet_cond

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = self.mid_block(sample, emb)

        # 5. Control net blocks

        controlnet_down_block_res_samples = ()

        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(sample)

        # 6. scaling
        if guess_mode and not self.config.global_pool_conditions:
            scales = torch.logspace(-1, 0, len(down_block_res_samples) + 1, device=sample.device)  # 0.1 to 1.0
            scales = scales * conditioning_scale
            down_block_res_samples = [sample * scale for sample, scale in zip(down_block_res_samples, scales)]
            mid_block_res_sample = mid_block_res_sample * scales[-1]  # last one
        else:
            down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
            mid_block_res_sample = mid_block_res_sample * conditioning_scale

        if self.config.global_pool_conditions:
            down_block_res_samples = [
                torch.mean(sample, dim=(2, 3), keepdim=True) for sample in down_block_res_samples
            ]
            mid_block_res_sample = torch.mean(mid_block_res_sample, dim=(2, 3), keepdim=True)

        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample)

        return ControlNetOutput(
            down_block_res_samples=down_block_res_samples, mid_block_res_sample=mid_block_res_sample
        )


def zero_module(module: nn.Module) -> nn.Module:
    """
    将给定模块的所有参数初始化为零。

    这是 ControlNet 架构中的一个关键技巧：所有 ControlNet 特定的层都以零权重初始化，
    确保在训练开始时 ControlNet 不会影响原始 UNet 的行为。这使得训练更加稳定，
    因为模型最初表现为原始 UNet，然后逐渐学习添加条件控制。

    Args:
        module (nn.Module):
            要零初始化的 PyTorch 模块。通常是卷积层或线性层。

    Returns:
        nn.Module:
            参数已零初始化的相同模块。

    Example:
        ```python
        # 创建一个卷积层并零初始化
        conv = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        conv = zero_module(conv)
        
        # 现在 conv 的所有权重和偏置都为零
        print(conv.weight.sum())  # 输出: tensor(0.)
        ```
    """
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
