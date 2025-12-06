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
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput
from ..embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from ..modeling_utils import ModelMixin
from .unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block


@dataclass
class UNet2DOutput(BaseOutput):
    """
    The output of [`UNet2DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output from the last layer of the model.
    """

    sample: torch.Tensor


class UNet2DModel(ModelMixin, ConfigMixin):
    r"""
    A 2D UNet model that takes a noisy sample and a timestep and returns a sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample. Dimensions must be a multiple of `2 ** (len(block_out_channels) -
            1)`.
        in_channels (`int`, *optional*, defaults to 3): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 3): Number of channels in the output.
        center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.
        time_embedding_type (`str`, *optional*, defaults to `"positional"`): Type of time embedding to use.
        freq_shift (`int`, *optional*, defaults to 0): Frequency shift for Fourier time embedding.
        flip_sin_to_cos (`bool`, *optional*, defaults to `True`):
            Whether to flip sin to cos for Fourier time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")`):
            Tuple of downsample block types.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock2D"`):
            Block type for middle of UNet, it can be either `UNetMidBlock2D` or `None`.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(224, 448, 672, 896)`):
            Tuple of block output channels.
        layers_per_block (`int`, *optional*, defaults to `2`): The number of layers per block.
        mid_block_scale_factor (`float`, *optional*, defaults to `1`): The scale factor for the mid block.
        downsample_padding (`int`, *optional*, defaults to `1`): The padding for the downsample convolution.
        downsample_type (`str`, *optional*, defaults to `conv`):
            The downsample type for downsampling layers. Choose between "conv" and "resnet"
        upsample_type (`str`, *optional*, defaults to `conv`):
            The upsample type for upsampling layers. Choose between "conv" and "resnet"
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        attention_head_dim (`int`, *optional*, defaults to `8`): The attention head dimension.
        norm_num_groups (`int`, *optional*, defaults to `32`): The number of groups for normalization.
        attn_norm_num_groups (`int`, *optional*, defaults to `None`):
            If set to an integer, a group norm layer will be created in the mid block's [`Attention`] layer with the
            given number of groups. If left as `None`, the group norm layer will only be created if
            `resnet_time_scale_shift` is set to `default`, and if created will have `norm_num_groups` groups.
        norm_eps (`float`, *optional*, defaults to `1e-5`): The epsilon for normalization.
        resnet_time_scale_shift (`str`, *optional*, defaults to `"default"`): Time scale shift config
            for ResNet blocks (see [`~models.resnet.ResnetBlock2D`]). Choose from `default` or `scale_shift`.
        class_embed_type (`str`, *optional*, defaults to `None`):
            The type of class embedding to use which is ultimately summed with the time embeddings. Choose from `None`,
            `"timestep"`, or `"identity"`.
        num_class_embeds (`int`, *optional*, defaults to `None`):
            Input dimension of the learnable embedding matrix to be projected to `time_embed_dim` when performing class
            conditioning with `class_embed_type` equal to `None`.
    """

    # 梯度检查，可以减少内存占用
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["norm"]

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str, ...] = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        mid_block_type: Optional[str] = "UNetMidBlock2D",
        up_block_types: Tuple[str, ...] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 32,
        attn_norm_num_groups: Optional[int] = None,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        num_train_timesteps: Optional[int] = None,
    ):
        super().__init__()

        self.sample_size = sample_size
        # 时间嵌入的维度，如果未指定，则使用block_out_channels[0] * 4
        time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        # input
        # 通过卷积层，将输入维度从in_channels转为block_out_channels[0]
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # time
        # 时间投影，输入维度timestep_input_dim，输出维度time_embed_dim
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(embedding_size=block_out_channels[0], scale=16)
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        elif time_embedding_type == "learned":
            self.time_proj = nn.Embedding(num_train_timesteps, block_out_channels[0])
            timestep_input_dim = block_out_channels[0]

        # 时间嵌入
        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # class embedding，输入维度可能有多种，输出维度time_embed_dim
        # 类别嵌入
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        # 降采样blocks
        self.down_blocks = nn.ModuleList([])

        # 中间层blocks
        self.mid_block = None

        # 升采样的blocks
        self.up_blocks = nn.ModuleList([])

        # down
        # 降采样模块，block_out_channels代表每个blocks输出通道数
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            # 当前block的输入通道数就是上个block的输出通道数量
            input_channel = output_channel
            
            # 当前block的输出通道数量由block_out_channels指定
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block, # 只有最后一个block，才添加降采样模块
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                downsample_padding=downsample_padding,
                resnet_time_scale_shift=resnet_time_scale_shift,
                downsample_type=downsample_type,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

        # mid
        if mid_block_type is None:
            self.mid_block = None
        else:
            self.mid_block = UNetMidBlock2D(
                in_channels=block_out_channels[-1], # 中间层的输入通道数量，就是最后一层降采样block的输出通道数量
                temb_channels=time_embed_dim,
                dropout=dropout,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else block_out_channels[-1],
                resnet_groups=norm_num_groups,
                attn_groups=attn_norm_num_groups,
                add_attention=add_attention,
            )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels)) # 升采样的输出通道数，和降采样的输出通道数你序
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block, # 如果是最后一层升采样block，最升采样
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                resnet_time_scale_shift=resnet_time_scale_shift,
                upsample_type=upsample_type,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)

        # out
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)

        # 输出做GroupNorm
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups_out, eps=norm_eps)

        # 然后做激活函数
        self.conv_act = nn.SiLU()

        # 通过卷积，将通道数量转为out_channels
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],   # 时间嵌入可以是tensor，float，int中的任意一种类型
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DOutput, Tuple]:
        r"""
        The [`UNet2DModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d.UNet2DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unets.unet_2d.UNet2DOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_2d.UNet2DOutput`] is returned, otherwise a `tuple` is
                returned where the first element is the sample tensor.
        """
        # 0. center input if necessary
        # 把样本的均值中心化
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        # 确保timesteps始终是形状为[1]的整数张量，与sample在相同设备上
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            # 处理0维张量
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        # 将timesteps广播为batch的size
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        # 处理时间信息，经过self.time_proj和self.time_embedding的计算
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        # 处理类别信息
        if self.class_embedding is not None:
            if class_labels is None:
                # 如果初始化的时候指定了需要类别信息但是forward的时候没有提供类别信息，则报错
                raise ValueError("class_labels should be provided when doing class conditioning")

            if self.config.class_embed_type == "timestep":
                # 如果类别的嵌入类型是timestep，则用time_proj处理
                class_labels = self.time_proj(class_labels)

            # 类别信息
            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)

            # 嵌入信息同时包含时间信息 + 类别信息
            emb = emb + class_emb
        elif self.class_embedding is None and class_labels is not None:
            # 如果初始化的时候没有指定需要类别信息但是forward的时候提供类别信息，则发出警告
            raise ValueError("class_embedding needs to be initialized in order to use class conditioning")

        # 2. pre-process
        # 预处理，通过卷积层将输入的通道数专为和第一层download block层的输入的通道数一致
        skip_sample = sample
        sample = self.conv_in(sample)

        # 3. down
        # 降采样，注意如何做skip connection
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                # 降采样block是否包含skip_conv的属性
                # 暂时不深入研究了，看起来skip_conv像是是否对skip做卷积操作
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        # 中间层
        if self.mid_block is not None:
            sample = self.mid_block(sample, emb)

        # 5. up
        # 升采样
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

        # 6. post-process
        # 后处理操作，卷积，激活函数等
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps

        if not return_dict:
            return (sample,)

        return UNet2DOutput(sample=sample)
