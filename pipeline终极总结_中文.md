# Diffusers Pipelines 详细总结报告

本报告基于对源码的自动分析，提取了每个 pipeline 类的文档字符串，并按其功能进行了分类。
共分析 253 个 pipeline 类。

## 分类概览
- **3d-generation**: 2 个 pipeline
- **audio-generation**: 6 个 pipeline
- **controlnet**: 2 个 pipeline
- **diffusion**: 4 个 pipeline
- **image-to-image**: 52 个 pipeline
- **other**: 65 个 pipeline
- **prior**: 5 个 pipeline
- **text-to-image**: 76 个 pipeline
- **video-generation**: 41 个 pipeline

## 详细列表

### 3d-generation
| Pipeline 类 | 用途 | 文件 |
| ------------- | ------ | ------ |
| ShapEImg2ImgPipeline | 用于生成 3D 资产的潜在表示并使用 NeRF 方法进行渲染的管道... | `src/diffusers/pipelines/shap_e/pipeline_shap_e_img2img.py` |
| ShapEPipeline | 用于生成 3D 资产的潜在表示并使用 NeRF 方法进行渲染的管道。 | `src/diffusers/pipelines/shap_e/pipeline_shap_e.py` |

### audio-generation
| Pipeline 类 | 用途 | 文件 |
| ------------- | ------ | ------ |
| AudioDiffusionPipeline | 音频扩散管道。 | `src/diffusers/pipelines/deprecated/audio_diffusion/pipeline_audio_diffusion.py` |
| AudioLDM2Pipeline | 使用 AudioLDM2 生成文本到音频的管道。 | `src/diffusers/pipelines/audioldm2/pipeline_audioldm2.py` |
| AudioLDMPipeline | 使用 AudioLDM 生成文本到音频的管道。 | `src/diffusers/pipelines/audioldm/pipeline_audioldm.py` |
| DanceDiffusionPipeline | 音频生成管道。 | `src/diffusers/pipelines/dance_diffusion/pipeline_dance_diffusion.py` |
| SpectrogramDiffusionPipeline | 无条件音频生成的管道。 | `src/diffusers/pipelines/deprecated/spectrogram_diffusion/pipeline_spectrogram_diffusion.py` |
| StableAudioPipeline | 使用 StableAudio 生成文本到音频的管道。 | `src/diffusers/pipelines/stable_audio/pipeline_stable_audio.py` |

### controlnet
| Pipeline 类 | 用途 | 文件 |
| ------------- | ------ | ------ |
| BlipDiffusionControlNetPipeline | 基于 Canny Edge 的管道使用 Blip Diffusion 进行受控主题驱动生成。 | `src/diffusers/pipelines/controlnet/pipeline_controlnet_blip_diffusion.py` |
| WanVACEPipeline | 使用广域网进行可控发电的管道。 | `src/diffusers/pipelines/wan/pipeline_wan_vace.py` |

### diffusion
| Pipeline 类 | 用途 | 文件 |
| ------------- | ------ | ------ |
| BlipDiffusionPipeline | 使用 Blip 扩散进行零样本主题驱动生成的管道。 | `src/diffusers/pipelines/blip_diffusion/pipeline_blip_diffusion.py` |
| LDMPipeline | 使用潜在扩散的无条件图像生成管道。 | `src/diffusers/pipelines/deprecated/latent_diffusion_uncond/pipeline_latent_diffusion_uncond.py` |
| StableDiffusionLatentUpscalePipeline | 用于将稳定扩散输出图像分辨率提高 2 倍的管道。 | `src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_latent_upscale.py` |
| VersatileDiffusionDualGuidedPipeline | 使用多功能扩散进行图像-文本双引导生成的管道。 | `src/diffusers/pipelines/deprecated/versatile_diffusion/pipeline_versatile_diffusion_dual_guided.py` |

### image-to-image
| Pipeline 类 | 用途 | 文件 |
| ------------- | ------ | ------ |
| AltDiffusionImg2ImgPipeline | 使用 Alt Diffusion 生成文本引导的图像到图像的管道。 | `src/diffusers/pipelines/deprecated/alt_diffusion/pipeline_alt_diffusion_img2img.py` |
| ChromaImg2ImgPipeline | 用于图像到图像生成的 Chroma 管道。 | `src/diffusers/pipelines/chroma/pipeline_chroma_img2img.py` |
| CycleDiffusionPipeline | 使用稳定扩散生成文本引导图像到图像的管道。 | `src/diffusers/pipelines/deprecated/stable_diffusion_variants/pipeline_cycle_diffusion.py` |
| FlaxStableDiffusionImg2ImgPipeline | 基于 Flax 的管道，使用稳定扩散进行文本引导的图像到图像生成。 | `src/diffusers/pipelines/stable_diffusion/pipeline_flax_stable_diffusion_img2img.py` |
| FlaxStableDiffusionInpaintPipeline | 基于 Flax 的管道，使用稳定扩散进行文本引导图像修复。 | `src/diffusers/pipelines/stable_diffusion/pipeline_flax_stable_diffusion_inpaint.py` |
| FluxControlImg2ImgPipeline | 用于图像修复的 Flux 管道。 | `src/diffusers/pipelines/flux/pipeline_flux_control_img2img.py` |
| FluxControlInpaintPipeline | 使用 Flux-dev-Depth/Canny 进行图像修复的 Flux 管道。 | `src/diffusers/pipelines/flux/pipeline_flux_control_inpaint.py` |
| FluxControlNetImg2ImgPipeline | 用于图像到图像生成的 Flux controlnet 管道。 | `src/diffusers/pipelines/flux/pipeline_flux_controlnet_image_to_image.py` |
| FluxControlNetInpaintPipeline | 用于修复的 Flux controlnet 管道。 | `src/diffusers/pipelines/flux/pipeline_flux_controlnet_inpainting.py` |
| FluxFillPipeline | 用于图像修复/修复的通量填充管道。 | `src/diffusers/pipelines/flux/pipeline_flux_fill.py` |
| FluxImg2ImgPipeline | 用于图像修复的 Flux 管道。 | `src/diffusers/pipelines/flux/pipeline_flux_img2img.py` |
| FluxInpaintPipeline | 用于图像修复的 Flux 管道。 | `src/diffusers/pipelines/flux/pipeline_flux_inpaint.py` |
| FluxPriorReduxPipeline | 用于图像到图像生成的 Flux Redux 管道。 | `src/diffusers/pipelines/flux/pipeline_flux_prior_redux.py` |
| KandinskyImg2ImgCombinedPipeline | 使用康定斯基进行图像到图像生成的组合管道 | `src/diffusers/pipelines/kandinsky/pipeline_kandinsky_combined.py` |
| KandinskyImg2ImgPipeline | 使用康定斯基进行图像到图像生成的管道 | `src/diffusers/pipelines/kandinsky/pipeline_kandinsky_img2img.py` |
| KandinskyInpaintPipeline | 使用 Kandinsky2.1 进行文本引导图像修复的管道 | `src/diffusers/pipelines/kandinsky/pipeline_kandinsky_inpaint.py` |
| KandinskyV22ControlnetImg2ImgPipeline | 使用康定斯基进行图像到图像生成的管道 | `src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_controlnet_img2img.py` |
| KandinskyV22Img2ImgCombinedPipeline | 使用康定斯基进行图像到图像生成的组合管道 | `src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py` |
| KandinskyV22Img2ImgPipeline | 使用康定斯基进行图像到图像生成的管道 | `src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_img2img.py` |
| KandinskyV22InpaintCombinedPipeline | 使用康定斯基进行修复生成的组合管道 | `src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py` |
| KandinskyV22InpaintPipeline | 使用 Kandinsky2.1 进行文本引导图像修复的管道 | `src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_inpainting.py` |
| LDMSuperResolutionPipeline | 使用潜在扩散的图像超分辨率管道。 | `src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion_superresolution.py` |
| LEditsPPPipelineStableDiffusion | 使用具有稳定扩散功能的 LEDits++ 进行文本图像编辑的管道。 | `src/diffusers/pipelines/ledits_pp/pipeline_leditspp_stable_diffusion.py` |
| LEditsPPPipelineStableDiffusionXL | 使用 LEDits++ 和 Stable Diffusion XL 进行文本图像编辑的管道。 | `src/diffusers/pipelines/ledits_pp/pipeline_leditspp_stable_diffusion_xl.py` |
| LatentConsistencyModelImg2ImgPipeline | 使用潜在一致性模型进行图像到图像生成的管道。 | `src/diffusers/pipelines/latent_consistency_models/pipeline_latent_consistency_img2img.py` |
| OnnxStableDiffusionImg2ImgPipeline | 使用稳定扩散生成文本引导图像到图像的管道。 | `src/diffusers/pipelines/stable_diffusion/pipeline_onnx_stable_diffusion_img2img.py` |
| OnnxStableDiffusionInpaintPipeline | 使用稳定扩散进行文本引导图像修复的管道。*这是一个实验性功能*。 | `src/diffusers/pipelines/stable_diffusion/pipeline_onnx_stable_diffusion_inpaint.py` |
| OnnxStableDiffusionInpaintPipelineLegacy | 使用稳定扩散进行文本引导图像修复的管道。这是 Onn 的*传统功能*... | `src/diffusers/pipelines/deprecated/stable_diffusion_variants/pipeline_onnx_stable_diffusion_inpaint_legacy.py` |
| QwenImageEditInpaintPipeline | 用于图像编辑的 Qwen-Image-Edit 管道。 | `src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_inpaint.py` |
| QwenImageEditPipeline | 用于图像编辑的 Qwen-Image-Edit 管道。 | `src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py` |
| QwenImageEditPlusPipeline | 用于图像编辑的 Qwen-Image-Edit 管道。 | `src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py` |
| RePaintPipeline | 使用 RePaint 进行图像修复的管道。 | `src/diffusers/pipelines/deprecated/repaint/pipeline_repaint.py` |
| StableDiffusion3PAGImg2ImgPipeline | [PAG 管道](https://huggingface.co/docs/diffusers/main/en/using-diffusers/pag) 用于图像到图像... | `src/diffusers/pipelines/pag/pipeline_pag_sd_3_img2img.py` |
| StableDiffusionControlNetImg2ImgPipeline | 使用稳定扩散和 ControlNet 指导进行图像到图像生成的管道。 | `src/diffusers/pipelines/controlnet/pipeline_controlnet_img2img.py` |
| StableDiffusionControlNetInpaintPipeline | 在 ControlNet 指导下使用稳定扩散进行图像修复的管道。 | `src/diffusers/pipelines/controlnet/pipeline_controlnet_inpaint.py` |
| StableDiffusionControlNetPAGInpaintPipeline | 在 ControlNet 指导下使用稳定扩散进行图像修复的管道。 | `src/diffusers/pipelines/pag/pipeline_pag_controlnet_sd_inpaint.py` |
| StableDiffusionDepth2ImgPipeline | 使用稳定扩散进行文本引导的基于深度的图像到图像生成的管道。 | `src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_depth2img.py` |
| StableDiffusionImageVariationPipeline | 使用稳定扩散从输入图像生成图像变化的管道。 | `src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_image_variation.py` |
| StableDiffusionImg2ImgPipeline | 使用稳定扩散进行文本引导的图像到图像生成的管道。 | `src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py` |
| StableDiffusionInpaintPipeline | 使用稳定扩散进行文本引导图像修复的管道。 | `src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py` |
| StableDiffusionInpaintPipelineLegacy | 使用稳定扩散进行文本引导图像修复的管道。*这是一个实验性功能*。 | `src/diffusers/pipelines/deprecated/stable_diffusion_variants/pipeline_stable_diffusion_inpaint_legacy.py` |
| StableDiffusionInstructPix2PixPipeline | 通过遵循文本指令进行像素级图像编辑的管道（基于稳定扩散）。 | `src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py` |
| StableDiffusionPAGImg2ImgPipeline | 使用稳定扩散进行文本引导的图像到图像生成的管道。 | `src/diffusers/pipelines/pag/pipeline_pag_sd_img2img.py` |
| StableDiffusionPix2PixZeroPipeline | 使用 Pix2Pix Zero 进行像素级图像编辑的管道。基于稳定扩散。 | `src/diffusers/pipelines/deprecated/stable_diffusion_variants/pipeline_stable_diffusion_pix2pix_zero.py` |
| StableDiffusionUpscalePipeline | 使用稳定扩散 2 进行文本引导图像超分辨率的管道。 | `src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_upscale.py` |
| StableDiffusionXLControlNetImg2ImgPipeline | 使用 Stable Diffusion XL 和 ControlNet 指导进行图像到图像生成的管道。 | `src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl_img2img.py` |
| StableDiffusionXLControlNetPAGImg2ImgPipeline | 使用 Stable Diffusion XL 和 ControlNet 指导进行图像到图像生成的管道。 | `src/diffusers/pipelines/pag/pipeline_pag_controlnet_sd_xl_img2img.py` |
| StableDiffusionXLControlNetUnionImg2ImgPipeline | 使用 Stable Diffusion XL 和 ControlNet 指导进行图像到图像生成的管道。 | `src/diffusers/pipelines/controlnet/pipeline_controlnet_union_sd_xl_img2img.py` |
| StableDiffusionXLInstructPix2PixPipeline | 按照文本说明进行像素级图像编辑的管道。基于稳定扩散 XL。 | `src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_instruct_pix2pix.py` |
| StableUnCLIPImg2ImgPipeline | 使用稳定的 unCLIP 进行文本引导的图像到图像生成的管道。 | `src/diffusers/pipelines/stable_diffusion/pipeline_stable_unclip_img2img.py` |
| UnCLIPImageVariationPipeline | 使用 UnCLIP 从输入图像生成图像变体的管道。 | `src/diffusers/pipelines/unclip/pipeline_unclip_image_variation.py` |
| VersatileDiffusionImageVariationPipeline | 使用多功能扩散的图像变化管道。 | `src/diffusers/pipelines/deprecated/versatile_diffusion/pipeline_versatile_diffusion_image_variation.py` |

### other
| Pipeline 类 | 用途 | 文件 |
| ------------- | ------ | ------ |
| AmusedImg2ImgPipeline | 无描述 | `src/diffusers/pipelines/amused/pipeline_amused_img2img.py` |
| AmusedInpaintPipeline | 无描述 | `src/diffusers/pipelines/amused/pipeline_amused_inpaint.py` |
| AmusedPipeline | 无描述 | `src/diffusers/pipelines/amused/pipeline_amused.py` |
| AuraFlowPipeline | 参数： | `src/diffusers/pipelines/aura_flow/pipeline_aura_flow.py` |
| BriaFiboPipeline | 参数： | `src/diffusers/pipelines/bria_fibo/pipeline_bria_fibo.py` |
| BriaPipeline | 基于 FluxPipeline，有几处变化： | `src/diffusers/pipelines/bria/pipeline_bria.py` |
| ConsistencyModelPipeline | 用于无条件或类条件图像生成的管道。 | `src/diffusers/pipelines/consistency_models/pipeline_consistency_models.py` |
| CosmosTextToWorldPipeline | 使用 [Cosmos Predict1](https://github.com/nvidia-cosmos/cosmo... 进行文本到世界生成的管道 | `src/diffusers/pipelines/cosmos/pipeline_cosmos_text2world.py` |
| DDIMPipeline | 图像生成管道。 | `src/diffusers/pipelines/ddim/pipeline_ddim.py` |
| DDPMPipeline | 图像生成管道。 | `src/diffusers/pipelines/ddpm/pipeline_ddpm.py` |
| DiTPipeline | 基于 Transformer 主干而不是 UNet 的图像生成管道。 | `src/diffusers/pipelines/dit/pipeline_dit.py` |
| FlaxStableDiffusionXLPipeline | 无描述 | `src/diffusers/pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.py` |
| HiDreamImagePipeline | 无描述 | `src/diffusers/pipelines/hidream_image/pipeline_hidream_image.py` |
| HunyuanDiTControlNetPipeline | 使用 HunyuanDiT 生成英语/中文到图像的管道。 | `src/diffusers/pipelines/controlnet_hunyuandit/pipeline_hunyuandit_controlnet.py` |
| HunyuanDiTPAGPipeline | 使用 HunyuanDiT 和 [Perturbed Attention] 生成英语/中文到图像的管道 | `src/diffusers/pipelines/pag/pipeline_pag_hunyuandit.py` |
| HunyuanDiTPipeline | 使用 HunyuanDiT 生成英语/中文到图像的管道。 | `src/diffusers/pipelines/hunyuandit/pipeline_hunyuandit.py` |
| I2VGenXLPipeline | 无描述 | `src/diffusers/pipelines/i2vgen_xl/pipeline_i2vgen_xl.py` |
| IFImg2ImgPipeline | 无描述 | `src/diffusers/pipelines/deepfloyd_if/pipeline_if_img2img.py` |
| IFImg2ImgSuperResolutionPipeline | 无描述 | `src/diffusers/pipelines/deepfloyd_if/pipeline_if_img2img_superresolution.py` |
| IFInpaintingPipeline | 无描述 | `src/diffusers/pipelines/deepfloyd_if/pipeline_if_inpainting.py` |
| IFInpaintingSuperResolutionPipeline | 无描述 | `src/diffusers/pipelines/deepfloyd_if/pipeline_if_inpainting_superresolution.py` |
| IFPipeline | 无描述 | `src/diffusers/pipelines/deepfloyd_if/pipeline_if.py` |
| IFSuperResolutionPipeline | 无描述 | `src/diffusers/pipelines/deepfloyd_if/pipeline_if_superresolution.py` |
| Kandinsky3Img2ImgPipeline | 无描述 | `src/diffusers/pipelines/kandinsky3/pipeline_kandinsky3_img2img.py` |
| Kandinsky3Pipeline | 无描述 | `src/diffusers/pipelines/kandinsky3/pipeline_kandinsky3.py` |
| KandinskyInpaintCombinedPipeline | 使用康定斯基生成的组合管道 | `src/diffusers/pipelines/kandinsky/pipeline_kandinsky_combined.py` |
| KarrasVePipeline | 无条件图像生成的管道。 | `src/diffusers/pipelines/deprecated/stochastic_karras_ve/pipeline_stochastic_karras_ve.py` |
| LTXLatentUpsamplePipeline | 无描述 | `src/diffusers/pipelines/ltx/pipeline_ltx_latent_upsample.py` |
| Lumina2Text2ImgPipeline | 无描述 | `src/diffusers/pipelines/lumina2/pipeline_lumina2.py` |
| LuminaText2ImgPipeline | 无描述 | `src/diffusers/pipelines/lumina/pipeline_lumina.py` |
| MarigoldDepthPipeline | 使用 Marigold 方法进行单目深度估计的管道：https://marigoldmonodepth.github.... | `src/diffusers/pipelines/marigold/pipeline_marigold_depth.py` |
| MarigoldIntrinsicsPipeline | 使用 Marigold 方法进行本征图像分解 (IID) 的管道： | `src/diffusers/pipelines/marigold/pipeline_marigold_intrinsics.py` |
| MarigoldNormalsPipeline | 使用 Marigold 方法进行单眼法线估计的管道：https://marigold monodepth.github... | `src/diffusers/pipelines/marigold/pipeline_marigold_normals.py` |
| MusicLDMPipeline | 无描述 | `src/diffusers/pipelines/musicldm/pipeline_musicldm.py` |
| OmniGenPipeline | 用于多模式到图像生成的 OmniGen 管道。 | `src/diffusers/pipelines/omnigen/pipeline_omnigen.py` |
| OnnxStableDiffusionPipeline | 无描述 | `src/diffusers/pipelines/stable_diffusion/pipeline_onnx_stable_diffusion.py` |
| OnnxStableDiffusionUpscalePipeline | 无描述 | `src/diffusers/pipelines/stable_diffusion/pipeline_onnx_stable_diffusion_upscale.py` |
| PIAPipeline | 无描述 | `src/diffusers/pipelines/pia/pipeline_pia.py` |
| PNDMPipeline | 无条件图像生成的管道。 | `src/diffusers/pipelines/deprecated/pndm/pipeline_pndm.py` |
| PaintByExamplePipeline | 无描述 | `src/diffusers/pipelines/paint_by_example/pipeline_paint_by_example.py` |
| ScoreSdeVePipeline | 无条件图像生成的管道。 | `src/diffusers/pipelines/deprecated/score_sde_ve/pipeline_score_sde_ve.py` |
| SemanticStableDiffusionPipeline | 无描述 | `src/diffusers/pipelines/semantic_stable_diffusion/pipeline_semantic_stable_diffusion.py` |
| StableCascadeDecoderPipeline | 用于从稳定级联模型生成图像的管道。 | `src/diffusers/pipelines/stable_cascade/pipeline_stable_cascade.py` |
| StableDiffusion3ControlNetInpaintingPipeline | 参数： | `src/diffusers/pipelines/controlnet_sd3/pipeline_stable_diffusion_3_controlnet_inpainting.py` |
| StableDiffusion3ControlNetPipeline | 参数： | `src/diffusers/pipelines/controlnet_sd3/pipeline_stable_diffusion_3_controlnet.py` |
| StableDiffusion3Img2ImgPipeline | 参数： | `src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3_img2img.py` |
| StableDiffusion3InpaintPipeline | 参数： | `src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3_inpaint.py` |
| StableDiffusion3Pipeline | 参数： | `src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py` |
| StableDiffusionDiffEditPipeline | > [!警告] > 这是一项实验性功能！ | `src/diffusers/pipelines/stable_diffusion_diffedit/pipeline_stable_diffusion_diffedit.py` |
| StableDiffusionLDM3DPipeline | 无描述 | `src/diffusers/pipelines/stable_diffusion_ldm3d/pipeline_stable_diffusion_ldm3d.py` |
| StableDiffusionOnnxPipeline | 无描述 | `src/diffusers/pipelines/stable_diffusion/pipeline_onnx_stable_diffusion.py` |
| StableDiffusionPanoramaPipeline | 无描述 | `src/diffusers/pipelines/stable_diffusion_panorama/pipeline_stable_diffusion_panorama.py` |
| StableDiffusionPipelineSafe | 无描述 | `src/diffusers/pipelines/stable_diffusion_safe/pipeline_stable_diffusion_safe.py` |
| StableDiffusionSAGPipeline | 无描述 | `src/diffusers/pipelines/stable_diffusion_sag/pipeline_stable_diffusion_sag.py` |
| StableDiffusionXLKDiffusionPipeline | 无描述 | `src/diffusers/pipelines/stable_diffusion_k_diffusion/pipeline_stable_diffusion_xl_k_diffusion.py` |
| TextToVideoSDPipeline | 无描述 | `src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth.py` |
| TextToVideoZeroPipeline | 无描述 | `src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_zero.py` |
| TextToVideoZeroSDXLPipeline | 无描述 | `src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_zero_sdxl.py` |
| UniDiffuserPipeline | 双峰图像文本模型的管道，支持无条件文本和图像生成、文本... | `src/diffusers/pipelines/unidiffuser/pipeline_unidiffuser.py` |
| VideoToVideoSDPipeline | 无描述 | `src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth_img2img.py` |
| VisualClozeGenerationPipeline | 用于生成具有视觉上下文的图像的 VisualCloze 管道。参考： | `src/diffusers/pipelines/visualcloze/pipeline_visualcloze_generation.py` |
| VisualClozePipeline | 用于生成具有视觉上下文的图像的 VisualCloze 管道。参考： | `src/diffusers/pipelines/visualcloze/pipeline_visualcloze_combined.py` |
| WanAnimatePipeline | 使用 Wan-Animate 进行统一角色动画和替换的管道。 | `src/diffusers/pipelines/wan/pipeline_wan_animate.py` |
| WuerstchenDecoderPipeline | 从 Wuerstchen 模型生成图像的管道。 | `src/diffusers/pipelines/wuerstchen/pipeline_wuerstchen.py` |
| ZImagePipeline | 无描述 | `src/diffusers/pipelines/z_image/pipeline_z_image.py` |

### prior
| Pipeline 类 | 用途 | 文件 |
| ------------- | ------ | ------ |
| KandinskyPriorPipeline | 用于生成康定斯基先验图像的管道 | `src/diffusers/pipelines/kandinsky/pipeline_kandinsky_prior.py` |
| KandinskyV22PriorEmb2EmbPipeline | 用于生成康定斯基先验图像的管道 | `src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_prior_emb2emb.py` |
| KandinskyV22PriorPipeline | 用于生成康定斯基先验图像的管道 | `src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_prior.py` |
| StableCascadePriorPipeline | 用于为稳定级联生成图像的管道。 | `src/diffusers/pipelines/stable_cascade/pipeline_stable_cascade_prior.py` |
| WuerstchenPriorPipeline | 用于为 Wuerstchen 生成图像先验的管道。 | `src/diffusers/pipelines/wuerstchen/pipeline_wuerstchen_prior.py` |

### text-to-image
| Pipeline 类 | 用途 | 文件 |
| ------------- | ------ | ------ |
| AltDiffusionPipeline | 使用 Alt Diffusion 生成文本到图像的管道。 | `src/diffusers/pipelines/deprecated/alt_diffusion/pipeline_alt_diffusion.py` |
| ChromaPipeline | 用于生成文本到图像的 Chroma 管道。 | `src/diffusers/pipelines/chroma/pipeline_chroma.py` |
| CogView3PlusPipeline | 使用 CogView3Plus 生成文本到图像的管道。 | `src/diffusers/pipelines/cogview3/pipeline_cogview3plus.py` |
| CogView4ControlPipeline | 使用 CogView4 生成文本到图像的管道。 | `src/diffusers/pipelines/cogview4/pipeline_cogview4_control.py` |
| CogView4Pipeline | 使用 CogView4 生成文本到图像的管道。 | `src/diffusers/pipelines/cogview4/pipeline_cogview4.py` |
| Cosmos2TextToImagePipeline | 使用 [Cosmos Predict2](https://github.com/nvidia-cosmos/cosmo... 生成文本到图像的管道 | `src/diffusers/pipelines/cosmos/pipeline_cosmos2_text2image.py` |
| FlaxStableDiffusionControlNetPipeline | 基于 Flax 的管道，使用稳定扩散和 ControlNet Guidance 生成文本到图像。 | `src/diffusers/pipelines/controlnet/pipeline_flax_controlnet.py` |
| FlaxStableDiffusionPipeline | 基于 Flax 的管道，使用稳定扩散生成文本到图像。 | `src/diffusers/pipelines/stable_diffusion/pipeline_flax_stable_diffusion.py` |
| Flux2Pipeline | 用于生成文本到图像的 Flux2 管道。 | `src/diffusers/pipelines/flux2/pipeline_flux2.py` |
| FluxControlNetPipeline | 用于生成文本到图像的 Flux 管道。 | `src/diffusers/pipelines/flux/pipeline_flux_controlnet.py` |
| FluxControlPipeline | Flux 管道，用于根据图像条件生成可控的文本到图像。 | `src/diffusers/pipelines/flux/pipeline_flux_control.py` |
| FluxKontextInpaintPipeline | 用于生成文本到图像的 Flux Kontext 管道。 | `src/diffusers/pipelines/flux/pipeline_flux_kontext_inpaint.py` |
| FluxKontextPipeline | 用于图像到图像和文本到图像生成的 Flux Kontext 管道。 | `src/diffusers/pipelines/flux/pipeline_flux_kontext.py` |
| FluxPipeline | 用于生成文本到图像的 Flux 管道。 | `src/diffusers/pipelines/flux/pipeline_flux.py` |
| HunyuanImagePipeline | 用于生成文本到图像的 HunyuanImage 管道。 | `src/diffusers/pipelines/hunyuan_image/pipeline_hunyuanimage.py` |
| HunyuanImageRefinerPipeline | 用于生成文本到图像的 HunyuanImage 管道。 | `src/diffusers/pipelines/hunyuan_image/pipeline_hunyuanimage_refiner.py` |
| KandinskyCombinedPipeline | 使用康定斯基生成文本到图像的组合管道 | `src/diffusers/pipelines/kandinsky/pipeline_kandinsky_combined.py` |
| KandinskyPipeline | 使用康定斯基生成文本到图像的管道 | `src/diffusers/pipelines/kandinsky/pipeline_kandinsky.py` |
| KandinskyV22CombinedPipeline | 使用康定斯基生成文本到图像的组合管道 | `src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py` |
| KandinskyV22ControlnetPipeline | 使用康定斯基生成文本到图像的管道 | `src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_controlnet.py` |
| KandinskyV22Pipeline | 使用康定斯基生成文本到图像的管道 | `src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2.py` |
| KolorsImg2ImgPipeline | 使用 Kolors 生成文本到图像的管道。 | `src/diffusers/pipelines/kolors/pipeline_kolors_img2img.py` |
| KolorsPAGPipeline | 使用 Kolors 生成文本到图像的管道。 | `src/diffusers/pipelines/pag/pipeline_pag_kolors.py` |
| KolorsPipeline | 使用 Kolors 生成文本到图像的管道。 | `src/diffusers/pipelines/kolors/pipeline_kolors.py` |
| LDMTextToImagePipeline | 使用潜在扩散生成文本到图像的管道。 | `src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion.py` |
| LatentConsistencyModelPipeline | 使用潜在一致性模型生成文本到图像的管道。 | `src/diffusers/pipelines/latent_consistency_models/pipeline_latent_consistency_text2img.py` |
| Lumina2Pipeline | 使用 Lumina-T2I 生成文本到图像的管道。 | `src/diffusers/pipelines/lumina2/pipeline_lumina2.py` |
| LuminaPipeline | 使用 Lumina-T2I 生成文本到图像的管道。 | `src/diffusers/pipelines/lumina/pipeline_lumina.py` |
| PRXPipeline | 使用 PRX Transformer 生成文本到图像的管道。 | `src/diffusers/pipelines/prx/pipeline_prx.py` |
| PixArtAlphaPipeline | 使用 PixArt-Alpha 生成文本到图像的管道。 | `src/diffusers/pipelines/pixart_alpha/pipeline_pixart_alpha.py` |
| PixArtSigmaPAGPipeline | [PAG 管道](https://huggingface.co/docs/diffusers/main/en/using-diffusers/pag) 用于文本到图像... | `src/diffusers/pipelines/pag/pipeline_pag_pixart_sigma.py` |
| PixArtSigmaPipeline | 使用 PixArt-Sigma 生成文本到图像的管道。 | `src/diffusers/pipelines/pixart_alpha/pipeline_pixart_sigma.py` |
| QwenImageControlNetInpaintPipeline | 用于生成文本到图像的 QwenImage 管道。 | `src/diffusers/pipelines/qwenimage/pipeline_qwenimage_controlnet_inpaint.py` |
| QwenImageControlNetPipeline | 用于生成文本到图像的 QwenImage 管道。 | `src/diffusers/pipelines/qwenimage/pipeline_qwenimage_controlnet.py` |
| QwenImageImg2ImgPipeline | 用于生成文本到图像的 QwenImage 管道。 | `src/diffusers/pipelines/qwenimage/pipeline_qwenimage_img2img.py` |
| QwenImageInpaintPipeline | 用于生成文本到图像的 QwenImage 管道。 | `src/diffusers/pipelines/qwenimage/pipeline_qwenimage_inpaint.py` |
| QwenImagePipeline | 用于生成文本到图像的 QwenImage 管道。 | `src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py` |
| SanaControlNetPipeline | 使用 [Sana](https://huggingface.co/papers/2410.10629) 生成文本到图像的管道。 | `src/diffusers/pipelines/sana/pipeline_sana_controlnet.py` |
| SanaPAGPipeline | 使用 [Sana](https://huggingface.co/papers/2410.10629) 生成文本到图像的管道。这个... | `src/diffusers/pipelines/pag/pipeline_pag_sana.py` |
| SanaPipeline | 使用 [Sana](https://huggingface.co/papers/2410.10629) 生成文本到图像的管道。 | `src/diffusers/pipelines/sana/pipeline_sana.py` |
| SanaSprintImg2ImgPipeline | 使用 [SANA-Sprint](https://huggingface.co/papers/2503.09641) 生成文本到图像的管道。 | `src/diffusers/pipelines/sana/pipeline_sana_sprint_img2img.py` |
| SanaSprintPipeline | 使用 [SANA-Sprint](https://huggingface.co/papers/2503.09641) 生成文本到图像的管道。 | `src/diffusers/pipelines/sana/pipeline_sana_sprint.py` |
| StableCascadeCombinedPipeline | 使用稳定级联生成文本到图像的组合管道。 | `src/diffusers/pipelines/stable_cascade/pipeline_stable_cascade_combined.py` |
| StableDiffusion3PAGPipeline | [PAG 管道](https://huggingface.co/docs/diffusers/main/en/using-diffusers/pag) 用于文本到图像... | `src/diffusers/pipelines/pag/pipeline_pag_sd_3.py` |
| StableDiffusionAdapterPipeline | 使用 T2I-Adapter 增强的稳定扩散生成文本到图像的管道 | `src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_adapter.py` |
| StableDiffusionAttendAndExcitePipeline | 使用稳定扩散和参加并激发的文本到图像生成管道。 | `src/diffusers/pipelines/stable_diffusion_attend_and_excite/pipeline_stable_diffusion_attend_and_excite.py` |
| StableDiffusionControlNetPAGPipeline | 在 ControlNet 指导下使用稳定扩散生成文本到图像的管道。 | `src/diffusers/pipelines/pag/pipeline_pag_controlnet_sd.py` |
| StableDiffusionControlNetPipeline | 在 ControlNet 指导下使用稳定扩散生成文本到图像的管道。 | `src/diffusers/pipelines/controlnet/pipeline_controlnet.py` |
| StableDiffusionControlNetXSPipeline | 使用稳定扩散和 ControlNet-XS 指导生成文本到图像的管道。 | `src/diffusers/pipelines/controlnet_xs/pipeline_controlnet_xs.py` |
| StableDiffusionGLIGENPipeline | 使用稳定扩散和接地语言到图像生成器生成文本到图像的管道... | `src/diffusers/pipelines/stable_diffusion_gligen/pipeline_stable_diffusion_gligen.py` |
| StableDiffusionGLIGENTextImagePipeline | 使用稳定扩散和接地语言到图像生成器生成文本到图像的管道... | `src/diffusers/pipelines/stable_diffusion_gligen/pipeline_stable_diffusion_gligen_text_image.py` |
| StableDiffusionKDiffusionPipeline | 使用稳定扩散生成文本到图像的管道。 | `src/diffusers/pipelines/stable_diffusion_k_diffusion/pipeline_stable_diffusion_k_diffusion.py` |
| StableDiffusionModelEditingPipeline | 用于文本到图像模型编辑的管道。 | `src/diffusers/pipelines/deprecated/stable_diffusion_variants/pipeline_stable_diffusion_model_editing.py` |
| StableDiffusionPAGInpaintPipeline | 使用稳定扩散生成文本到图像的管道。 | `src/diffusers/pipelines/pag/pipeline_pag_sd_inpaint.py` |
| StableDiffusionPAGPipeline | 使用稳定扩散生成文本到图像的管道。 | `src/diffusers/pipelines/pag/pipeline_pag_sd.py` |
| StableDiffusionParadigmsPipeline | 使用稳定扩散的并行版本生成文本到图像的管道。 | `src/diffusers/pipelines/deprecated/stable_diffusion_variants/pipeline_stable_diffusion_paradigms.py` |
| StableDiffusionPipeline | 使用稳定扩散生成文本到图像的管道。 | `src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py` |
| StableDiffusionXLAdapterPipeline | 使用 T2I-Adapter 增强的稳定扩散生成文本到图像的管道 | `src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_xl_adapter.py` |
| StableDiffusionXLControlNetInpaintPipeline | 使用 Stable Diffusion XL 生成文本到图像的管道。 | `src/diffusers/pipelines/controlnet/pipeline_controlnet_inpaint_sd_xl.py` |
| StableDiffusionXLControlNetPAGPipeline | 使用 Stable Diffusion XL 在 ControlNet 指导下生成文本到图像的管道。 | `src/diffusers/pipelines/pag/pipeline_pag_controlnet_sd_xl.py` |
| StableDiffusionXLControlNetPipeline | 使用 Stable Diffusion XL 在 ControlNet 指导下生成文本到图像的管道。 | `src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl.py` |
| StableDiffusionXLControlNetUnionInpaintPipeline | 使用 Stable Diffusion XL 生成文本到图像的管道。 | `src/diffusers/pipelines/controlnet/pipeline_controlnet_union_inpaint_sd_xl.py` |
| StableDiffusionXLControlNetUnionPipeline | 使用 Stable Diffusion XL 在 ControlNet 指导下生成文本到图像的管道。 | `src/diffusers/pipelines/controlnet/pipeline_controlnet_union_sd_xl.py` |
| StableDiffusionXLControlNetXSPipeline | 使用 Stable Diffusion XL 和 ControlNet-XS 指导生成文本到图像的管道。 | `src/diffusers/pipelines/controlnet_xs/pipeline_controlnet_xs_sd_xl.py` |
| StableDiffusionXLImg2ImgPipeline | 使用 Stable Diffusion XL 生成文本到图像的管道。 | `src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py` |
| StableDiffusionXLInpaintPipeline | 使用 Stable Diffusion XL 生成文本到图像的管道。 | `src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_inpaint.py` |
| StableDiffusionXLPAGImg2ImgPipeline | 使用 Stable Diffusion XL 生成文本到图像的管道。 | `src/diffusers/pipelines/pag/pipeline_pag_sd_xl_img2img.py` |
| StableDiffusionXLPAGInpaintPipeline | 使用 Stable Diffusion XL 生成文本到图像的管道。 | `src/diffusers/pipelines/pag/pipeline_pag_sd_xl_inpaint.py` |
| StableDiffusionXLPAGPipeline | 使用 Stable Diffusion XL 生成文本到图像的管道。 | `src/diffusers/pipelines/pag/pipeline_pag_sd_xl.py` |
| StableDiffusionXLPipeline | 使用 Stable Diffusion XL 生成文本到图像的管道。 | `src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py` |
| StableUnCLIPPipeline | 使用稳定的 unCLIP 生成文本到图像的管道。 | `src/diffusers/pipelines/stable_diffusion/pipeline_stable_unclip.py` |
| UnCLIPPipeline | 使用 unCLIP 生成文本到图像的管道。 | `src/diffusers/pipelines/unclip/pipeline_unclip.py` |
| VQDiffusionPipeline | 使用 VQ 扩散生成文本到图像的管道。 | `src/diffusers/pipelines/deprecated/vq_diffusion/pipeline_vq_diffusion.py` |
| VersatileDiffusionPipeline | 使用稳定扩散生成文本到图像的管道。 | `src/diffusers/pipelines/deprecated/versatile_diffusion/pipeline_versatile_diffusion.py` |
| VersatileDiffusionTextToImagePipeline | 使用 Versatile Diffusion 生成文本到图像的管道。 | `src/diffusers/pipelines/deprecated/versatile_diffusion/pipeline_versatile_diffusion_text_to_image.py` |
| WuerstchenCombinedPipeline | 使用 Wuerstchen 生成文本到图像的组合管道 | `src/diffusers/pipelines/wuerstchen/pipeline_wuerstchen_combined.py` |

### video-generation
| Pipeline 类 | 用途 | 文件 |
| ------------- | ------ | ------ |
| AllegroPipeline | 使用 Allegro 生成文本到视频的管道。 | `src/diffusers/pipelines/allegro/pipeline_allegro.py` |
| AnimateDiffControlNetPipeline | 在 ControlNet 指导下生成文本到视频的管道。 | `src/diffusers/pipelines/animatediff/pipeline_animatediff_controlnet.py` |
| AnimateDiffPAGPipeline | 使用以下方式生成文本到视频的管道 | `src/diffusers/pipelines/pag/pipeline_pag_sd_animatediff.py` |
| AnimateDiffPipeline | 用于生成文本到视频的管道。 | `src/diffusers/pipelines/animatediff/pipeline_animatediff.py` |
| AnimateDiffSDXLPipeline | 使用 Stable Diffusion XL 生成文本到视频的管道。 | `src/diffusers/pipelines/animatediff/pipeline_animatediff_sdxl.py` |
| AnimateDiffSparseControlNetPipeline | 使用 [SparseCtrl：添加 S... 中描述的方法进行受控文本到视频生成的管道 | `src/diffusers/pipelines/animatediff/pipeline_animatediff_sparsectrl.py` |
| AnimateDiffVideoToVideoControlNetPipeline | 在 ControlNet 指导下生成视频到视频的管道。 | `src/diffusers/pipelines/animatediff/pipeline_animatediff_video2video_controlnet.py` |
| AnimateDiffVideoToVideoPipeline | 视频到视频生成的管道。 | `src/diffusers/pipelines/animatediff/pipeline_animatediff_video2video.py` |
| ChronoEditPipeline | 使用 Wan 生成图像到视频的管道。 | `src/diffusers/pipelines/chronoedit/pipeline_chronoedit.py` |
| CogVideoXFunControlPipeline | 使用 CogVideoX Fun 进行受控文本到视频生成的管道。 | `src/diffusers/pipelines/cogvideo/pipeline_cogvideox_fun_control.py` |
| CogVideoXImageToVideoPipeline | 使用 CogVideoX 生成图像到视频的管道。 | `src/diffusers/pipelines/cogvideo/pipeline_cogvideox_image2video.py` |
| CogVideoXPipeline | 使用 CogVideoX 生成文本到视频的管道。 | `src/diffusers/pipelines/cogvideo/pipeline_cogvideox.py` |
| CogVideoXVideoToVideoPipeline | 使用 CogVideoX 生成视频到视频的管道。 | `src/diffusers/pipelines/cogvideo/pipeline_cogvideox_video2video.py` |
| ConsisIDPipeline | 使用 ConsisID 生成图像到视频的管道。 | `src/diffusers/pipelines/consisid/pipeline_consisid.py` |
| Cosmos2VideoToWorldPipeline | 使用 [Cosmos Predict2](https://github.com/nvidia-cosmos/cosm... 生成视频到世界的管道 | `src/diffusers/pipelines/cosmos/pipeline_cosmos2_video2world.py` |
| CosmosVideoToWorldPipeline | 使用 [Cosmos] 生成图像到世界和视频到世界的管道 | `src/diffusers/pipelines/cosmos/pipeline_cosmos_video2world.py` |
| EasyAnimateControlPipeline | 使用 EasyAnimate 生成文本到视频的管道。 | `src/diffusers/pipelines/easyanimate/pipeline_easyanimate_control.py` |
| EasyAnimateInpaintPipeline | 使用 EasyAnimate 生成文本到视频的管道。 | `src/diffusers/pipelines/easyanimate/pipeline_easyanimate_inpaint.py` |
| EasyAnimatePipeline | 使用 EasyAnimate 生成文本到视频的管道。 | `src/diffusers/pipelines/easyanimate/pipeline_easyanimate.py` |
| HunyuanSkyreelsImageToVideoPipeline | 使用 HunyuanVideo 生成图像到视频的管道。 | `src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_skyreels_image2video.py` |
| HunyuanVideoFramepackPipeline | 使用 HunyuanVideo 生成文本到视频的管道。 | `src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video_framepack.py` |
| HunyuanVideoImageToVideoPipeline | 使用 HunyuanVideo 生成图像到视频的管道。 | `src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video_image2video.py` |
| HunyuanVideoPipeline | 使用 HunyuanVideo 生成文本到视频的管道。 | `src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video.py` |
| Kandinsky5T2VPipeline | 使用 Kandinsky 5.0 生成文本到视频的管道。 | `src/diffusers/pipelines/kandinsky5/pipeline_kandinsky.py` |
| LTXConditionPipeline | 用于文本/图像/视频到视频生成的管道。 | `src/diffusers/pipelines/ltx/pipeline_ltx_condition.py` |
| LTXImageToVideoPipeline | 图像到视频生成的管道。 | `src/diffusers/pipelines/ltx/pipeline_ltx_image2video.py` |
| LTXPipeline | 用于生成文本到视频的管道。 | `src/diffusers/pipelines/ltx/pipeline_ltx.py` |
| LattePipeline | 使用 Latte 生成文本到视频的管道。 | `src/diffusers/pipelines/latte/pipeline_latte.py` |
| LucyEditPipeline | 使用 Lucy Edit 生成视频到视频的管道。 | `src/diffusers/pipelines/lucy/pipeline_lucy_edit.py` |
| MochiPipeline | 用于生成文本到视频的 mochi 管道。 | `src/diffusers/pipelines/mochi/pipeline_mochi.py` |
| SanaImageToVideoPipeline | 使用 [Sana](https://huggingface.co/papers/2509.24695) 生成图像/文本到视频的管道。... | `src/diffusers/pipelines/sana_video/pipeline_sana_video_i2v.py` |
| SanaVideoPipeline | 使用 [Sana](https://huggingface.co/papers/2509.24695) 生成文本到视频的管道。这... | `src/diffusers/pipelines/sana_video/pipeline_sana_video.py` |
| SkyReelsV2DiffusionForcingImageToVideoPipeline | 使用具有扩散强迫的 SkyReels-V2 生成图像到视频 (i2v) 的管道。 | `src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_i2v.py` |
| SkyReelsV2DiffusionForcingPipeline | 使用带有扩散强制的 SkyReels-V2 生成文本到视频 (t2v) 的管道。 | `src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing.py` |
| SkyReelsV2DiffusionForcingVideoToVideoPipeline | 使用带有扩散强制的 SkyReels-V2 生成视频到视频 (v2v) 的管道。 | `src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_v2v.py` |
| SkyReelsV2ImageToVideoPipeline | 使用 SkyReels-V2 生成图像到视频 (i2v) 的管道。 | `src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_i2v.py` |
| SkyReelsV2Pipeline | 使用 SkyReels-V2 生成文本到视频 (t2v) 的管道。 | `src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2.py` |
| StableVideoDiffusionPipeline | 使用稳定视频扩散从输入图像生成视频的管道。 | `src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py` |
| WanImageToVideoPipeline | 使用 Wan 生成图像到视频的管道。 | `src/diffusers/pipelines/wan/pipeline_wan_i2v.py` |
| WanPipeline | 使用 Wan 生成文本到视频的管道。 | `src/diffusers/pipelines/wan/pipeline_wan.py` |
| WanVideoToVideoPipeline | 使用 Wan 生成视频到视频的管道。 | `src/diffusers/pipelines/wan/pipeline_wan_video2video.py` |