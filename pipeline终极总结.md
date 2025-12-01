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
|-------------|------|------|
| ShapEImg2ImgPipeline | Pipeline for generating latent representation of a 3D asset and rendering with the NeRF method from ... | `src/diffusers/pipelines/shap_e/pipeline_shap_e_img2img.py` |
| ShapEPipeline | Pipeline for generating latent representation of a 3D asset and rendering with the NeRF method. | `src/diffusers/pipelines/shap_e/pipeline_shap_e.py` |

### audio-generation
| Pipeline 类 | 用途 | 文件 |
|-------------|------|------|
| AudioDiffusionPipeline | Pipeline for audio diffusion. | `src/diffusers/pipelines/deprecated/audio_diffusion/pipeline_audio_diffusion.py` |
| AudioLDM2Pipeline | Pipeline for text-to-audio generation using AudioLDM2. | `src/diffusers/pipelines/audioldm2/pipeline_audioldm2.py` |
| AudioLDMPipeline | Pipeline for text-to-audio generation using AudioLDM. | `src/diffusers/pipelines/audioldm/pipeline_audioldm.py` |
| DanceDiffusionPipeline | Pipeline for audio generation. | `src/diffusers/pipelines/dance_diffusion/pipeline_dance_diffusion.py` |
| SpectrogramDiffusionPipeline | Pipeline for unconditional audio generation. | `src/diffusers/pipelines/deprecated/spectrogram_diffusion/pipeline_spectrogram_diffusion.py` |
| StableAudioPipeline | Pipeline for text-to-audio generation using StableAudio. | `src/diffusers/pipelines/stable_audio/pipeline_stable_audio.py` |

### controlnet
| Pipeline 类 | 用途 | 文件 |
|-------------|------|------|
| BlipDiffusionControlNetPipeline | Pipeline for Canny Edge based Controlled subject-driven generation using Blip Diffusion. | `src/diffusers/pipelines/controlnet/pipeline_controlnet_blip_diffusion.py` |
| WanVACEPipeline | Pipeline for controllable generation using Wan. | `src/diffusers/pipelines/wan/pipeline_wan_vace.py` |

### diffusion
| Pipeline 类 | 用途 | 文件 |
|-------------|------|------|
| BlipDiffusionPipeline | Pipeline for Zero-Shot Subject Driven Generation using Blip Diffusion. | `src/diffusers/pipelines/blip_diffusion/pipeline_blip_diffusion.py` |
| LDMPipeline | Pipeline for unconditional image generation using latent diffusion. | `src/diffusers/pipelines/deprecated/latent_diffusion_uncond/pipeline_latent_diffusion_uncond.py` |
| StableDiffusionLatentUpscalePipeline | Pipeline for upscaling Stable Diffusion output image resolution by a factor of 2. | `src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_latent_upscale.py` |
| VersatileDiffusionDualGuidedPipeline | Pipeline for image-text dual-guided generation using Versatile Diffusion. | `src/diffusers/pipelines/deprecated/versatile_diffusion/pipeline_versatile_diffusion_dual_guided.py` |

### image-to-image
| Pipeline 类 | 用途 | 文件 |
|-------------|------|------|
| AltDiffusionImg2ImgPipeline | Pipeline for text-guided image-to-image generation using Alt Diffusion. | `src/diffusers/pipelines/deprecated/alt_diffusion/pipeline_alt_diffusion_img2img.py` |
| ChromaImg2ImgPipeline | The Chroma pipeline for image-to-image generation. | `src/diffusers/pipelines/chroma/pipeline_chroma_img2img.py` |
| CycleDiffusionPipeline | Pipeline for text-guided image to image generation using Stable Diffusion. | `src/diffusers/pipelines/deprecated/stable_diffusion_variants/pipeline_cycle_diffusion.py` |
| FlaxStableDiffusionImg2ImgPipeline | Flax-based pipeline for text-guided image-to-image generation using Stable Diffusion. | `src/diffusers/pipelines/stable_diffusion/pipeline_flax_stable_diffusion_img2img.py` |
| FlaxStableDiffusionInpaintPipeline | Flax-based pipeline for text-guided image inpainting using Stable Diffusion. | `src/diffusers/pipelines/stable_diffusion/pipeline_flax_stable_diffusion_inpaint.py` |
| FluxControlImg2ImgPipeline | The Flux pipeline for image inpainting. | `src/diffusers/pipelines/flux/pipeline_flux_control_img2img.py` |
| FluxControlInpaintPipeline | The Flux pipeline for image inpainting using Flux-dev-Depth/Canny. | `src/diffusers/pipelines/flux/pipeline_flux_control_inpaint.py` |
| FluxControlNetImg2ImgPipeline | The Flux controlnet pipeline for image-to-image generation. | `src/diffusers/pipelines/flux/pipeline_flux_controlnet_image_to_image.py` |
| FluxControlNetInpaintPipeline | The Flux controlnet pipeline for inpainting. | `src/diffusers/pipelines/flux/pipeline_flux_controlnet_inpainting.py` |
| FluxFillPipeline | The Flux Fill pipeline for image inpainting/outpainting. | `src/diffusers/pipelines/flux/pipeline_flux_fill.py` |
| FluxImg2ImgPipeline | The Flux pipeline for image inpainting. | `src/diffusers/pipelines/flux/pipeline_flux_img2img.py` |
| FluxInpaintPipeline | The Flux pipeline for image inpainting. | `src/diffusers/pipelines/flux/pipeline_flux_inpaint.py` |
| FluxPriorReduxPipeline | The Flux Redux pipeline for image-to-image generation. | `src/diffusers/pipelines/flux/pipeline_flux_prior_redux.py` |
| KandinskyImg2ImgCombinedPipeline | Combined Pipeline for image-to-image generation using Kandinsky | `src/diffusers/pipelines/kandinsky/pipeline_kandinsky_combined.py` |
| KandinskyImg2ImgPipeline | Pipeline for image-to-image generation using Kandinsky | `src/diffusers/pipelines/kandinsky/pipeline_kandinsky_img2img.py` |
| KandinskyInpaintPipeline | Pipeline for text-guided image inpainting using Kandinsky2.1 | `src/diffusers/pipelines/kandinsky/pipeline_kandinsky_inpaint.py` |
| KandinskyV22ControlnetImg2ImgPipeline | Pipeline for image-to-image generation using Kandinsky | `src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_controlnet_img2img.py` |
| KandinskyV22Img2ImgCombinedPipeline | Combined Pipeline for image-to-image generation using Kandinsky | `src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py` |
| KandinskyV22Img2ImgPipeline | Pipeline for image-to-image generation using Kandinsky | `src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_img2img.py` |
| KandinskyV22InpaintCombinedPipeline | Combined Pipeline for inpainting generation using Kandinsky | `src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py` |
| KandinskyV22InpaintPipeline | Pipeline for text-guided image inpainting using Kandinsky2.1 | `src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_inpainting.py` |
| LDMSuperResolutionPipeline | A pipeline for image super-resolution using latent diffusion. | `src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion_superresolution.py` |
| LEditsPPPipelineStableDiffusion | Pipeline for textual image editing using LEDits++ with Stable Diffusion. | `src/diffusers/pipelines/ledits_pp/pipeline_leditspp_stable_diffusion.py` |
| LEditsPPPipelineStableDiffusionXL | Pipeline for textual image editing using LEDits++ with Stable Diffusion XL. | `src/diffusers/pipelines/ledits_pp/pipeline_leditspp_stable_diffusion_xl.py` |
| LatentConsistencyModelImg2ImgPipeline | Pipeline for image-to-image generation using a latent consistency model. | `src/diffusers/pipelines/latent_consistency_models/pipeline_latent_consistency_img2img.py` |
| OnnxStableDiffusionImg2ImgPipeline | Pipeline for text-guided image to image generation using Stable Diffusion. | `src/diffusers/pipelines/stable_diffusion/pipeline_onnx_stable_diffusion_img2img.py` |
| OnnxStableDiffusionInpaintPipeline | Pipeline for text-guided image inpainting using Stable Diffusion. *This is an experimental feature*. | `src/diffusers/pipelines/stable_diffusion/pipeline_onnx_stable_diffusion_inpaint.py` |
| OnnxStableDiffusionInpaintPipelineLegacy | Pipeline for text-guided image inpainting using Stable Diffusion. This is a *legacy feature* for Onn... | `src/diffusers/pipelines/deprecated/stable_diffusion_variants/pipeline_onnx_stable_diffusion_inpaint_legacy.py` |
| QwenImageEditInpaintPipeline | The Qwen-Image-Edit pipeline for image editing. | `src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_inpaint.py` |
| QwenImageEditPipeline | The Qwen-Image-Edit pipeline for image editing. | `src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py` |
| QwenImageEditPlusPipeline | The Qwen-Image-Edit pipeline for image editing. | `src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py` |
| RePaintPipeline | Pipeline for image inpainting using RePaint. | `src/diffusers/pipelines/deprecated/repaint/pipeline_repaint.py` |
| StableDiffusion3PAGImg2ImgPipeline | [PAG pipeline](https://huggingface.co/docs/diffusers/main/en/using-diffusers/pag) for image-to-image... | `src/diffusers/pipelines/pag/pipeline_pag_sd_3_img2img.py` |
| StableDiffusionControlNetImg2ImgPipeline | Pipeline for image-to-image generation using Stable Diffusion with ControlNet guidance. | `src/diffusers/pipelines/controlnet/pipeline_controlnet_img2img.py` |
| StableDiffusionControlNetInpaintPipeline | Pipeline for image inpainting using Stable Diffusion with ControlNet guidance. | `src/diffusers/pipelines/controlnet/pipeline_controlnet_inpaint.py` |
| StableDiffusionControlNetPAGInpaintPipeline | Pipeline for image inpainting using Stable Diffusion with ControlNet guidance. | `src/diffusers/pipelines/pag/pipeline_pag_controlnet_sd_inpaint.py` |
| StableDiffusionDepth2ImgPipeline | Pipeline for text-guided depth-based image-to-image generation using Stable Diffusion. | `src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_depth2img.py` |
| StableDiffusionImageVariationPipeline | Pipeline to generate image variations from an input image using Stable Diffusion. | `src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_image_variation.py` |
| StableDiffusionImg2ImgPipeline | Pipeline for text-guided image-to-image generation using Stable Diffusion. | `src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py` |
| StableDiffusionInpaintPipeline | Pipeline for text-guided image inpainting using Stable Diffusion. | `src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py` |
| StableDiffusionInpaintPipelineLegacy | Pipeline for text-guided image inpainting using Stable Diffusion. *This is an experimental feature*. | `src/diffusers/pipelines/deprecated/stable_diffusion_variants/pipeline_stable_diffusion_inpaint_legacy.py` |
| StableDiffusionInstructPix2PixPipeline | Pipeline for pixel-level image editing by following text instructions (based on Stable Diffusion). | `src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py` |
| StableDiffusionPAGImg2ImgPipeline | Pipeline for text-guided image-to-image generation using Stable Diffusion. | `src/diffusers/pipelines/pag/pipeline_pag_sd_img2img.py` |
| StableDiffusionPix2PixZeroPipeline | Pipeline for pixel-level image editing using Pix2Pix Zero. Based on Stable Diffusion. | `src/diffusers/pipelines/deprecated/stable_diffusion_variants/pipeline_stable_diffusion_pix2pix_zero.py` |
| StableDiffusionUpscalePipeline | Pipeline for text-guided image super-resolution using Stable Diffusion 2. | `src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_upscale.py` |
| StableDiffusionXLControlNetImg2ImgPipeline | Pipeline for image-to-image generation using Stable Diffusion XL with ControlNet guidance. | `src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl_img2img.py` |
| StableDiffusionXLControlNetPAGImg2ImgPipeline | Pipeline for image-to-image generation using Stable Diffusion XL with ControlNet guidance. | `src/diffusers/pipelines/pag/pipeline_pag_controlnet_sd_xl_img2img.py` |
| StableDiffusionXLControlNetUnionImg2ImgPipeline | Pipeline for image-to-image generation using Stable Diffusion XL with ControlNet guidance. | `src/diffusers/pipelines/controlnet/pipeline_controlnet_union_sd_xl_img2img.py` |
| StableDiffusionXLInstructPix2PixPipeline | Pipeline for pixel-level image editing by following text instructions. Based on Stable Diffusion XL. | `src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_instruct_pix2pix.py` |
| StableUnCLIPImg2ImgPipeline | Pipeline for text-guided image-to-image generation using stable unCLIP. | `src/diffusers/pipelines/stable_diffusion/pipeline_stable_unclip_img2img.py` |
| UnCLIPImageVariationPipeline | Pipeline to generate image variations from an input image using UnCLIP. | `src/diffusers/pipelines/unclip/pipeline_unclip_image_variation.py` |
| VersatileDiffusionImageVariationPipeline | Pipeline for image variation using Versatile Diffusion. | `src/diffusers/pipelines/deprecated/versatile_diffusion/pipeline_versatile_diffusion_image_variation.py` |

### other
| Pipeline 类 | 用途 | 文件 |
|-------------|------|------|
| AmusedImg2ImgPipeline | No description | `src/diffusers/pipelines/amused/pipeline_amused_img2img.py` |
| AmusedInpaintPipeline | No description | `src/diffusers/pipelines/amused/pipeline_amused_inpaint.py` |
| AmusedPipeline | No description | `src/diffusers/pipelines/amused/pipeline_amused.py` |
| AuraFlowPipeline | Args: | `src/diffusers/pipelines/aura_flow/pipeline_aura_flow.py` |
| BriaFiboPipeline | Args: | `src/diffusers/pipelines/bria_fibo/pipeline_bria_fibo.py` |
| BriaPipeline | Based on FluxPipeline with several changes: | `src/diffusers/pipelines/bria/pipeline_bria.py` |
| ConsistencyModelPipeline | Pipeline for unconditional or class-conditional image generation. | `src/diffusers/pipelines/consistency_models/pipeline_consistency_models.py` |
| CosmosTextToWorldPipeline | Pipeline for text-to-world generation using [Cosmos Predict1](https://github.com/nvidia-cosmos/cosmo... | `src/diffusers/pipelines/cosmos/pipeline_cosmos_text2world.py` |
| DDIMPipeline | Pipeline for image generation. | `src/diffusers/pipelines/ddim/pipeline_ddim.py` |
| DDPMPipeline | Pipeline for image generation. | `src/diffusers/pipelines/ddpm/pipeline_ddpm.py` |
| DiTPipeline | Pipeline for image generation based on a Transformer backbone instead of a UNet. | `src/diffusers/pipelines/dit/pipeline_dit.py` |
| FlaxStableDiffusionXLPipeline | No description | `src/diffusers/pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.py` |
| HiDreamImagePipeline | No description | `src/diffusers/pipelines/hidream_image/pipeline_hidream_image.py` |
| HunyuanDiTControlNetPipeline | Pipeline for English/Chinese-to-image generation using HunyuanDiT. | `src/diffusers/pipelines/controlnet_hunyuandit/pipeline_hunyuandit_controlnet.py` |
| HunyuanDiTPAGPipeline | Pipeline for English/Chinese-to-image generation using HunyuanDiT and [Perturbed Attention | `src/diffusers/pipelines/pag/pipeline_pag_hunyuandit.py` |
| HunyuanDiTPipeline | Pipeline for English/Chinese-to-image generation using HunyuanDiT. | `src/diffusers/pipelines/hunyuandit/pipeline_hunyuandit.py` |
| I2VGenXLPipeline | No description | `src/diffusers/pipelines/i2vgen_xl/pipeline_i2vgen_xl.py` |
| IFImg2ImgPipeline | No description | `src/diffusers/pipelines/deepfloyd_if/pipeline_if_img2img.py` |
| IFImg2ImgSuperResolutionPipeline | No description | `src/diffusers/pipelines/deepfloyd_if/pipeline_if_img2img_superresolution.py` |
| IFInpaintingPipeline | No description | `src/diffusers/pipelines/deepfloyd_if/pipeline_if_inpainting.py` |
| IFInpaintingSuperResolutionPipeline | No description | `src/diffusers/pipelines/deepfloyd_if/pipeline_if_inpainting_superresolution.py` |
| IFPipeline | No description | `src/diffusers/pipelines/deepfloyd_if/pipeline_if.py` |
| IFSuperResolutionPipeline | No description | `src/diffusers/pipelines/deepfloyd_if/pipeline_if_superresolution.py` |
| Kandinsky3Img2ImgPipeline | No description | `src/diffusers/pipelines/kandinsky3/pipeline_kandinsky3_img2img.py` |
| Kandinsky3Pipeline | No description | `src/diffusers/pipelines/kandinsky3/pipeline_kandinsky3.py` |
| KandinskyInpaintCombinedPipeline | Combined Pipeline for generation using Kandinsky | `src/diffusers/pipelines/kandinsky/pipeline_kandinsky_combined.py` |
| KarrasVePipeline | Pipeline for unconditional image generation. | `src/diffusers/pipelines/deprecated/stochastic_karras_ve/pipeline_stochastic_karras_ve.py` |
| LTXLatentUpsamplePipeline | No description | `src/diffusers/pipelines/ltx/pipeline_ltx_latent_upsample.py` |
| Lumina2Text2ImgPipeline | No description | `src/diffusers/pipelines/lumina2/pipeline_lumina2.py` |
| LuminaText2ImgPipeline | No description | `src/diffusers/pipelines/lumina/pipeline_lumina.py` |
| MarigoldDepthPipeline | Pipeline for monocular depth estimation using the Marigold method: https://marigoldmonodepth.github.... | `src/diffusers/pipelines/marigold/pipeline_marigold_depth.py` |
| MarigoldIntrinsicsPipeline | Pipeline for Intrinsic Image Decomposition (IID) using the Marigold method: | `src/diffusers/pipelines/marigold/pipeline_marigold_intrinsics.py` |
| MarigoldNormalsPipeline | Pipeline for monocular normals estimation using the Marigold method: https://marigoldmonodepth.githu... | `src/diffusers/pipelines/marigold/pipeline_marigold_normals.py` |
| MusicLDMPipeline | No description | `src/diffusers/pipelines/musicldm/pipeline_musicldm.py` |
| OmniGenPipeline | The OmniGen pipeline for multimodal-to-image generation. | `src/diffusers/pipelines/omnigen/pipeline_omnigen.py` |
| OnnxStableDiffusionPipeline | No description | `src/diffusers/pipelines/stable_diffusion/pipeline_onnx_stable_diffusion.py` |
| OnnxStableDiffusionUpscalePipeline | No description | `src/diffusers/pipelines/stable_diffusion/pipeline_onnx_stable_diffusion_upscale.py` |
| PIAPipeline | No description | `src/diffusers/pipelines/pia/pipeline_pia.py` |
| PNDMPipeline | Pipeline for unconditional image generation. | `src/diffusers/pipelines/deprecated/pndm/pipeline_pndm.py` |
| PaintByExamplePipeline | No description | `src/diffusers/pipelines/paint_by_example/pipeline_paint_by_example.py` |
| ScoreSdeVePipeline | Pipeline for unconditional image generation. | `src/diffusers/pipelines/deprecated/score_sde_ve/pipeline_score_sde_ve.py` |
| SemanticStableDiffusionPipeline | No description | `src/diffusers/pipelines/semantic_stable_diffusion/pipeline_semantic_stable_diffusion.py` |
| StableCascadeDecoderPipeline | Pipeline for generating images from the Stable Cascade model. | `src/diffusers/pipelines/stable_cascade/pipeline_stable_cascade.py` |
| StableDiffusion3ControlNetInpaintingPipeline | Args: | `src/diffusers/pipelines/controlnet_sd3/pipeline_stable_diffusion_3_controlnet_inpainting.py` |
| StableDiffusion3ControlNetPipeline | Args: | `src/diffusers/pipelines/controlnet_sd3/pipeline_stable_diffusion_3_controlnet.py` |
| StableDiffusion3Img2ImgPipeline | Args: | `src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3_img2img.py` |
| StableDiffusion3InpaintPipeline | Args: | `src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3_inpaint.py` |
| StableDiffusion3Pipeline | Args: | `src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py` |
| StableDiffusionDiffEditPipeline | > [!WARNING] > This is an experimental feature! | `src/diffusers/pipelines/stable_diffusion_diffedit/pipeline_stable_diffusion_diffedit.py` |
| StableDiffusionLDM3DPipeline | No description | `src/diffusers/pipelines/stable_diffusion_ldm3d/pipeline_stable_diffusion_ldm3d.py` |
| StableDiffusionOnnxPipeline | No description | `src/diffusers/pipelines/stable_diffusion/pipeline_onnx_stable_diffusion.py` |
| StableDiffusionPanoramaPipeline | No description | `src/diffusers/pipelines/stable_diffusion_panorama/pipeline_stable_diffusion_panorama.py` |
| StableDiffusionPipelineSafe | No description | `src/diffusers/pipelines/stable_diffusion_safe/pipeline_stable_diffusion_safe.py` |
| StableDiffusionSAGPipeline | No description | `src/diffusers/pipelines/stable_diffusion_sag/pipeline_stable_diffusion_sag.py` |
| StableDiffusionXLKDiffusionPipeline | No description | `src/diffusers/pipelines/stable_diffusion_k_diffusion/pipeline_stable_diffusion_xl_k_diffusion.py` |
| TextToVideoSDPipeline | No description | `src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth.py` |
| TextToVideoZeroPipeline | No description | `src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_zero.py` |
| TextToVideoZeroSDXLPipeline | No description | `src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_zero_sdxl.py` |
| UniDiffuserPipeline | Pipeline for a bimodal image-text model which supports unconditional text and image generation, text... | `src/diffusers/pipelines/unidiffuser/pipeline_unidiffuser.py` |
| VideoToVideoSDPipeline | No description | `src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth_img2img.py` |
| VisualClozeGenerationPipeline | The VisualCloze pipeline for image generation with visual context. Reference: | `src/diffusers/pipelines/visualcloze/pipeline_visualcloze_generation.py` |
| VisualClozePipeline | The VisualCloze pipeline for image generation with visual context. Reference: | `src/diffusers/pipelines/visualcloze/pipeline_visualcloze_combined.py` |
| WanAnimatePipeline | Pipeline for unified character animation and replacement using Wan-Animate. | `src/diffusers/pipelines/wan/pipeline_wan_animate.py` |
| WuerstchenDecoderPipeline | Pipeline for generating images from the Wuerstchen model. | `src/diffusers/pipelines/wuerstchen/pipeline_wuerstchen.py` |
| ZImagePipeline | No description | `src/diffusers/pipelines/z_image/pipeline_z_image.py` |

### prior
| Pipeline 类 | 用途 | 文件 |
|-------------|------|------|
| KandinskyPriorPipeline | Pipeline for generating image prior for Kandinsky | `src/diffusers/pipelines/kandinsky/pipeline_kandinsky_prior.py` |
| KandinskyV22PriorEmb2EmbPipeline | Pipeline for generating image prior for Kandinsky | `src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_prior_emb2emb.py` |
| KandinskyV22PriorPipeline | Pipeline for generating image prior for Kandinsky | `src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_prior.py` |
| StableCascadePriorPipeline | Pipeline for generating image prior for Stable Cascade. | `src/diffusers/pipelines/stable_cascade/pipeline_stable_cascade_prior.py` |
| WuerstchenPriorPipeline | Pipeline for generating image prior for Wuerstchen. | `src/diffusers/pipelines/wuerstchen/pipeline_wuerstchen_prior.py` |

### text-to-image
| Pipeline 类 | 用途 | 文件 |
|-------------|------|------|
| AltDiffusionPipeline | Pipeline for text-to-image generation using Alt Diffusion. | `src/diffusers/pipelines/deprecated/alt_diffusion/pipeline_alt_diffusion.py` |
| ChromaPipeline | The Chroma pipeline for text-to-image generation. | `src/diffusers/pipelines/chroma/pipeline_chroma.py` |
| CogView3PlusPipeline | Pipeline for text-to-image generation using CogView3Plus. | `src/diffusers/pipelines/cogview3/pipeline_cogview3plus.py` |
| CogView4ControlPipeline | Pipeline for text-to-image generation using CogView4. | `src/diffusers/pipelines/cogview4/pipeline_cogview4_control.py` |
| CogView4Pipeline | Pipeline for text-to-image generation using CogView4. | `src/diffusers/pipelines/cogview4/pipeline_cogview4.py` |
| Cosmos2TextToImagePipeline | Pipeline for text-to-image generation using [Cosmos Predict2](https://github.com/nvidia-cosmos/cosmo... | `src/diffusers/pipelines/cosmos/pipeline_cosmos2_text2image.py` |
| FlaxStableDiffusionControlNetPipeline | Flax-based pipeline for text-to-image generation using Stable Diffusion with ControlNet Guidance. | `src/diffusers/pipelines/controlnet/pipeline_flax_controlnet.py` |
| FlaxStableDiffusionPipeline | Flax-based pipeline for text-to-image generation using Stable Diffusion. | `src/diffusers/pipelines/stable_diffusion/pipeline_flax_stable_diffusion.py` |
| Flux2Pipeline | The Flux2 pipeline for text-to-image generation. | `src/diffusers/pipelines/flux2/pipeline_flux2.py` |
| FluxControlNetPipeline | The Flux pipeline for text-to-image generation. | `src/diffusers/pipelines/flux/pipeline_flux_controlnet.py` |
| FluxControlPipeline | The Flux pipeline for controllable text-to-image generation with image conditions. | `src/diffusers/pipelines/flux/pipeline_flux_control.py` |
| FluxKontextInpaintPipeline | The Flux Kontext pipeline for text-to-image generation. | `src/diffusers/pipelines/flux/pipeline_flux_kontext_inpaint.py` |
| FluxKontextPipeline | The Flux Kontext pipeline for image-to-image and text-to-image generation. | `src/diffusers/pipelines/flux/pipeline_flux_kontext.py` |
| FluxPipeline | The Flux pipeline for text-to-image generation. | `src/diffusers/pipelines/flux/pipeline_flux.py` |
| HunyuanImagePipeline | The HunyuanImage pipeline for text-to-image generation. | `src/diffusers/pipelines/hunyuan_image/pipeline_hunyuanimage.py` |
| HunyuanImageRefinerPipeline | The HunyuanImage pipeline for text-to-image generation. | `src/diffusers/pipelines/hunyuan_image/pipeline_hunyuanimage_refiner.py` |
| KandinskyCombinedPipeline | Combined Pipeline for text-to-image generation using Kandinsky | `src/diffusers/pipelines/kandinsky/pipeline_kandinsky_combined.py` |
| KandinskyPipeline | Pipeline for text-to-image generation using Kandinsky | `src/diffusers/pipelines/kandinsky/pipeline_kandinsky.py` |
| KandinskyV22CombinedPipeline | Combined Pipeline for text-to-image generation using Kandinsky | `src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py` |
| KandinskyV22ControlnetPipeline | Pipeline for text-to-image generation using Kandinsky | `src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_controlnet.py` |
| KandinskyV22Pipeline | Pipeline for text-to-image generation using Kandinsky | `src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2.py` |
| KolorsImg2ImgPipeline | Pipeline for text-to-image generation using Kolors. | `src/diffusers/pipelines/kolors/pipeline_kolors_img2img.py` |
| KolorsPAGPipeline | Pipeline for text-to-image generation using Kolors. | `src/diffusers/pipelines/pag/pipeline_pag_kolors.py` |
| KolorsPipeline | Pipeline for text-to-image generation using Kolors. | `src/diffusers/pipelines/kolors/pipeline_kolors.py` |
| LDMTextToImagePipeline | Pipeline for text-to-image generation using latent diffusion. | `src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion.py` |
| LatentConsistencyModelPipeline | Pipeline for text-to-image generation using a latent consistency model. | `src/diffusers/pipelines/latent_consistency_models/pipeline_latent_consistency_text2img.py` |
| Lumina2Pipeline | Pipeline for text-to-image generation using Lumina-T2I. | `src/diffusers/pipelines/lumina2/pipeline_lumina2.py` |
| LuminaPipeline | Pipeline for text-to-image generation using Lumina-T2I. | `src/diffusers/pipelines/lumina/pipeline_lumina.py` |
| PRXPipeline | Pipeline for text-to-image generation using PRX Transformer. | `src/diffusers/pipelines/prx/pipeline_prx.py` |
| PixArtAlphaPipeline | Pipeline for text-to-image generation using PixArt-Alpha. | `src/diffusers/pipelines/pixart_alpha/pipeline_pixart_alpha.py` |
| PixArtSigmaPAGPipeline | [PAG pipeline](https://huggingface.co/docs/diffusers/main/en/using-diffusers/pag) for text-to-image ... | `src/diffusers/pipelines/pag/pipeline_pag_pixart_sigma.py` |
| PixArtSigmaPipeline | Pipeline for text-to-image generation using PixArt-Sigma. | `src/diffusers/pipelines/pixart_alpha/pipeline_pixart_sigma.py` |
| QwenImageControlNetInpaintPipeline | The QwenImage pipeline for text-to-image generation. | `src/diffusers/pipelines/qwenimage/pipeline_qwenimage_controlnet_inpaint.py` |
| QwenImageControlNetPipeline | The QwenImage pipeline for text-to-image generation. | `src/diffusers/pipelines/qwenimage/pipeline_qwenimage_controlnet.py` |
| QwenImageImg2ImgPipeline | The QwenImage pipeline for text-to-image generation. | `src/diffusers/pipelines/qwenimage/pipeline_qwenimage_img2img.py` |
| QwenImageInpaintPipeline | The QwenImage pipeline for text-to-image generation. | `src/diffusers/pipelines/qwenimage/pipeline_qwenimage_inpaint.py` |
| QwenImagePipeline | The QwenImage pipeline for text-to-image generation. | `src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py` |
| SanaControlNetPipeline | Pipeline for text-to-image generation using [Sana](https://huggingface.co/papers/2410.10629). | `src/diffusers/pipelines/sana/pipeline_sana_controlnet.py` |
| SanaPAGPipeline | Pipeline for text-to-image generation using [Sana](https://huggingface.co/papers/2410.10629). This p... | `src/diffusers/pipelines/pag/pipeline_pag_sana.py` |
| SanaPipeline | Pipeline for text-to-image generation using [Sana](https://huggingface.co/papers/2410.10629). | `src/diffusers/pipelines/sana/pipeline_sana.py` |
| SanaSprintImg2ImgPipeline | Pipeline for text-to-image generation using [SANA-Sprint](https://huggingface.co/papers/2503.09641). | `src/diffusers/pipelines/sana/pipeline_sana_sprint_img2img.py` |
| SanaSprintPipeline | Pipeline for text-to-image generation using [SANA-Sprint](https://huggingface.co/papers/2503.09641). | `src/diffusers/pipelines/sana/pipeline_sana_sprint.py` |
| StableCascadeCombinedPipeline | Combined Pipeline for text-to-image generation using Stable Cascade. | `src/diffusers/pipelines/stable_cascade/pipeline_stable_cascade_combined.py` |
| StableDiffusion3PAGPipeline | [PAG pipeline](https://huggingface.co/docs/diffusers/main/en/using-diffusers/pag) for text-to-image ... | `src/diffusers/pipelines/pag/pipeline_pag_sd_3.py` |
| StableDiffusionAdapterPipeline | Pipeline for text-to-image generation using Stable Diffusion augmented with T2I-Adapter | `src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_adapter.py` |
| StableDiffusionAttendAndExcitePipeline | Pipeline for text-to-image generation using Stable Diffusion and Attend-and-Excite. | `src/diffusers/pipelines/stable_diffusion_attend_and_excite/pipeline_stable_diffusion_attend_and_excite.py` |
| StableDiffusionControlNetPAGPipeline | Pipeline for text-to-image generation using Stable Diffusion with ControlNet guidance. | `src/diffusers/pipelines/pag/pipeline_pag_controlnet_sd.py` |
| StableDiffusionControlNetPipeline | Pipeline for text-to-image generation using Stable Diffusion with ControlNet guidance. | `src/diffusers/pipelines/controlnet/pipeline_controlnet.py` |
| StableDiffusionControlNetXSPipeline | Pipeline for text-to-image generation using Stable Diffusion with ControlNet-XS guidance. | `src/diffusers/pipelines/controlnet_xs/pipeline_controlnet_xs.py` |
| StableDiffusionGLIGENPipeline | Pipeline for text-to-image generation using Stable Diffusion with Grounded-Language-to-Image Generat... | `src/diffusers/pipelines/stable_diffusion_gligen/pipeline_stable_diffusion_gligen.py` |
| StableDiffusionGLIGENTextImagePipeline | Pipeline for text-to-image generation using Stable Diffusion with Grounded-Language-to-Image Generat... | `src/diffusers/pipelines/stable_diffusion_gligen/pipeline_stable_diffusion_gligen_text_image.py` |
| StableDiffusionKDiffusionPipeline | Pipeline for text-to-image generation using Stable Diffusion. | `src/diffusers/pipelines/stable_diffusion_k_diffusion/pipeline_stable_diffusion_k_diffusion.py` |
| StableDiffusionModelEditingPipeline | Pipeline for text-to-image model editing. | `src/diffusers/pipelines/deprecated/stable_diffusion_variants/pipeline_stable_diffusion_model_editing.py` |
| StableDiffusionPAGInpaintPipeline | Pipeline for text-to-image generation using Stable Diffusion. | `src/diffusers/pipelines/pag/pipeline_pag_sd_inpaint.py` |
| StableDiffusionPAGPipeline | Pipeline for text-to-image generation using Stable Diffusion. | `src/diffusers/pipelines/pag/pipeline_pag_sd.py` |
| StableDiffusionParadigmsPipeline | Pipeline for text-to-image generation using a parallelized version of Stable Diffusion. | `src/diffusers/pipelines/deprecated/stable_diffusion_variants/pipeline_stable_diffusion_paradigms.py` |
| StableDiffusionPipeline | Pipeline for text-to-image generation using Stable Diffusion. | `src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py` |
| StableDiffusionXLAdapterPipeline | Pipeline for text-to-image generation using Stable Diffusion augmented with T2I-Adapter | `src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_xl_adapter.py` |
| StableDiffusionXLControlNetInpaintPipeline | Pipeline for text-to-image generation using Stable Diffusion XL. | `src/diffusers/pipelines/controlnet/pipeline_controlnet_inpaint_sd_xl.py` |
| StableDiffusionXLControlNetPAGPipeline | Pipeline for text-to-image generation using Stable Diffusion XL with ControlNet guidance. | `src/diffusers/pipelines/pag/pipeline_pag_controlnet_sd_xl.py` |
| StableDiffusionXLControlNetPipeline | Pipeline for text-to-image generation using Stable Diffusion XL with ControlNet guidance. | `src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl.py` |
| StableDiffusionXLControlNetUnionInpaintPipeline | Pipeline for text-to-image generation using Stable Diffusion XL. | `src/diffusers/pipelines/controlnet/pipeline_controlnet_union_inpaint_sd_xl.py` |
| StableDiffusionXLControlNetUnionPipeline | Pipeline for text-to-image generation using Stable Diffusion XL with ControlNet guidance. | `src/diffusers/pipelines/controlnet/pipeline_controlnet_union_sd_xl.py` |
| StableDiffusionXLControlNetXSPipeline | Pipeline for text-to-image generation using Stable Diffusion XL with ControlNet-XS guidance. | `src/diffusers/pipelines/controlnet_xs/pipeline_controlnet_xs_sd_xl.py` |
| StableDiffusionXLImg2ImgPipeline | Pipeline for text-to-image generation using Stable Diffusion XL. | `src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py` |
| StableDiffusionXLInpaintPipeline | Pipeline for text-to-image generation using Stable Diffusion XL. | `src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_inpaint.py` |
| StableDiffusionXLPAGImg2ImgPipeline | Pipeline for text-to-image generation using Stable Diffusion XL. | `src/diffusers/pipelines/pag/pipeline_pag_sd_xl_img2img.py` |
| StableDiffusionXLPAGInpaintPipeline | Pipeline for text-to-image generation using Stable Diffusion XL. | `src/diffusers/pipelines/pag/pipeline_pag_sd_xl_inpaint.py` |
| StableDiffusionXLPAGPipeline | Pipeline for text-to-image generation using Stable Diffusion XL. | `src/diffusers/pipelines/pag/pipeline_pag_sd_xl.py` |
| StableDiffusionXLPipeline | Pipeline for text-to-image generation using Stable Diffusion XL. | `src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py` |
| StableUnCLIPPipeline | Pipeline for text-to-image generation using stable unCLIP. | `src/diffusers/pipelines/stable_diffusion/pipeline_stable_unclip.py` |
| UnCLIPPipeline | Pipeline for text-to-image generation using unCLIP. | `src/diffusers/pipelines/unclip/pipeline_unclip.py` |
| VQDiffusionPipeline | Pipeline for text-to-image generation using VQ Diffusion. | `src/diffusers/pipelines/deprecated/vq_diffusion/pipeline_vq_diffusion.py` |
| VersatileDiffusionPipeline | Pipeline for text-to-image generation using Stable Diffusion. | `src/diffusers/pipelines/deprecated/versatile_diffusion/pipeline_versatile_diffusion.py` |
| VersatileDiffusionTextToImagePipeline | Pipeline for text-to-image generation using Versatile Diffusion. | `src/diffusers/pipelines/deprecated/versatile_diffusion/pipeline_versatile_diffusion_text_to_image.py` |
| WuerstchenCombinedPipeline | Combined Pipeline for text-to-image generation using Wuerstchen | `src/diffusers/pipelines/wuerstchen/pipeline_wuerstchen_combined.py` |

### video-generation
| Pipeline 类 | 用途 | 文件 |
|-------------|------|------|
| AllegroPipeline | Pipeline for text-to-video generation using Allegro. | `src/diffusers/pipelines/allegro/pipeline_allegro.py` |
| AnimateDiffControlNetPipeline | Pipeline for text-to-video generation with ControlNet guidance. | `src/diffusers/pipelines/animatediff/pipeline_animatediff_controlnet.py` |
| AnimateDiffPAGPipeline | Pipeline for text-to-video generation using | `src/diffusers/pipelines/pag/pipeline_pag_sd_animatediff.py` |
| AnimateDiffPipeline | Pipeline for text-to-video generation. | `src/diffusers/pipelines/animatediff/pipeline_animatediff.py` |
| AnimateDiffSDXLPipeline | Pipeline for text-to-video generation using Stable Diffusion XL. | `src/diffusers/pipelines/animatediff/pipeline_animatediff_sdxl.py` |
| AnimateDiffSparseControlNetPipeline | Pipeline for controlled text-to-video generation using the method described in [SparseCtrl: Adding S... | `src/diffusers/pipelines/animatediff/pipeline_animatediff_sparsectrl.py` |
| AnimateDiffVideoToVideoControlNetPipeline | Pipeline for video-to-video generation with ControlNet guidance. | `src/diffusers/pipelines/animatediff/pipeline_animatediff_video2video_controlnet.py` |
| AnimateDiffVideoToVideoPipeline | Pipeline for video-to-video generation. | `src/diffusers/pipelines/animatediff/pipeline_animatediff_video2video.py` |
| ChronoEditPipeline | Pipeline for image-to-video generation using Wan. | `src/diffusers/pipelines/chronoedit/pipeline_chronoedit.py` |
| CogVideoXFunControlPipeline | Pipeline for controlled text-to-video generation using CogVideoX Fun. | `src/diffusers/pipelines/cogvideo/pipeline_cogvideox_fun_control.py` |
| CogVideoXImageToVideoPipeline | Pipeline for image-to-video generation using CogVideoX. | `src/diffusers/pipelines/cogvideo/pipeline_cogvideox_image2video.py` |
| CogVideoXPipeline | Pipeline for text-to-video generation using CogVideoX. | `src/diffusers/pipelines/cogvideo/pipeline_cogvideox.py` |
| CogVideoXVideoToVideoPipeline | Pipeline for video-to-video generation using CogVideoX. | `src/diffusers/pipelines/cogvideo/pipeline_cogvideox_video2video.py` |
| ConsisIDPipeline | Pipeline for image-to-video generation using ConsisID. | `src/diffusers/pipelines/consisid/pipeline_consisid.py` |
| Cosmos2VideoToWorldPipeline | Pipeline for video-to-world generation using [Cosmos Predict2](https://github.com/nvidia-cosmos/cosm... | `src/diffusers/pipelines/cosmos/pipeline_cosmos2_video2world.py` |
| CosmosVideoToWorldPipeline | Pipeline for image-to-world and video-to-world generation using [Cosmos | `src/diffusers/pipelines/cosmos/pipeline_cosmos_video2world.py` |
| EasyAnimateControlPipeline | Pipeline for text-to-video generation using EasyAnimate. | `src/diffusers/pipelines/easyanimate/pipeline_easyanimate_control.py` |
| EasyAnimateInpaintPipeline | Pipeline for text-to-video generation using EasyAnimate. | `src/diffusers/pipelines/easyanimate/pipeline_easyanimate_inpaint.py` |
| EasyAnimatePipeline | Pipeline for text-to-video generation using EasyAnimate. | `src/diffusers/pipelines/easyanimate/pipeline_easyanimate.py` |
| HunyuanSkyreelsImageToVideoPipeline | Pipeline for image-to-video generation using HunyuanVideo. | `src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_skyreels_image2video.py` |
| HunyuanVideoFramepackPipeline | Pipeline for text-to-video generation using HunyuanVideo. | `src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video_framepack.py` |
| HunyuanVideoImageToVideoPipeline | Pipeline for image-to-video generation using HunyuanVideo. | `src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video_image2video.py` |
| HunyuanVideoPipeline | Pipeline for text-to-video generation using HunyuanVideo. | `src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video.py` |
| Kandinsky5T2VPipeline | Pipeline for text-to-video generation using Kandinsky 5.0. | `src/diffusers/pipelines/kandinsky5/pipeline_kandinsky.py` |
| LTXConditionPipeline | Pipeline for text/image/video-to-video generation. | `src/diffusers/pipelines/ltx/pipeline_ltx_condition.py` |
| LTXImageToVideoPipeline | Pipeline for image-to-video generation. | `src/diffusers/pipelines/ltx/pipeline_ltx_image2video.py` |
| LTXPipeline | Pipeline for text-to-video generation. | `src/diffusers/pipelines/ltx/pipeline_ltx.py` |
| LattePipeline | Pipeline for text-to-video generation using Latte. | `src/diffusers/pipelines/latte/pipeline_latte.py` |
| LucyEditPipeline | Pipeline for video-to-video generation using Lucy Edit. | `src/diffusers/pipelines/lucy/pipeline_lucy_edit.py` |
| MochiPipeline | The mochi pipeline for text-to-video generation. | `src/diffusers/pipelines/mochi/pipeline_mochi.py` |
| SanaImageToVideoPipeline | Pipeline for image/text-to-video generation using [Sana](https://huggingface.co/papers/2509.24695). ... | `src/diffusers/pipelines/sana_video/pipeline_sana_video_i2v.py` |
| SanaVideoPipeline | Pipeline for text-to-video generation using [Sana](https://huggingface.co/papers/2509.24695). This m... | `src/diffusers/pipelines/sana_video/pipeline_sana_video.py` |
| SkyReelsV2DiffusionForcingImageToVideoPipeline | Pipeline for Image-to-Video (i2v) generation using SkyReels-V2 with diffusion forcing. | `src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_i2v.py` |
| SkyReelsV2DiffusionForcingPipeline | Pipeline for Text-to-Video (t2v) generation using SkyReels-V2 with diffusion forcing. | `src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing.py` |
| SkyReelsV2DiffusionForcingVideoToVideoPipeline | Pipeline for Video-to-Video (v2v) generation using SkyReels-V2 with diffusion forcing. | `src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_v2v.py` |
| SkyReelsV2ImageToVideoPipeline | Pipeline for Image-to-Video (i2v) generation using SkyReels-V2. | `src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_i2v.py` |
| SkyReelsV2Pipeline | Pipeline for Text-to-Video (t2v) generation using SkyReels-V2. | `src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2.py` |
| StableVideoDiffusionPipeline | Pipeline to generate video from an input image using Stable Video Diffusion. | `src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py` |
| WanImageToVideoPipeline | Pipeline for image-to-video generation using Wan. | `src/diffusers/pipelines/wan/pipeline_wan_i2v.py` |
| WanPipeline | Pipeline for text-to-video generation using Wan. | `src/diffusers/pipelines/wan/pipeline_wan.py` |
| WanVideoToVideoPipeline | Pipeline for video-to-video generation using Wan. | `src/diffusers/pipelines/wan/pipeline_wan_video2video.py` |