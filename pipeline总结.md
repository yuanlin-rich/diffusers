# Diffusers Pipelines 总结

本目录包含以下 pipeline 模块，每个模块提供不同的生成模型。

| 模块 | Pipeline 数量 | 描述 | Pipeline 类列表 |
|------|---------------|------|----------------|
| allegro | 1 | 暂无描述 | AllegroPipeline |
| amused | 3 | 暂无描述 | AmusedInpaintPipeline, AmusedImg2ImgPipeline, AmusedPipeline |
| animatediff | 6 | AnimateDiff 视频生成 | AnimateDiffPipeline, AnimateDiffControlNetPipeline, AnimateDiffVideoToVideoControlNetPipeline, AnimateDiffSDXLPipeline, AnimateDiffSparseControlNetPipeline ... 等 6 个 |
| audioldm | 1 | 音频生成 | AudioLDMPipeline |
| audioldm2 | 1 | 音频生成第二代 | AudioLDM2Pipeline |
| aura_flow | 1 | Aura Flow 风格迁移 | AuraFlowPipeline |
| blip_diffusion | 1 | BLIP 扩散 | BlipDiffusionPipeline |
| bria | 1 | BRIA 图像生成 | BriaPipeline |
| bria_fibo | 1 | BRIA FIBO 模型 | BriaFiboPipeline |
| chroma | 2 | Chroma 颜色控制生成 | ChromaPipeline, ChromaImg2ImgPipeline |
| chronoedit | 1 | 时序编辑 | ChronoEditPipeline |
| cogvideo | 4 | CogVideo 视频生成 | CogVideoXPipeline, CogVideoXFunControlPipeline, CogVideoXVideoToVideoPipeline, CogVideoXImageToVideoPipeline |
| cogview3 | 1 | CogView3 文本到图像 | CogView3PlusPipeline |
| cogview4 | 2 | CogView4 文本到图像 | CogView4Pipeline, CogView4ControlPipeline |
| consisid | 1 | 一致性 ID | ConsisIDPipeline |
| consistency_models | 1 | 一致性模型 | ConsistencyModelPipeline |
| controlnet | 11 | ControlNet 条件控制生成 | BlipDiffusionControlNetPipeline, StableDiffusionXLControlNetImg2ImgPipeline, StableDiffusionXLControlNetPipeline, StableDiffusionControlNetPipeline, StableDiffusionControlNetInpaintPipeline ... 等 11 个 |
| controlnet_hunyuandit | 1 | 暂无描述 | HunyuanDiTControlNetPipeline |
| controlnet_sd3 | 2 | 暂无描述 | StableDiffusion3ControlNetInpaintingPipeline, StableDiffusion3ControlNetPipeline |
| controlnet_xs | 2 | 暂无描述 | StableDiffusionControlNetXSPipeline, StableDiffusionXLControlNetXSPipeline |
| cosmos | 4 | Cosmos 多模态 | CosmosVideoToWorldPipeline, Cosmos2TextToImagePipeline, Cosmos2VideoToWorldPipeline, CosmosTextToWorldPipeline |
| dance_diffusion | 1 | 暂无描述 | DanceDiffusionPipeline |
| ddim | 1 | 去噪扩散隐式模型 | DDIMPipeline |
| ddpm | 1 | 去噪扩散概率模型 | DDPMPipeline |
| deepfloyd_if | 6 | DeepFloyd IF 级联文本到图像 | IFInpaintingPipeline, IFPipeline, IFSuperResolutionPipeline, IFImg2ImgSuperResolutionPipeline, IFImg2ImgPipeline ... 等 6 个 |
| deprecated | 18 | 暂无描述 | LDMPipeline, SpectrogramDiffusionPipeline, AudioDiffusionPipeline, CycleDiffusionPipeline, VersatileDiffusionPipeline ... 等 18 个 |
| dit | 1 | DiT (Diffusion Transformer) | DiTPipeline |
| easyanimate | 3 | EasyAnimate 视频生成 | EasyAnimateInpaintPipeline, EasyAnimatePipeline, EasyAnimateControlPipeline |
| flux | 13 | FLUX 文本到图像生成模型 | FluxPipeline, FluxControlInpaintPipeline, FluxControlNetPipeline, FluxControlImg2ImgPipeline, FluxControlNetInpaintPipeline ... 等 13 个 |
| flux2 | 1 | FLUX 2.0 模型 | Flux2Pipeline |
| hidream_image | 1 | HiDream 图像生成 | HiDreamImagePipeline |
| hunyuan_image | 2 | 混元图像生成 | HunyuanImageRefinerPipeline, HunyuanImagePipeline |
| hunyuan_video | 4 | 混元视频生成 | HunyuanVideoImageToVideoPipeline, HunyuanVideoFramepackPipeline, HunyuanVideoPipeline, HunyuanSkyreelsImageToVideoPipeline |
| hunyuandit | 1 | 混元 DiT 模型 | HunyuanDiTPipeline |
| i2vgen_xl | 1 | 图像到视频生成 | I2VGenXLPipeline |
| kandinsky | 7 | Kandinsky 多模态生成 | KandinskyImg2ImgPipeline, KandinskyPriorPipeline, KandinskyInpaintCombinedPipeline, KandinskyCombinedPipeline, KandinskyInpaintPipeline ... 等 7 个 |
| kandinsky2_2 | 10 | Kandinsky 2.2 | KandinskyV22ControlnetPipeline, KandinskyV22InpaintPipeline, KandinskyV22InpaintCombinedPipeline, KandinskyV22Img2ImgPipeline, KandinskyV22Img2ImgCombinedPipeline ... 等 10 个 |
| kandinsky3 | 2 | Kandinsky 3 | Kandinsky3Img2ImgPipeline, Kandinsky3Pipeline |
| kandinsky5 | 1 | 暂无描述 | Kandinsky5T2VPipeline |
| kolors | 2 | Kolors 颜色生成 | KolorsImg2ImgPipeline, KolorsPipeline |
| latent_consistency_models | 2 | 潜在一致性模型 | LatentConsistencyModelImg2ImgPipeline, LatentConsistencyModelPipeline |
| latent_diffusion | 2 | 潜在扩散模型 | LDMTextToImagePipeline, LDMSuperResolutionPipeline |
| latte | 1 | 暂无描述 | LattePipeline |
| ltx | 4 | LTX 视频生成 | LTXPipeline, LTXImageToVideoPipeline, LTXLatentUpsamplePipeline, LTXConditionPipeline |
| lucy | 1 | Lucy 编辑 | LucyEditPipeline |
| lumina | 2 | Lumina 文本到图像 | LuminaPipeline, LuminaText2ImgPipeline |
| lumina2 | 2 | Lumina 2 | Lumina2Pipeline, Lumina2Text2ImgPipeline |
| marigold | 3 | 深度估计 | MarigoldIntrinsicsPipeline, MarigoldDepthPipeline, MarigoldNormalsPipeline |
| mochi | 1 | Mochi 模型 | MochiPipeline |
| musicldm | 1 | 音乐生成 | MusicLDMPipeline |
| omnigen | 1 | OmniGen 通用生成 | OmniGenPipeline |
| pag | 17 | PAG (Prompt‑Adjusted Guidance) 引导 | SanaPAGPipeline, AnimateDiffPAGPipeline, StableDiffusionXLPAGInpaintPipeline, StableDiffusion3PAGPipeline, StableDiffusionXLControlNetPAGImg2ImgPipeline ... 等 17 个 |
| paint_by_example | 1 | 示例绘画 | PaintByExamplePipeline |
| pia | 1 | PIA 模型 | PIAPipeline |
| pipeline_flax_utils | 1 | 暂无描述 | FlaxDiffusionPipeline |
| pipeline_utils | 1 | 暂无描述 | DiffusionPipeline |
| pixart_alpha | 2 | PixArt-α 高质量图像生成 | PixArtAlphaPipeline, PixArtSigmaPipeline |
| prx | 1 | PRX 模型 | PRXPipeline |
| qwenimage | 8 | QwenImage 图像生成 | QwenImageInpaintPipeline, QwenImageEditPipeline, QwenImageControlNetPipeline, QwenImageImg2ImgPipeline, QwenImagePipeline ... 等 8 个 |
| sana | 4 | SANA 图像生成 | SanaControlNetPipeline, SanaSprintImg2ImgPipeline, SanaPipeline, SanaSprintPipeline |
| sana_video | 2 | SANA 视频生成 | SanaVideoPipeline, SanaImageToVideoPipeline |
| semantic_stable_diffusion | 1 | 语义稳定扩散 | SemanticStableDiffusionPipeline |
| shap_e | 2 | 3D 形状生成 | ShapEImg2ImgPipeline, ShapEPipeline |
| skyreels_v2 | 5 | SkyReels V2 视频生成 | SkyReelsV2DiffusionForcingPipeline, SkyReelsV2DiffusionForcingImageToVideoPipeline, SkyReelsV2DiffusionForcingVideoToVideoPipeline, SkyReelsV2ImageToVideoPipeline, SkyReelsV2Pipeline |
| stable_audio | 1 | 稳定音频生成 | StableAudioPipeline |
| stable_cascade | 3 | 稳定级联模型 | StableCascadePriorPipeline, StableCascadeCombinedPipeline, StableCascadeDecoderPipeline |
| stable_diffusion | 19 | Stable Diffusion 文本到图像生成 | StableDiffusionLatentUpscalePipeline, FlaxStableDiffusionInpaintPipeline, StableDiffusionLDM3DPipeline, StableDiffusionImageVariationPipeline, OnnxStableDiffusionUpscalePipeline ... 等 19 个 |
| stable_diffusion_3 | 3 | Stable Diffusion 3 最新版本 | StableDiffusion3InpaintPipeline, StableDiffusion3Img2ImgPipeline, StableDiffusion3Pipeline |
| stable_diffusion_attend_and_excite | 1 | 暂无描述 | StableDiffusionAttendAndExcitePipeline |
| stable_diffusion_diffedit | 1 | 暂无描述 | StableDiffusionDiffEditPipeline |
| stable_diffusion_gligen | 2 | 暂无描述 | StableDiffusionGLIGENPipeline, StableDiffusionGLIGENTextImagePipeline |
| stable_diffusion_k_diffusion | 2 | 暂无描述 | StableDiffusionXLKDiffusionPipeline, StableDiffusionKDiffusionPipeline |
| stable_diffusion_ldm3d | 1 | 暂无描述 | StableDiffusionLDM3DPipeline |
| stable_diffusion_panorama | 1 | 暂无描述 | StableDiffusionPanoramaPipeline |
| stable_diffusion_sag | 1 | 暂无描述 | StableDiffusionSAGPipeline |
| stable_diffusion_xl | 5 | Stable Diffusion XL 更高分辨率文本到图像 | FlaxStableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInstructPix2PixPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLPipeline |
| stable_video_diffusion | 1 | 稳定视频扩散模型 | StableVideoDiffusionPipeline |
| t2i_adapter | 2 | T2I-Adapter 轻量条件控制 | StableDiffusionAdapterPipeline, StableDiffusionXLAdapterPipeline |
| text_to_video_synthesis | 4 | 文本到视频合成 | TextToVideoZeroSDXLPipeline, TextToVideoZeroPipeline, VideoToVideoSDPipeline, TextToVideoSDPipeline |
| unclip | 2 | UnCLIP 图像生成 | UnCLIPImageVariationPipeline, UnCLIPPipeline |
| unidiffuser | 1 | 统一扩散模型 | UniDiffuserPipeline |
| visualcloze | 2 | 视觉填空 | VisualClozeGenerationPipeline, VisualClozePipeline |
| wan | 5 | WAN 视频生成 | WanVideoToVideoPipeline, WanAnimatePipeline, WanImageToVideoPipeline, WanPipeline, WanVACEPipeline |
| wuerstchen | 3 | Wuerstchen 高效生成 | WuerstchenDecoderPipeline, WuerstchenPriorPipeline, WuerstchenCombinedPipeline |
| z_image | 1 | Z 图像生成 | ZImagePipeline |