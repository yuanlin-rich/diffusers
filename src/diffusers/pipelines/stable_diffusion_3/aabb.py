import torch
from diffusers import DiffusionPipeline
import warnings
warnings.filterwarnings('ignore')

# 初始化管道 - 核心配置针对M3 Mac优化
pipeline = DiffusionPipeline(
    model="argmaxinc/stable-diffusion",  # DiffusionKit内置标识
    w16=True,      # 使用16位权重，节省内存
    a16=True,      # 使用16位激活，节省内存
    model_size="2b",  # 对应SD3-Medium
    low_memory_mode=True,  # **关键：启用低内存模式**，防止OOM
    local_ckpt="./sd3_medium_coreml",  # 指向你转换好的Core ML模型目录
    shift=3.0,     # 采样偏移，保持默认
    use_t5=False   # 如果模型需要T5文本编码器则设为True，但会增加负担
)

# 生成图像 (分辨率设为768x768以进一步降低内存压力)
prompt = "A beautiful sunset over a mountain lake, digital art"
HEIGHT = 768
WIDTH = 768

print("开始生成图像，这可能需要一些时间...")
try:
    image, _ = pipeline.generate_image(
        prompt,
        cfg_weight=5.0,          # 提示词引导强度，默认即可
        num_steps=30,            # 推理步数，可调整（步数越多耗时越长，质量可能越高）
        latent_size=(HEIGHT // 8, WIDTH // 8),  # 必须与转换时的--latent-size对应
        seed=42                  # 随机种子，固定种子可复现结果
    )
    # 保存图像
    output_path = "sd3_generated_image.png"
    image.save(output_path)
    print(f"✅ 图像生成成功！已保存至: {output_path}")
except Exception as e:
    print(f"❌ 生成过程中出现错误: {e}")
    print("提示：这可能是由于内存不足。请尝试：1. 关闭所有其他应用；2. 将low_memory_mode设为False后重试（风险较高）。")