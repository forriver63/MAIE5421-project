"""在小组统一的 14 种测试条件下评估 TrustMark-C 或 TrustMark-Q。"""
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
import torch
from trustmark import TrustMark

# 从仓库根目录导入主线实验的旋转角度、随机种子和消息生成函数。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.robustness.run import ANGLES, SEED, source_for
from watermarking import attacks, io, metrics
from watermarking.data import loader
from watermarking.wam.transforms import default_transform


def as_pil(tensor):
    """把主线使用的图像张量转回 TrustMark 接收的 RGB 图片。"""
    return Image.fromarray(np.rint(attacks.tensor_rgb(tensor) * 255).astype(np.uint8))


def conditions(marked, index, seed):
    """每次返回一个条件名和一张待解码的图；每种条件都从同一张水印图开始。"""
    yield "clean", marked
    for quality in (75, 50):
        yield f"jpeg{quality}", attacks.jpeg_attack(marked, quality)

    # 沿用主线的随机种子，让同一图片和编号受到相同的噪声。
    for sigma in (2, 5):
        noise_seed = SEED + index * 1009 + seed * 97 + sigma
        yield f"noise{sigma}", attacks.noise_attack(marked, sigma, noise_seed)
    for angle in ANGLES:
        yield f"rotate{angle:+d}", attacks.rotate_attack(marked, angle)

    # 主线组合攻击：取第一个保留约 75% 面积的裁剪框，裁完再做 JPEG Q75。
    crop = next(c for c in attacks.fixed_crops(index, seed) if c[0] == 0.75)
    _, _, _, x, y, width, height = crop
    cropped = marked[:, :, y:y + height, x:x + width]
    yield "crop75_jpeg75", attacks.jpeg_attack(cropped, 75)


def run(variant):
    # 只加载一次所选模型的官方预训练权重。C 可装 40 位，Q 可装 61 位。
    schema = {"C": TrustMark.Encoding.BCH_SUPER, "Q": TrustMark.Encoding.BCH_5}[variant]
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TrustMark(model_type=variant, encoding_type=schema, use_ECC=True,
                      loadRemover=False, device=device)

    successes = defaultdict(list)
    psnr_values, ssim_values = [], []
    # 主线加载器会读取这些图片，转成 RGB，再统一裁剪和缩放到 512 × 512，保存成 PNG和清单。放进临时目录，结束后自动清理。
    with TemporaryDirectory() as temp:
        for index, record in enumerate(loader.load_images(temp)):
            original = loader.load_image(record)
            cover = as_pil(original)
            for seed in range(3):
                # 每张图取主线的 3 个编号。每个编号由 4 个字节组成，共 32 位。
                bits = "".join(f"{byte:08b}" for byte in source_for(index, seed))
                payload = bits.ljust(model.schemaCapacity(), "0")
                watermarked = model.encode(cover, payload, MODE="binary", WM_STRENGTH=1.0)

                # 写入后先算画质。主线函数比较原图与水印图，每次嵌入只算一次。
                marked = default_transform(watermarked.convert("RGB")).unsqueeze(0)
                psnr, ssim = metrics.quality(original, marked)
                psnr_values.append(psnr)
                ssim_values.append(ssim)

                # 用 TrustMark 解码；只核对原始 32 位，忽略末尾用于补齐的零。
                # 未检出或错一位都记为失败，仍然计入该条件的样本总数。
                for name, attacked in conditions(marked, index, seed):
                    decoded, detected, _ = model.decode(as_pil(attacked), MODE="binary")
                    successes[name].append(bool(detected and decoded[:32] == bits))
            print(f"TrustMark-{variant}: {index + 1}/10 {record['name']}", flush=True)

    # 每个条件单独统计 30 次解码。画质是同一批 30 张水印图的平均值。
    rows = []
    for name, results in successes.items():
        rows.append(dict(model=variant, condition=name, total=len(results),
                         exact=sum(results), exact_rate=mean(results),
                         mean_clean_psnr=mean(psnr_values), mean_clean_ssim=mean(ssim_values)))
    output = Path(__file__).with_name(f"TrustMark-{variant}.csv")
    io.write_csv(output, rows)
    print(f"结果已保存到 {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("C", "Q"), help="选择一个模型")
    run(parser.parse_args().model)
