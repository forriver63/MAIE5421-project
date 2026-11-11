"""MBRS A 组（补零）；运行 run_B.py 使用随机填充。"""
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.nn.functional import interpolate

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.trustmark.run import SEED, conditions, source_for
from watermarking import attacks, io, metrics
from watermarking.data import loader

SIZE = 256
CAPACITY = 256


def load_model():
    # 对应作者 test_settings.json 的 MBRS_256_m256，不使用 diffusion block。
    path = HERE / "weights" / "EC_42.pth"
    if not path.is_file():
        raise FileNotFoundError(f"请把 MBRS_256_m256 的官方 EC_42.pth 放在 {path}")
    sys.path.insert(0, str(HERE / "upstream"))
    from network.Encoder_MP import Encoder_MP
    from network.Decoder import Decoder

    # 作者的 EC 权重同时保存 encoder 和 decoder。严格匹配各层参数。
    state = torch.load(path, map_location="cpu", weights_only=True)
    encoder = Encoder_MP(SIZE, SIZE, CAPACITY)
    decoder = Decoder(SIZE, SIZE, CAPACITY)
    for name, module in (("encoder", encoder), ("decoder", decoder)):
        prefix = name + "."
        module.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
    return encoder.eval(), decoder.eval()


def resize(tensor, shape):
    return interpolate(tensor, size=shape, mode="bilinear", align_corners=False, antialias=True)


def rgb_tensor(image):
    # 小组张量 → RGB [0,1]，排列为 MBRS 使用的 batch、通道、高、宽。
    return torch.from_numpy(attacks.tensor_rgb(image)).permute(2, 0, 1).unsqueeze(0)


@torch.inference_mode()
def run(group):
    torch.set_num_threads(4)
    encoder, decoder = load_model()
    results = defaultdict(list)
    full_results = defaultdict(list)
    psnrs, ssims = [], []
    with TemporaryDirectory() as temp:
        for index, record in enumerate(loader.load_images(temp)):
            original = loader.load_image(record)
            rgb = rgb_tensor(original)
            # 官方模型接收 [-1,1]，内部处理 256×256。
            small = resize(rgb, (SIZE, SIZE)) * 2 - 1
            for seed in range(3):
                bits = np.unpackbits(np.array(source_for(index, seed), dtype=np.uint8))
                padding = np.zeros(CAPACITY - 32, dtype=np.uint8)
                if group == "B":
                    rng = np.random.default_rng([SEED, index, seed, 1])
                    padding = rng.integers(0, 2, size=CAPACITY - 32).astype(np.uint8)
                payload = torch.from_numpy(np.concatenate((bits, padding))).float()[None]

                # 强度为 1。只放大水印残差，保留原始照片的高频细节。
                residual = (encoder(small, payload) - small) / 2
                marked_rgb = (rgb + resize(residual, rgb.shape[-2:])).clamp(0, 1)
                # 与 TrustMark 的 RGB 输出一样，量化为 8 位后进入主线。
                marked = attacks.tensor_from_rgb(marked_rgb[0].permute(1, 2, 0).numpy())
                p, s = metrics.quality(original, marked)
                psnrs.append(p)
                ssims.append(s)
                for condition, attacked in conditions(marked, index, seed):
                    # 攻击已经在主线尺度完成，再将收到的整张图缩放给解码器。
                    prediction = decoder(resize(rgb_tensor(attacked), (SIZE, SIZE)) * 2 - 1)
                    if prediction.shape != payload.shape:
                        raise ValueError(f"解码长度异常：{prediction.shape}")
                    # 官方逐位阈值为 > 0.5，无独立的 detected 标志。
                    correct = (prediction > 0.5) == payload.bool()
                    results[condition].append(bool(correct[:, :32].all()))
                    full_results[condition].append(bool(correct.all()))
            print(f"MBRS-{group}: {index + 1}/10 {record['name']}", flush=True)
    rows = []
    for condition, scores in results.items():
        rows.append(dict(model="MBRS_256_m256", group=group, condition=condition,
                         total=len(scores), exact=sum(scores), exact_rate=mean(scores),
                         full_exact=sum(full_results[condition]), full_exact_rate=mean(full_results[condition]),
                         mean_clean_psnr=mean(psnrs), mean_clean_ssim=mean(ssims)))
    output = HERE / f"MBRS-{group}.csv"
    io.write_csv(output, rows)
    print(output)


if __name__ == "__main__":
    run("A")
