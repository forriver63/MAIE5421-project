"""在小组的 30 个图片与编号组合上评估 SSL Watermark 的多比特版本。"""
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models import resnet50

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.trustmark.run import as_pil, conditions, source_for
from watermarking import io, metrics
from watermarking.data import loader
from watermarking.wam.transforms import default_transform

# upstream 是作者仓库 facebookresearch/ssl_watermarking 的本地副本。
sys.path.insert(0, str(HERE / "upstream"))
import data_augmentation
import decode
import encode
import utils
import utils_img


def load_model():
    # 直接装入作者的 DINO 权重，省去官方初始化中额外下载 ImageNet 权重的一步。
    base = resnet50(weights=None)
    base.fc = torch.nn.Identity()
    # 作者的训练快照还保存了 NumPy 标量，仅允许读取这些数值类型。
    scalar_types = [np.core.multiarray.scalar, np.dtype, type(np.dtype("float64"))]
    with torch.serialization.safe_globals(scalar_types):
        weights = torch.load(HERE / "dino_r50_plus.pth", map_location="cpu", weights_only=True)
    for key in ("state_dict", "model_state_dict", "teacher"):
        if key in weights:
            weights = weights[key]
            break
    weights = {k.removeprefix("module.").removeprefix("backbone."): v for k, v in weights.items()}
    # 仅取特征网络用到的参数；如果缺了卷积层等关键参数，立即报告。
    base.load_state_dict({k: weights[k] for k in base.state_dict()})
    head = utils.load_normalization_layer(str(HERE / "out2048_yfcc_orig.pth"))
    return utils.NormLayerWrapper(base, head).to(utils.device).eval().requires_grad_(False)


def run():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    model = load_model()
    # 每一位对应一个固定方向，所有图片共用同一组方向。32 位全部用于原编号。
    carrier = utils.generate_carriers(32, model.head.out_features).to(utils.device)
    aug = data_augmentation.All()
    params = SimpleNamespace(batch_size=1, epochs=10, target_psnr=41, optimizer="Adam,lr=0.01",
                             scheduler=None, lambda_w=5e4, lambda_i=1, verbose=0)
    successes = defaultdict(list)
    psnrs, ssims = [], []
    with TemporaryDirectory() as temp:
        for index, record in enumerate(loader.load_images(temp)):
            original = loader.load_image(record)
            frame = utils_img.default_transform(as_pil(original))
            one_image = DataLoader([(frame, 0)], batch_size=1)
            for seed in range(3):
                # 主线的四个字节按高位在前展开，与 TrustMark 输入的编号逐位一致。
                bits = np.unpackbits(np.array(source_for(index, seed), dtype=np.uint8))
                message = torch.from_numpy(bits).bool()
                # SSL 每次只优化这张图片的像素，网络权重保持冻结。
                encoded, = encode.watermark_multibit(one_image, message[None], carrier, model, aug, params)
                rgb = utils_img.unnormalize_img(encoded).squeeze(0).permute(1, 2, 0)
                pil = Image.fromarray(rgb.mul(255).round().clamp(0, 255).byte().cpu().numpy())
                marked = default_transform(pil).unsqueeze(0)
                psnr, ssim = metrics.quality(original, marked)
                psnrs.append(psnr)
                ssims.append(ssim)
                # 直接复用 TrustMark 的攻击顺序和参数；SSL 自己读取攻击后的图片。
                with torch.no_grad():
                    for name, attacked in conditions(marked, index, seed):
                        result, = decode.decode_multibit([as_pil(attacked)], carrier, model)
                        successes[name].append(torch.equal(result["msg"], message))
            print(f"SSL Watermark: {index + 1}/10 {record['name']}", flush=True)

    if len(successes) != 14 or any(len(values) != 30 for values in successes.values()):
        raise RuntimeError("测试条件或样本数量与主线不一致")
    rows = [dict(model="SSL Watermark", condition=name, total=len(values), exact=sum(values),
                 exact_rate=mean(values), mean_clean_psnr=mean(psnrs), mean_clean_ssim=mean(ssims))
            for name, values in successes.items()]
    io.write_csv(HERE / "SSL-Watermark.csv", rows)


if __name__ == "__main__":
    run()
