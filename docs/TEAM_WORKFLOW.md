# 团队协作分工与 Workflow

本文档说明重构后的代码框架下，每个角色的职责、要改哪些文件、以及日常运行/验证的命令。

## 0. 通用准备（所有人第一次都要做）

```sh
# 1) 环境 + 安装包（详见 README「Setup」）
python3.10 -m venv .venv && . .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .

# 2) 克隆 WAM 源码 + 下载 checkpoint（README「Setup」）
#    WAM 目录：validation_wam_20261007/watermark-anything
#    checkpoint：.../checkpoints/wam_mit.pth（SHA-256 见 README）

# 3) 确认环境可用
.venv/bin/python -m pytest tests/
```

## 目录速查

| 位置 | 作用 |
|---|---|
| `src/watermarking/codec/` | 纠错码（BCH/CRC/packet/outer）——**纯数学，无 torch/图像** |
| `src/watermarking/wam/` | WAM 模型封装（加载/嵌入/检测） |
| `src/watermarking/data/loader.py` | 图像加载 + manifest |
| `src/watermarking/attacks.py` | 攻击（JPEG/噪声/旋转/裁剪）+ crop 几何 |
| `src/watermarking/eval.py` | 解码 + 打分 + 全局还原 |
| `src/watermarking/pipeline.py` | 高层 embed/detect API（CLI/Web 共用） |
| `src/watermarking/{cli,web}.py` | 命令行 / FastAPI 服务 |
| `experiments/*/run.py` | 各实验的编排入口 |
| `tests/` | pytest 测试 |
| `validation_wam_*/` | **证据存档**（只读，不修改） |

---

## 角色 1：端到端测试（E2E Testing）

**负责**：保证整条链路（embed → 攻击 → detect → decode）正确，防止别人改动破坏结果。

**改哪些文件**：
- 新增测试：`tests/` 下加 `test_*.py`（纯 codec）或 `tests/e2e/`（需模型）
- 参考高层 API：`src/watermarking/pipeline.py` 的 `embed_image` / `detect_image`

**怎么加一个测试**（`tests/test_pipeline_roundtrip.py`）：

```python
from skimage import data
from PIL import Image
from watermarking.pipeline import embed_image, detect_image

def test_embed_detect_roundtrip():
    img = Image.fromarray(data.astronaut()).convert("RGB").resize((512, 512))
    payload = 0x12345678
    watermarked = embed_image(img, payload)
    result = detect_image(watermarked)
    # clean 条件恢复率不是 100%，这里只断言链路不崩、返回结构正确
    assert set(result) >= {"status", "payload", "valid_ids"}
```

**运行/验证**：
```sh
.venv/bin/python -m pytest tests/ -v
```

**对照标准**：结果文件在 `validation_wam_*/results/`（只读标准答案）。改完代码后用 `experiments/*/run.py` 重跑对照。

---

## 角色 2：修改 / 添加 baseline（外码 / 纠错方案）

**负责**：新增或替换外码方案（例如加一个新的纠错码，或替换 spatial repetition）。

**改哪些文件**：
- 核心：`src/watermarking/codec/outer.py`（加 `encode_xxx` / `decode_xxx`）
- 注册：`experiments/ablation/run.py` 的 `METHODS` 字典

**怎么加一个新外码**（在 `outer.py`）：

```python
def encode_mycode(source):
    # source: (A,B,C,D) 4 字节 -> 8 个符号
    ...

def decode_mycode(unique: dict[int, int]) -> dict:
    # unique: {region_id: symbol} -> {"status": ..., "payload": ...}
    ...
```

然后在 `experiments/ablation/run.py` 注册：

```python
METHODS = {
    "repetition": (outer.encode_repetition, outer.decode_repetition),
    "mds": (outer.encode_mds, outer.decode_mds),
    "mycode": (outer.encode_mycode, outer.decode_mycode),   # 新增
}
```

**运行/验证**：
```sh
.venv/bin/python -m pytest tests/test_outer.py        # 先保证现有外码没坏
.venv/bin/python experiments/ablation/run.py         # 新外码 vs 旧外码对比
```

> 注意：`decode_global(detection, method=...)` 在 `src/watermarking/eval.py`，目前只分 `repetition` / `mds` 两分支，加第三种需要在 `eval.decode_global` 里补分支。

---

## 角色 3：评估画图（Evaluation & Plotting）

**负责**：从评测产出的 csv/json 生成汇总指标、图表和报告。

**改哪些文件**：
- 聚合+画图：`experiments/robustness/finalize.py`
- 新指标：`src/watermarking/metrics.py`
- 打分逻辑：`src/watermarking/eval.py` 的 `score`

**workflow**：
```sh
.venv/bin/python experiments/robustness/run.py       # 1) 跑评测 -> results/cases.csv 等
.venv/bin/python experiments/robustness/finalize.py  # 2) 聚合 -> summary.json + figures/*.png
```

**产物位置**：
- `experiments/robustness/results/summary.json`（机器可读汇总）
- `experiments/robustness/figures/*.png`（图表）

> 想加新图：在 `finalize.py` 里加 `matplotlib` 绘图代码即可，数据源就是 `results/cases.csv` / `packets.csv`。

---

## 角色 4：收集更多 data（Data Collection）

**负责**：扩充测试图像集（现在只有 scikit-image 的 10 张公开图）。

**改哪些文件**：
- `src/watermarking/data/loader.py` 的 `IMAGE_FILES` 字典 + `load_images`

**怎么加图**：

```python
IMAGE_FILES = {
    "astronaut": "astronaut.png",
    ...
    "my_new_image": "my_new_image.jpg",   # 新增
}
```

**运行/验证**：
```sh
.venv/bin/python experiments/robustness/run.py   # 会重新生成 image_manifest.json + 处理图
```

> 注意：新图会改变 manifest，从而影响评测结果（更多 case）。旧的 `validation_wam_*/results/` 是历史证据，不要覆盖；新结果写在 `experiments/*/results/`。

---

## 角色 5：前端交互（Frontend / CLI / Web）

**负责**：提供命令行和 Web 服务，让用户能上传图+payload 得到水印图、或上传图提取 payload。

**改哪些文件**：
- 高层 API：`src/watermarking/pipeline.py`（`embed_image` / `detect_image`）
- 命令行：`src/watermarking/cli.py`
- Web 服务：`src/watermarking/web.py`（FastAPI）

**workflow**：

```sh
# CLI
.venv/bin/wm embed input.png out.png --payload 0x12345678
.venv/bin/wm detect out.png --json

# Web
.venv/bin/uvicorn watermarking.web:app --host 0.0.0.0 --port 8000
# 然后：POST /embed  (form: image 文件 + payload 整数) -> 返回水印图 PNG
#       POST /detect (form: image 文件) -> 返回 {"status", "payload", ...}
```

**怎么加一个新端点**（在 `web.py`）：

```python
@app.post("/detect_batch")
async def detect_batch(images: list[UploadFile] = File(...)):
    return [detect_image(Image.open(io.BytesIO(await f.read())), wam=_model) for f in images]
```

---

## 角色 6：消融实验（Ablation Study）

**负责**：系统对比不同配置（外码、区域数、攻击强度）的恢复率。

**改哪些文件**：
- `experiments/ablation/run.py`（`METHODS` 注册表 + `run_ablation` 的条件列表）

**workflow**：
```sh
.venv/bin/python experiments/ablation/run.py
```

**产物**：
- `experiments/ablation/results/ablation_cases.csv`（逐条）
- `experiments/ablation/results/ablation_summary.json`（按 method × condition 汇总的恢复率）

**怎么加一个消融维度**：在 `run_ablation` 的 `conditions` 里加条件（如 `"noise5"`），或加新的 `METHODS` 条目。

---

## 通用规范

1. **不改证据**：`validation_wam_*/` 是只读证据存档，任何改动只在 `src/`、`experiments/`、`tests/` 进行。
2. **改完必测**：至少跑 `.venv/bin/python -m pytest tests/`；涉及 WAM 的改动跑对应 `experiments/*/run.py` 对照旧结果。
3. **结果可复现**：种子、参数写在代码或 `configs/`（如需参数化）里，不要散落。
4. **提交前自查**：`git status` 确认没有误提交 `.venv/`、`*.pth`、`*.egg-info/`（已在 `.gitignore`）。
