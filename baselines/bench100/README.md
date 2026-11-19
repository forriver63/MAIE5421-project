# DIV2K 100 图传统水印实验

[中文结果和 28 条件对照](../records/div2k100/REPORT_ZH.md) · [并列 CSV](../records/div2k100/tables/comparison.csv) · [完整性记录](../records/div2k100/integrity.json)

本目录是 SVD 和 DWT-DCT 的 100 图入口。算法来自 `baselines/svd/rgb_adapter.py` 和 `baselines/dwt_dct/watermark.py`，参数直接读取各自的 `configs/team32.json`。SVD 使用 Δ=12、4×4 块和 RGB-Y，DWT-DCT 使用 α=20、PN 种子 2007、HL2、RGB-Y 和一次重复。每张图分别嵌入三条独立消息，每次均从原图开始，每条消息携带 32 个有效位。全程未调参。

## 数据与提取

实验以 `baseline/image_manifest.json` 冻结的 100 张 test 图为准。对官方 HR PNG 和处理后的 RGB PNG 分别校验 SHA-256。仓库通用数据准备同时读取训练包和验证包，本实验只准备验证包。20 张 calibration 图不参与。消息取自 `payload_for`，按最高有效位在前展开。编码输出经四舍六入五成双取整、截断到 [0,255] 后保存为 RGB PNG，从磁盘读回，同一份像素用于画质评分、clean 和其余 27 项攻击。

接收器只获得受攻击像素和算法要求的提取设置。SVD 还需要 32 个由原图选出的、有顺序的块位置，随每次嵌入归档。DWT-DCT 使用固定设置。裁剪按实际尺寸传入，两个原接收器的尺寸异常记为 `erasure/unsupported_geometry`，逐样本 BER 留空。其他执行异常保存上下文与 traceback 并中止。方法与共享攻击均在 CPU 上运行，按图并行。

每个方法另做 100 次未嵌入图的 clean 提取。SVD 对照取同图 seed 0 嵌入保存的选块 key，DWT-DCT 对照取固定配置。负样本没有参考消息，`accepted` 留空。正样本的有效输出覆盖率只表示提取器返回完整 32 位的比例，错误输出也计入覆盖率。

## 结果文件

| 文件（相对于 `baselines/records/div2k100/`） | 内容 |
|---|---|
| `{svd,dwt_dct}/cases.csv` | 图号、seed、条件、原始与恢复位串、结局、BER、擦除原因、受攻击宽高与 RGB 像素哈希 |
| `{svd,dwt_dct}/embeddings.csv` | 300 次嵌入的 PNG 质量与 PNG/key 哈希 |
| `{svd,dwt_dct}/negatives.csv` | 100 个独立的未嵌入对照、key 来源及解码输出 |
| `{svd,dwt_dct}/attack_plans.json.gz` | 8,400 个正样本和 100 个负样本的共享攻击计划 |
| `{svd,dwt_dct}/extraction_settings.json.gz` | 每次嵌入的提取设置，包含 SVD 块位置 |
| `{svd,dwt_dct}/provenance.json` | Git 版本、代码内容哈希、配置、库版本、100 对图像哈希、缓存标识 |
| `tables/` | 28 条件计数与比率、六类核心指标、扩展指标、两方法并列表及按图配对区间 |
| `smoke_verification.json`、`integrity.json` | 单图复核、全量键集合、数量、重复、缺失、执行错误与导出 CSV 哈希 |
| `machine.json`、`python-packages.txt` | 本机运行设备、并行数、起始提交及实际环境版本 |

直接查看 CSV 时将 `payload` 和 `recovered` 作为文本读取，以保留前导零。BER 与各类比率在 CSV 中使用 0–1 尺度。核心分数按图、按六类等权计算。归档的 WAM 100 图材料提供汇总值，本报告的配对区间只使用本次实测的两个传统方法。

## 在本机复跑

在 PowerShell 中执行以下命令。环境和数据已准备好时可直接运行最后一行。默认将复跑结果放在仓库外的 `FINAL/rerun/results`，嵌入 PNG 和断点放在 `FINAL/rerun/cache`。

```powershell
Set-Location 'E:\香港科技大学\CV\FINAL\MAIE5421-project'
$py = '..\envs\watermark\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
& $py baselines/bench100/prepare_test.py --archive ../data/DIV2K_valid_HR.zip --output ../data
& ./baselines/bench100/reproduce.ps1 -Manifest ../data/images.json
```

运行脚本先对第一张图的 168 条正样本和两次未嵌入提取做独立复核，再运行全量实验和完整性检查。相同命令可续跑，缓存匹配运行代码、依赖、配置与图像哈希。更改这些输入后请指定新的 `-Cache` 和 `-Output` 目录。只需重新汇总正式结果时，从仓库根目录执行：

```powershell
& ../envs/watermark/python.exe baselines/bench100/analyze.py --output baselines/records/div2k100 --cache ../cache/div2k100/full
```

在其他电脑从零准备时，可把仓库也放在一个 `FINAL` 文件夹下。在仓库根目录运行以下命令，再执行上面的数据准备和实验命令。Python 环境中的包版本见本次随结果保存的清单。

```powershell
$baseDir = (Resolve-Path ..).Path
$env:CONDA_PKGS_DIRS = "$baseDir/cache/conda"
$env:PIP_CACHE_DIR = "$baseDir/cache/pip"
$env:TEMP = "$baseDir/cache/tmp"
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP, ../data | Out-Null
conda create -y -p "$baseDir/envs/watermark" python=3.10 pip
& ../envs/watermark/python.exe -m pip install -r baselines/records/div2k100/python-packages.txt
curl.exe --fail --location --continue-at - --output ../data/DIV2K_valid_HR.zip https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
```
