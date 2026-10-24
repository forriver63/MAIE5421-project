# WAM 方法验证 sprint：Phase 0 停止报告

> Archive navigation note (2026-10-22): this is a historical protocol/report. Script names and machine-local artifacts below describe the original run and may no longer exist in this checkout. See [current commands](../README.md), [migration coverage](../docs/RESEARCH_HISTORY.md), and [中文实验流程](../docs/WORKFLOW_ZH.md). Original measurements and protocol text below are retained.

日期：2026-10-07

**决策：MODIFY。需要先修复或更换执行环境；方法本身仍未验证。**

本轮没有 WAM 推理结果，不能将环境阻塞解释为模型定位失败，也不能据此宣布方法可行或不可行。这里的 MODIFY 仅针对执行条件，尚无证据要求改动水印算法。遵守任务的明确停止条件：官方 WAM / checkpoint 不能运行时停止，不用自实现替代。

## A. 成功执行的工作

- 检查旧代码、数据、结果及环境，在 `validation_wam_20261007/` 创建独立环境和日志目录。
- 为旧实验的 35 个文件记录 SHA-256，并在结束时验证全部未变。
- 经 GitHub 连接器核实官方源码 commit，保存该 commit 下的 requirements 和模型配置。
- 成功安装 PyTorch 2.5.1、Torchvision 0.20.1；导入成功，CPU tensor 运算 `[1,1]+1=[2,2]` 通过。
- 尝试安装官方依赖；保存准确失败信息、已安装包锁定清单及下载诊断。

**未执行：**模型加载、自然图像嵌入、局部检测、裁剪、全局 payload 恢复、JPEG/噪声、旋转、confidence filtering。未实现 MDS/WAM 完整系统、训练、ECC 或角度搜索。

## B. 实际环境与版本

| 项目 | 实际状态 |
|---|---|
| Python | CPython 3.12.14 |
| 平台 | macOS 26.6.2 / arm64 |
| PyTorch / Torchvision | 已安装并验证：2.5.1 / 0.20.1 |
| 设备 | 只验证 CPU；当前进程报告 MPS unavailable，不推断物理硬件不存在 GPU |
| NumPy / Pillow | 实际安装：2.5.3 / 12.3.0 |
| WAM 源码固定版本 | `2c08af04d037d5667c02f6ddebbda9ff04581c3e`，已核实，尚未完整下载或安装 |
| 官方模型 | `wam_mit.pth`，尚未下载、计算校验和或加载 |
| 完整依赖 | 未安装完成，不能声称已经有可复现的 WAM 运行环境 |

`environment.installed.lock.txt` 记录实际已安装的全部包。`requirements.target.txt` 只是目标直接依赖，不是成功解析的完整 lockfile；其中 NumPy/Pillow 目标版本与当前安装版本不同，因为后续安装未完成。官方 README 的测试栈是 Python 3.10.14、PyTorch 2.5.1、Torchvision 0.20.1、CUDA 12.4；本地 Python/macOS 组合的 WAM 兼容性仍未验证。

模型计划从 [官方 MIT checkpoint 地址](https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth) 获取。由于未取得文件，checkpoint SHA-256 必须保持空值，不能声称已固定到不可变权重版本。恢复下载后应先计算 hash，再记录成功加载状态。

## 精确阻塞原因

1. 普通 curl 请求报 DNS 解析失败；提权后对 PyPI、GitHub、官方权重域名及 Hugging Face 的探测均在约 7 秒时返回 `curl: (28) Resolving timed out`。但 pip 实际下载成功，说明不能简单概括为整个环境完全断网。
2. 完整依赖安装在当前包解析过程中失败：`Could not find a version that satisfies the requirement lazy-loader>=0.4 (from scikit-image)`。这只说明此次解析未找到可用版本，不表示该包在公共 PyPI 不存在，也尚未证明版本本身不兼容。
3. 尝试用 Python 获取官方源码与权重时，自动审批服务返回 HTTP 403，操作未执行。工具明确说明这是审批流程故障，不是安全性否决。没有绕过审批或再次执行同一被阻止操作。

日志目录包含各网络探测、PyTorch 安装、完整依赖安装、成功 CPU 检查与审批故障记录。`environment_probe.json` 是首次诊断快照，包含一次 15 秒导入超时；随后重新导入成功，最新状态以 `status.json` 和 `logs/torch_import_verified.json` 为准。

## C–I. 按要求逐项报告

| 项目 | 本轮结果 |
|---|---|
| C. 干净自然图像，多区域 WAM | 未运行；2 区域初始测试尚未开始，Gate 1 未评估 |
| D. 未知偏移裁剪 | 未运行；Gate 2 未评估 |
| E. MDS / repetition / RLNC | 未进入 Phase 3；无 WAM 对照结果 |
| F. JPEG / 噪声 | 未运行；无 inner ECC 必要性证据 |
| G. 旋转 / 角度搜索 | 未运行；未实现适配器 |
| H. 代表性可视化 | 无。未生成伪造定位图，未用旧合成图替代自然图像结果 |
| I. 失败案例 | 环境与权重获取失败；没有可供归因的模型失败样本 |

所有模型指标应记作 **N/A / NOT RUN**，不能记作 0%：没有 PSNR、SSIM、packet accuracy、定位率、64-bit recovery 或每图推理耗时。

## J. 八个最终问题

| 问题 | 基于已执行证据的回答 |
|---|---|
| 1. WAM 能否恢复干净自然图像中的不同局部消息？ | 未知，尚未执行官方模型。论文与示例不能代替本项目测试。 |
| 2. 未知偏移裁剪后能否盲定位和解码？ | 未知，没有裁剪实验。 |
| 3. 擦除码是否在相近画质下优于强 repetition？ | 未知，没有 WAM 图像对照结果。 |
| 4. 固定 4-of-8 时 MDS 是否优于 RLNC？ | 旧编码层实测支持优先把 MDS 作为候选：70/70 对 RLNC 69/70；不构成 WAM 端到端优越性的证据。本轮没有重跑该实验。 |
| 5. JPEG 是否需要 inner ECC？ | 未知，不建议在没有误码诊断前添加。 |
| 6. 小角度搜索是否有效？ | 未知，不应把角度搜索写成已经验证的模块。 |
| 7. Confidence-aware rejection 是否有效？ | 未知，无输出置信度数据，更未在开发集校准阈值。 |
| 8. 最终课程 method 应该是什么？ | 目前不能确定。保留 WAM localized packets + MDS 4-of-8 为待验证候选；先验证定位瓶颈，不能承诺这就是最终路线。 |

## 恢复执行的最小条件

解决依赖解析与官方文件获取后，保存完整依赖 lock、源码 commit、checkpoint SHA-256 和成功加载日志。然后只用一张公开自然图像、两个不同 32-bit 消息做初始测试；成功后再扩展至 10–20 张、4 区域，最后才尝试 8 区域。继续遵守原请求的 Gate 1 / Gate 2，decoder 不接收原图、原始区域坐标或 crop 信息。

本轮不生成定稿 proposal，因为尚无本次方法验证证据。

## 官方来源

- [WAM 固定源码版本](https://github.com/facebookresearch/watermark-anything/tree/2c08af04d037d5667c02f6ddebbda9ff04581c3e)
- [官方安装与预训练权重说明](https://github.com/facebookresearch/watermark-anything#requirements)
- [固定版本 requirements](https://github.com/facebookresearch/watermark-anything/blob/2c08af04d037d5667c02f6ddebbda9ff04581c3e/requirements.txt)
- [固定版本模型配置](https://github.com/facebookresearch/watermark-anything/blob/2c08af04d037d5667c02f6ddebbda9ff04581c3e/checkpoints/params.json)
