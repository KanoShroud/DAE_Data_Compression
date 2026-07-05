# AGENTS.md

本文件是写给 Codex 的项目级长期规则。进入本仓库工作时，先读本文件，再读 `对话上下文.md`、`当前任务.md`、`修改记录.md`、`实验结果记录.md`。

## 项目背景

- 项目名称：`DAE Data Compression`。
- 本地参考路径：`F:\PythonWorkspace\DAE Data Compression`，如仓库迁移，以当前工作目录为准。
- 项目运行虚拟环境：`D:\Software\anaconda3\envs\PyTorch`，如果缺少需要的包，可通过`conda install`安装。
- 研究主题：基于深度去噪自编码器（DAE）的多 UAV 协作定位数据压缩。
- 基准论文：Chen 等，`Collaborative Localization Using Multiple UAVs: A Deep Denoising Autoencoder-Based Data Compression Approach`，IEEE TVT 2025。
- 研究主线：面向 TDOA/GCC 定位任务的数据压缩，而非单纯最小化波形 MSE。
- 当前版本、实验结果、分支状态、下一步任务，以 `对话上下文.md` 和 `当前任务.md` 为准。

## 工作原则

- 先读上下文，再改代码；不要凭记忆假设当前版本。
- 用户明确要求“只给方案/不执行”时，不得修改文件。
- 用户要求执行时，保持改动小而可解释，优先沿用现有代码风格。
- 每次修改后检查原理、逻辑、可读性和结果展示，不只检查语法。
- 不编造实验结果、指标、模型权重、运行目录或结论；不确定就写“待确认”。
- 不在未确认前大规模重构无关文件。
- 如果发现工作树已有未提交修改，默认是用户或前序工作留下的，不要回滚。

## 协作分工

- Codex 负责改代码、轻量验证、读取结果、分析数据和给出方案。
- 用户默认用 PyCharm 运行 `main.py`、完整训练、完整 eval-only 和长耗时脚本。
- 用户默认用 PyCharm Git 插件管理提交、分支、合并、推送和回滚；Codex 只做必要的只读 Git 检查。
- Codex 不主动 commit、push、merge、rebase、checkout 或创建分支，除非用户明确要求。
- 代码修改通过轻量验证后，如仍需正式结果，Codex 应说明“待 PyCharm 运行”，并列出入口脚本、关键配置和需回读的结果文件。

## 编码与中文路径规则

- 读取中文路径或中文文档时，在 PowerShell 中优先使用 `Get-Content -Encoding UTF8 -LiteralPath ...`；路径含空格、中文或特殊字符时使用 `-LiteralPath`。
- 看到中文乱码时，先按 UTF-8 重新读取确认，不要基于乱码内容修改文件。
- 编辑含中文注释的 `.py` 文件时，只改任务相关行；避免因格式化、批量替换或编码转换重写整文件。
- 新增或修改 Python 文本读写逻辑时，显式使用 `encoding="utf-8"`；处理中文路径时优先使用 `pathlib.Path` 或原始字符串。
- 不把 PowerShell 默认显示乱码当作文件损坏证据；是否需要统一编码应单独确认。

## 代码修改规则

- 手工编辑文件使用 `apply_patch`。
- 搜索文件优先使用 `rg` / `rg --files`。
- 修改核心 Python 文件后，至少编译本轮涉及的 `.py` 文件。
- 若涉及主流程、训练、评估、绘图、baseline 或结果导出，优先执行：
  - `python -m py_compile main.py train.py signal_gen.py model.py evaluate.py replot.py baselines.py export_results.py task_baselines.py`
  - 必要时追加本轮新增或修改的 Python 文件，例如阶段性脚本 `update_dft_fisher_direct.py`。
  - 必要时用 `D:\Software\anaconda3\envs\PyTorch\python.exe` 做 PyTorch 冒烟测试。
- 涉及损失函数时必须同步检查训练集与验证集目标是否一致。
- 涉及 GCC/TDOA 时必须检查训练、验证、评估使用的 lag window 和相关定义是否一致。
- 涉及图片生成时必须检查图例、坐标轴、曲线含义和 pkl 字段是否一致。

## Git 与分支规则

- 当任务涉及修改文件、运行脚本、分析工作树状态或准备提交前，必须检查：
  - `git status --short --branch`
  - `git branch --show-current`
  - `git log -5 --oneline --decorate`
- 纯讨论、纯方案、纯文本解释时可不检查 Git。
- 不要提交 Git，除非用户明确要求。
- 不要使用 `git reset --hard`、`git checkout --` 回滚用户修改，除非用户明确要求。
- 若需要新分支，默认使用 `codex/` 前缀，除非用户指定。
- 如用户只需要版本管理建议，Codex 只给出建议文件列表和 commit message 草稿，不直接执行 Git 写操作。

## 忽略规则

当前 `.gitignore` 已忽略：

- `运行结果/`
- `__pycache__/`
- `参考文献/`
- `*.pyc`
- `*.pyo`
- `*.pyd`
- `*.xml`

建议继续忽略但未必已写入 `.gitignore` 的内容：

- 模型权重：`*.pt`、`*.pth`
- 大型绘图/中间数据：`*.pkl`，但历史 `plot_data.pkl` 若作为本地实验证据，保留在被忽略的 `运行结果/` 下即可。
- 临时文件：`.pytest_cache/`、`.mypy_cache/`、`*.tmp`
- 本地环境：`.venv/`、`venv/`

修改 `.gitignore` 前需确认是否会影响用户希望纳入版本管理的文件。

## 常用检查命令

```powershell
git status --short --branch
git branch --show-current
git log -5 --oneline --decorate
rg --files
python -m py_compile main.py train.py signal_gen.py model.py evaluate.py replot.py baselines.py export_results.py task_baselines.py
D:\Software\anaconda3\envs\PyTorch\python.exe -m py_compile main.py train.py signal_gen.py model.py evaluate.py replot.py baselines.py export_results.py task_baselines.py
```

完整训练、完整 `paper_repro_eval_only` 和正式结果生成默认由用户在 PyCharm 中运行。当前常用配置集中在 `main.py` 顶部 `USER_*` 普通变量；默认 `ALLOW_ENV_OVERRIDES=False`，PyCharm Run Configuration 或 PowerShell 中的 `DAE_*` 环境变量通常不会生效。

## 项目文档职责与更新规则

本项目根目录维护以下长期文档，更新采用事件驱动，不要求每轮都更新所有文档。

### 文档职责

- `AGENTS.md`：项目级长期规则。只记录稳定工作原则、审批机制、验证要求、Git 安全规则和文档更新规则；不得记录当前版本号、当前实验结果、当前分支状态或下一步任务。
- `对话上下文.md`：项目当前研究状态摘要。记录当前代码版本、研究阶段、核心方法、关键结论、未解决问题和下一步方向；避免重复完整实验表。
- `当前任务.md`：当前唯一任务看板。只记录正在推进的任务、用户审批状态、是否允许修改代码/运行脚本、范围内外文件、待办清单和完成标准；旧任务完成后移入 `修改记录.md`，不得长期残留在正文。
- `修改记录.md`：已执行修改的短日志。记录有项目意义的代码或文档修改、修改原因、涉及文件、验证方式、验证结果和遗留问题；未执行修改的方案不写入本文件。
- `实验结果记录.md`：实验和验证事实表。只记录实际运行、读取日志/pkl/svg、或明确来自用户反馈的实验信息；必须标注证据来源、路径、关键参数、指标、结论、可信度和是否需复核。

### 更新触发规则

- 只分析或讨论、不修改文件：通常不更新 `修改记录.md`；若读取了新实验结果，应更新 `实验结果记录.md`；若形成新的下一步任务，应更新 `当前任务.md`。
- 提出修改方案但等待用户审批：更新 `当前任务.md` 为“方案待审批”；不更新 `修改记录.md`。
- 用户批准方案后：更新 `当前任务.md` 为“已批准待执行”，再开始修改。
- 执行代码或文档修改后：更新 `修改记录.md`；若研究状态或下一步发生变化，同步更新 `对话上下文.md` 和 `当前任务.md`。
- 运行 `py_compile`、冒烟测试、评估脚本或完整训练后：更新 `实验结果记录.md`；如果这些验证属于某次代码修改，也在 `修改记录.md` 中简要引用。
- 完整训练或重要结果分析完成后：必须更新 `实验结果记录.md` 和 `对话上下文.md`，并根据结论更新 `当前任务.md`。
- Codex 修改代码并通过轻量验证后，若正式结果需用户在 PyCharm 运行，应在交付说明或 `当前任务.md` 中标明入口脚本、关键配置、是否训练、预期输出目录和回读文件。
- `AGENTS.md` 仅在项目级规则变化时更新，禁止频繁写入动态状态。
- `修改记录.md` 记录有项目意义的代码或文档修改。轻微排版、错别字、仅更新任务状态、仅追加修改记录本身，通常不单独记录。
- 更新 `修改记录.md` 是记录流程的一部分，不触发新的修改记录，避免递归记录。

## 结果记录底线

- 只有实际读取过日志、pkl、svg 或实际运行过命令，才能写具体结果。
- 只根据用户口述时，必须标注“用户反馈”或“待复核”。
- 实验结论必须包含路径、脚本、关键参数、指标或现象、可信度。
- 用户在 PyCharm 运行后，优先读取 `run.log`、`plot_data.pkl`、相关 SVG、`tables/*.csv` 和 `tables/*.md`；分析时区分正式运行、小样本 smoke、单方法增量更新、仅重绘/导出和用户反馈。
