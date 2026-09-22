"""Refresh navigation and file-type indexes without reading binary artifact contents."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    report = json.loads((ROOT / "tools/maintenance/layout_20260922.json").read_text(encoding="utf-8"))
    inventories = {"datasets": [], "checkpoints": []}
    for path in sorted((ROOT / "运行结果").rglob("*")):
        if not path.is_file():
            continue
        category = "checkpoints" if path.suffix.lower() in (".pt", ".pth", ".ckpt") else "datasets" if path.suffix.lower() in (".npy", ".npz", ".h5", ".hdf5") else None
        if category:
            inventories[category].append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size})
    for category, items in inventories.items():
        (ROOT / category / "artifact_index.json").write_text(json.dumps({"scope": "Existing experiment bundles; index only, no copied artifacts", "files": items}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# 项目导航", "", "## 1. 目录职责", "",
        "路径均相对项目根目录；不依赖当前 PowerShell/PyCharm 工作目录查找实现。",
        "", "| 位置 | 内容 |", "|---|---|",
        "| 根目录长期 md | 当前状态、授权、研究方案、结论和记录 |",
        "| [研究总入口](任务驱动压缩与定位研究.md) | 各路线分阶段的实施方案与执行结论 |",
        "| research_code/ | 按路线/阶段分类的唯一实现；shared/ 为已核对依赖的公共模块 |",
        "| 运行入口/ | PyCharm 薄入口，不复制算法和默认配置 |",
        "| tools/audits/ | 跨路线完整性与门禁审计 |",
        "| tools/maintenance/ | 索引生成及有重复执行保护的一次性迁移工具 |",
        "| [datasets](datasets/README.md) / [checkpoints](checkpoints/README.md) | 独立数据/权重的登记入口，既有实验产物保持原位 |",
        "| [运行结果清单](运行结果/结果清单.md) | 原路线/原运行编号的完整实验包，未拆分重命名 |",
        "| [文档归档](文档归档/README.md) | 历史快照，不代表最新状态 |",
        "| 参考文献/、Obsidian Vault/ | 保持原位置和内容 |", "",
        "## 2. PyCharm 运行方式", "",
        "1. 解释器保持 `D:\\Software\\anaconda3\\envs\\PyTorch\\python.exe`。选择下表入口；工作目录建议项目根目录。",
        "2. 配置只在对应实现中的既有 `RUN_MODE`、`USER_*` 或命令行参数维护；薄入口只转交运行，不覆盖配置。",
        "3. 根目录 [main.py](main.py) 保持可运行，调用 [实际主流程](research_code/dae_pipeline/main.py)。其他旧脚本路径按迁移映射更新运行配置，不保留散落根目录的重复脚本。",
        "4. 本次迁移不批准新的正式运行；主线仍按 [当前任务](当前任务.md) 的授权和门禁执行。", "",
        "| PyCharm 入口 | 配置与实际实现 |", "|---|---|",
    ]
    for entry, target in sorted(report["entries"].items()):
        lines.append(f"| [{entry}]({entry}) | [{target}]({target}) |")
    lines += ["", "## 3. 实现文件索引", "", "测试随所属路线保留；不将仍被其他阶段依赖的旧实现挪到失活归档。", ""]
    folders = sorted({(ROOT / new).parent for new in report["mapping"].values()})
    for folder in folders:
        relative = folder.relative_to(ROOT).as_posix()
        lines += ["### " + relative, ""]
        for path in sorted(folder.iterdir()):
            if path.suffix in (".py", ".md") and path.name != "__init__.py":
                rel = path.relative_to(ROOT).as_posix()
                lines.append(f"- [{path.name}]({rel})")
        lines.append("")
    lines += ["## 4. 数据与权重", "",
              f"- 数据数组/缓存候选共 {len(inventories['datasets'])} 个，精确路径见 [数据索引](datasets/artifact_index.json)。这不是独立数据集认定，也不自动授权协议间复用。",
              f"- 权重文件共 {len(inventories['checkpoints'])} 个，精确路径见 [权重索引](checkpoints/artifact_index.json)。既有权重仍与日志、配置和结果在同一实验包。",
              "- 索引更新工具：[index_layout.py](tools/maintenance/index_layout.py)，只读取目录和文件大小，不重算实验。", "",
              "## 5. 迁移与兼容", "",
              "- [runtime_paths.py](runtime_paths.py) 提供项目根目录、旧模块名兼容和旧源码路径映射。新代码使用 research_code 的规范导入；独立反序列化旧 pickle 前安装旧模块别名。",
              "- [layout_20260922.json](tools/maintenance/layout_20260922.json) 记录每个源文件的新旧位置及仅迁移造成的哈希变化，不把它当作新的实验结果。",
              "- 本地原源码和元数据备份：`运行结果/维护记录/layout_20260922/source_before.zip`、`metadata_before.zip`、`references_before.zip`。历史数值文件未因目录整理重写。",
              "- 归档正文保留历史路径，通过映射查找现位置；当前文档和现用入口采用新路径。", ""]
    (ROOT / "项目导航.md").write_text("\n".join(lines), encoding="utf-8")
    print({category: len(items) for category, items in inventories.items()})


if __name__ == "__main__":
    main()
