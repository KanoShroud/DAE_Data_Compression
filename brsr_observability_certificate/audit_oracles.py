"""审计BRSR历史Oracle标签的信息边界和合法解释。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class OracleDefinition:
    legacy_label: str
    oracle_class: str
    information_used: str
    valid_use: str
    invalid_inference: str
    protocol_status: str


ORACLE_DEFINITIONS = (
    OracleDefinition(
        legacy_label="Oracle-OneAcquire",
        oracle_class="ClairvoyantOutcomeOracle",
        information_used="真实TDOA及每个一次新增频点动作的实际平方误差",
        valid_use="不可部署的逐样本事后误差下界",
        invalid_inference="不能单独证明动作价值可由动作前观测预测",
        protocol_status="DESCRIPTIVE_LOWER_BOUND",
    ),
    OracleDefinition(
        legacy_label="Oracle-B24",
        oracle_class="ClairvoyantOutcomeOracle",
        information_used="真实TDOA及stop/acquire各结果的实际平方误差",
        valid_use="不可部署的同bit事后误差下界",
        invalid_inference="不能单独证明stop/acquire策略存在可学习余量",
        protocol_status="DESCRIPTIVE_LOWER_BOUND",
    ),
    OracleDefinition(
        legacy_label="OracleAction-B29",
        oracle_class="ClairvoyantOutcomeOracle",
        information_used="真实TDOA及全部29-bit动作的实际平方误差",
        valid_use="不可部署的逐样本动作下界",
        invalid_inference="不能作为安全门控或动作评分可学习性的证据",
        protocol_status="DESCRIPTIVE_LOWER_BOUND",
    ),
    OracleDefinition(
        legacy_label="BlockOracle",
        oracle_class="ClairvoyantBlockOutcomeOracle",
        information_used="块内全部独立样本真值及所有动作的实际平方误差",
        valid_use="仅描述把独立样本事后分组后的数学下界",
        invalid_inference="不能证明块内存在共享物理状态或可部署共享动作",
        protocol_status="NO_GO_PROTOCOL_IID_BLOCK",
    ),
    OracleDefinition(
        legacy_label="Linear-OracleRho",
        oracle_class="NuisanceParameterOracle",
        information_used="真实复增益rho",
        valid_use="量化未知复增益建模损失的诊断下界",
        invalid_inference="不能作为可部署似然或动作策略",
        protocol_status="DIAGNOSTIC_ONLY",
    ),
    OracleDefinition(
        legacy_label="OracleRho",
        oracle_class="NuisanceParameterOracle",
        information_used="真实复增益rho",
        valid_use="量化未知复增益建模损失的诊断下界",
        invalid_inference="不能作为可部署似然或动作策略",
        protocol_status="DIAGNOSTIC_ONLY",
    ),
)


SCAN_PATHS = (
    ROOT / "brsr_tdoa",
    ROOT / "brsr_route_feasibility",
    ROOT / "brsr_stage2_rate_adaptive",
    ROOT / "当前任务.md",
    ROOT / "对话上下文.md",
    ROOT / "研究总结.md",
    ROOT / "研究门禁注册表.md",
    ROOT / "实验结果记录.md",
    ROOT / "修改记录.md",
)


def _source_files() -> list[Path]:
    files: list[Path] = []
    for path in SCAN_PATHS:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
            files.extend(sorted(path.rglob("*.md")))
    return files


def audit_oracles() -> pd.DataFrame:
    """返回当前源码和长期文档中的BRSR Oracle语义清单。"""

    files = _source_files()
    rows: list[dict[str, object]] = []
    for definition in ORACLE_DEFINITIONS:
        occurrences: list[str] = []
        for path in files:
            text = path.read_text(encoding="utf-8")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if definition.legacy_label in line:
                    relative = path.relative_to(ROOT)
                    occurrences.append(f"{relative}:{line_number}")
        rows.append(
            {
                **asdict(definition),
                "occurrence_count": len(occurrences),
                "occurrences": " | ".join(occurrences),
            }
        )
    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="可选CSV输出路径；省略时只打印审计表。",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    frame = audit_oracles()
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        print(f"[Saved] {output}", flush=True)
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
