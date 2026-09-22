"""One-shot, bounded code-layout migration; no experiments are executed."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "tools/maintenance/layout_20260922.json"
PACKAGES = {
    "brsr_tdoa": "research_code/brsr/stage_ab",
    "brsr_route_feasibility": "research_code/brsr/stage1",
    "brsr_stage2_rate_adaptive": "research_code/brsr/stage2",
    "brsr_observability_certificate": "research_code/brsr/observability",
    "brsr_state_oracle_certificate": "research_code/brsr/state_oracle",
    "brsr_s1_precision_replication": "research_code/brsr/precision_replication",
    "brsr_transition_regime": "research_code/brsr/transition_regime",
    "np_nbq_dpd": "research_code/np_nbq_dpd",
    "raptq_tdoa": "research_code/raptq",
    "sitq_tdoa": "research_code/sitq",
    "pasr_tdoa": "research_code/pasr",
    "pfrs_frequency_mask": "research_code/pfrs",
    "standalone_frequency_alignment": "research_code/frequency_alignment",
}
GROUPS = {
    "research_code/dae_pipeline": "main model train evaluate experiment_profiles replot export_results diagnose_go_no_go analyze_results",
    "research_code/v5_compressed_tdoa": "compressed_tdoa topk_localization diagnose_v5b_topk diagnose_v5b_topk_wls evaluate_topk_fairness benchmark_fairness_complexity",
    "research_code/shared": "signal_gen baselines task_baselines experiment_cache experiment_integrity",
    "tools/audits": "audit_research_gates test_audit_research_gates validate_project_integrity",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if REPORT.exists():
        raise SystemExit("Migration already recorded; do not run twice.")
    mapping = {}
    modules = {old: new.replace("/", ".") for old, new in PACKAGES.items()}
    for old, new in PACKAGES.items():
        for path in (ROOT / old).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                mapping[path.relative_to(ROOT).as_posix()] = new + "/" + path.relative_to(ROOT / old).as_posix()
    for folder, names in GROUPS.items():
        for name in names.split():
            mapping[name + ".py"] = folder + "/" + name + ".py"
            modules[name] = folder.replace("/", ".") + "." + name
    for name in (ROOT / "pfrs_frequency_mask").glob("*.py"):
        modules[name.stem] = "research_code.pfrs." + name.stem
    for name in (ROOT / "standalone_frequency_alignment").glob("*.py"):
        if name.stem != "__init__":
            modules[name.stem] = "research_code.frequency_alignment." + name.stem
    originals = {old: (ROOT / old).read_bytes() for old in mapping}
    result_stats = {p.relative_to(ROOT).as_posix(): [p.stat().st_size, p.stat().st_mtime_ns]
                    for p in (ROOT / "运行结果").rglob("*") if p.is_file()}
    path_map = {**PACKAGES, **mapping}

    def rename_module(name):
        for old in sorted(modules, key=len, reverse=True):
            if name == old or name.startswith(old + "."):
                return modules[old] + name[len(old):]
        return name

    def rewrite_code(text, old):
        tree = ast.parse(text)
        lines = text.splitlines(keepends=True)
        edits = []
        # AST locations are byte columns; import statements here are ASCII.
        starts = [0]
        for line in lines:
            starts.append(starts[-1] + len(line))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and not node.level and node.module:
                renamed = rename_module(node.module)
                if renamed != node.module:
                    start = starts[node.lineno - 1] + node.col_offset
                    end = starts[node.end_lineno - 1] + node.end_col_offset
                    fragment = text[start:end]
                    edits.append((start, end, fragment.replace("from " + node.module + " import", "from " + renamed + " import", 1)))
            elif isinstance(node, ast.Import):
                changed = False
                imports = []
                for alias in node.names:
                    renamed = rename_module(alias.name)
                    changed |= renamed != alias.name
                    if renamed != alias.name and "." in alias.name and not alias.asname:
                        raise RuntimeError("Review dotted import: " + old)
                    binding = alias.asname or (alias.name if renamed != alias.name else None)
                    imports.append(renamed + (" as " + binding if binding else ""))
                if changed:
                    edits.append((starts[node.lineno - 1] + node.col_offset, starts[node.end_lineno - 1] + node.end_col_offset, "import " + ", ".join(imports)))
        for start, end, replacement in sorted(edits, reverse=True):
            text = text[:start] + replacement + text[end:]
        # Only known repository-root expressions are changed; sibling-file paths stay local.
        if "/" in old:
            text = text.replace("Path(__file__).resolve().parents[1]", "_LAYOUT_ROOT")
        else:
            text = text.replace("Path(__file__).resolve().parent", "_LAYOUT_ROOT")
        text = text.replace("os.path.dirname(os.path.abspath(__file__))", "str(_LAYOUT_ROOT)")
        text = text.replace("PROJECT_ROOT = MODULE_ROOT.parent", "PROJECT_ROOT = _LAYOUT_ROOT")
        text = text.replace("PROJECT_DIRECTORY = MODULE_DIRECTORY.parent", "PROJECT_DIRECTORY = _LAYOUT_ROOT")
        # Rewrite literal project-relative paths, including multi-part Path joins.
        for before, after in sorted(path_map.items(), key=lambda item: len(item[0]), reverse=True):
            text = text.replace('"' + before + '"', '"' + after + '"')
            text = text.replace("'" + before + "'", "'" + after + "'")
        parsed = ast.parse(text)
        insert = 0
        for node in parsed.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                insert = node.end_lineno
            elif isinstance(node, ast.ImportFrom) and node.module == "__future__":
                insert = node.end_lineno
            else:
                break
        bootstrap = [
            "\n# Resolve the repository independently of the entry's directory.\n",
            "from pathlib import Path as _LayoutPath\n",
            "import sys as _layout_sys\n",
            "_LAYOUT_ROOT = next(p for p in _LayoutPath(__file__).resolve().parents if (p / 'runtime_paths.py').is_file())\n",
            "if str(_LAYOUT_ROOT) not in _layout_sys.path:\n",
            "    _layout_sys.path.insert(0, str(_LAYOUT_ROOT))\n",
            "from runtime_paths import install_legacy_imports as _install_layout_imports\n",
            "_install_layout_imports()\n\n",
        ]
        lines = text.splitlines(keepends=True)
        lines[insert:insert] = bootstrap
        return "".join(lines)

    changes = []
    for old, new in mapping.items():
        src, dst = ROOT / old, ROOT / new
        if not src.resolve().is_relative_to(ROOT) or not dst.resolve().is_relative_to(ROOT):
            raise RuntimeError("Out-of-workspace migration")
        if dst.exists():
            raise RuntimeError("Target exists: " + new)
        data = originals[old]
        if src.suffix == ".py":
            text = data.decode("utf-8-sig").replace("\r\n", "\n")
            result = rewrite_code(text, old)
            if b"\r\n" in data:
                result = result.replace("\n", "\r\n")
            output = (b"\xef\xbb\xbf" if data.startswith(b"\xef\xbb\xbf") else b"") + result.encode("utf-8")
        else:
            output = data
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        if output != data:
            dst.write_bytes(output)
        changes.append({"old": old, "new": new, "before_sha256": sha(data), "after_sha256": sha(output)})
    # Package markers and legacy alias map are generated, not duplicate implementations.
    for path in [ROOT / "research_code", ROOT / "tools"] + [p.parent for p in (ROOT / "research_code").rglob("*.py")]:
        marker = path / "__init__.py"
        if not marker.exists():
            marker.write_text('"""Research code package."""\n', encoding="utf-8")
    REPORT.write_text(json.dumps({"mapping": mapping, "modules": modules, "files": changes, "result_file_stats": result_stats}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Original code is retained locally for hash mapping and bounded numerical comparison.
    import zipfile
    backup = ROOT / "运行结果/维护记录/layout_20260922"
    backup.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(backup / "source_before.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for old, data in originals.items():
            archive.writestr(old, data)
    print(f"Moved {len(mapping)} files; original bytes saved once for migration audit.")


if __name__ == "__main__":
    main()
