"""Bounded migration accounting and non-computational AST difference check."""

import ast
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[2]
BACKUP = ROOT / "运行结果/维护记录/layout_20260922"


def main():
    report = json.loads((ROOT / "tools/maintenance/layout_20260922.json").read_text(encoding="utf-8"))
    reverse = {v: k for k, v in report["mapping"].items()}
    for old, new in report["modules"].items():
        if "." not in old:
            reverse.setdefault(new.replace(".", "/"), old)

    class Normalize(ast.NodeTransformer):
        def visit_Import(self, node):
            return None

        def visit_ImportFrom(self, node):
            return None

        def visit_Constant(self, node):
            if isinstance(node.value, str):
                value = node.value
                if re.fullmatch(r"[a-f0-9]{64}", value):
                    return ast.Constant("SOURCE_OR_MANIFEST_DIGEST")
                if value in reverse:
                    return ast.Constant(reverse[value])
            return node

        def visit(self, node):
            if isinstance(node, ast.expr):
                text = ast.unparse(node)
                if text in ("Path('运行结果')", "_LAYOUT_ROOT / '运行结果'", "Path(__file__).resolve().parents[1] / '运行结果'", "Path(__file__).resolve().parent / '运行结果'"):
                    return ast.Name(id="RESULT_ROOT", ctx=ast.Load())
                if isinstance(node, ast.Call) and ast.unparse(node.func) == "os.path.join" and len(node.args) == 2 and ast.unparse(node.args[0]) == "str(_LAYOUT_ROOT)" and isinstance(node.args[1], ast.Constant) and str(node.args[1].value).startswith("运行结果/"):
                    return node.args[1]
                if text in ("str(_LAYOUT_ROOT)", "str(Path(__file__).resolve().parents[1])", "str(Path(__file__).resolve().parent)", "os.path.dirname(os.path.abspath(__file__))"):
                    return ast.Name(id="REPO_STRING", ctx=ast.Load())
                if text in ("_LAYOUT_ROOT", "Path(__file__).resolve().parent", "Path(__file__).resolve().parents[1]", "MODULE_ROOT.parent", "MODULE_DIRECTORY.parent"):
                    return ast.Name(id="REPO_ROOT", ctx=ast.Load())
            return super().visit(node)

    differences = []
    with zipfile.ZipFile(BACKUP / "source_before.zip") as archive:
        for old, new in report["mapping"].items():
            assert (ROOT / new).is_file(), new
            if not old.endswith(".py"):
                continue
            before = ast.parse(archive.read(old).decode("utf-8-sig"))
            after = ast.parse((ROOT / new).read_text(encoding="utf-8-sig"))
            # Strip only the inserted repository bootstrap, ending at its installer call.
            end = next(i for i, node in enumerate(after.body) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "_install_layout_imports")
            start = next(i for i, node in enumerate(after.body) if isinstance(node, ast.ImportFrom) and node.module == "pathlib" and any(a.asname == "_LayoutPath" for a in node.names))
            del after.body[start:end + 1]
            left = ast.dump(Normalize().visit(before), include_attributes=False)
            right = ast.dump(Normalize().visit(after), include_attributes=False)
            if left != right:
                differences.append(new)
    stats = json.loads((BACKUP / "result_file_stats.json").read_text(encoding="utf-8"))
    unexpected_artifacts = []
    for relative, expected in stats.items():
        path = ROOT / relative
        if relative in report["metadata_updates"]:
            continue
        if not path.is_file() or [path.stat().st_size, path.stat().st_mtime_ns] != expected:
            unexpected_artifacts.append(relative)
    print(json.dumps({"mapped_files": len(report["mapping"]), "review_ast_differences": differences, "unexpected_artifact_changes": unexpected_artifacts}, ensure_ascii=False, indent=2))
    assert not unexpected_artifacts
    assert set(differences) == {"tools/audits/validate_project_integrity.py", "research_code/shared/experiment_integrity.py"}, differences


if __name__ == "__main__":
    main()
