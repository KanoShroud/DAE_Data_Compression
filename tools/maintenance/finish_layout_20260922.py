"""One-shot mechanical entries, references and metadata update for the approved move."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "tools/maintenance/layout_20260922.json"
BACKUP = ROOT / "运行结果/维护记录/layout_20260922"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_like(path, text, original):
    text = text.replace("\r\n", "\n")
    if b"\r\n" in original:
        text = text.replace("\n", "\r\n")
    path.write_bytes((b"\xef\xbb\xbf" if original.startswith(b"\xef\xbb\xbf") else b"") + text.encode("utf-8"))


def main():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if "entries" in report:
        raise SystemExit("Already finalized; do not repeat.")
    mapping = report["mapping"]
    entries = {}
    for old, new in mapping.items():
        path = ROOT / new
        if path.suffix != ".py" or path.name.startswith("test_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        runnable = any(isinstance(node, ast.If) and "__name__" in ast.unparse(node.test) for node in tree.body)
        if not runnable and path.name != "main.py":
            continue
        if new.startswith("tools/"):
            continue
        relative = "运行入口/" + new.removeprefix("research_code/")
        target = ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            f'"""PyCharm entry for {new}; configure the implementation, not this wrapper."""\n\n'
            'from pathlib import Path\nimport runpy\nimport sys\n\n'
            'ROOT = next(p for p in Path(__file__).resolve().parents if (p / "runtime_paths.py").is_file())\n'
            'if str(ROOT) not in sys.path:\n    sys.path.insert(0, str(ROOT))\n\n'
            'if __name__ == "__main__":\n'
            f'    runpy.run_path(str(ROOT / "{new}"), run_name="__main__")\n', encoding="utf-8")
        entries[relative] = new

    # Replace explicit source references, never result-directory names or scientific text.
    package_map = {k: v.replace(".", "/") for k, v in report["modules"].items() if "." not in k and (ROOT / v.replace(".", "/")).is_dir()}
    replacements = {**package_map, **mapping}
    pattern = re.compile(r"(?<![\w/\\.])(" + "|".join(re.escape(k) for k in sorted(replacements, key=len, reverse=True)) + r")(?=$|[\s`'\"/)\\:，。；、）])")
    changed_docs = []
    docs = list(ROOT.glob("*.md")) + list((ROOT / "research_code").rglob("*.md"))
    reverse = {v: k for k, v in mapping.items()}
    with zipfile.ZipFile(BACKUP / "references_before.zip", "w", zipfile.ZIP_DEFLATED) as backup:
        for path in docs:
            data = path.read_bytes()
            text = data.decode("utf-8-sig")
            current = path.relative_to(ROOT).as_posix()
            old_parent = (ROOT / reverse.get(current, current)).parent
            links = []

            def protect_link(match):
                destination = match.group(2)
                plain = destination.strip("<>").split("#", 1)[0]
                if "://" not in plain and plain and not plain.startswith("#"):
                    resolved = (old_parent / plain).resolve()
                    if resolved.is_relative_to(ROOT):
                        old_rel = resolved.relative_to(ROOT).as_posix()
                        new_target = ROOT / mapping.get(old_rel, old_rel)
                        suffix = "#" + destination.split("#", 1)[1] if "#" in destination else ""
                        destination = os.path.relpath(new_target, path.parent).replace("\\", "/") + suffix
                        if " " in destination:
                            destination = "<" + destination + ">"
                links.append("[" + match.group(1) + "](" + destination + ")")
                return f"LAYOUTLINKTOKEN{len(links) - 1}END"

            text = re.sub(r"\[([^\]]*)\]\(([^)]+)\)", protect_link, text)
            text = pattern.sub(lambda m: replacements[m.group(1)], text)
            for index, link in enumerate(links):
                text = text.replace(f"LAYOUTLINKTOKEN{index}END", link)
            if text != data.decode("utf-8-sig"):
                backup.writestr(current, data)
                write_like(path, text, data)
                changed_docs.append(current)

    # Update only JSON strings/keys referring to moved sources or their exact known hashes.
    # Historic code revisions whose digests differ from the pre-move bytes stay historic.
    metadata = {}
    for path in (ROOT / "运行结果").rglob("*.json"):
        if BACKUP in path.parents:
            continue
        data = path.read_bytes()
        try:
            value = json.loads(data.decode("utf-8-sig"))
        except (ValueError, UnicodeError):
            continue
        metadata[path] = (data, value)
    sources = {ROOT / row["new"]: (ROOT / row["new"]).read_bytes() for row in report["files"] if row["new"].endswith(".py")}
    with zipfile.ZipFile(BACKUP / "source_before.zip") as archive:
        originals = {name: archive.read(name) for name in mapping if name.endswith(".py")}
    groups = [
        ["sitq_tdoa/run_stage_a.py", "sitq_tdoa/core.py"],
        ["np_nbq_dpd/run_stage0.py", "np_nbq_dpd/core.py"],
    ]
    for name, middle in [("run_stage0", []), ("diagnose_stage0_numerical", []), ("diagnose_sparse_integrator", ["sparse_integrator"]), ("recompute_stage0_sparse_amp_phase", ["sparse_integrator"]), ("run_relative_phase_rate", ["relative_phase_rate"])]:
        groups.append(["raptq_tdoa/" + n + ".py" for n in [name, *middle, "core"]])

    def path_string(value):
        normalized = value.replace("\\", "/")
        for old, new in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
            if normalized == old:
                return new
            prefix = ROOT.as_posix() + "/"
            if normalized.casefold() == (prefix + old).casefold():
                return str(ROOT / new) if "\\" in value else (ROOT / new).as_posix()
        return value

    hashes = {}
    changed_metadata = {}
    for iteration in range(24):
        for old, before in originals.items():
            hashes[sha(before)] = sha((ROOT / mapping[old]).read_bytes())
        for group in groups:
            before = b"".join(Path(old).name.encode() + originals[old] for old in group)
            after = b"".join(Path(old).name.encode() + (ROOT / mapping[old]).read_bytes() for old in group)
            hashes[sha(before)] = sha(after)

        def rewrite(value):
            if isinstance(value, dict):
                return {path_string(k): rewrite(v) for k, v in value.items()}
            if isinstance(value, list):
                return [rewrite(v) for v in value]
            if isinstance(value, str):
                return hashes.get(value, path_string(value))
            return value

        changed = False
        for path, (data, value) in metadata.items():
            updated = rewrite(value)
            output = (json.dumps(updated, ensure_ascii=False, indent=2) + "\n").encode("utf-8") if updated != value else data
            if path.read_bytes() != output:
                path.write_bytes(output)
                changed_metadata[path.relative_to(ROOT).as_posix()] = [sha(data), sha(output)]
                changed = True
            hashes[sha(data)] = sha(output)
        for path, data in sources.items():
            text = data.decode("utf-8-sig")
            updated = re.sub(r"[a-f0-9]{64}", lambda m: hashes.get(m.group(), m.group()), text)
            old_output = path.read_bytes()
            if updated != old_output.decode("utf-8-sig"):
                write_like(path, updated, data)
                changed = True
        if not changed:
            break
    else:
        raise RuntimeError("Metadata dependencies did not converge; inspect before proceeding")
    with zipfile.ZipFile(BACKUP / "metadata_before.zip", "w", zipfile.ZIP_DEFLATED) as backup:
        for relative in changed_metadata:
            backup.writestr(relative, metadata[ROOT / relative][0])

    # Keep bulky local stat bookkeeping out of the permanent runtime alias map.
    stats = report.pop("result_file_stats")
    (BACKUP / "result_file_stats.json").write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")
    report["entries"] = entries
    report["changed_document_paths"] = changed_docs
    report["metadata_updates"] = changed_metadata
    report["hash_update_rounds"] = iteration + 1
    for row in report["files"]:
        row["after_sha256"] = sha((ROOT / row["new"]).read_bytes())
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Entries={len(entries)}; documents={len(changed_docs)}; metadata={len(changed_metadata)}; rounds={iteration + 1}")


if __name__ == "__main__":
    main()
