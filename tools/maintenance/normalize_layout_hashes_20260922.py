"""Rebase LF-normalized historical source hashes after the path-only migration."""

import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[2]
BACKUP = ROOT / "运行结果/维护记录/layout_20260922"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    report_path = ROOT / "tools/maintenance/layout_20260922.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("normalized_hashes_updated"):
        raise SystemExit("Already applied.")
    with zipfile.ZipFile(BACKUP / "source_before.zip") as archive:
        old_source = {name: archive.read(name) for name in archive.namelist() if name.endswith(".py")}
    sources = {row["new"]: (ROOT / row["new"]).read_bytes() for row in report["files"] if row["new"].endswith(".py")}
    originals = {}
    with zipfile.ZipFile(BACKUP / "metadata_before.zip") as archive:
        originals = {name: archive.read(name) for name in archive.namelist()}
    json_base = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in (ROOT / "运行结果").rglob("*.json") if BACKUP not in p.parents}
    hashes = {}
    for _ in range(20):
        for old, data in old_source.items():
            target = report["mapping"][old]
            digest = sha((ROOT / target).read_bytes())
            for prior in (sha(data), sha(data.replace(b"\r\n", b"\n")), sha(sources[target])):
                hashes[prior] = digest

        def update(value):
            if isinstance(value, dict):
                return {k: update(v) for k, v in value.items()}
            if isinstance(value, list):
                return [update(v) for v in value]
            return hashes.get(value, value) if isinstance(value, str) else value

        changed = False
        for relative, data in json_base.items():
            try:
                value = json.loads(data)
            except (ValueError, UnicodeError):
                continue
            updated = update(value)
            output = (json.dumps(updated, ensure_ascii=False, indent=2) + "\n").encode("utf-8") if updated != value else data
            path = ROOT / relative
            if path.read_bytes() != output:
                originals.setdefault(relative, data)
                path.write_bytes(output)
                changed = True
            hashes[sha(data)] = sha(output)
            if relative in originals:
                hashes[sha(originals[relative])] = sha(output)
        for relative, data in sources.items():
            output = re.sub(rb"[a-f0-9]{64}", lambda m: hashes.get(m.group().decode(), m.group().decode()).encode(), data)
            path = ROOT / relative
            if path.read_bytes() != output:
                path.write_bytes(output)
                changed = True
        if not changed:
            break
    else:
        raise RuntimeError("Hash dependency cycle")
    with zipfile.ZipFile(BACKUP / "metadata_before.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for relative, data in originals.items():
            archive.writestr(relative, data)
            report["metadata_updates"][relative] = [sha(data), sha((ROOT / relative).read_bytes())]
    for row in report["files"]:
        row["after_sha256"] = sha((ROOT / row["new"]).read_bytes())
    report["normalized_hashes_updated"] = True
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Updated metadata:", len(report["metadata_updates"]))


if __name__ == "__main__":
    main()
