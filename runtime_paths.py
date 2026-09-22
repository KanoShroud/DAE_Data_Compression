"""Stable project paths and legacy Python import compatibility."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_ROOT = PROJECT_ROOT / "运行结果"
DATASET_ROOT = PROJECT_ROOT / "datasets"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
_INSTALLED = False


class _LegacyLoader(importlib.abc.Loader):
    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        return importlib.import_module(self.target)

    def exec_module(self, module):
        pass


class _LegacyFinder(importlib.abc.MetaPathFinder):
    def __init__(self, aliases):
        self.aliases = aliases

    def find_spec(self, fullname, path=None, target=None):
        for old in sorted(self.aliases, key=len, reverse=True):
            if fullname == old or fullname.startswith(old + "."):
                canonical = self.aliases[old] + fullname[len(old):]
                return importlib.util.spec_from_loader(fullname, _LegacyLoader(canonical))
        return None


def install_legacy_imports():
    """Keep old pickle/module names loadable without duplicated model classes."""
    global _INSTALLED
    if _INSTALLED:
        return
    manifest = PROJECT_ROOT / "tools/maintenance/layout_20260922.json"
    aliases = json.loads(manifest.read_text(encoding="utf-8"))["modules"]
    sys.meta_path.insert(0, _LegacyFinder(aliases))
    _INSTALLED = True


def resolve_project_path(relative):
    """Resolve a current or pre-migration repository-relative source path."""
    text = str(relative).replace("\\", "/")
    manifest = PROJECT_ROOT / "tools/maintenance/layout_20260922.json"
    mapping = json.loads(manifest.read_text(encoding="utf-8"))["mapping"]
    return PROJECT_ROOT / mapping.get(text, text)


def equivalent_code_fingerprints(reference, current):
    """Accept only explicitly recorded path-migration fingerprint pairs."""
    if reference == current:
        return True
    if not isinstance(reference, str) or not isinstance(current, str):
        return False
    manifest = PROJECT_ROOT / "tools/maintenance/layout_20260922.json"
    aliases = json.loads(manifest.read_text(encoding="utf-8")).get("fingerprint_aliases", {})
    return aliases.get(reference, reference) == aliases.get(current, current)
