"""The migration allowance must not weaken any unrelated protocol check."""

import json

import pytest

from runtime_paths import PROJECT_ROOT
from research_code.shared.experiment_integrity import assert_protocol_compatible


def test_exact_migration_pair_is_accepted():
    aliases = json.loads((PROJECT_ROOT / "tools/maintenance/layout_20260922.json").read_text(encoding="utf-8"))["fingerprint_aliases"]
    for old, new in aliases.items():
        assert_protocol_compatible({"baseline_code_fingerprint": old}, {"baseline_code_fingerprint": new}, keys=("baseline_code_fingerprint",))


def test_unknown_code_or_changed_protocol_is_rejected():
    with pytest.raises(RuntimeError):
        assert_protocol_compatible({"baseline_code_fingerprint": "unregistered"}, {"baseline_code_fingerprint": "different"}, keys=("baseline_code_fingerprint",))
    with pytest.raises(RuntimeError):
        assert_protocol_compatible({"seed": 1}, {"seed": 2}, keys=("seed",))
