"""
Small profile layer for main.py user-facing experiment routes.

The project still keeps main.py as the canonical PyCharm entry point. Profiles
only set the USER_* variables that select a route; they do not duplicate any
training, evaluation or plotting logic.
"""


EXPERIMENT_PROFILES = {
    "compressed_tdoa_v5a_fast": {
        "USER_EXPERIMENT_MODE": "compressed_tdoa_v5a_fast",
        "USER_REUSE_STATIC_BASELINE_CACHE": False,
        "USER_RUN_TRADITIONAL_BASELINES": True,
        "USER_RUN_TASK_AWARE_BASELINES": True,
        "USER_RUN_STRONG_BASELINES": True,
        "USER_RUN_STRONG_DIAGNOSTIC_SUPPLEMENT": False,
        "USER_RUN_GEOHYBRID_BASELINES": True,
        "USER_BASELINE_INCLUDE_DIAGNOSTIC_VARIANTS": True,
        "USER_BASELINE_DFT_MODE": "scs_lite",
        "USER_BASELINE_DFT_DIRECT_SOURCE": "DFT-train-power",
        "USER_DIRECT_ONLY_OUTPUT": True,
        "USER_EXPORT_CORE_FIGURES": False,
        "USER_EXPORT_CORE_TABLES": False,
        "USER_EXPORT_FIG6_RESULTS": False,
        "USER_EXPORT_MARKDOWN_TABLES": False,
        "USER_V5A_EPOCHS": 40,
        "USER_V5A_N_PAIRS": 10000,
    },
    "freq_task_v4_min_fast": {
        "USER_EXPERIMENT_MODE": "freq_task_v4_min_fast",
        "USER_FREQ_TASK_V4_FAST_EPOCHS": 40,
        "USER_REUSE_STATIC_BASELINE_CACHE": True,
        "USER_RUN_GEOHYBRID_BASELINES": False,
        "USER_EXPORT_MARKDOWN_TABLES": False,
    },
    "geohybrid_direct_only_eval": {
        "USER_EXPERIMENT_MODE": "paper_repro_eval_only",
        "USER_REUSE_STATIC_BASELINE_CACHE": False,
        "USER_RUN_TRADITIONAL_BASELINES": True,
        "USER_RUN_TASK_AWARE_BASELINES": True,
        "USER_RUN_STRONG_BASELINES": True,
        "USER_RUN_STRONG_DIAGNOSTIC_SUPPLEMENT": True,
        "USER_RUN_GEOHYBRID_BASELINES": True,
        "USER_BASELINE_INCLUDE_DIAGNOSTIC_VARIANTS": True,
        "USER_BASELINE_DFT_MODE": "scs_lite",
        "USER_BASELINE_DFT_DIRECT_SOURCE": "DFT-train-power",
        "USER_DIRECT_ONLY_OUTPUT": True,
        "USER_EXPORT_CORE_FIGURES": False,
        "USER_EXPORT_CORE_TABLES": False,
        "USER_EXPORT_FIG6_RESULTS": False,
        "USER_EXPORT_MARKDOWN_TABLES": False,
    },
    "geohybrid_v2_eval": {
        "USER_EXPERIMENT_MODE": "paper_repro_eval_only",
        "USER_REUSE_STATIC_BASELINE_CACHE": False,
        "USER_RUN_TRADITIONAL_BASELINES": True,
        "USER_RUN_TASK_AWARE_BASELINES": True,
        "USER_RUN_STRONG_BASELINES": True,
        "USER_RUN_STRONG_DIAGNOSTIC_SUPPLEMENT": True,
        "USER_RUN_GEOHYBRID_BASELINES": True,
        "USER_BASELINE_INCLUDE_DIAGNOSTIC_VARIANTS": True,
        "USER_BASELINE_DFT_MODE": "scs_lite",
        "USER_BASELINE_DFT_DIRECT_SOURCE": "DFT-train-power",
    },
    "geoambi_direct_eval": {
        "USER_EXPERIMENT_MODE": "paper_repro_eval_only",
        "USER_REUSE_STATIC_BASELINE_CACHE": False,
        "USER_RUN_TRADITIONAL_BASELINES": True,
        "USER_RUN_TASK_AWARE_BASELINES": True,
        "USER_RUN_STRONG_BASELINES": True,
        "USER_RUN_STRONG_DIAGNOSTIC_SUPPLEMENT": True,
        "USER_RUN_GEOHYBRID_BASELINES": False,
        "USER_BASELINE_INCLUDE_DIAGNOSTIC_VARIANTS": True,
        "USER_BASELINE_DFT_MODE": "scs_lite",
        "USER_BASELINE_DFT_DIRECT_SOURCE": "DFT-train-power",
    },
    "paper_repro_eval": {
        "USER_EXPERIMENT_MODE": "paper_repro_eval_only",
        "USER_REUSE_STATIC_BASELINE_CACHE": False,
    },
    "freq_task_v2_final": {
        "USER_EXPERIMENT_MODE": "freq_task_v2_200_final",
        "USER_REUSE_STATIC_BASELINE_CACHE": True,
    },
    "freq_task_v3_nested_fast": {
        "USER_EXPERIMENT_MODE": "freq_task_v3_fast_final",
        "USER_FREQ_TASK_V3_FAST_EPOCHS": 60,
        "USER_REUSE_STATIC_BASELINE_CACHE": True,
    },
    "freq_task_v3_nested_200": {
        "USER_EXPERIMENT_MODE": "freq_task_v3_200_final",
        "USER_FREQ_TASK_V3_FULL_EPOCHS": 200,
        "USER_REUSE_STATIC_BASELINE_CACHE": True,
    },
}


def apply_experiment_profile(namespace, profile_name):
    """Apply a profile to a globals-like namespace in main.py."""
    if profile_name is None:
        return {}
    if profile_name not in EXPERIMENT_PROFILES:
        valid = ", ".join(sorted(EXPERIMENT_PROFILES))
        raise ValueError(f"Unknown experiment profile '{profile_name}'. Valid profiles: {valid}")
    profile = dict(EXPERIMENT_PROFILES[profile_name])
    for key, value in profile.items():
        namespace[key] = value
    return profile
