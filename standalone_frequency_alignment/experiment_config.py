"""独立验证路线的普通变量配置。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """一次验证运行所需的全部配置。"""

    profile: str
    seed: int
    n_fft: int
    selected_bins: int
    interpolation_factor: int
    snr_values_db: tuple[float, ...]
    overlap_values: tuple[float, ...]
    signal_kinds: tuple[str, ...]
    trials_per_point: int
    lag_limit_samples: float
    delay_grid_step_samples: float
    selector_mode: str = "controlled_overlap"
    zhai_max_sweeps: int = 50
    zhai_relative_tolerance: float = 1e-10
    bootstrap_repetitions: int = 1000

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def get_profile(name: str) -> ExperimentConfig:
    """返回可直接在 PyCharm 中切换的预定义配置。"""

    profiles = {
        "sanity": ExperimentConfig(
            profile="sanity",
            seed=20260714,
            n_fft=32,
            selected_bins=8,
            interpolation_factor=4,
            snr_values_db=(20.0,),
            overlap_values=(0.0, 0.5, 1.0),
            signal_kinds=("broadband_random",),
            trials_per_point=2,
            lag_limit_samples=6.0,
            delay_grid_step_samples=0.02,
            bootstrap_repetitions=100,
        ),
        "smoke": ExperimentConfig(
            profile="smoke",
            seed=20260714,
            n_fft=64,
            selected_bins=12,
            interpolation_factor=4,
            snr_values_db=(2.0, 14.0),
            overlap_values=(0.0, 0.5, 1.0),
            signal_kinds=("smooth_complex", "broadband_random"),
            trials_per_point=4,
            lag_limit_samples=10.0,
            delay_grid_step_samples=0.02,
            bootstrap_repetitions=200,
        ),
        "synthetic_full": ExperimentConfig(
            profile="synthetic_full",
            seed=20260714,
            n_fft=300,
            selected_bins=20,
            interpolation_factor=8,
            snr_values_db=(2.0, 6.0, 10.0, 14.0, 18.0, 22.0),
            overlap_values=(0.0, 0.25, 0.5, 0.75, 1.0),
            signal_kinds=(
                "smooth_complex",
                "smooth_magnitude_random_phase",
                "broadband_random",
                "multiband",
            ),
            trials_per_point=500,
            lag_limit_samples=30.0,
            delay_grid_step_samples=0.01,
            bootstrap_repetitions=2000,
        ),
        "local_zhai_smoke": ExperimentConfig(
            profile="local_zhai_smoke",
            seed=20260714,
            n_fft=64,
            selected_bins=12,
            interpolation_factor=4,
            snr_values_db=(2.0, 14.0),
            overlap_values=(-1.0,),
            signal_kinds=("smooth_complex", "broadband_random"),
            trials_per_point=4,
            lag_limit_samples=10.0,
            delay_grid_step_samples=0.02,
            selector_mode="local_zhai_thesis",
            bootstrap_repetitions=200,
        ),
        "project_smoke": ExperimentConfig(
            profile="project_smoke",
            seed=20260714,
            n_fft=1024,
            selected_bins=64,
            interpolation_factor=4,
            snr_values_db=(0.0, 12.0),
            overlap_values=(-1.0,),
            signal_kinds=("urban8_los",),
            trials_per_point=2,
            lag_limit_samples=48.0,
            delay_grid_step_samples=0.05,
            selector_mode="local_zhai_thesis",
            bootstrap_repetitions=100,
        ),
        "project_full": ExperimentConfig(
            profile="project_full",
            seed=20260714,
            n_fft=1024,
            selected_bins=64,
            interpolation_factor=8,
            snr_values_db=(-10.0, -6.0, -2.0, 0.0, 2.0, 6.0, 12.0, 20.0),
            overlap_values=(-1.0,),
            signal_kinds=("urban8_los",),
            trials_per_point=200,
            lag_limit_samples=48.0,
            delay_grid_step_samples=0.02,
            selector_mode="local_zhai_thesis",
            bootstrap_repetitions=2000,
        ),
    }
    try:
        return profiles[name]
    except KeyError as exc:
        raise ValueError(f"未知配置 {name!r}，可选值为 {sorted(profiles)}") from exc
