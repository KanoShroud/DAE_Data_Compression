"""读取现有 SignalSimulator 的适配层；不修改项目主流程。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProjectPairBatch:
    station_a_noisy: np.ndarray
    station_a_clean: np.ndarray
    station_b_noisy: np.ndarray
    station_b_clean: np.ndarray
    delay_a_minus_b: np.ndarray
    sample_rate_hz: float


def generate_project_pairs(
    batch_size: int,
    snr_db: float,
    seed: int,
    signal_length: int = 1024,
) -> ProjectPairBatch:
    """按冻结 paper-repro 场景生成双站数据，仅供独立验证调用。"""

    from signal_gen import SignalSimulator

    simulator = SignalSimulator(
        signal_len=signal_length,
        beta=0.2,
        rrc_span=8,
        channel_mode="fixed",
        n_fixed_channels=50,
        channel_pool_seed=42,
        nlos_prob=0.0,
        delay_label_mode="los",
        multipath_scale=0.2,
        scenario_mode="urban8",
        n_uavs=8,
        area_size=(200.0, 260.0),
        normalization_mode="per_observation_noisy_rms",
        urban_base_delay=16,
        urban_min_los=5,
        urban_train_los_only=True,
    )
    x_a_noisy, x_a_clean, x_b_noisy, x_b_clean, delays_a, delays_b = (
        simulator.generate_pair_batch(batch_size, snr_db=snr_db, seed=seed)
    )

    def to_numpy_complex(tensor: object) -> np.ndarray:
        array = tensor.detach().cpu().numpy()
        if array.ndim != 3 or array.shape[1] != 2:
            raise ValueError(f"项目波形应为 [batch, 2, N]，实际为 {array.shape}")
        return array[:, 0, :] + 1j * array[:, 1, :]

    return ProjectPairBatch(
        station_a_noisy=to_numpy_complex(x_a_noisy),
        station_a_clean=to_numpy_complex(x_a_clean),
        station_b_noisy=to_numpy_complex(x_b_noisy),
        station_b_clean=to_numpy_complex(x_b_clean),
        delay_a_minus_b=np.asarray(delays_a, dtype=float)
        - np.asarray(delays_b, dtype=float),
        sample_rate_hz=float(simulator.fs),
    )
