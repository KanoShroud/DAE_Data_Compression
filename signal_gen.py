# signal_gen.py
import numpy as np
import torch
from scipy import signal


def rrc_filter(beta, span, sps):
    """
    Root Raised Cosine (RRC) 脉冲成形滤波器

    参数:
        beta: 滚降系数 (roll-off factor)，典型值 0.22~0.35
        span: 滤波器跨度，以符号数为单位
        sps:  每符号采样点数 (samples per symbol)

    返回:
        h: RRC 滤波器系数 (归一化能量)
    """
    num_taps = span * sps
    t = np.arange(-num_taps // 2, num_taps // 2 + 1) / sps
    h = np.zeros_like(t)

    # t == 0 的情况
    mask_zero = np.abs(t) < 1e-12
    h[mask_zero] = 1.0 + beta * (4.0 / np.pi - 1.0)

    # t == ±1/(4β) 的情况
    mask_edge = np.abs(np.abs(4.0 * beta * t) - 1.0) < 1e-10
    mask_edge &= ~mask_zero
    if np.any(mask_edge):
        h[mask_edge] = (beta / np.sqrt(2.0)
                        * ((1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                           + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))))

    # 一般情况
    mask_normal = ~mask_zero & ~mask_edge
    t_norm = t[mask_normal]
    h[mask_normal] = (
        (np.sin(np.pi * t_norm * (1.0 - beta))
         + 4.0 * beta * t_norm * np.cos(np.pi * t_norm * (1.0 + beta)))
        / (np.pi * t_norm * (1.0 - (4.0 * beta * t_norm) ** 2))
    )

    # 能量归一化
    h /= np.sqrt(np.sum(h ** 2))
    return h


class SignalSimulator:
    """
    信号生成器
    模拟 BPSK 调制的非协作辐射源，经 RRC 脉冲成形、多径信道和接收匹配滤波。
    """

    def __init__(self, signal_len=1024, beta=0.2, rrc_span=8,
                 channel_mode="random", n_fixed_channels=50, channel_pool_seed=42,
                 nlos_prob=0.2, delay_label_mode="strongest",
                 snr_train_range=(-10, 10), multipath_scale=0.3,
                 scenario_mode="sv_pair", n_uavs=8,
                 area_size=(200.0, 260.0), normalization_mode="none",
                 urban_base_delay=16, urban_min_los=4,
                 urban_train_los_only=False):
        """
        参数:
            signal_len:         信号长度（采样点数）
            beta:               RRC 滚降系数 (0.2 → BW = (1+0.2)×20 = 24 MHz)
            rrc_span:           RRC 滤波器跨度（符号数）
            channel_mode:       "random" 每样本随机信道；"fixed" 从预生成信道池中随机选取
            n_fixed_channels:   固定信道池大小（仅 channel_mode="fixed" 时生效）
            channel_pool_seed:  信道池生成种子（仅 channel_mode="fixed" 时生效）
            nlos_prob:          固定信道池中触发 NLOS 遮挡的概率
            delay_label_mode:   "strongest" 使用最强径标签；"los" 使用几何/直达径标签
            snr_train_range:    snr_db=None 时训练样本的 SNR 均匀采样范围
            multipath_scale:    多径幅度缩放，默认保持历史 S-V 设定
            scenario_mode:      "sv_pair" 历史两路信道；"urban8" 论文近似 8 UAV 城市场景
            n_uavs:             urban8 模式下 UAV 数量
            area_size:          urban8 区域尺寸，单位 m
            normalization_mode: "none"、"per_sample_rms"、"per_observation_rms" 或
                                "per_observation_noisy_rms"
            urban_base_delay:   urban8 直达径绝对保护延迟；TDOA 由不同 UAV 延迟差给出，可正可负
            urban_min_los:      urban8 snapshot 至少包含的 LOS UAV 数量
            urban_train_los_only: urban8 训练数据只展开 LOS UAV，与 LOS-only 定位评估对齐
        """
        if delay_label_mode not in ("strongest", "los"):
            raise ValueError("delay_label_mode must be 'strongest' or 'los'")
        if scenario_mode not in ("sv_pair", "urban8"):
            raise ValueError("scenario_mode must be 'sv_pair' or 'urban8'")
        if normalization_mode not in ("none", "per_sample_rms", "per_observation_rms",
                                      "per_observation_noisy_rms"):
            raise ValueError("normalization_mode must be 'none', 'per_sample_rms', "
                             "'per_observation_rms', or 'per_observation_noisy_rms'")
        self.signal_len = signal_len
        self.samples_per_symbol = 2
        self.fs = 40e6
        self.c = 3e8
        self.sample_distance_m = self.c / self.fs
        self.beta = beta
        self.rrc_span = rrc_span
        self.channel_mode = channel_mode
        self.nlos_prob = float(nlos_prob)
        self.delay_label_mode = delay_label_mode
        self.snr_train_range = tuple(snr_train_range)
        self.multipath_scale = float(multipath_scale)
        self.scenario_mode = scenario_mode
        self.n_uavs = int(n_uavs)
        self.area_size = tuple(float(v) for v in area_size)
        self.normalization_mode = normalization_mode
        self.normalization_uses_clean_target = normalization_mode in (
            "per_sample_rms", "per_observation_rms"
        )
        self.urban_base_delay = float(urban_base_delay)
        self.urban_min_los = int(urban_min_los)
        self.urban_train_los_only = bool(urban_train_los_only)
        # RRC 滤波器群延迟（采样点）：mode='same' 引入 (len(rrc)-1)//2 = 8 样本延迟
        # 发射端 + 接收端 RRC 各一次 → 总计 16 样本
        # TDOA 计算中两路抵消，不影响差值；仅在需要绝对延迟时需要补偿
        self._rrc_group_delay = (rrc_span * self.samples_per_symbol) // 2 * 2  # = 16

        # RRC 脉冲成形滤波器（替代原 Butterworth 滤波器）
        self.rrc = rrc_filter(beta=beta, span=rrc_span, sps=self.samples_per_symbol)

        # 固定信道池
        self._channel_pool = None
        self._channel_delays = None
        self._urban_snapshots = None
        self._urban_buildings = None
        if self.scenario_mode == "urban8":
            self._build_urban_snapshots(n_fixed_channels, channel_pool_seed)
        elif channel_mode == "fixed":
            self._build_channel_pool(n_fixed_channels, channel_pool_seed)

    def _apply_rrc(self, x, axis=-1):
        """对信号施加 RRC 匹配滤波，输出长度与输入一致"""
        if x.ndim == 1:
            return signal.convolve(x, self.rrc, mode='same')
        else:
            return np.array([signal.convolve(x[i], self.rrc, mode='same')
                             for i in range(x.shape[0])])

    def _normalize_complex_pair(self, noisy, clean):
        if self.normalization_mode in ("none", "per_observation_rms", "per_observation_noisy_rms"):
            return noisy, clean
        scale = np.sqrt(np.mean(np.abs(clean) ** 2)) + 1e-9
        return noisy / scale, clean / scale

    def _to_tensor(self, x_complex):
        return torch.stack([torch.tensor(x_complex.real),
                            torch.tensor(x_complex.imag)], dim=0).float()

    @staticmethod
    def _segment_intersects_rect(p0, p1, rect):
        x0, y0 = p0
        x1, y1 = p1
        rx0, ry0, rx1, ry1 = rect
        dx = x1 - x0
        dy = y1 - y0
        t0, t1 = 0.0, 1.0
        for p, q in [(-dx, x0 - rx0), (dx, rx1 - x0), (-dy, y0 - ry0), (dy, ry1 - y0)]:
            if abs(p) < 1e-12:
                if q < 0:
                    return False
            else:
                r = q / p
                if p < 0:
                    if r > t1:
                        return False
                    t0 = max(t0, r)
                else:
                    if r < t0:
                        return False
                    t1 = min(t1, r)
        return t0 < t1

    def _point_in_building(self, point):
        x, y = point
        for rx0, ry0, rx1, ry1 in self._urban_buildings:
            if rx0 <= x <= rx1 and ry0 <= y <= ry1:
                return True
        return False

    def _is_los(self, source, uav):
        for rect in self._urban_buildings:
            if self._segment_intersects_rect(source, uav, rect):
                return False
        return True

    def _sample_free_point(self, rng):
        area_w, area_h = self.area_size
        for _ in range(10000):
            pt = np.array([rng.uniform(8.0, area_w - 8.0),
                           rng.uniform(8.0, area_h - 8.0)])
            if not self._point_in_building(pt):
                return pt
        raise RuntimeError("failed to sample a free point in urban scene")

    def _make_urban_buildings(self):
        # 近似论文 200m×260m 城市十字路口：中心十字道路保持开阔，周边放置 19 个矩形建筑。
        rects = [
            (8, 8, 38, 48), (48, 8, 78, 46), (122, 8, 154, 50), (164, 8, 194, 48),
            (8, 60, 38, 100), (48, 58, 78, 104), (122, 60, 154, 102), (164, 58, 194, 104),
            (8, 156, 38, 200), (48, 156, 78, 206), (122, 156, 154, 204), (164, 156, 194, 204),
            (8, 214, 38, 252), (48, 216, 78, 252), (122, 214, 154, 252), (164, 216, 194, 252),
            (88, 8, 112, 58), (88, 202, 112, 252), (82, 112, 118, 148),
        ]
        return [(float(a), float(b), float(c), float(d)) for a, b, c, d in rects]

    def _add_fractional_tap(self, h, delay, gain, half_width=8):
        center = int(np.floor(delay))
        idx = np.arange(center - half_width + 1, center + half_width + 1)
        valid = (idx >= 0) & (idx < len(h))
        idx_valid = idx[valid]
        if idx_valid.size == 0:
            return
        x = idx_valid - delay
        window = np.hamming(idx_valid.size)
        kernel = np.sinc(x) * window
        norm = np.sqrt(np.sum(kernel ** 2)) + 1e-12
        h[idx_valid] += gain * kernel / norm

    def _build_urban_snapshots(self, n_snapshots, seed):
        rng = np.random.default_rng(seed)
        self._urban_buildings = self._make_urban_buildings()
        snapshots = []
        area_w, area_h = self.area_size
        base_angles = np.linspace(0, 2 * np.pi, self.n_uavs, endpoint=False)
        for sid in range(n_snapshots):
            for _ in range(2000):
                source = self._sample_free_point(rng)
                radius = rng.uniform(65.0, 115.0, size=self.n_uavs)
                jitter = rng.normal(0.0, 0.20, size=self.n_uavs)
                uavs = []
                for r, a, j in zip(radius, base_angles, jitter):
                    pt = source + r * np.array([np.cos(a + j), np.sin(a + j)])
                    pt[0] = np.clip(pt[0], 5.0, area_w - 5.0)
                    pt[1] = np.clip(pt[1], 5.0, area_h - 5.0)
                    if self._point_in_building(pt):
                        pt = self._sample_free_point(rng)
                    uavs.append(pt)
                uavs = np.asarray(uavs, dtype=float)
                los = np.asarray([self._is_los(source, u) for u in uavs], dtype=bool)
                if np.sum(los) >= self.urban_min_los:
                    distances = np.linalg.norm(uavs - source[None, :], axis=1)
                    channels, delays, delay_float = [], [], []
                    for ui in range(self.n_uavs):
                        channel_h, label_delay, float_delay = self._urban_channel(
                            distances[ui], bool(los[ui]), rng=rng
                        )
                        channels.append(channel_h)
                        delays.append(label_delay)
                        delay_float.append(float_delay)
                    snapshots.append({
                        'snapshot_id': sid,
                        'source': source,
                        'uavs': uavs,
                        'los': los,
                        'distances': distances,
                        'channels': np.asarray(channels, dtype=complex),
                        'delays': np.asarray(delays, dtype=int),
                        'delay_float': np.asarray(delay_float, dtype=float),
                    })
                    break
            else:
                raise RuntimeError(
                    f"failed to build an urban snapshot with at least {self.urban_min_los} LOS UAVs"
                )
        self._urban_snapshots = snapshots
        los_counts = [int(np.sum(s['los'])) for s in snapshots]
        print(f"[SignalSimulator] urban8 snapshots generated: {len(snapshots)} "
              f"(seed={seed}, LOS count min/mean/max="
              f"{min(los_counts)}/{np.mean(los_counts):.1f}/{max(los_counts)})")

    def _make_base_signal(self, batch_size):
        num_symbols = self.signal_len // self.samples_per_symbol
        bits = np.random.choice([-1, 1], size=(batch_size, num_symbols))
        base_signal = np.repeat(bits, self.samples_per_symbol, axis=1)
        if base_signal.shape[1] < self.signal_len:
            pad = np.zeros((batch_size, self.signal_len - base_signal.shape[1]))
            base_signal = np.hstack([base_signal, pad])
        else:
            base_signal = base_signal[:, :self.signal_len]
        base_signal_shaped = self._apply_rrc(base_signal, axis=1)
        return base_signal_shaped + 1j * np.zeros_like(base_signal_shaped)

    def _urban_channel(self, distance_m, is_los, rng=None):
        rng_uniform = rng.uniform if rng is not None else np.random.uniform
        rng_integers = rng.integers if rng is not None else np.random.randint
        h = np.zeros(self.signal_len, dtype=complex)
        los_delay_float = self.urban_base_delay + distance_m / self.sample_distance_m
        los_delay = int(np.clip(round(los_delay_float), 0, self.signal_len - 1))
        path_gain = (50.0 / max(distance_m, 20.0)) ** 1.2
        direct_gain = path_gain if is_los else path_gain * 0.12
        self._add_fractional_tap(h, los_delay_float, direct_gain)

        n_paths = int(rng_integers(2, 5)) if is_los else int(rng_integers(4, 8))
        for _ in range(n_paths):
            extra = int(rng_integers(3, 38 if is_los else 70))
            tap_delay = min(los_delay_float + extra + rng_uniform(-0.5, 0.5),
                            self.signal_len - 1)
            decay = np.exp(-extra / (18.0 if is_los else 28.0))
            rayleigh = np.sqrt(-2 * np.log(rng_uniform(1e-10, 1.0)))
            nlos_boost = 1.0 if is_los else 2.5
            gain = path_gain * self.multipath_scale * nlos_boost * decay * rayleigh
            phase = rng_uniform(0, 2 * np.pi)
            self._add_fractional_tap(h, tap_delay, gain * np.exp(1j * phase))

        if self.delay_label_mode == "los":
            label_delay = los_delay
        else:
            label_delay = int(np.argmax(np.abs(h)))
        return h, label_delay, los_delay_float

    def _apply_channel_noise_complex(self, u_t, h, snr_db):
        full_conv = signal.convolve(u_t, h, mode='full')
        x_clean_raw = full_conv[:self.signal_len]
        clean_real = self._apply_rrc(x_clean_raw.real)
        clean_imag = self._apply_rrc(x_clean_raw.imag)
        clean_complex = clean_real + 1j * clean_imag
        sig_p = np.mean(np.abs(clean_complex) ** 2)
        noise_p = sig_p / (10 ** (snr_db / 10))
        noise = (np.random.normal(0, 1, self.signal_len)
                 + 1j * np.random.normal(0, 1, self.signal_len)) * np.sqrt(noise_p / 2)
        raw_noisy = x_clean_raw + noise
        noisy_real = self._apply_rrc(raw_noisy.real)
        noisy_imag = self._apply_rrc(raw_noisy.imag)
        noisy_complex = noisy_real + 1j * noisy_imag
        return self._normalize_complex_pair(noisy_complex, clean_complex)

    def generate_urban_batch(self, batch_size, snr_db=None, seed=None, snapshot_indices=None):
        if self.scenario_mode != "urban8":
            raise RuntimeError("generate_urban_batch requires scenario_mode='urban8'")
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if snr_db is not None and np.asarray(snr_db).ndim != 0:
            raise ValueError("snr_db must be a scalar or None")
        if seed is not None:
            rng_state = np.random.get_state()
            np.random.seed(seed)
        else:
            rng_state = None

        if snr_db is None:
            snrs = np.random.uniform(self.snr_train_range[0], self.snr_train_range[1],
                                     size=batch_size)
        else:
            snrs = np.ones(batch_size) * snr_db
        if snapshot_indices is None:
            snapshot_indices = np.random.randint(0, len(self._urban_snapshots), size=batch_size)
        else:
            snapshot_indices = np.asarray(snapshot_indices, dtype=int)
            if snapshot_indices.shape != (batch_size,):
                raise ValueError(
                    "snapshot_indices must contain exactly batch_size entries"
                )
            if np.any(snapshot_indices < 0) or np.any(
                snapshot_indices >= len(self._urban_snapshots)
            ):
                raise IndexError("snapshot_indices contains an out-of-range snapshot")

        u_batch = self._make_base_signal(batch_size)
        noisy = np.zeros((batch_size, self.n_uavs, 2, self.signal_len), dtype=np.float32)
        clean = np.zeros_like(noisy)
        delays = np.zeros((batch_size, self.n_uavs), dtype=int)
        delay_float = np.zeros((batch_size, self.n_uavs), dtype=float)
        sources = np.zeros((batch_size, 2), dtype=float)
        uavs = np.zeros((batch_size, self.n_uavs, 2), dtype=float)
        los_mask = np.zeros((batch_size, self.n_uavs), dtype=bool)
        snapshot_ids = np.zeros(batch_size, dtype=int)

        for bi in range(batch_size):
            snap = self._urban_snapshots[int(snapshot_indices[bi])]
            sources[bi] = snap['source']
            uavs[bi] = snap['uavs']
            los_mask[bi] = snap['los']
            snapshot_ids[bi] = snap['snapshot_id']
            for ui in range(self.n_uavs):
                h = snap['channels'][ui]
                label_delay = int(snap['delays'][ui])
                float_delay = float(snap['delay_float'][ui])
                x_noisy, x_clean = self._apply_channel_noise_complex(u_batch[bi], h, snrs[bi])
                noisy[bi, ui] = np.stack([x_noisy.real, x_noisy.imag], axis=0)
                clean[bi, ui] = np.stack([x_clean.real, x_clean.imag], axis=0)
                delays[bi, ui] = label_delay
                delay_float[bi, ui] = float_delay
            if self.normalization_mode == "per_observation_rms":
                scale = np.sqrt(np.mean(clean[bi] ** 2)) + 1e-9
                noisy[bi] /= scale
                clean[bi] /= scale
            elif self.normalization_mode == "per_observation_noisy_rms":
                scale = np.sqrt(np.mean(noisy[bi] ** 2)) + 1e-9
                noisy[bi] /= scale
                clean[bi] /= scale

        if rng_state is not None:
            np.random.set_state(rng_state)

        meta = {
            'source': sources,
            'uavs': uavs,
            'los': los_mask,
            'distances': np.asarray([self._urban_snapshots[int(si)]['distances']
                                     for si in snapshot_ids], dtype=float),
            'delays': delays,
            'delay_float': delay_float,
            'snapshot_id': snapshot_ids,
            'snr_db': snrs,
            'buildings': self._urban_buildings,
        }
        return torch.tensor(noisy).float(), torch.tensor(clean).float(), meta

    def build_urban_training_plan(self, n_samples, seed=42):
        """
        为 urban8 训练构造固定的 waveform 采样计划。

        计划只固定 snapshot 与 UAV 选择；具体基带符号、SNR 和噪声可在后续按
        不同 seed 重新生成。这样 CV 分组稳定，同时训练集可做轻量重采样增强。
        """
        if self.scenario_mode != "urban8":
            raise RuntimeError("build_urban_training_plan requires scenario_mode='urban8'")

        rng_state = np.random.get_state()
        np.random.seed(seed)

        snapshot_indices = []
        uav_indices = []
        n_snapshots = len(self._urban_snapshots)
        while len(snapshot_indices) < n_samples:
            si = int(np.random.randint(0, n_snapshots))
            snap = self._urban_snapshots[si]
            if self.urban_train_los_only:
                candidates = np.where(snap['los'])[0]
            else:
                candidates = np.arange(self.n_uavs)
            if len(candidates) == 0:
                continue
            # 随机化同一 observation 内的 UAV 展开顺序，避免固定低编号 UAV 偏置。
            candidates = np.random.permutation(candidates)
            for ui in candidates:
                snapshot_indices.append(si)
                uav_indices.append(int(ui))
                if len(snapshot_indices) >= n_samples:
                    break

        np.random.set_state(rng_state)
        return (np.asarray(snapshot_indices, dtype=int),
                np.asarray(uav_indices, dtype=int))

    def build_urban_pair_training_plan(self, n_pairs, seed=42):
        """
        为 urban8 pair-aware 训练构造固定的 snapshot/UAV-pair 计划。

        每个样本对应同一 observation 内的两个 UAV waveform。pair 从 LOS UAV
        中随机抽取，训练目标的 TDOA 使用几何距离差的浮点采样值，和最终
        all-pair WLS 定位评估保持同一物理定义。
        """
        if self.scenario_mode != "urban8":
            raise RuntimeError("build_urban_pair_training_plan requires scenario_mode='urban8'")

        rng_state = np.random.get_state()
        np.random.seed(seed)

        snapshot_indices = []
        uav_i_indices = []
        uav_j_indices = []
        n_snapshots = len(self._urban_snapshots)
        while len(snapshot_indices) < n_pairs:
            si = int(np.random.randint(0, n_snapshots))
            snap = self._urban_snapshots[si]
            if self.urban_train_los_only:
                candidates = np.where(snap['los'])[0]
            else:
                candidates = np.arange(self.n_uavs)
            if len(candidates) < 2:
                continue
            pair = np.random.choice(candidates, size=2, replace=False)
            snapshot_indices.append(si)
            uav_i_indices.append(int(pair[0]))
            uav_j_indices.append(int(pair[1]))

        np.random.set_state(rng_state)
        return (
            np.asarray(snapshot_indices, dtype=int),
            np.asarray(uav_i_indices, dtype=int),
            np.asarray(uav_j_indices, dtype=int),
        )

    def generate_urban_training_dataset_from_plan(self, snapshot_indices, uav_indices,
                                                  seed=42, return_groups=False):
        """
        按固定 snapshot/UAV 计划生成 urban8 单 UAV waveform 训练集。

        每次调用可使用不同 seed 重新采样基带符号、训练 SNR 和噪声；snapshot
        几何与信道保持固定，CV group 仍由 snapshot 决定。
        """
        if self.scenario_mode != "urban8":
            raise RuntimeError("generate_urban_training_dataset_from_plan requires scenario_mode='urban8'")
        snapshot_indices = np.asarray(snapshot_indices, dtype=int)
        uav_indices = np.asarray(uav_indices, dtype=int)
        if len(snapshot_indices) != len(uav_indices):
            raise ValueError("snapshot_indices and uav_indices must have the same length")
        if np.any(snapshot_indices < 0) or np.any(
            snapshot_indices >= len(self._urban_snapshots)
        ):
            raise IndexError("training snapshot index out of range")
        if np.any(uav_indices < 0) or np.any(uav_indices >= self.n_uavs):
            raise IndexError("training UAV index out of range")

        rng_state = np.random.get_state()
        np.random.seed(seed)

        batch_cap = 500
        noisy_list, clean_list = [], []
        for start in range(0, len(snapshot_indices), batch_cap):
            end = min(start + batch_cap, len(snapshot_indices))
            snap_chunk = snapshot_indices[start:end]
            uav_chunk = uav_indices[start:end]
            Xn, Xc, _ = self.generate_urban_batch(len(snap_chunk), snr_db=None,
                                                  snapshot_indices=snap_chunk)
            row_idx = torch.arange(len(snap_chunk), dtype=torch.long)
            uav_idx = torch.tensor(uav_chunk, dtype=torch.long)
            noisy_list.append(Xn[row_idx, uav_idx])
            clean_list.append(Xc[row_idx, uav_idx])

        X_noisy = torch.cat(noisy_list, dim=0)
        X_clean = torch.cat(clean_list, dim=0)
        groups = torch.tensor(snapshot_indices, dtype=torch.long)

        np.random.set_state(rng_state)
        if return_groups:
            return X_noisy, X_clean, groups
        return X_noisy, X_clean

    def generate_urban_pair_training_dataset_from_plan(self, snapshot_indices,
                                                       uav_i_indices, uav_j_indices,
                                                       seed=42, return_groups=False):
        """
        按固定 snapshot/UAV-pair 计划生成 urban8 成对 waveform 训练集。

        返回的 tdoa 为 tau_i_minus_j，单位是 samples，使用几何距离差而非
        整数 delay label，从而和定位评估中的 pairwise TDOA 定义一致。
        """
        if self.scenario_mode != "urban8":
            raise RuntimeError("generate_urban_pair_training_dataset_from_plan requires scenario_mode='urban8'")
        snapshot_indices = np.asarray(snapshot_indices, dtype=int)
        uav_i_indices = np.asarray(uav_i_indices, dtype=int)
        uav_j_indices = np.asarray(uav_j_indices, dtype=int)
        if not (len(snapshot_indices) == len(uav_i_indices) == len(uav_j_indices)):
            raise ValueError(
                "snapshot_indices, uav_i_indices and uav_j_indices must have the same length"
            )
        if np.any(snapshot_indices < 0) or np.any(
            snapshot_indices >= len(self._urban_snapshots)
        ):
            raise IndexError("pair-training snapshot index out of range")
        if (np.any(uav_i_indices < 0) or np.any(uav_i_indices >= self.n_uavs)
                or np.any(uav_j_indices < 0) or np.any(uav_j_indices >= self.n_uavs)):
            raise IndexError("pair-training UAV index out of range")
        if np.any(uav_i_indices == uav_j_indices):
            raise ValueError("pair-training UAV indices must be distinct")

        rng_state = np.random.get_state()
        np.random.seed(seed)

        batch_cap = 500
        x1n_list, x1c_list = [], []
        x2n_list, x2c_list = [], []
        tdoa_list = []
        for start in range(0, len(snapshot_indices), batch_cap):
            end = min(start + batch_cap, len(snapshot_indices))
            snap_chunk = snapshot_indices[start:end]
            ui_chunk = uav_i_indices[start:end]
            uj_chunk = uav_j_indices[start:end]
            Xn, Xc, meta = self.generate_urban_batch(
                len(snap_chunk), snr_db=None, snapshot_indices=snap_chunk
            )
            row_idx = torch.arange(len(snap_chunk), dtype=torch.long)
            ui_idx = torch.tensor(ui_chunk, dtype=torch.long)
            uj_idx = torch.tensor(uj_chunk, dtype=torch.long)
            x1n_list.append(Xn[row_idx, ui_idx])
            x1c_list.append(Xc[row_idx, ui_idx])
            x2n_list.append(Xn[row_idx, uj_idx])
            x2c_list.append(Xc[row_idx, uj_idx])
            distances = np.asarray(meta['distances'], dtype=float)
            rows = np.arange(len(snap_chunk))
            tdoa = (distances[rows, ui_chunk] - distances[rows, uj_chunk]) / (self.c / self.fs)
            tdoa_list.append(torch.tensor(tdoa, dtype=torch.float32))

        X1_noisy = torch.cat(x1n_list, dim=0)
        X1_clean = torch.cat(x1c_list, dim=0)
        X2_noisy = torch.cat(x2n_list, dim=0)
        X2_clean = torch.cat(x2c_list, dim=0)
        tdoa = torch.cat(tdoa_list, dim=0)
        groups = torch.tensor(snapshot_indices, dtype=torch.long)

        np.random.set_state(rng_state)
        if return_groups:
            return X1_noisy, X1_clean, X2_noisy, X2_clean, tdoa, groups
        return X1_noisy, X1_clean, X2_noisy, X2_clean, tdoa

    def _build_channel_pool(self, n_channels, seed):
        """
        预生成 n_channels 条固定多径信道，确保可复现。

        Saleh-Valenzuela 聚簇多径信道模型（匹配论文 Wireless InSite 城市场景）：
        - 多径按簇到达，更接近真实城市散射环境
        - 簇到达服从 Poisson 过程（率 Λ）
        - 簇内径到达服从 Poisson 过程（率 λ，λ > Λ）
        - 簇幅度指数衰减（时间常数 Γ）
        - 簇内径幅度指数衰减（时间常数 γ）
        - 默认 20% 概率 NLOS（最强径不是第一径）；paper_repro 模式可设为 0

        参数说明（采样点为单位，1 采样点 = 25ns @ 40MHz）：
        - Λ = 0.1: 簇到达率，平均簇间距 10 采样点（250ns）
        - λ = 0.5: 簇内径到达率，平均径间距 2 采样点（50ns）
        - Γ = 50: 簇衰减时间常数（1.25μs）
        - γ = 10: 簇内径衰减时间常数（250ns）
        """
        rng_state = np.random.get_state()
        np.random.seed(seed)

        # Saleh-Valenzuela 参数
        Lambda = 0.1      # 簇到达率（每采样点）
        lam = 0.5         # 簇内径到达率（每采样点）
        Gamma = 50        # 簇衰减时间常数（采样点）
        gamma = 10        # 簇内径衰减时间常数（采样点）

        self._channel_pool = []
        self._channel_delays = []
        for _ in range(n_channels):
            main_delay = np.random.randint(10, 50)
            h = np.zeros(self.signal_len, dtype=complex)
            h[main_delay] += 1.0  # LOS 分量

            # 生成簇到达时间（Poisson 过程）
            cluster_delays = []
            t = main_delay + 5
            while t < min(main_delay + 300, self.signal_len):
                cluster_delays.append(t)
                t += np.random.exponential(1.0 / Lambda)

            # 对每个簇生成径
            for cluster_delay in cluster_delays:
                # 簇幅度：指数衰减
                cluster_amp = np.exp(-(cluster_delay - main_delay) / Gamma)

                # 簇内径到达时间（Poisson 过程）
                ray_t = cluster_delay
                while ray_t < min(cluster_delay + 50, self.signal_len):
                    if int(ray_t) < self.signal_len:
                        # 径幅度 = 簇幅度 × 簇内径衰减 × Rayleigh 衰落
                        ray_amp = cluster_amp * np.exp(-(ray_t - cluster_delay) / gamma)
                        rayleigh = np.sqrt(-2 * np.log(np.random.uniform(1e-10, 1.0)))
                        gain = ray_amp * rayleigh * self.multipath_scale
                        phase = np.random.uniform(0, 2 * np.pi)
                        h[int(ray_t)] += gain * np.exp(1j * phase)
                    ray_t += np.random.exponential(1.0 / lam)

            # NLOS：LOS 径被遮挡，NLOS 径成为主导。
            # paper_repro 模式将 nlos_prob 设为 0，并用 LOS 标签对齐论文 LOS UAV 前提。
            if self.nlos_prob > 0 and np.random.random() < self.nlos_prob:
                # LOS 径衰减 20 倍（模拟物理遮挡）
                h[main_delay] *= 0.05
                # 非 LOS 径中选取最强径放大为新的主导
                non_los_gains = [(i, abs(h[i])) for i in range(len(h))
                                 if i != main_delay and abs(h[i]) > 0]
                if non_los_gains:
                    strongest = max(non_los_gains, key=lambda x: x[1])
                    h[strongest[0]] *= 5.0

            if self.delay_label_mode == "los":
                label_delay = main_delay
            else:
                label_delay = int(np.argmax(np.abs(h)))
            self._channel_pool.append(h)
            self._channel_delays.append(label_delay)

        np.random.set_state(rng_state)
        print(f"[SignalSimulator] 固定信道池已生成: {n_channels} 条信道 "
              f"(seed={seed}, nlos_prob={self.nlos_prob}, label={self.delay_label_mode})")

    def generate_pair_batch(self, batch_size, snr_db=None, seed=None):
        """
        生成配对的 UAV 接收信号（两路独立链路）

        参数:
            batch_size: 批量大小
            snr_db:     SNR (dB)。若为 None，训练模式使用 [-10, 10] dB 混合（与评估范围一致）

        返回:
            X1_noisy, X1_clean, X2_noisy, X2_clean, delays1, delays2
        """
        if self.scenario_mode == "urban8":
            X_noisy, X_clean, meta = self.generate_urban_batch(batch_size, snr_db=snr_db,
                                                               seed=seed)
            pair_rng = (
                np.random if seed is None
                else np.random.default_rng(int(seed) + 7919)
            )
            x1n, x1c, x2n, x2c = [], [], [], []
            d1, d2 = [], []
            for i in range(batch_size):
                los_idx = np.where(meta['los'][i])[0]
                if len(los_idx) >= 2:
                    selected = pair_rng.choice(los_idx, size=2, replace=False)
                    i1, i2 = int(selected[0]), int(selected[1])
                else:
                    i1, i2 = 0, 1
                x1n.append(X_noisy[i, i1])
                x1c.append(X_clean[i, i1])
                x2n.append(X_noisy[i, i2])
                x2c.append(X_clean[i, i2])
                d1.append(float(meta['delay_float'][i, i1]))
                d2.append(float(meta['delay_float'][i, i2]))
            return (torch.stack(x1n, dim=0), torch.stack(x1c, dim=0),
                    torch.stack(x2n, dim=0), torch.stack(x2c, dim=0),
                    d1, d2)

        if seed is not None:
            rng_state = np.random.get_state()
            np.random.seed(seed)
        else:
            rng_state = None

        u_t = self._make_base_signal(batch_size)

        def apply_channel_and_noise(u_t_batch):
            # 注意：不保存/恢复RNG状态，确保两次调用(X1和X2)产生不同的随机采样
            X_noisy_out = np.zeros_like(u_t_batch)
            X_clean_out = np.zeros_like(u_t_batch)
            delays = []

            if snr_db is None:
                snrs = np.random.uniform(self.snr_train_range[0], self.snr_train_range[1],
                                         size=batch_size)
            else:
                snrs = np.ones(batch_size) * snr_db

            for i in range(batch_size):
                if self.channel_mode == "fixed":
                    ch_idx = np.random.randint(0, len(self._channel_pool))
                    h = self._channel_pool[ch_idx]
                    main_delay = self._channel_delays[ch_idx]
                else:
                    main_delay = np.random.randint(10, 50)
                    h = np.zeros(self.signal_len, dtype=complex)
                    h[main_delay] = 1.0
                    for _ in range(np.random.randint(2, 4)):
                        tap_delay = np.random.randint(main_delay + 5,
                                                      min(main_delay + 100, self.signal_len))
                        h[tap_delay] = (np.random.uniform(0.1, 0.4) * self.multipath_scale / 0.3
                                        * np.exp(1j * np.random.uniform(0, 2 * np.pi)))
                if self.delay_label_mode == "los":
                    delays.append(main_delay)
                else:
                    delays.append(int(np.argmax(np.abs(h))))

                full_conv = signal.convolve(u_t_batch[i], h, mode='full')
                x_clean_raw = full_conv[:self.signal_len]

                # 接收端 RRC 匹配滤波 → 纯净目标
                clean_real = self._apply_rrc(x_clean_raw.real)
                clean_imag = self._apply_rrc(x_clean_raw.imag)
                clean_complex = clean_real + 1j * clean_imag
                X_clean_out[i] = clean_complex

                # 加噪 — 以 Rx RRC 滤波后的信号功率为 SNR 基准
                # RRC 滤波器单位能量归一化(Σ|h|²=1),白噪声通过后功率不变
                sig_p = np.mean(np.abs(clean_complex) ** 2)
                noise_p = sig_p / (10 ** (snrs[i] / 10))
                noise = (np.random.normal(0, 1, self.signal_len)
                         + 1j * np.random.normal(0, 1, self.signal_len)) * np.sqrt(noise_p / 2)
                raw_noisy = x_clean_raw + noise

                # 接收端 RRC 匹配滤波 → 含噪输入
                noisy_real = self._apply_rrc(raw_noisy.real)
                noisy_imag = self._apply_rrc(raw_noisy.imag)
                noisy_complex = noisy_real + 1j * noisy_imag
                noisy_complex, clean_complex = self._normalize_complex_pair(noisy_complex, clean_complex)
                X_noisy_out[i] = noisy_complex
                X_clean_out[i] = clean_complex

            T_noisy = torch.stack([torch.tensor(X_noisy_out.real),
                                   torch.tensor(X_noisy_out.imag)], dim=1).float()
            T_clean = torch.stack([torch.tensor(X_clean_out.real),
                                   torch.tensor(X_clean_out.imag)], dim=1).float()

            return T_noisy, T_clean, delays

        X1_noisy, X1_clean, delays1 = apply_channel_and_noise(u_t)
        X2_noisy, X2_clean, delays2 = apply_channel_and_noise(u_t)

        if rng_state is not None:
            np.random.set_state(rng_state)

        return X1_noisy, X1_clean, X2_noisy, X2_clean, delays1, delays2

    def generate_training_dataset(self, n_samples, seed=42, return_groups=False):
        """
        生成固定训练/验证数据集，确保实验可复现。

        参数:
            n_samples: 样本总数（论文为 10,000）
            seed:      随机种子

        返回:
            X_noisy:  (n_samples, 2, 1024) 含噪输入张量
            X_clean:  (n_samples, 2, 1024) 纯净目标张量
        """
        rng_state = np.random.get_state()
        np.random.seed(seed)

        # 分批次生成，避免内存溢出
        batch_cap = 500
        noisy_list, clean_list = [], []
        if self.scenario_mode == "urban8":
            snapshot_indices, uav_indices = self.build_urban_training_plan(n_samples, seed=seed)
            X_noisy, X_clean, groups = self.generate_urban_training_dataset_from_plan(
                snapshot_indices, uav_indices, seed=seed, return_groups=True
            )
            np.random.set_state(rng_state)
            if return_groups:
                return X_noisy, X_clean, groups
            return X_noisy, X_clean

        for start in range(0, n_samples, batch_cap):
            end = min(start + batch_cap, n_samples)
            bs = end - start
            X1_n, X1_c, _, _, _, _ = self.generate_pair_batch(bs, snr_db=None)
            noisy_list.append(X1_n)
            clean_list.append(X1_c)

        np.random.set_state(rng_state)

        X_noisy = torch.cat(noisy_list, dim=0)
        X_clean = torch.cat(clean_list, dim=0)
        if return_groups:
            groups = torch.arange(n_samples, dtype=torch.long)
            return X_noisy, X_clean, groups
        return X_noisy, X_clean

    def generate_paired_training_dataset(self, n_samples, seed=42):
        """
        生成配对训练数据集（同一发射波形、两条独立接收链路），用于相关性损失训练。

        返回:
            X1_noisy, X1_clean, X2_noisy, X2_clean: (n_samples, 2, signal_len)
            tdoa: (n_samples,) 真实 TDOA（采样点）
        """
        rng_state = np.random.get_state()
        np.random.seed(seed)

        batch_cap = 500
        x1n_list, x1c_list, x2n_list, x2c_list, tdoa_list = [], [], [], [], []

        for start in range(0, n_samples, batch_cap):
            end = min(start + batch_cap, n_samples)
            bs = end - start
            X1_n, X1_c, X2_n, X2_c, delays1, delays2 = self.generate_pair_batch(bs, snr_db=None)
            x1n_list.append(X1_n)
            x1c_list.append(X1_c)
            x2n_list.append(X2_n)
            x2c_list.append(X2_c)
            tdoa_list.append(torch.tensor(np.array(delays1) - np.array(delays2)))

        np.random.set_state(rng_state)

        return (torch.cat(x1n_list, dim=0), torch.cat(x1c_list, dim=0),
                torch.cat(x2n_list, dim=0), torch.cat(x2c_list, dim=0),
                torch.cat(tdoa_list, dim=0))
