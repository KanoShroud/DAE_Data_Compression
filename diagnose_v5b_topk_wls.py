"""V5-B top-K WLS diagnostic v5 — progress, partial save, optimized beam."""

import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
from evaluate import _localize_from_tdoa_pairs
from signal_gen import SignalSimulator


# PyCharm direct-run settings. The low-SNR gate has passed, so the default now
# performs the full audit run. Command-line arguments remain available for smoke tests.
PYCHARM_RESULT_DIR = "运行结果/压缩域TDOA_V5/20260709_120019_V5B专家门控"
PYCHARM_RUN_MODE = "full"  # smoke | low_gate | full
PYCHARM_PROGRESS_EVERY = 20
PYCHARM_BEAM_MAX_PAIRS = 12
PYCHARM_BEAM_PASSES = 1


def _pair_weight(margin, entropy, variance):
    ec = float(np.clip(1.0 - entropy, 0.0, 1.0))
    w = float(np.clip(
        0.35 * np.clip(10.0 * margin, 0.02, 10.0)
        + 0.20 * np.clip(8.0 / (np.sqrt(max(variance,0.01)) + 0.5), 0.02, 10.0),
        0.02, 10.0))
    w *= float(np.clip(0.25 + 0.75 * ec, 0.02, 1.0))
    return float(np.clip(w, 0.02, 10.0))


def _localize_wls(uav_pos, pairs, lags, fs, c, area_size, weights=None,
                  grid_size=3):
    if len(pairs) < 3:
        return None, float("inf")
    uav_pos = np.asarray(uav_pos, dtype=float)
    pw = np.ones(len(pairs)) if weights is None else np.asarray(weights, dtype=float)
    ok = np.isfinite(lags) & (pw > 0)
    pairs = [pairs[i] for i in range(len(pairs)) if ok[i]]
    lags = np.asarray(lags, dtype=float)[ok]
    pw = pw[ok]
    if len(pairs) < 3:
        return None, float("inf")

    measurements = [
        (int(i), int(j), float(tau), float(w))
        for (i, j), tau, w in zip(pairs, lags, pw)
    ]
    try:
        pos, info = _localize_from_tdoa_pairs(
            uav_pos,
            measurements,
            fs=float(fs),
            c=float(c),
            area_size=area_size,
            robust_mode="standard",
        )
    except Exception:
        return None, float("inf")
    if pos is None or not np.all(np.isfinite(pos)):
        return None, float("inf")
    return pos, _compute_cost(pos, uav_pos, pairs, lags, fs, c, pw)


def _conf(entropy, margin, variance, eps=1e-9):
    return float(margin * max(0.0, 1.0 - entropy) / np.sqrt(max(variance, 0.01) + eps))


def _beam_wls(uav_pos, pairs, lags_list, confs, fs, c, area_size,
              weights=None, eligible_mask=None, max_passes=1,
              beam_max_pairs=12, prev_best=None):
    """Coordinate beam search over distinct peaks of low-confidence pairs."""
    best = [L[0] for L in lags_list]
    init_pos = prev_best
    n_wls_solves = 0
    if init_pos is None:
        init_pos, _ = _localize_wls(uav_pos, pairs, best, fs, c, area_size,
                                     weights=weights, grid_size=1)
        n_wls_solves += 1
    if init_pos is None:
        return None, float("inf"), best, {
            "n_pairs_considered": 0, "n_pairs_changed": 0,
            "n_wls_solves": n_wls_solves,
            "considered_indices": [], "changed_indices": [],
        }
    best_pos, best_cost = init_pos, _compute_cost(
        init_pos, uav_pos, pairs, best, fs, c, weights
    )

    eligible = (
        np.ones(len(pairs), dtype=bool)
        if eligible_mask is None else np.asarray(eligible_mask, dtype=bool)
    )
    if eligible.shape != (len(pairs),):
        raise ValueError("eligible_mask must match the number of pairs")
    order = np.asarray(
        [idx for idx in np.argsort(confs) if eligible[idx]], dtype=int
    )
    trial_indices = order[:min(beam_max_pairs, len(order))]
    changed_indices = set()

    for _ in range(max_passes):
        improved = False
        for idx in trial_indices:
            current_lag = float(best[idx])
            local_best_lag = current_lag
            local_best_pos = best_pos
            local_best_cost = best_cost
            for alt in lags_list[idx][1:3]:
                if not np.isfinite(alt) or abs(float(alt) - current_lag) < 1e-3:
                    continue
                proposal = list(best)
                proposal[idx] = float(alt)
                np_pos, np_cost = _localize_wls(uav_pos, pairs, proposal, fs, c, area_size,
                                                 weights=weights, grid_size=1)
                n_wls_solves += 1
                if (np_pos is not None and np.isfinite(np_cost)
                        and np_cost < local_best_cost * 0.999):
                    local_best_lag = float(alt)
                    local_best_pos = np_pos
                    local_best_cost = float(np_cost)
            if abs(local_best_lag - current_lag) >= 1e-3:
                best[idx] = local_best_lag
                best_pos = local_best_pos
                best_cost = local_best_cost
                changed_indices.add(int(idx))
                improved = True
        if not improved:
            break
    return best_pos, best_cost, best, {
        "n_pairs_considered": int(len(trial_indices)),
        "n_pairs_changed": int(len(changed_indices)),
        "n_wls_solves": int(n_wls_solves),
        "considered_indices": [int(v) for v in trial_indices.tolist()],
        "changed_indices": sorted(int(v) for v in changed_indices),
    }


def _compute_cost(pos, uav_pos, pairs, lags, fs, c, weights=None):
    """Compute WLS residual cost at a given position."""
    w = np.ones(len(pairs)) if weights is None else np.asarray(weights)
    w = w / (np.median(w) + 1e-12)
    w = np.clip(w, 0.05, 20.0)
    dr = np.array(lags, dtype=float) * c / fs
    cost = 0.0
    for (i, j), d, wi in zip(pairs, dr, w):
        r = np.linalg.norm(pos - uav_pos[i]) - np.linalg.norm(pos - uav_pos[j]) - d
        cost += wi * r * r
    return cost


def _paired_bootstrap_rmse_gain(top1_errors, beam_errors, seed=42, n_boot=2000):
    """Paired bootstrap CI for RMSE(Top1) - RMSE(Beam), common-valid only."""
    e1 = np.asarray(top1_errors, dtype=float)
    eb = np.asarray(beam_errors, dtype=float)
    valid = np.isfinite(e1) & np.isfinite(eb)
    e1, eb = e1[valid], eb[valid]
    if e1.size == 0:
        return np.nan, np.nan, np.nan, 0
    point = float(np.sqrt(np.mean(e1 ** 2)) - np.sqrt(np.mean(eb ** 2)))
    rng = np.random.RandomState(int(seed))
    gains = np.empty(int(n_boot), dtype=float)
    for bi in range(int(n_boot)):
        idx = rng.randint(0, e1.size, size=e1.size)
        gains[bi] = (
            np.sqrt(np.mean(e1[idx] ** 2))
            - np.sqrt(np.mean(eb[idx] ** 2))
        )
    return (
        point,
        float(np.percentile(gains, 2.5)),
        float(np.percentile(gains, 97.5)),
        int(e1.size),
    )


def _cluster_bootstrap_gain(data, top1_col, beam_col, cluster_col,
                            metric="rmse", seed=42, n_boot=2000):
    """Paired cluster bootstrap for repeated observations of urban snapshots."""
    required = {top1_col, beam_col, cluster_col}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"cluster bootstrap missing columns: {missing}")
    frame = data[[top1_col, beam_col, cluster_col]].copy()
    valid = (
        np.isfinite(frame[top1_col].to_numpy(dtype=float))
        & np.isfinite(frame[beam_col].to_numpy(dtype=float))
        & frame[cluster_col].notna().to_numpy()
    )
    frame = frame.loc[valid]
    clusters = frame[cluster_col].drop_duplicates().to_numpy()
    if frame.empty or clusters.size == 0:
        return np.nan, np.nan, np.nan, 0, 0

    grouped = {
        cluster: frame[frame[cluster_col] == cluster][[top1_col, beam_col]].to_numpy(dtype=float)
        for cluster in clusters
    }

    def score(values):
        if metric == "rmse":
            return float(np.sqrt(np.mean(values ** 2)))
        if metric == "mae":
            return float(np.mean(np.abs(values)))
        raise ValueError(f"unsupported cluster-bootstrap metric: {metric}")

    top1 = frame[top1_col].to_numpy(dtype=float)
    beam = frame[beam_col].to_numpy(dtype=float)
    point = score(top1) - score(beam)
    rng = np.random.RandomState(int(seed))
    gains = np.empty(int(n_boot), dtype=float)
    for bi in range(int(n_boot)):
        sampled = rng.choice(clusters, size=clusters.size, replace=True)
        blocks = [grouped[cluster] for cluster in sampled]
        values = np.concatenate(blocks, axis=0)
        gains[bi] = score(values[:, 0]) - score(values[:, 1])
    return (
        float(point),
        float(np.percentile(gains, 2.5)),
        float(np.percentile(gains, 97.5)),
        int(clusters.size),
        int(len(frame)),
    )


def _snapshot_indices_from_diag(diag_df, snr_db, n_trials):
    """Return per-trial snapshot ids from top-K CSV when available.

    Older diagnostic CSVs do not contain ``snapshot_id``.  In that case the
    caller must reproduce geometry with the same seeded generator call used by
    ``diagnose_v5b_topk.py``.
    """
    if "snapshot_id" not in diag_df.columns:
        return None
    rows = diag_df[(diag_df["SNR_dB"] == float(snr_db)) & (diag_df["trial_idx"] < int(n_trials))]
    if rows.empty:
        return None
    snap_by_trial = rows.groupby("trial_idx")["snapshot_id"].first()
    if len(snap_by_trial) < int(n_trials):
        missing = sorted(set(range(int(n_trials))) - set(int(v) for v in snap_by_trial.index))
        raise RuntimeError(
            f"snapshot_id preflight failed for SNR={snr_db}: "
            f"missing trial ids {missing[:8]}"
        )
    return [int(snap_by_trial.loc[i]) for i in range(int(n_trials))]


def _check_tdoa_consistency(diag_df, geo, eval_snr, trial_counts,
                            tol=1e-4, max_examples=5):
    """Verify CSV pair labels match rebuilt metadata before expensive WLS."""
    required = {"SNR_dB", "trial_idx", "uav_i", "uav_j", "true_tdoa"}
    missing = sorted(required - set(diag_df.columns))
    if missing:
        raise RuntimeError(f"top-K diagnostic CSV missing required columns: {missing}")

    checked = 0
    failures = []
    for snr_db in eval_snr:
        n_trials = int(trial_counts[float(snr_db)])
        meta = geo[float(snr_db)]
        sub = diag_df[(diag_df["SNR_dB"] == float(snr_db)) & (diag_df["trial_idx"] < int(n_trials))]
        for _, row in sub.iterrows():
            ti = int(row["trial_idx"])
            i = int(row["uav_i"])
            j = int(row["uav_j"])
            csv_tau = float(row["true_tdoa"])
            meta_tau = float(meta["delay_float"][ti, i] - meta["delay_float"][ti, j])
            checked += 1
            if abs(csv_tau - meta_tau) > float(tol):
                if len(failures) < int(max_examples):
                    failures.append((float(snr_db), ti, i, j, csv_tau, meta_tau))
    if failures:
        msg = [
            f"TDOA consistency preflight FAILED: checked={checked}, "
            f"showing {len(failures)} examples"
        ]
        for snr_db, ti, i, j, csv_tau, meta_tau in failures:
            msg.append(
                f"  SNR={snr_db:g}, trial={ti}, pair=({i},{j}), "
                f"csv={csv_tau:.6f}, meta={meta_tau:.6f}, diff={abs(csv_tau-meta_tau):.6g}"
            )
        raise RuntimeError("\n".join(msg))
    print(f"[OK] TDOA preflight passed: checked {checked} CSV pairs", flush=True)
    return checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", nargs="?", default=PYCHARM_RESULT_DIR)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--run-mode", choices=("smoke", "low_gate", "full"),
        default=PYCHARM_RUN_MODE,
    )
    parser.add_argument("--progress-every", type=int, default=PYCHARM_PROGRESS_EVERY)
    parser.add_argument("--beam-max-pairs", type=int, default=PYCHARM_BEAM_MAX_PAIRS)
    parser.add_argument("--beam-passes", type=int, default=PYCHARM_BEAM_PASSES)
    parser.add_argument("--save-partial-every-snr", action="store_true", default=True)
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument(
        "--strict-align", action="store_true",
        help="Raise after saving outputs if Top1 alignment exceeds 1 m."
    )
    args = parser.parse_args()
    result_dir = os.path.abspath(args.result_dir)

    with open(os.path.join(result_dir, "plot_data.pkl"), "rb") as f:
        data = pickle.load(f)
    cfg = data.get("config", {})

    smoke_csv_path = os.path.join(
        result_dir, "tables", "v5b_pair_topk_diagnostics_smoke.csv"
    )
    full_csv_path = os.path.join(
        result_dir, "tables", "v5b_pair_topk_diagnostics.csv"
    )
    csv_path = (
        smoke_csv_path
        if args.smoke and os.path.exists(smoke_csv_path)
        else full_csv_path
    )
    diag_df = pd.read_csv(csv_path)
    required_topk = {
        "estimate_lag", "top2_lag", "top3_lag", "pair_weight",
        "low_confidence_prior", "topk_candidate_mode",
    }
    missing_topk = sorted(required_topk - set(diag_df.columns))
    if missing_topk:
        raise RuntimeError(
            "Top-K CSV uses an obsolete schema. Run diagnose_v5b_topk.py first; "
            f"missing columns: {missing_topk}"
        )
    candidate_modes = set(diag_df["topk_candidate_mode"].dropna().astype(str))
    if candidate_modes != {"distinct_local_maxima_subsample"}:
        raise RuntimeError(
            "Top-K CSV does not contain distinct sub-sample local maxima. "
            "Run diagnose_v5b_topk.py first; "
            f"found modes: {sorted(candidate_modes)}"
        )

    sim = SignalSimulator(
        channel_mode=cfg.get("channel_mode", "fixed"),
        n_fixed_channels=int(cfg.get("n_fixed_channels", 50)),
        channel_pool_seed=cfg.get("seed", 42),
        nlos_prob=float(cfg.get("nlos_prob", 0.0)),
        delay_label_mode=cfg.get("delay_label_mode", "los"),
        snr_train_range=tuple(cfg.get("train_snr_range", (-10, 20))),
        multipath_scale=float(cfg.get("multipath_scale", 0.2)),
        scenario_mode=cfg.get("scenario_mode", "urban8"),
        normalization_mode=cfg.get("normalization_mode", "per_observation_noisy_rms"),
        urban_base_delay=int(cfg.get("urban_base_delay", 16)),
        urban_min_los=int(cfg.get("urban_min_los", 5)),
        urban_train_los_only=bool(cfg.get("urban_train_los_only", True)),
    )
    fs = float(sim.fs)
    c_speed = float(sim.c)
    area_size = (float(sim.area_size[0]), float(sim.area_size[1]))

    full_eval_snr = [
        float(x) for x in cfg.get("eval_snr_range", np.arange(-10, 21, 2))
    ]
    full_n_trials = int(cfg.get("monte_carlo_trials", 200))
    seed = int(cfg.get("seed", 42))

    run_mode = "smoke" if args.smoke else args.run_mode
    if run_mode == "smoke":
        eval_snr = full_eval_snr[:1]
        trial_counts = {float(eval_snr[0]): 20}
    elif run_mode == "low_gate":
        eval_snr = [snr for snr in full_eval_snr if snr <= 0.0]
        eval_snr.extend(snr for snr in (4.0, 12.0) if snr in full_eval_snr)
        trial_counts = {
            float(snr): (
                full_n_trials if snr <= 0.0 else min(50, full_n_trials)
            )
            for snr in eval_snr
        }
    else:
        eval_snr = list(full_eval_snr)
        trial_counts = {float(snr): full_n_trials for snr in eval_snr}

    # ── config summary ───────────────────────────────────────────────────────
    total_samples = int(sum(trial_counts.values()))
    print(f"[Config] run_mode={run_mode}", flush=True)
    print(f"[Config] top-K CSV={csv_path}", flush=True)
    print(f"[Config] SNR range: {len(eval_snr)} points ({eval_snr[0]:.0f}..{eval_snr[-1]:.0f} dB)", flush=True)
    print(f"[Config] Trials by SNR: {trial_counts}  |  Total samples: {total_samples}", flush=True)
    print(f"[Config] beam_max_pairs={args.beam_max_pairs}  beam_passes={args.beam_passes}  "
          f"grid_size={args.grid_size}  progress_every={args.progress_every}", flush=True)
    print(f"[Config] save_partial_every_snr={args.save_partial_every_snr}", flush=True)
    print(f"[Config] area={area_size}", flush=True)
    geom_mode = "CSV snapshot_id" if "snapshot_id" in diag_df.columns else "seeded generator"
    print(f"[Config] geometry reconstruction: {geom_mode}", flush=True)

    print(f"[WLS] Generating geometry for {total_samples} total trials...", flush=True)
    geo = {}
    for snr_db in eval_snr:
        n_trials = int(trial_counts[float(snr_db)])
        snap_idx = _snapshot_indices_from_diag(diag_df, snr_db, n_trials)
        _, _, meta = sim.generate_urban_batch(
            n_trials, snr_db=float(snr_db), snapshot_indices=snap_idx, seed=seed)
        geo[snr_db] = meta

    _check_tdoa_consistency(diag_df, geo, eval_snr, trial_counts)

    methods = ["Top1-WLS", "Top3-BeamWLS"]
    all_rows = []
    all_worst = []
    all_pair_audit = []
    tdoa_assert_fails = 0
    t_start = time.time()
    done_global = 0

    tables_dir = os.path.join(result_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    prefix = {
        "smoke": "v5b_topk_wls_smoke",
        "low_gate": "v5b_topk_wls_gate",
        "full": "v5b_topk_wls",
    }[run_mode]
    top1_lag_col = "estimate_lag" if "estimate_lag" in diag_df.columns else "top1_lag"
    print(
        f"[Config] localizer=evaluate._localize_from_tdoa_pairs(standard), "
        f"top1_lag_col={top1_lag_col}",
        flush=True,
    )

    for snr_idx, snr_db in enumerate(eval_snr):
        n_trials = int(trial_counts[float(snr_db)])
        t_snr_start = time.time()
        print(f"\n[SNR {snr_db:5.0f} dB] start  ({snr_idx+1}/{len(eval_snr)})", flush=True)
        meta = geo[snr_db]
        snr_rows = []
        snr_worst = []

        for ti in range(n_trials):
            trial_started = time.time()
            done_global += 1
            los = meta["los"][ti]
            los_idx = np.where(los)[0]
            if len(los_idx) < 2:
                continue
            src_true = meta["source"][ti]
            uav_pos = meta["uavs"][ti]
            dfloat = meta["delay_float"][ti]

            pr = diag_df[(diag_df["SNR_dB"] == snr_db) & (diag_df["trial_idx"] == ti)]
            if len(pr) < 3:
                continue

            pairs = []
            lags_list = []
            ws = []
            confs = []
            low_conf_flags = []
            pair_source_rows = []
            for _, r in pr.iterrows():
                i, j = int(r["uav_i"]), int(r["uav_j"])
                true_csv = float(r["true_tdoa"])
                true_meta = float(dfloat[i] - dfloat[j])
                if abs(true_csv - true_meta) > 1e-4:
                    tdoa_assert_fails += 1
                l1_raw = r.get(top1_lag_col, np.nan)
                l1 = float(l1_raw) if pd.notna(l1_raw) else float(r["top1_lag"])
                l2 = float(r["top2_lag"]) if not pd.isna(r["top2_lag"]) else l1
                l3 = float(r["top3_lag"]) if not pd.isna(r["top3_lag"]) else l1
                pairs.append((i, j))
                lags_list.append([l1, l2, l3])
                pw_csv = r.get("pair_weight")
                if pd.notna(pw_csv) and np.isfinite(float(pw_csv)) and float(pw_csv) > 0:
                    w = float(pw_csv)
                else:
                    w = _pair_weight(float(r.get("peak_margin", 0.1)),
                                     float(r.get("posterior_entropy", 0.5)),
                                     float(r.get("posterior_variance", 1.0)))
                ws.append(w)
                confs.append(_conf(float(r.get("posterior_entropy", 0.5)),
                                   float(r.get("peak_margin", 0.1)),
                                   float(r.get("posterior_variance", 1.0))))
                low_conf_raw = r.get("low_confidence_prior", False)
                low_conf_flags.append(
                    bool(low_conf_raw) if pd.notna(low_conf_raw) else False
                )
                pair_source_rows.append(r.to_dict())
            ws_arr = np.asarray(ws, dtype=float)
            confs_arr = np.asarray(confs, dtype=float)
            low_conf_arr = np.asarray(low_conf_flags, dtype=bool)

            # Top1
            top1_lags = [L[0] for L in lags_list]
            p1, top1_cost = _localize_wls(
                uav_pos, pairs, top1_lags, fs, c_speed,
                area_size, weights=ws_arr, grid_size=args.grid_size,
            )
            e1 = float(np.linalg.norm(p1 - src_true)) if p1 is not None else np.nan

            # Beam only considers pairs already marked low-confidence by V5-B.
            p3b, beam_cost, beam_lags, beam_info = _beam_wls(
                uav_pos, pairs, lags_list, confs_arr, fs, c_speed, area_size,
                weights=ws_arr, eligible_mask=low_conf_arr,
                max_passes=args.beam_passes,
                beam_max_pairs=args.beam_max_pairs, prev_best=p1,
            )
            e3b = float(np.linalg.norm(p3b - src_true)) if p3b is not None else np.nan
            trial_elapsed = float(time.time() - trial_started)
            snapshot_id = int(
                meta["snapshot_id"][ti]
                if "snapshot_id" in meta else pair_source_rows[0].get("snapshot_id", -1)
            )

            common = {
                "SNR_dB": snr_db,
                "trial_idx": ti,
                "snapshot_id": snapshot_id,
                "n_pairs_total": len(pairs),
                "n_pairs_kept": len(pairs),
                "n_pairs_rejected": 0,
            }
            method_rows = [
                ("Top1-WLS", e1, p1 is not None, 0, 0, 1),
                (
                    "Top3-BeamWLS", e3b, p3b is not None,
                    beam_info["n_pairs_considered"],
                    beam_info["n_pairs_changed"],
                    1 + beam_info["n_wls_solves"],
                ),
            ]
            for mn, e, success, n_considered, n_changed, n_wls_solves in method_rows:
                r2 = dict(common)
                r2.update({
                    "method": mn,
                    "error_m": float(e),
                    "solver_success": bool(success),
                    "n_pairs_considered": int(n_considered),
                    "n_pairs_changed": int(n_changed),
                    "n_wls_solves": int(n_wls_solves),
                    "trial_elapsed_s": trial_elapsed,
                })
                snr_rows.append(r2)
                all_rows.append(r2)
            snr_worst.append({
                "SNR_dB": snr_db, "trial_idx": ti, "n_pairs": len(pairs),
                "snapshot_id": snapshot_id,
                "top1_rmse_m": float(e1), "top3_beam_rmse_m": float(e3b),
                "top1_success": bool(p1 is not None),
                "top3_beam_success": bool(p3b is not None),
                "n_low_confidence_pairs": int(np.sum(low_conf_arr)),
                "n_pairs_changed": int(beam_info["n_pairs_changed"]),
                "n_wls_solves": int(1 + beam_info["n_wls_solves"]),
                "top1_wls_cost": float(top1_cost),
                "top3_beam_wls_cost": float(beam_cost),
                "trial_elapsed_s": trial_elapsed,
            })
            all_worst.append(snr_worst[-1])

            considered_set = set(beam_info["considered_indices"])
            for pair_idx, ((ui, uj), candidates, source_row) in enumerate(zip(
                    pairs, lags_list, pair_source_rows)):
                selected_lag = float(beam_lags[pair_idx])
                candidate_arr = np.asarray(candidates, dtype=float)
                selected_rank = int(np.argmin(np.abs(candidate_arr - selected_lag))) + 1
                true_tdoa = float(source_row["true_tdoa"])
                selected_prob_raw = source_row.get(f"top{selected_rank}_prob", np.nan)
                selected_prob = (
                    float(selected_prob_raw) if pd.notna(selected_prob_raw) else np.nan
                )
                top1_abs_error = abs(float(candidates[0]) - true_tdoa)
                beam_abs_error = abs(selected_lag - true_tdoa)
                all_pair_audit.append({
                    "SNR_dB": float(snr_db),
                    "trial_idx": int(ti),
                    "snapshot_id": snapshot_id,
                    "uav_i": int(ui),
                    "uav_j": int(uj),
                    "true_tdoa": true_tdoa,
                    "top1_lag": float(candidates[0]),
                    "top2_lag": float(candidates[1]),
                    "top3_lag": float(candidates[2]),
                    "beam_selected_lag": selected_lag,
                    "beam_selected_rank": selected_rank,
                    "beam_changed": bool(
                        abs(selected_lag - float(candidates[0])) >= 1e-3
                    ),
                    "beam_considered": bool(pair_idx in considered_set),
                    "top1_abs_error_samples": float(top1_abs_error),
                    "beam_abs_error_samples": float(beam_abs_error),
                    "tdoa_abs_error_gain_samples": float(top1_abs_error - beam_abs_error),
                    "top1_prob": float(source_row.get("top1_prob", np.nan)),
                    "top2_prob": float(source_row.get("top2_prob", np.nan)),
                    "top3_prob": float(source_row.get("top3_prob", np.nan)),
                    "beam_selected_prob": selected_prob,
                    "posterior_entropy": float(source_row.get("posterior_entropy", np.nan)),
                    "posterior_variance": float(source_row.get("posterior_variance", np.nan)),
                    "peak_margin": float(source_row.get("peak_margin", np.nan)),
                    "pair_weight": float(ws_arr[pair_idx]),
                    "low_confidence_prior": bool(low_conf_arr[pair_idx]),
                    "top1_localization_error_m": float(e1),
                    "beam_localization_error_m": float(e3b),
                    "top1_wls_cost": float(top1_cost),
                    "beam_wls_cost": float(beam_cost),
                    "top1_est_x": float(p1[0]) if p1 is not None else np.nan,
                    "top1_est_y": float(p1[1]) if p1 is not None else np.nan,
                    "beam_est_x": float(p3b[0]) if p3b is not None else np.nan,
                    "beam_est_y": float(p3b[1]) if p3b is not None else np.nan,
                    "source_x": float(src_true[0]),
                    "source_y": float(src_true[1]),
                    "sample_n_pairs_changed": int(beam_info["n_pairs_changed"]),
                    "sample_n_wls_solves": int(1 + beam_info["n_wls_solves"]),
                    "sample_elapsed_s": trial_elapsed,
                })

            if done_global % args.progress_every == 0:
                elapsed = time.time() - t_start
                avg_per = elapsed / max(done_global, 1)
                eta = avg_per * (total_samples - done_global)
                n_pairs_this = len(pairs)
                print(f"  [{snr_db:5.0f} dB] trial {ti+1:3d}/{n_trials}  "
                      f"done {done_global}/{total_samples}  "
                      f"pairs={n_pairs_this}  "
                      f"elapsed={elapsed:.0f}s  avg={avg_per:.1f}s/trial  ETA={eta:.0f}s",
                      flush=True)

        # ── save partial after each SNR ──────────────────────────────────────
        if args.save_partial_every_snr and all_rows:
            pd.DataFrame(all_rows).to_csv(
                os.path.join(tables_dir, f"{prefix}_partial_rows.csv"),
                index=False, encoding="utf-8")
            pd.DataFrame(all_worst).to_csv(
                os.path.join(tables_dir, f"{prefix}_partial_worst.csv"),
                index=False, encoding="utf-8")
            pd.DataFrame(all_pair_audit).to_csv(
                os.path.join(tables_dir, f"{prefix}_partial_pair_decisions.csv"),
                index=False, encoding="utf-8")
        t_snr_elapsed = time.time() - t_snr_start
        print(f"[SNR {snr_db:5.0f} dB] done  elapsed={t_snr_elapsed:.0f}s  "
              f"({t_snr_elapsed/max(n_trials,1):.1f}s/trial)", flush=True)

    if tdoa_assert_fails > 0:
        raise RuntimeError(
            f"TDOA consistency check FAILED: {tdoa_assert_fails} pairs have "
            f"|CSV true_tdoa - meta delay_float| > 1e-4.")
    print("[OK] All TDOA assertions passed", flush=True)

    res_df = pd.DataFrame(all_rows)
    pair_audit_df = pd.DataFrame(all_pair_audit)

    def _rmse_stats(errors):
        raw = np.asarray(errors, dtype=float)
        e = raw[np.isfinite(raw)]
        n_total = len(raw)
        n_valid = len(e)
        if n_valid == 0:
            return {"rmse": np.nan, "median": np.nan, "trimmed_rmse": np.nan,
                    "outlier_gt20_pct": np.nan, "outlier_gt50_pct": np.nan,
                    "valid_count": 0, "failure_count": n_total,
                    "failure_rate": 1.0 if n_total else np.nan}
        return {"rmse": float(np.sqrt(np.mean(e**2))),
                "median": float(np.median(e)),
                "trimmed_rmse": float(np.sqrt(np.mean(
                    np.sort(e)[:max(1, int(n_valid * 0.95))] ** 2
                ))),
                "outlier_gt20_pct": float(np.mean(e > 20)*100),
                "outlier_gt50_pct": float(np.mean(e > 50)*100),
                "valid_count": n_valid,
                "failure_count": n_total - n_valid,
                "failure_rate": float((n_total - n_valid) / max(n_total, 1))}

    summary_rows = []
    for (snr, method), grp in res_df.groupby(["SNR_dB", "method"]):
        s = _rmse_stats(grp["error_m"].values)
        s["SNR_dB"] = snr
        s["method"] = method
        s["mean_n_pairs_total"] = float(grp["n_pairs_total"].mean())
        s["mean_n_pairs_kept"] = float(grp["n_pairs_kept"].mean())
        s["mean_n_pairs_rejected"] = float(grp["n_pairs_rejected"].mean())
        s["mean_n_pairs_considered"] = float(grp["n_pairs_considered"].mean())
        s["mean_n_pairs_changed"] = float(grp["n_pairs_changed"].mean())
        summary_rows.append(s)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(tables_dir, f"{prefix}_baselines.csv"), index=False, encoding="utf-8")

    diag = res_df.groupby(["SNR_dB", "method"]).agg(
        rmse=("error_m", lambda x: _rmse_stats(x)["rmse"]),
        median_error_m=("error_m", "median"),
        valid_count=("solver_success", "sum"),
        failure_rate=("solver_success", lambda x: 1.0 - float(np.mean(x))),
        mean_n_pairs_total=("n_pairs_total", "mean"),
        mean_n_pairs_kept=("n_pairs_kept", "mean"),
        mean_n_pairs_rejected=("n_pairs_rejected", "mean"),
        mean_n_pairs_considered=("n_pairs_considered", "mean"),
        mean_n_pairs_changed=("n_pairs_changed", "mean"),
    ).reset_index()
    diag.to_csv(os.path.join(tables_dir, f"{prefix}_method_diagnostics.csv"), index=False, encoding="utf-8")

    pd.DataFrame(all_worst).to_csv(
        os.path.join(tables_dir, f"{prefix}_worst_localization_errors.csv"), index=False, encoding="utf-8")
    pair_audit_path = os.path.join(tables_dir, f"{prefix}_pair_decisions.csv")
    pair_audit_df.to_csv(pair_audit_path, index=False, encoding="utf-8")

    snr_arr = res_df["SNR_dB"].values
    buckets = np.full(len(res_df), "other", dtype=object)
    buckets[snr_arr <= 0.0] = "low"
    buckets[(snr_arr >= 2.0) & (snr_arr <= 6.0)] = "mid"
    buckets[snr_arr >= 8.0] = "high"
    res_df["snr_bucket"] = buckets
    bucket_rows = []
    for (bk, method), grp in res_df.groupby(["snr_bucket", "method"]):
        s = _rmse_stats(grp["error_m"].values)
        s["snr_bucket"] = bk
        s["method"] = method
        bucket_rows.append(s)
    bs = pd.DataFrame(bucket_rows)
    bs.to_csv(os.path.join(tables_dir, f"{prefix}_summary_by_snr_bucket.csv"), index=False, encoding="utf-8")

    def _tdoa_stats(abs_errors):
        values = np.asarray(abs_errors, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return {
                "mae_samples": np.nan, "rmse_samples": np.nan,
                "median_samples": np.nan, "p95_samples": np.nan,
                "within_1_rate": np.nan, "within_2_rate": np.nan,
                "pair_count": 0,
            }
        return {
            "mae_samples": float(np.mean(values)),
            "rmse_samples": float(np.sqrt(np.mean(values ** 2))),
            "median_samples": float(np.median(values)),
            "p95_samples": float(np.percentile(values, 95.0)),
            "within_1_rate": float(np.mean(values <= 1.0)),
            "within_2_rate": float(np.mean(values <= 2.0)),
            "pair_count": int(values.size),
        }

    tdoa_rows = []
    tdoa_methods = [
        ("V5B-Top1-TDOA", "top1_abs_error_samples"),
        ("V5B-Top3-GeoBeam-TDOA", "beam_abs_error_samples"),
    ]
    for snr_db, grp in pair_audit_df.groupby("SNR_dB"):
        for method, error_col in tdoa_methods:
            item = _tdoa_stats(grp[error_col].values)
            item.update({"SNR_dB": float(snr_db), "method": method})
            tdoa_rows.append(item)
    tdoa_summary = pd.DataFrame(tdoa_rows)
    tdoa_summary.to_csv(
        os.path.join(tables_dir, f"{prefix}_tdoa_metrics_by_snr.csv"),
        index=False, encoding="utf-8",
    )

    pair_snr = pair_audit_df["SNR_dB"].to_numpy(dtype=float)
    pair_buckets = np.full(len(pair_audit_df), "other", dtype=object)
    pair_buckets[pair_snr <= 0.0] = "low"
    pair_buckets[(pair_snr >= 2.0) & (pair_snr <= 6.0)] = "mid"
    pair_buckets[pair_snr >= 8.0] = "high"
    pair_audit_df["snr_bucket"] = pair_buckets
    pair_audit_df.to_csv(pair_audit_path, index=False, encoding="utf-8")
    tdoa_bucket_rows = []
    for bucket, grp in pair_audit_df.groupby("snr_bucket"):
        for method, error_col in tdoa_methods:
            item = _tdoa_stats(grp[error_col].values)
            item.update({"snr_bucket": bucket, "method": method})
            tdoa_bucket_rows.append(item)
    tdoa_bucket_summary = pd.DataFrame(tdoa_bucket_rows)
    tdoa_bucket_summary.to_csv(
        os.path.join(tables_dir, f"{prefix}_tdoa_metrics_by_snr_bucket.csv"),
        index=False, encoding="utf-8",
    )

    complexity = (
        res_df[res_df["method"] == "Top3-BeamWLS"]
        .groupby("SNR_dB")
        .agg(
            sample_count=("trial_idx", "count"),
            mean_pairs_total=("n_pairs_total", "mean"),
            mean_pairs_considered=("n_pairs_considered", "mean"),
            mean_pairs_changed=("n_pairs_changed", "mean"),
            changed_sample_rate=("n_pairs_changed", lambda x: float(np.mean(x > 0))),
            mean_wls_solves=("n_wls_solves", "mean"),
            max_wls_solves=("n_wls_solves", "max"),
            mean_elapsed_s=("trial_elapsed_s", "mean"),
            p95_elapsed_s=("trial_elapsed_s", lambda x: float(np.percentile(x, 95.0))),
        )
        .reset_index()
    )
    complexity.to_csv(
        os.path.join(tables_dir, f"{prefix}_complexity_by_snr.csv"),
        index=False, encoding="utf-8",
    )

    print(f"\n[Saved] {tables_dir}/{prefix}_*.csv", flush=True)
    print("\n" + "=" * 84, flush=True)
    print("V5-B Top-K WLS  |  low: SNR<=0  mid: 2-6  high: >=8", flush=True)
    print("=" * 84, flush=True)
    print(f"{'Method':<22s} {'Bucket':<8s} {'RMSE':>7s} {'Median':>7s} "
          f"{'Trim':>7s} {'>20m':>6s} {'>50m':>6s} {'Valid':>6s} {'Fail':>6s}",
          flush=True)
    print("-" * 84, flush=True)
    for method in methods:
        for bk_key, bk_label in [("low","low"),("mid","mid"),("high","high")]:
            sub = bs[(bs["method"]==method) & (bs["snr_bucket"]==bk_key)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            print(f"{method:<22s} {bk_label:<8s} {r['rmse']:7.2f} {r['median']:7.2f} "
                  f"{r['trimmed_rmse']:7.2f} {r['outlier_gt20_pct']:5.1f}% "
                  f"{r['outlier_gt50_pct']:5.1f}% {int(r['valid_count']):6d} "
                  f"{100.0*r['failure_rate']:5.1f}%",
                  flush=True)
        grp_all = res_df[res_df["method"]==method]["error_m"].values
        s_all = _rmse_stats(grp_all)
        print(f"{method:<22s} {'all':<8s} {s_all['rmse']:7.2f} {s_all['median']:7.2f} "
              f"{s_all['trimmed_rmse']:7.2f} {s_all['outlier_gt20_pct']:5.1f}% "
              f"{s_all['outlier_gt50_pct']:5.1f}% {int(s_all['valid_count']):6d} "
              f"{100.0*s_all['failure_rate']:5.1f}%",
              flush=True)
        print("-" * 84, flush=True)

    v5b_ref = data.get("fig9_results")
    alignment_failed = False
    if v5b_ref is not None:
        ref_results, ref_snr = v5b_ref
        ref_methods = ref_results.get("methods", {})
        v5b_method = ref_methods.get("V5B-Expert64", {})
        v5b_rmse = v5b_method.get("rmse", [])
        v5b_valid = v5b_method.get("valid_count", [])
        top1_sum = summary[summary["method"] == "Top1-WLS"]
        align_rows = []
        for snr in eval_snr:
            if int(trial_counts[float(snr)]) != full_n_trials:
                continue
            sub = top1_sum[top1_sum["SNR_dB"] == snr]
            if not sub.empty:
                align_rows.append({
                    "SNR_dB": float(snr),
                    "top1_wls_rmse": float(sub.iloc[0]["rmse"]),
                    "top1_valid_count": int(sub.iloc[0]["valid_count"]),
                })
        if v5b_rmse and align_rows:
            ref_map = {
                float(snr): float(rmse)
                for snr, rmse in zip(np.asarray(ref_snr, dtype=float), v5b_rmse)
            }
            ref_valid_map = {
                float(snr): int(count)
                for snr, count in zip(np.asarray(ref_snr, dtype=float), v5b_valid)
            } if v5b_valid else {}
            for row in align_rows:
                ref_value = ref_map.get(float(row["SNR_dB"]), np.nan)
                row["saved_v5b_rmse"] = ref_value
                row["abs_diff_m"] = abs(row["top1_wls_rmse"] - ref_value)
                row["saved_v5b_valid_count"] = ref_valid_map.get(
                    float(row["SNR_dB"]), np.nan
                )
                row["valid_count_diff"] = (
                    row["top1_valid_count"] - row["saved_v5b_valid_count"]
                    if np.isfinite(row["saved_v5b_valid_count"]) else np.nan
                )
            align_df = pd.DataFrame(align_rows)
            align_path = os.path.join(tables_dir, f"{prefix}_alignment.csv")
            align_df.to_csv(align_path, index=False, encoding="utf-8")
            md = float(np.nanmax(align_df["abs_diff_m"].values))
            max_valid_diff = float(np.nanmax(np.abs(align_df["valid_count_diff"].values)))
            print(f"\n[Align] Top1-WLS vs V5B-Expert64 max per-SNR RMSE diff = {md:.4f} m", flush=True)
            print(f"[Align] max valid-count diff = {max_valid_diff:.0f}", flush=True)
            print(f"[Align] Saved per-SNR alignment diagnostics: {align_path}", flush=True)
            alignment_failed = bool(md > 1.0 or max_valid_diff > 0.0)
            if alignment_failed:
                print(
                    "[Align] NO-GO: alignment exceeds 1 m or valid-count sets differ. "
                    "Beam results remain diagnostic only.",
                    flush=True,
                )
            else:
                print("[Align] PASSED: Top1-WLS matches V5B-Expert64 within 1 m.", flush=True)

    # Paired, common-valid low-SNR gate. This prevents one-sided solver failures
    # or a few catastrophic outliers from masquerading as a stable improvement.
    wide = res_df.pivot_table(
        index=["SNR_dB", "trial_idx"], columns="method",
        values="error_m", aggfunc="first",
    )
    low_wide = wide[wide.index.get_level_values("SNR_dB") <= 0.0]
    gain, ci_low, ci_high, common_valid = _paired_bootstrap_rmse_gain(
        low_wide.get("Top1-WLS", pd.Series(dtype=float)).values,
        low_wide.get("Top3-BeamWLS", pd.Series(dtype=float)).values,
        seed=seed + 30000,
    )
    sample_audit_df = pd.DataFrame(all_worst)
    low_sample_audit = sample_audit_df[sample_audit_df["SNR_dB"] <= 0.0]
    (
        cluster_loc_gain,
        cluster_loc_ci_low,
        cluster_loc_ci_high,
        cluster_loc_count,
        cluster_loc_rows,
    ) = _cluster_bootstrap_gain(
        low_sample_audit,
        "top1_rmse_m",
        "top3_beam_rmse_m",
        "snapshot_id",
        metric="rmse",
        seed=seed + 31000,
    )
    low_pair_audit = pair_audit_df[pair_audit_df["SNR_dB"] <= 0.0]
    (
        cluster_tdoa_mae_gain,
        cluster_tdoa_mae_ci_low,
        cluster_tdoa_mae_ci_high,
        cluster_tdoa_count,
        cluster_tdoa_rows,
    ) = _cluster_bootstrap_gain(
        low_pair_audit,
        "top1_abs_error_samples",
        "beam_abs_error_samples",
        "snapshot_id",
        metric="mae",
        seed=seed + 32000,
    )
    (
        cluster_tdoa_rmse_gain,
        cluster_tdoa_rmse_ci_low,
        cluster_tdoa_rmse_ci_high,
        _,
        _,
    ) = _cluster_bootstrap_gain(
        low_pair_audit,
        "top1_abs_error_samples",
        "beam_abs_error_samples",
        "snapshot_id",
        metric="rmse",
        seed=seed + 33000,
    )

    def _bucket_row(bucket, method):
        sub = bs[(bs["snr_bucket"] == bucket) & (bs["method"] == method)]
        return None if sub.empty else sub.iloc[0]

    low_top1 = _bucket_row("low", "Top1-WLS")
    low_beam = _bucket_row("low", "Top3-BeamWLS")
    checks = {
        "low_gain_ge_3m": bool(np.isfinite(gain) and gain >= 3.0),
        "low_gain_ci_positive": bool(np.isfinite(ci_low) and ci_low > 0.0),
        "low_cluster_loc_ci_positive": bool(
            np.isfinite(cluster_loc_ci_low) and cluster_loc_ci_low > 0.0
        ),
        "low_cluster_tdoa_mae_ci_positive": bool(
            np.isfinite(cluster_tdoa_mae_ci_low) and cluster_tdoa_mae_ci_low > 0.0
        ),
        "low_trim_not_worse": bool(
            low_top1 is not None and low_beam is not None
            and low_beam["trimmed_rmse"] <= low_top1["trimmed_rmse"]
        ),
        "low_failure_not_worse": bool(
            low_top1 is not None and low_beam is not None
            and low_beam["failure_rate"] <= low_top1["failure_rate"] + 1e-12
        ),
        "alignment_passed": not alignment_failed,
    }
    for bucket in ("mid", "high"):
        r1 = _bucket_row(bucket, "Top1-WLS")
        rb = _bucket_row(bucket, "Top3-BeamWLS")
        if r1 is None or rb is None:
            checks[f"{bucket}_rmse_within_5pct"] = run_mode == "smoke"
            checks[f"{bucket}_trim_within_5pct"] = run_mode == "smoke"
            continue
        checks[f"{bucket}_rmse_within_5pct"] = bool(
            rb["rmse"] <= 1.05 * r1["rmse"]
        )
        checks[f"{bucket}_trim_within_5pct"] = bool(
            rb["trimmed_rmse"] <= 1.05 * r1["trimmed_rmse"]
        )

    method_check_names = [key for key in checks if key != "alignment_passed"]
    gate_evaluated = run_mode != "smoke"
    method_gate_passed = bool(
        all(checks[key] for key in method_check_names)
    ) if gate_evaluated else False
    protocol_gate_passed = bool(checks["alignment_passed"])
    gate_passed = bool(
        method_gate_passed and protocol_gate_passed
    ) if run_mode != "smoke" else False
    gate_rows = [{
        "run_mode": run_mode,
        "gate_evaluated": gate_evaluated,
        "low_common_valid": common_valid,
        "low_rmse_gain_m": gain,
        "low_rmse_gain_ci95_low_m": ci_low,
        "low_rmse_gain_ci95_high_m": ci_high,
        "low_cluster_loc_gain_m": cluster_loc_gain,
        "low_cluster_loc_gain_ci95_low_m": cluster_loc_ci_low,
        "low_cluster_loc_gain_ci95_high_m": cluster_loc_ci_high,
        "low_cluster_loc_count": cluster_loc_count,
        "low_cluster_loc_rows": cluster_loc_rows,
        "low_cluster_tdoa_mae_gain_samples": cluster_tdoa_mae_gain,
        "low_cluster_tdoa_mae_gain_ci95_low_samples": cluster_tdoa_mae_ci_low,
        "low_cluster_tdoa_mae_gain_ci95_high_samples": cluster_tdoa_mae_ci_high,
        "low_cluster_tdoa_rmse_gain_samples": cluster_tdoa_rmse_gain,
        "low_cluster_tdoa_rmse_gain_ci95_low_samples": cluster_tdoa_rmse_ci_low,
        "low_cluster_tdoa_rmse_gain_ci95_high_samples": cluster_tdoa_rmse_ci_high,
        "low_cluster_tdoa_count": cluster_tdoa_count,
        "low_cluster_tdoa_rows": cluster_tdoa_rows,
        "method_gate_passed": method_gate_passed,
        "protocol_gate_passed": protocol_gate_passed,
        "gate_passed": gate_passed,
        **checks,
    }]
    gate_path = os.path.join(tables_dir, f"{prefix}_go_no_go.csv")
    pd.DataFrame(gate_rows).to_csv(gate_path, index=False, encoding="utf-8")
    print(
        f"\n[Gate] low-SNR paired RMSE gain={gain:.3f} m "
        f"(95% CI {ci_low:.3f}..{ci_high:.3f}, n={common_valid})",
        flush=True,
    )
    print(
        f"[Audit] snapshot-cluster localization RMSE gain={cluster_loc_gain:.3f} m "
        f"(95% CI {cluster_loc_ci_low:.3f}..{cluster_loc_ci_high:.3f}, "
        f"clusters={cluster_loc_count})",
        flush=True,
    )
    print(
        f"[Audit] snapshot-cluster TDOA MAE gain={cluster_tdoa_mae_gain:.3f} samples "
        f"(95% CI {cluster_tdoa_mae_ci_low:.3f}..{cluster_tdoa_mae_ci_high:.3f}); "
        f"RMSE gain={cluster_tdoa_rmse_gain:.3f} samples "
        f"(95% CI {cluster_tdoa_rmse_ci_low:.3f}..{cluster_tdoa_rmse_ci_high:.3f})",
        flush=True,
    )
    print(f"[Gate] checks={checks}", flush=True)
    if gate_evaluated:
        gate_line = (
            f"method={'PASS' if method_gate_passed else 'NO-GO'} | "
            f"protocol={'PASS' if protocol_gate_passed else 'NO-GO'} | "
            f"final={'PASS' if gate_passed else 'NO-GO'}"
        )
    else:
        gate_line = "method=NOT-EVALUATED(smoke) | final=NOT-EVALUATED(smoke)"
    print(f"[Gate] {gate_line} | saved={gate_path}", flush=True)

    t_total = time.time() - t_start
    print(f"\n[Done] Total elapsed: {t_total:.0f}s ({t_total/60:.1f} min)", flush=True)
    if alignment_failed and args.strict_align:
        raise RuntimeError("Top1-WLS alignment failed; see alignment CSV.")


if __name__ == "__main__":
    main()
