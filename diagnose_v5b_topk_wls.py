"""V5-B top-K WLS diagnostic v5 — progress, partial save, optimized beam."""

import os, sys, pickle, argparse, time
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from signal_gen import SignalSimulator


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
    lags = np.asarray(lags, dtype=float)[ok]; pw = pw[ok]
    if len(pairs) < 3:
        return None, float("inf")

    bounds = ([0.0, 0.0], [float(area_size[0]), float(area_size[1])])
    delta_ranges = lags * c / fs
    pw = pw / (np.median(pw) + 1e-12)
    sqrt_w = np.sqrt(np.clip(pw, 0.05, 20.0))

    def residual(p):
        vals = []
        for (i,j), dr in zip(pairs, delta_ranges):
            vals.append(np.linalg.norm(p - uav_pos[i]) - np.linalg.norm(p - uav_pos[j]) - dr)
        return sqrt_w * np.array(vals, dtype=float)

    used = sorted(set([i for i,_ in pairs] + [j for _,j in pairs]))
    gs = max(2, int(grid_size))
    x0_list = [np.mean(uav_pos[used], axis=0),
               np.array([area_size[0]/2, area_size[1]/2])]
    for x in np.linspace(0.1*bounds[1][0], 0.9*bounds[1][0], gs):
        for y in np.linspace(0.1*bounds[1][1], 0.9*bounds[1][1], gs):
            x0_list.append(np.array([x, y]))

    best_pos, best_cost = None, float("inf")
    for x0 in x0_list:
        x0 = np.clip(np.asarray(x0, dtype=float), bounds[0], bounds[1])
        try:
            res = least_squares(residual, x0=x0, bounds=bounds, loss="linear", max_nfev=250)
        except Exception:
            continue
        if res.success and np.all(np.isfinite(res.x)):
            c = float(np.sum(res.fun ** 2))
            if c < best_cost:
                best_cost = c
                best_pos = res.x
    return best_pos, best_cost


def _conf(entropy, margin, variance, eps=1e-9):
    return float(margin * max(0.0, 1.0 - entropy) / np.sqrt(max(variance, 0.01) + eps))


def _beam_wls(uav_pos, pairs, lags_list, confs, fs, c, area_size,
              weights=None, max_passes=1, beam_max_pairs=12, prev_best=None):
    """Beam search: only try top2/3 for the beam_max_pairs lowest-confidence pairs."""
    best = [L[0] for L in lags_list]
    init_pos = prev_best
    if init_pos is None:
        init_pos, _ = _localize_wls(uav_pos, pairs, best, fs, c, area_size,
                                     weights=weights, grid_size=1)
    if init_pos is None:
        return None, float("inf"), best
    best_pos, best_cost = init_pos, _compute_cost(
        init_pos, uav_pos, pairs, best, fs, c, weights
    )

    order = np.argsort(confs)
    trial_indices = order[:min(beam_max_pairs, len(order))]

    for _ in range(max_passes):
        improved = False
        for idx in trial_indices:
            cur = best[idx]
            for alt in lags_list[idx][1:3]:
                if abs(alt - cur) < 1e-3:
                    continue
                best[idx] = alt
                np_pos, np_cost = _localize_wls(uav_pos, pairs, best, fs, c, area_size,
                                                 weights=weights, grid_size=1)
                if np_pos is not None and np.isfinite(np_cost) and np_cost < best_cost * 0.999:
                    best_pos, best_cost = np_pos, np_cost
                    improved = True
                else:
                    best[idx] = cur
        if not improved:
            break
    return best_pos, best_cost, best


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


def _check_tdoa_consistency(diag_df, geo, eval_snr, n_trials, tol=1e-4, max_examples=5):
    """Verify CSV pair labels match rebuilt metadata before expensive WLS."""
    required = {"SNR_dB", "trial_idx", "uav_i", "uav_j", "true_tdoa"}
    missing = sorted(required - set(diag_df.columns))
    if missing:
        raise RuntimeError(f"top-K diagnostic CSV missing required columns: {missing}")

    checked = 0
    failures = []
    for snr_db in eval_snr:
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
    parser.add_argument("result_dir", nargs="?", default="运行结果/20260709_120019")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--beam-max-pairs", type=int, default=12)
    parser.add_argument("--beam-passes", type=int, default=1)
    parser.add_argument("--save-partial-every-snr", action="store_true", default=True)
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument(
        "--strict-align", action="store_true",
        help="Raise if Top1-WLS differs from saved V5B-Expert64 by more than 5 m."
    )
    args = parser.parse_args()
    result_dir = os.path.abspath(args.result_dir)

    with open(os.path.join(result_dir, "plot_data.pkl"), "rb") as f:
        data = pickle.load(f)
    cfg = data.get("config", {})

    csv_path = os.path.join(result_dir, "tables", "v5b_pair_topk_diagnostics.csv")
    diag_df = pd.read_csv(csv_path)

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

    eval_snr = [float(x) for x in cfg.get("eval_snr_range", np.arange(-10, 21, 2))]
    n_trials = int(cfg.get("monte_carlo_trials", 200))
    seed = int(cfg.get("seed", 42))

    if args.smoke:
        eval_snr = eval_snr[:1]
        n_trials = 20

    # ── config summary ───────────────────────────────────────────────────────
    total_samples = len(eval_snr) * n_trials
    print(f"[Config] SNR range: {len(eval_snr)} points ({eval_snr[0]:.0f}..{eval_snr[-1]:.0f} dB)", flush=True)
    print(f"[Config] Trials per SNR: {n_trials}  |  Total samples: {total_samples}", flush=True)
    print(f"[Config] beam_max_pairs={args.beam_max_pairs}  beam_passes={args.beam_passes}  "
          f"grid_size={args.grid_size}  progress_every={args.progress_every}", flush=True)
    print(f"[Config] save_partial_every_snr={args.save_partial_every_snr}", flush=True)
    print(f"[Config] area={area_size}", flush=True)
    geom_mode = "CSV snapshot_id" if "snapshot_id" in diag_df.columns else "seeded generator"
    print(f"[Config] geometry reconstruction: {geom_mode}", flush=True)

    print(f"[WLS] Generating geometry for {len(eval_snr)} SNR x {n_trials} trials...", flush=True)
    geo = {}
    for snr_db in eval_snr:
        snap_idx = _snapshot_indices_from_diag(diag_df, snr_db, n_trials)
        _, _, meta = sim.generate_urban_batch(
            n_trials, snr_db=float(snr_db), snapshot_indices=snap_idx, seed=seed)
        geo[snr_db] = meta

    _check_tdoa_consistency(diag_df, geo, eval_snr, n_trials)

    methods = ["Top1-WLS", "Top3-BeamWLS", "Top3-RejectBeamWLS"]
    all_rows = []
    all_worst = []
    tdoa_assert_fails = 0
    centroid_fallbacks = 0
    t_start = time.time()
    done_global = 0

    tables_dir = os.path.join(result_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    prefix = "v5b_topk_wls_smoke" if args.smoke else "v5b_topk_wls"

    for snr_idx, snr_db in enumerate(eval_snr):
        t_snr_start = time.time()
        print(f"\n[SNR {snr_db:5.0f} dB] start  ({snr_idx+1}/{len(eval_snr)})", flush=True)
        meta = geo[snr_db]
        snr_rows = []
        snr_worst = []

        for ti in range(n_trials):
            done_global += 1
            los = meta["los"][ti]
            los_idx = np.where(los)[0]
            if len(los_idx) < 2:
                continue
            src_true = meta["source"][ti]
            uav_pos = meta["uavs"][ti]
            dfloat = meta["delay_float"][ti]
            centroid = np.mean(uav_pos[los_idx], axis=0)

            pr = diag_df[(diag_df["SNR_dB"] == snr_db) & (diag_df["trial_idx"] == ti)]
            if len(pr) < 3:
                continue

            pairs = []; lags_list = []; ws = []; confs = []
            for _, r in pr.iterrows():
                i, j = int(r["uav_i"]), int(r["uav_j"])
                true_csv = float(r["true_tdoa"])
                true_meta = float(dfloat[i] - dfloat[j])
                if abs(true_csv - true_meta) > 1e-4:
                    tdoa_assert_fails += 1
                l1 = float(r["top1_lag"])
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
            ws_arr = np.array(ws); confs_arr = np.array(confs)

            def solve_or_fallback(pairs_in, lags_in, weights_in=None):
                p, c = _localize_wls(uav_pos, pairs_in, lags_in, fs, c_speed, area_size,
                                      weights=weights_in, grid_size=args.grid_size)
                if p is None:
                    nonlocal centroid_fallbacks
                    centroid_fallbacks += 1
                    return centroid.copy()
                return p

            # Top1
            p1 = solve_or_fallback(pairs, [L[0] for L in lags_list], weights_in=ws_arr)
            e1 = np.linalg.norm(p1 - src_true)

            # Top3-BeamWLS (use Top1 position as init)
            p3b, c3b, _ = _beam_wls(uav_pos, pairs, lags_list, confs_arr, fs, c_speed, area_size,
                                     weights=ws_arr, max_passes=args.beam_passes,
                                     beam_max_pairs=args.beam_max_pairs, prev_best=p1)
            if p3b is None:
                p3b = centroid.copy(); centroid_fallbacks += 1
            e3b = np.linalg.norm(p3b - src_true)

            # Top3-RejectBeamWLS
            med = np.median(confs_arr)
            keep = confs_arr >= med
            if np.sum(keep) < 4:
                n_need = 4 - int(np.sum(keep))
                rej_idx = np.where(~keep)[0]
                add = sorted(rej_idx[np.argsort(confs_arr[~keep])[::-1][:n_need]])
                keep = np.zeros(len(confs_arr), dtype=bool)
                keep[list(np.where(confs_arr >= med)[0]) + add] = True
            if np.sum(keep) < 3:
                keep = np.ones(len(confs_arr), dtype=bool)
            r_pairs = [pairs[i] for i in range(len(pairs)) if keep[i]]
            r_lags = [lags_list[i] for i in range(len(pairs)) if keep[i]]
            r_confs = confs_arr[keep]
            r_ws = ws_arr[keep]
            # Use centroid as init for reject (since pair set changed)
            p3r, c3r, _ = _beam_wls(uav_pos, r_pairs, r_lags, r_confs, fs, c_speed, area_size,
                                     weights=r_ws, max_passes=args.beam_passes,
                                     beam_max_pairs=args.beam_max_pairs, prev_best=None)
            if p3r is None:
                p3r = centroid.copy(); centroid_fallbacks += 1
            e3r = np.linalg.norm(p3r - src_true)

            n_kept = int(np.sum(keep))
            n_rej = len(pairs) - n_kept
            base = {"SNR_dB": snr_db, "trial_idx": ti,
                    "n_pairs_total": len(pairs),
                    "n_pairs_kept": n_kept, "n_pairs_rejected": n_rej}
            for mn, e in [("Top1-WLS", e1), ("Top3-BeamWLS", e3b), ("Top3-RejectBeamWLS", e3r)]:
                r2 = dict(base); r2["method"] = mn; r2["error_m"] = float(e)
                snr_rows.append(r2); all_rows.append(r2)
            snr_worst.append({
                "SNR_dB": snr_db, "trial_idx": ti, "n_pairs": len(pairs),
                "top1_rmse_m": float(e1), "top3_beam_rmse_m": float(e3b),
                "top3_reject_rmse_m": float(e3r),
            })
            all_worst.append(snr_worst[-1])

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
        t_snr_elapsed = time.time() - t_snr_start
        print(f"[SNR {snr_db:5.0f} dB] done  elapsed={t_snr_elapsed:.0f}s  "
              f"({t_snr_elapsed/max(n_trials,1):.1f}s/trial)", flush=True)

    if tdoa_assert_fails > 0:
        raise RuntimeError(
            f"TDOA consistency check FAILED: {tdoa_assert_fails} pairs have "
            f"|CSV true_tdoa - meta delay_float| > 1e-4.")
    print("[OK] All TDOA assertions passed", flush=True)
    if centroid_fallbacks > 0:
        print(f"[WARN] {centroid_fallbacks} WLS fallbacks to centroid", flush=True)

    res_df = pd.DataFrame(all_rows)

    def _rmse_stats(errors):
        e = np.asarray(errors, dtype=float); e = e[np.isfinite(e)]; n = len(e)
        if n == 0:
            return {"rmse": np.nan, "median": np.nan, "trimmed_rmse": np.nan,
                    "outlier_gt20_pct": np.nan, "outlier_gt50_pct": np.nan, "valid_count": 0}
        return {"rmse": float(np.sqrt(np.mean(e**2))),
                "median": float(np.median(e)),
                "trimmed_rmse": float(np.sqrt(np.mean(np.sort(e)[:max(1,int(n*0.95))]**2))),
                "outlier_gt20_pct": float(np.mean(e > 20)*100),
                "outlier_gt50_pct": float(np.mean(e > 50)*100),
                "valid_count": n}

    summary_rows = []
    for (snr, method), grp in res_df.groupby(["SNR_dB", "method"]):
        s = _rmse_stats(grp["error_m"].values)
        s["SNR_dB"] = snr; s["method"] = method
        s["mean_n_pairs_total"] = float(grp["n_pairs_total"].mean())
        s["mean_n_pairs_kept"] = float(grp["n_pairs_kept"].mean())
        s["mean_n_pairs_rejected"] = float(grp["n_pairs_rejected"].mean())
        summary_rows.append(s)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(tables_dir, f"{prefix}_baselines.csv"), index=False, encoding="utf-8")

    diag = res_df.groupby(["SNR_dB", "method"]).agg(
        rmse=("error_m", lambda x: np.sqrt(np.mean(x**2))),
        median_error_m=("error_m", "median"),
        mean_n_pairs_total=("n_pairs_total", "mean"),
        mean_n_pairs_kept=("n_pairs_kept", "mean"),
        mean_n_pairs_rejected=("n_pairs_rejected", "mean"),
    ).reset_index()
    diag.to_csv(os.path.join(tables_dir, f"{prefix}_method_diagnostics.csv"), index=False, encoding="utf-8")

    pd.DataFrame(all_worst).to_csv(
        os.path.join(tables_dir, f"{prefix}_worst_localization_errors.csv"), index=False, encoding="utf-8")

    snr_arr = res_df["SNR_dB"].values
    buckets = np.full(len(res_df), "other", dtype=object)
    buckets[snr_arr <= 0.0] = "low"
    buckets[(snr_arr >= 2.0) & (snr_arr <= 6.0)] = "mid"
    buckets[snr_arr >= 8.0] = "high"
    res_df["snr_bucket"] = buckets
    bucket_rows = []
    for (bk, method), grp in res_df.groupby(["snr_bucket", "method"]):
        s = _rmse_stats(grp["error_m"].values)
        s["snr_bucket"] = bk; s["method"] = method
        bucket_rows.append(s)
    bs = pd.DataFrame(bucket_rows)
    bs.to_csv(os.path.join(tables_dir, f"{prefix}_summary_by_snr_bucket.csv"), index=False, encoding="utf-8")

    print(f"\n[Saved] {tables_dir}/{prefix}_*.csv", flush=True)
    print("\n" + "=" * 72, flush=True)
    print("V5-B Top-K WLS  |  low: SNR<=0  mid: 2-6  high: >=8", flush=True)
    print("=" * 72, flush=True)
    print(f"{'Method':<22s} {'Bucket':<8s} {'RMSE':>7s} {'Median':>7s} {'Trim':>7s} {'>20m':>6s} {'>50m':>6s}", flush=True)
    print("-" * 72, flush=True)
    for method in methods:
        for bk_key, bk_label in [("low","low"),("mid","mid"),("high","high")]:
            sub = bs[(bs["method"]==method) & (bs["snr_bucket"]==bk_key)]
            if sub.empty: continue
            r = sub.iloc[0]
            print(f"{method:<22s} {bk_label:<8s} {r['rmse']:7.2f} {r['median']:7.2f} "
                  f"{r['trimmed_rmse']:7.2f} {r['outlier_gt20_pct']:5.1f}% {r['outlier_gt50_pct']:5.1f}%",
                  flush=True)
        grp_all = res_df[res_df["method"]==method]["error_m"].values
        s_all = _rmse_stats(grp_all)
        print(f"{method:<22s} {'all':<8s} {s_all['rmse']:7.2f} {s_all['median']:7.2f} "
              f"{s_all['trimmed_rmse']:7.2f} {s_all['outlier_gt20_pct']:5.1f}% {s_all['outlier_gt50_pct']:5.1f}%",
              flush=True)
        print("-" * 72, flush=True)

    v5b_ref = data.get("fig9_results")
    if v5b_ref is not None:
        ref_results, ref_snr = v5b_ref
        ref_methods = ref_results.get("methods", {})
        v5b_rmse = ref_methods.get("V5B-Expert64", {}).get("rmse", [])
        top1_sum = summary[summary["method"] == "Top1-WLS"]
        align_rows = []
        for snr in eval_snr:
            sub = top1_sum[top1_sum["SNR_dB"] == snr]
            if not sub.empty:
                align_rows.append({
                    "SNR_dB": float(snr),
                    "top1_wls_rmse": float(sub.iloc[0]["rmse"]),
                })
        if v5b_rmse and align_rows:
            ref_map = {
                float(snr): float(rmse)
                for snr, rmse in zip(np.asarray(ref_snr, dtype=float), v5b_rmse)
            }
            for row in align_rows:
                ref_value = ref_map.get(float(row["SNR_dB"]), np.nan)
                row["saved_v5b_rmse"] = ref_value
                row["abs_diff_m"] = abs(row["top1_wls_rmse"] - ref_value)
            align_df = pd.DataFrame(align_rows)
            align_path = os.path.join(tables_dir, f"{prefix}_alignment.csv")
            align_df.to_csv(align_path, index=False, encoding="utf-8")
            md = float(np.nanmax(align_df["abs_diff_m"].values))
            print(f"\n[Align] Top1-WLS vs V5B-Expert64 max per-SNR RMSE diff = {md:.4f} m", flush=True)
            print(f"[Align] Saved per-SNR alignment diagnostics: {align_path}", flush=True)
            if md > 5.0 and args.strict_align:
                raise RuntimeError(f"Top1-WLS alignment failure: max diff {md:.4f} m > 5 m.")
            if md > 5.0:
                print(
                    "[Align] WARN: diff > 5 m. This usually means the pair-level "
                    "top-K CSV was generated on a different fixed eval set than "
                    "the saved Fig9 V5B curve. Stage-2A Top1/Top3 comparisons "
                    "within this script remain valid because they share the same "
                    "CSV and rebuilt geometry.",
                    flush=True,
                )
            elif md > 1.0:
                print(f"[Align] WARN: max diff {md:.4f} m > 1 m (finite-trial/eval-set tolerance).", flush=True)
            else:
                print("[Align] PASSED: Top1-WLS matches V5B-Expert64 within 1 m.", flush=True)

    t_total = time.time() - t_start
    print(f"\n[Done] Total elapsed: {t_total:.0f}s ({t_total/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
