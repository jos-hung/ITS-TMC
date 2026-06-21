"""
SCA Upper Bound Runner
======================
Runs the SCA-based LP-relaxation upper bound solver on the current problem
instance and, optionally, evaluates the rounded assignment via full simulation.

Output
------
- Console: UB convergence table
- ./results/sca_upper_bound.csv        : UB history across iterations
- ./results/sca_vs_drl_comparison.png  : UB line vs DRL learning curve (if CSV paths provided)

Usage
-----
Run from project root:
    python -m src.meta_heuristic.script_sca_upper_bound

Or to skip the simulation evaluation (faster, UB only):
    python -m src.meta_heuristic.script_sca_upper_bound --ub-only
"""

import os
import sys
import argparse
import json
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.physic_definition.system_base.ITS_based import Mission
from src.models.SCA import SCAUpperBound
from src.utils import write_config
from configs.systemcfg import mission_cfg, task_cfg


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="SCA Upper Bound for Problem (P1)")
    p.add_argument("--ub-only",   action="store_true",
                help="Only compute the LP upper bound; skip full simulation of rounded solution.")
    p.add_argument("--max-iter",  type=int,   default=30,
                help="Maximum SCA iterations (default: 30).")
    p.add_argument("--tol",       type=float, default=1e-3,
                help="Convergence tolerance on UB change (default: 1e-3).")
    p.add_argument("--n-trials",  type=int,   default=5,
                help="Number of independent trials to average over (default: 5).")
    p.add_argument("--drl-csv",   type=str,   default=None,
                help="Path to DRL reward CSV for comparison plot.")
    p.add_argument("--out-dir",   type=str,   default="./results",
                help="Output directory for CSV and figures.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading helper (reuses write_config from utils.py)
# ---------------------------------------------------------------------------

def load_problem_data():
    """Load map, tasks, missions; return (data_dict, graph, lmap, missions)."""
    data, graph, lmap = write_config()
    missions = []
    for item in data["decoded_data"]:
        m = Mission(item['depart_p'], item['depart_s'], 1,
                    graph=data["graph"], verbose=False)
        m.set_depends(item["depends"])
        missions.append(m)
    return data, graph, lmap, missions


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_sca_upper_bound(args):
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("SCA Upper Bound — loading problem data …")
    print("=" * 60)
    t0 = time.perf_counter()
    data, graph, lmap, missions = load_problem_data()
    print(f"  Loaded {len(missions)} missions in {time.perf_counter()-t0:.2f}s")

    n_vehicles     = mission_cfg['n_vehicle']
    n_miss_per_veh = mission_cfg['n_miss_per_vec']

    all_ub_histories = []
    all_sim_results  = []

    for trial in range(args.n_trials):
        print(f"\n--- Trial {trial + 1} / {args.n_trials} ---")

        # ---- SCA solver ----
        sca = SCAUpperBound(
            missions       = missions,
            n_vehicles     = n_vehicles,
            n_miss_per_veh = n_miss_per_veh,
            tau_sec        = task_cfg['tau'] * 60,
            max_iter       = args.max_iter,
            tol            = args.tol,
            seed           = 42 + trial,
        )
        t_sca = time.perf_counter()
        ub_hist, x_lp, x_int, info = sca.solve(verbose=True)
        print(f"  SCA finished in {time.perf_counter()-t_sca:.3f}s  "
            f"| Final UB = {ub_hist[-1]:.4f}  "
            f"| Converged = {info['converged']}")

        all_ub_histories.append(ub_hist)

        # ---- Full simulation of rounded solution ----
        if not args.ub_only:
            print("  Evaluating rounded assignment via simulation …")
            t_sim = time.perf_counter()
            sim_res = sca.evaluate_rounded(
                x_int        = x_int,
                decoded_data = data["decoded_data"],
                segments     = data["segments"],
                graph        = graph,
                lmap         = lmap,
                verbose      = False,
            )
            print(f"  Sim done in {time.perf_counter()-t_sim:.3f}s  "
                f"| profit={sim_res['total_profit']:.4f}  "
                f"| completed={sim_res['completed_tasks']}  "
                f"| benefit={sim_res['total_benefit']:.4f}")
            all_sim_results.append(sim_res)

    # -------------------------------------------------------------------------
    # Aggregate and save UB history
    # -------------------------------------------------------------------------
    max_iters = max(len(h) for h in all_ub_histories)
    # Pad shorter histories with their final value
    padded = np.array([
        h + [h[-1]] * (max_iters - len(h)) for h in all_ub_histories
    ])
    ub_mean = padded.mean(axis=0)
    ub_std  = padded.std(axis=0)

    df_ub = pd.DataFrame({
        'iteration': list(range(max_iters)),
        'ub_mean'  : ub_mean,
        'ub_std'   : ub_std,
    })
    csv_path = os.path.join(args.out_dir, "sca_upper_bound.csv")
    df_ub.to_csv(csv_path, index=False)
    print(f"\nUB history saved → {csv_path}")

    if all_sim_results:
        df_sim = pd.DataFrame(all_sim_results)
        sim_path = os.path.join(args.out_dir, "sca_simulation_results.csv")
        df_sim.to_csv(sim_path, index=False)
        print(f"Simulation results saved → {sim_path}")
        print(f"\n  SCA rounded solution (mean ± std over {args.n_trials} trials):")
        print(f"    total_profit   : {df_sim['total_profit'].mean():.4f} ± {df_sim['total_profit'].std():.4f}")
        print(f"    completed_tasks: {df_sim['completed_tasks'].mean():.2f} ± {df_sim['completed_tasks'].std():.2f}")
        print(f"    total_benefit  : {df_sim['total_benefit'].mean():.4f} ± {df_sim['total_benefit'].std():.4f}")

    print(f"\n  SCA Upper Bound (final, mean over {args.n_trials} trials): {ub_mean[-1]:.4f}")

    # -------------------------------------------------------------------------
    # Comparison plot (if DRL CSV provided)
    # -------------------------------------------------------------------------
    _plot_comparison(args, ub_mean, ub_std, all_sim_results)

    return ub_mean[-1], all_sim_results


def _plot_comparison(args, ub_mean, ub_std, sim_results):
    """Generate comparison figure: SCA UB + (optionally) DRL reward curve."""
    try:
        import matplotlib.pyplot as plt
        try:
            import scienceplots
            plt.style.use(["science", "ieee"])
        except ImportError:
            pass  # scienceplots optional

        fig, ax = plt.subplots(figsize=(7, 5))
        iters = np.arange(len(ub_mean))

        # SCA UB convergence line
        ax.plot(iters, ub_mean, 'b-o', markersize=4, label='SCA Upper Bound')
        ax.fill_between(iters,
                        ub_mean - ub_std,
                        ub_mean + ub_std,
                        alpha=0.2, color='blue')

        # Simulation achieved value (horizontal dashed line at mean)
        if sim_results:
            mean_profit = np.mean([r['total_profit'] for r in sim_results])
            ax.axhline(mean_profit, color='green', linestyle='--',
                       label=f'SCA Rounded (sim): {mean_profit:.2f}')

        # DRL learning curve (optional)
        if args.drl_csv and os.path.exists(args.drl_csv):
            df_drl = pd.read_csv(args.drl_csv)
            agent_cols = [c for c in df_drl.columns if c.startswith('Agent')]
            if agent_cols:
                df_drl['sum'] = df_drl[agent_cols].sum(axis=1)
                smoothed = df_drl['sum'].rolling(window=500, min_periods=1).mean()
                # Normalise x-axis: show only up to len(ub_mean) conceptual "budget"
                ax2 = ax.twiny()
                ax2.plot(smoothed.values, 'r:', alpha=0.7, label='DRL (smoothed)')
                ax2.set_xlabel('DRL Training Epochs', fontsize=10)
                ax2.legend(loc='upper right', fontsize=9)

        ax.set_xlabel('SCA Iteration', fontsize=11)
        ax.set_ylabel('System Profit (Upper Bound)', fontsize=11)
        ax.set_title('SCA Upper Bound vs. DRL', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        fig_path = os.path.join(args.out_dir, "sca_vs_drl_comparison.png")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        print(f"Comparison figure saved → {fig_path}")

    except Exception as exc:
        print(f"  (Plotting skipped: {exc})")


if __name__ == "__main__":
    args = _parse_args()
    run_sca_upper_bound(args)
