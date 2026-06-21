"""
SCA-Based LP-Relaxation Upper Bound for Problem (P1)
=====================================================

Paper: "MoRo: Mobility-Aware Robot Mission Orchestration under
        Computation-Induced Motion Degradation"

Algorithm
---------
The original Problem (P1) is a Mixed-Integer Nonlinear Program (MINLP):

    max  sum_k profit_k * y_k
    s.t. C1  : sum_v x_{k,v} <= 1          (each mission at most one vehicle)
         C2  : sum_v x_{k,v} >= y_k
         C3  : sum_k x_{k,v} <= M_v        (vehicle capacity)
         C4  : dependency ordering
         C5  : v^eff_{k,v} * tau >= road_k (mobility feasibility, Eq. 26)
         x_{k,v}, y_k in {0,1}

This module finds an upper bound via successive LP relaxations:

  Iter 0  : assume d^(0)_{k,v} = 0  (zero delay → maximum feasibility)
             → solve LP relaxation → UB_0   (loosest valid upper bound)
  Iter t  : update delay estimate d^(t) from x^(t-1)
             → recompute feasibility cuts
             → solve LP relaxation → UB_t <= UB_{t-1}  (tightening)

Upper-Bound Validity
--------------------
At every iteration t:
  - d^(t) <= true delays  (we under-estimate, starting from 0 and increasing)
  - Under-estimated delays → v^eff over-estimated → feasibility set is a
    *superset* of the true IP feasible set
  - The LP optimises over a superset → LP objective >= IP optimal  (UB valid)

The sequence {UB_t} is non-increasing and converges to a tight upper bound.
"""

import numpy as np
import copy
from scipy.optimize import linprog

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

from src.physic_definition.system_base.ITS_based import (
    Mission, Vehicle, mobility_v_eff
)
from configs.systemcfg import task_cfg, vehicle_cfg, network_cfg, mission_cfg


# ---------------------------------------------------------------------------
# Analytical helpers (no simulation needed)
# ---------------------------------------------------------------------------

def _n_tasks_estimate(road_len: float, v_nom: float, lambda_avg: float,
                       d_task: float = 0.0) -> int:
    """
    Number of offloading tasks that can physically be processed during
    a traversal of road_len at speed v_nom.

    Two limits apply:
      1. Arrival limit  : tasks that arrive in d_nom = road_len / v_nom seconds
                          n_arrive = lambda_avg * d_nom
      2. Capacity limit : each task occupies ~2*d_task seconds
                          (d_task wait + d_task recovery, both ≈ d_task for small delays)
                          n_fit = floor(d_nom / (2 * d_task))

    Without the capacity cap, when lambda_avg * d_task >> 0.5 the formula in
    mobility_S overflows (sum_slow > d_nom) and gives S > road_len (v_eff > v_nom),
    which breaks the SCA tightening direction.
    """
    if v_nom <= 0 or road_len <= 0:
        return 0
    d_nom = road_len / v_nom
    n_arrive = max(0, int(lambda_avg * d_nom))
    if d_task > 0:
        # Physical capacity: at most one task per ~2*d_task seconds
        n_fit = max(0, int(d_nom / (2.0 * d_task)))
        return min(n_arrive, n_fit)
    return n_arrive


def _mission_veff(road_len: float, d_task: float,
                  v_nom: float, rho_down: float, rho_up: float,
                  lambda_avg: float) -> float:
    """
    Compute v^eff (Eqs. 19-20) given a uniform per-task delay d_task.

    All tasks on the segment are assumed to impose the same delay d_task,
    which is a conservative under-estimate when d_task is small.
    """
    n = _n_tasks_estimate(road_len, v_nom, lambda_avg, d_task)
    d_nom = road_len / v_nom if v_nom > 0 else float('inf')
    delays = [d_task] * n
    return mobility_v_eff(d_nom, delays, v_nom, rho_down, rho_up)


def _feasible(road_len: float, d_task: float,
              v_nom: float, rho_down: float, rho_up: float,
              lambda_avg: float, tau_sec: float) -> bool:
    """
    Eq. (26): v^eff * tau >= road_len  (mission completion feasibility).
    """
    v_eff = _mission_veff(road_len, d_task, v_nom, rho_down, rho_up, lambda_avg)
    return v_eff * tau_sec >= road_len


# ---------------------------------------------------------------------------
# LP sub-problem solver
# ---------------------------------------------------------------------------

def _solve_lp(W: np.ndarray, n_miss_per_veh: int,
              dep_pairs: list) -> tuple:
    """
    Solve the LP relaxation of the assignment sub-problem at one SCA iteration.

        max   sum_{k,v} W[k,v] * x_{k,v}
        s.t.  C1  : sum_v x_{k,v} <= 1                  for all k
              C3  : sum_k x_{k,v} <= n_miss_per_veh      for all v
              C4r : x_{k,v} <= x_{k',v}                  for each (k depends on k')
                      (relaxed: same-vehicle but not strictly ordered)
              0 <= x_{k,v} <= 1  (relaxed binary)
              x_{k,v} = 0  when W[k,v] = 0  (enforced via upper bound = 0)

    Parameters
    ----------
    W             : (K, V) profit/feasibility-weighted matrix
    n_miss_per_veh: vehicle capacity M
    dep_pairs     : list of (k, k_prime) meaning "k depends on k_prime"

    Returns
    -------
    x_lp  : (K, V) fractional assignment
    ub_val: float  LP objective value (upper bound on IP optimal)
    """
    K, V = W.shape
    n = K * V

    # ---------- objective (negate for scipy minimise) ----------
    c = -W.flatten().astype(float)

    # ---------- inequality constraints ----------
    rows_ub = []
    rhs_ub  = []

    # C1: for each k, sum_v x_{k,v} <= 1
    for k in range(K):
        row = np.zeros(n)
        row[k * V:(k + 1) * V] = 1.0
        rows_ub.append(row)
        rhs_ub.append(1.0)

    # C3: for each v, sum_k x_{k,v} <= n_miss_per_veh
    for v in range(V):
        row = np.zeros(n)
        row[v::V] = 1.0
        rows_ub.append(row)
        rhs_ub.append(float(n_miss_per_veh))

    # C4r (dependency relaxation): x_{k,v} <= x_{k',v}  for each (k, k')
    # Equivalently: x_{k,v} - x_{k',v} <= 0
    for (k, k_prime) in dep_pairs:
        if 0 <= k < K and 0 <= k_prime < K:
            for v in range(V):
                row = np.zeros(n)
                row[k * V + v]       =  1.0
                row[k_prime * V + v] = -1.0
                rows_ub.append(row)
                rhs_ub.append(0.0)

    A_ub = np.array(rows_ub, dtype=float)
    b_ub = np.array(rhs_ub,  dtype=float)

    # ---------- variable bounds ----------
    bounds = []
    for k in range(K):
        for v in range(V):
            hi = 1.0 if W[k, v] > 0 else 0.0
            bounds.append((0.0, hi))

    # ---------- solve ----------
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if result.success:
        x_lp  = result.x.reshape(K, V)
        ub_val = float(-result.fun)
    else:
        x_lp  = np.zeros((K, V))
        ub_val = 0.0

    return x_lp, ub_val


# ---------------------------------------------------------------------------
# SCA Upper Bound Solver
# ---------------------------------------------------------------------------

class SCAUpperBound:
    """
    SCA LP-relaxation upper bound for the MoRo mission assignment Problem (P1).

    Parameters
    ----------
    missions       : list[Mission]  – Mission objects with get_long(), get_profit(),
                                      get_depends(), get_mid()
    n_vehicles     : int            – number of vehicles V
    n_miss_per_veh : int            – vehicle mission capacity M
    tau_sec        : float          – time budget in seconds (default: task_cfg['tau']*60)
    avg_rate_bps   : float          – average uplink data rate (bps)
                                      default: heuristic ~33 Mbps
    avg_cpu        : float          – average MEC CPU frequency
                                      (same unit as network_cfg['CPU_freq'])
    max_iter       : int            – max SCA iterations
    tol            : float          – convergence threshold on |UB_t - UB_{t-1}|
    seed           : int            – RNG seed

    Usage
    -----
    sca = SCAUpperBound(missions, n_vehicles=5, n_miss_per_veh=5)
    ub_hist, x_lp, x_int, info = sca.solve(verbose=True)
    print(f"Upper bound = {ub_hist[-1]:.2f}")
    """

    def __init__(self, missions, n_vehicles=None, n_miss_per_veh=None,
                 tau_sec=None, avg_rate_bps=None, avg_cpu=None,
                 max_iter=30, tol=1e-3, seed=42):

        self.missions  = missions
        self.K         = len(missions)
        self.V         = n_vehicles     if n_vehicles     is not None else mission_cfg['n_vehicle']
        self.M         = n_miss_per_veh if n_miss_per_veh is not None else mission_cfg['n_miss_per_vec']
        self.tau_sec   = tau_sec        if tau_sec        is not None else task_cfg['tau'] * 60
        self.max_iter  = max_iter
        self.tol       = tol
        self.rng       = np.random.default_rng(seed)

        # --- Mobility params ---
        self.v_nom    = vehicle_cfg['v_nominal']
        self.rho_down = vehicle_cfg['rho_down']
        self.rho_up   = vehicle_cfg['rho_up']

        # --- Average task arrival rate ---
        lambdas = task_cfg.get('lambdas', [10, 30, 50])
        self.lambda_avg = float(np.mean(lambdas))

        # --- Average task delay (comm + comp, no queuing) ---
        alpha_mid = (task_cfg['comm_size'][0] + task_cfg['comm_size'][1]) / 2.0 * 8000  # bits
        beta_mid  = (task_cfg['comp_size'][0]  + task_cfg['comp_size'][1])  / 2.0        # mcycles

        if avg_rate_bps is not None:
            self.avg_rate = avg_rate_bps
        else:
            # Heuristic: 10 MHz bandwidth, SNR ~ 10 dB → ~33 Mbps
            self.avg_rate = 33e6

        if avg_cpu is not None:
            self.avg_cpu = avg_cpu
        else:
            lo, hi = network_cfg['CPU_freq'][0], network_cfg['CPU_freq'][1]
            self.avg_cpu = (lo + hi) / 2.0

        # avg_d_task: expected delay per task under average channel + MEC conditions
        self.avg_d_task = alpha_mid / self.avg_rate + beta_mid / self.avg_cpu

        # --- Pre-compute dependency pairs (k depends on k') ---
        mid_to_idx = {m.get_mid(): i for i, m in enumerate(missions)}
        self._dep_pairs = []
        for i, m in enumerate(missions):
            for dep_mid in m.get_depends():
                if dep_mid in mid_to_idx:
                    self._dep_pairs.append((i, mid_to_idx[dep_mid]))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_profit_matrix(self, d_kv: np.ndarray) -> np.ndarray:
        """
        W[k,v] = profit_k  if mission k is feasible for vehicle v (C5 check),
               = 0          otherwise.

        Feasibility uses Eq. (26): v^eff(d^(t)) * tau >= road_len.
        Since d^(t) <= true delays, v^eff is over-estimated and the feasible
        set is a superset of the true set → W >= true W → LP UB is valid.
        """
        W = np.zeros((self.K, self.V))
        for k, mission in enumerate(self.missions):
            road_len = float(mission.get_long()[0])
            profit   = float(mission.get_profit())
            if road_len <= 0:
                W[k, :] = profit   # trivially feasible (zero-length mission)
                continue
            for v in range(self.V):
                if _feasible(road_len, d_kv[k, v],
                             self.v_nom, self.rho_down, self.rho_up,
                             self.lambda_avg, self.tau_sec):
                    W[k, v] = profit
        return W

    def _update_delays(self, x_lp: np.ndarray,
                       d_kv: np.ndarray) -> np.ndarray:
        """
        SCA delay tightening step.

        For pairs where the LP assigns x_{k,v} > 0, pull the delay estimate
        toward avg_d_task proportionally.  The monotone max() ensures the
        delay sequence is non-decreasing → UB sequence is non-increasing.

            d^(t+1)_{k,v} = max(d^(t)_{k,v},  x^(t)_{k,v} * avg_d_task)
        """
        updated = x_lp * self.avg_d_task
        return np.maximum(d_kv, updated)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, verbose: bool = True) -> tuple:
        """
        Run SCA iterations and return results.

        Returns
        -------
        ub_history : list[float]   UB value at each SCA iteration
        x_lp       : (K, V) ndarray  final fractional LP assignment
        x_int      : (K, V) ndarray  greedy-rounded integer assignment
        info       : dict {
            'converged'         : bool,
            'n_iter'            : int,
            'feasibility_matrix': W at final iteration,
            'd_task_final'      : per-(k,v) delay estimate at convergence,
            'dep_pairs'         : dependency list,
        }
        """
        # Iteration 0: d^(0) = 0  →  loosest (highest) upper bound
        d_kv = np.zeros((self.K, self.V))
        ub_history: list = []
        x_lp = np.zeros((self.K, self.V))
        converged = False

        if verbose:
            print(f"SCA Upper Bound  |  K={self.K} missions, V={self.V} vehicles, "
                  f"M={self.M}, tau={self.tau_sec}s")
            print(f"  avg_d_task = {self.avg_d_task:.6f} s, "
                  f"lambda_avg = {self.lambda_avg:.1f} tasks/s")

        W_final = np.zeros((self.K, self.V))
        for t in range(self.max_iter):
            W = self._build_profit_matrix(d_kv)
            x_lp, ub = _solve_lp(W, self.M, self._dep_pairs)
            ub_history.append(ub)

            if verbose:
                n_feas = int((W > 0).any(axis=1).sum())
                print(f"  iter {t:3d} | UB = {ub:10.4f} | "
                      f"feasible missions = {n_feas}/{self.K}")

            if t > 0 and abs(ub_history[-1] - ub_history[-2]) < self.tol:
                converged = True
                W_final = W
                if verbose:
                    print(f"  Converged at iteration {t}  "
                          f"(delta = {abs(ub_history[-1]-ub_history[-2]):.2e})")
                break

            d_kv  = self._update_delays(x_lp, d_kv)
            W_final = W

        # ------------------------------------------------------------------
        # Greedy rounding: high-confidence missions assigned first
        # ------------------------------------------------------------------
        x_int = np.zeros((self.K, self.V), dtype=int)
        capacity = np.zeros(self.V, dtype=int)

        # Sort missions by max LP value (most "certain" first)
        order = np.argsort(-x_lp.max(axis=1))
        for k in order:
            # Try vehicles in decreasing LP value order
            for v in np.argsort(-x_lp[k]):
                if capacity[v] < self.M and x_lp[k, v] > 0:
                    x_int[k, v] = 1
                    capacity[v] += 1
                    break

        info = {
            'converged'         : converged,
            'n_iter'            : len(ub_history),
            'feasibility_matrix': W_final,
            'd_task_final'      : d_kv,
            'dep_pairs'         : self._dep_pairs,
        }
        return ub_history, x_lp, x_int, info

    def assignment_to_sol(self, x_int: np.ndarray) -> list:
        """
        Convert integer assignment matrix (K×V) to the (order, vehicle_id)
        tuple list expected by Vehicle.set_mission(..., mtuple=True).

        sol[k] = (order_within_vehicle, vehicle_id)
        Unassigned missions are placed on the least-loaded vehicle.
        """
        sol = [None] * self.K
        order_counter = np.zeros(self.V, dtype=int)

        for k in range(self.K):
            assigned_v = int(np.argmax(x_int[k]))
            if x_int[k, assigned_v] == 1:
                sol[k] = (int(order_counter[assigned_v]), assigned_v)
                order_counter[assigned_v] += 1
            else:
                # Unassigned: send to least-loaded vehicle as fallback
                v = int(np.argmin(order_counter))
                sol[k] = (int(order_counter[v]), v)
                order_counter[v] += 1

        return sol

    def evaluate_rounded(self, x_int: np.ndarray, decoded_data: list,
                         segments: list, graph, lmap,
                         verbose: bool = False) -> dict:
        """
        Run the actual simulation with the rounded integer assignment and
        return performance metrics comparable to DRL / meta-heuristic results.

        Parameters
        ----------
        x_int        : (K, V) integer assignment from solve()
        decoded_data : list of mission dicts (same format as ItsProblem.data)
        segments     : map segment list
        graph        : Graph object
        lmap         : Map object
        verbose      : print per-vehicle details

        Returns
        -------
        dict with keys:
            total_profit     : float
            completed_tasks  : int
            total_benefit    : float
        """
        from src.physic_definition.system_base.ITS_based import Mission, Vehicle
        import copy

        sol = self.assignment_to_sol(x_int)

        # Build vehicles
        vehicles = []
        for v in range(self.V):
            seg = self.rng.choice(segments)
            veh = Vehicle(0.5, seg.get_endpoints()[0], lmap,
                          task_cfg['tau'], verbose=verbose)
            vehicles.append(veh)

        # Build missions
        missions = []
        for item in decoded_data:
            m = Mission(item['depart_p'], item['depart_s'], 1,
                        graph=graph, verbose=verbose)
            m.set_depends(item['depends'])
            m.set_observers(vehicles)
            missions.append(m)
        if missions:
            missions[0].reset()

        for veh in vehicles:
            veh.set_mission(sol, missions, mtuple=True)
        for veh in vehicles:
            veh.fit_order()

        while True:
            terminate = True
            for veh in vehicles:
                veh.process_mission()
            for veh in vehicles:
                veh.verify_ready()
            for veh in vehicles:
                if veh.inprocess():
                    terminate = False
                    break
            if terminate:
                break

        total_profit    = sum(v.get_vhicle_prof()    for v in vehicles)
        completed_tasks = sum(v.get_earn_completes() for v in vehicles)
        total_benefit   = sum(v.get_earn_profit()    for v in vehicles)
        vehicles[0].reset()

        return {
            'total_profit'   : total_profit,
            'completed_tasks': completed_tasks,
            'total_benefit'  : total_benefit,
        }
