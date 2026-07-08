"""SCA benchmark solver aligned with the revised paper formulation.

Key modelling choices in this implementation:
- continuous relaxation on assignment/completion/offloading variables
    (x, y, z_loc, z_mec, z_cld in [0, 1])
- linearized mission-completion constraint C5 using chain-rule gradients
    w.r.t. offloading variables
- queue-aware delay coefficients updated between SCA iterations
- damped SCA iterate update with step size gamma
"""

import numpy as np
from scipy.optimize import linprog

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from src.physic_definition.system_base.ITS_based import mobility_v_eff
from src.physic_definition.network.rate import queue_delay
from configs.systemcfg import task_cfg, vehicle_cfg, network_cfg, mission_cfg


def _n_tasks_estimate(road_len: float, v_nom: float, lambda_avg: float) -> int:
    """Estimate number of control tasks over one mission route."""
    if road_len <= 0 or v_nom <= 0:
        return 0
    d_nom = road_len / v_nom
    return max(1, int(lambda_avg * d_nom))


def _mission_time_from_delay(
    road_len: float,
    d_task: float,
    n_tasks: int,
    v_nom: float,
    rho_down: float,
    rho_up: float,
) -> float:
    """h(d) = road_len / v_eff(d), with v_eff from Eq. (19)-(20)."""
    if road_len <= 0 or v_nom <= 0:
        return 0.0

    d_nom = road_len / v_nom
    n_tasks = int(max(0, n_tasks))
    v_eff = mobility_v_eff(d_nom, [d_task] * n_tasks, v_nom, rho_down, rho_up)
    v_eff = max(min(v_eff, v_nom), 1e-9)
    return road_len / v_eff


def _mission_time_grad_delay(
    road_len: float,
    d_task: float,
    n_tasks: int,
    v_nom: float,
    rho_down: float,
    rho_up: float,
    eps: float = 1e-4,
) -> float:
    """Numerical derivative d h(d) / d d_task for C5 linearization."""
    d0 = max(0.0, d_task)
    h0 = _mission_time_from_delay(road_len, d0, n_tasks, v_nom, rho_down, rho_up)
    h1 = _mission_time_from_delay(road_len, d0 + eps, n_tasks, v_nom, rho_down, rho_up)
    return (h1 - h0) / eps


class SCAUpperBound:
    def __init__(
        self,
        missions,
        n_vehicles=None,
        n_miss_per_veh=None,
        tau_sec=None,
        avg_rate_bps=None,
        avg_cpu=None,
        max_iter=30,
        tol=1e-3,
        seed=42,
        gamma=0.5,
        omega1=1.0,
        omega2=0.0,
        queue_alpha=1.0,
        initial_positions=None,
        min_iter=8,
        obj_rel_tol=1e-4,
        var_tol=1e-3,
        conv_patience=3,
    ):
        self.missions = missions
        self.K = len(missions)
        self.V = n_vehicles if n_vehicles is not None else mission_cfg["n_vehicle"]
        self.M = n_miss_per_veh if n_miss_per_veh is not None else mission_cfg["n_miss_per_vec"]
        self.tau_sec = tau_sec if tau_sec is not None else task_cfg["tau"] * 60
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.default_rng(seed)
        self.gamma = float(np.clip(gamma, 1e-3, 1.0))
        self.omega1 = float(omega1)
        self.omega2 = float(omega2)
        self.queue_alpha = float(max(queue_alpha, 0.0))
        self.min_iter = int(max(1, min_iter))
        self.obj_rel_tol = float(max(obj_rel_tol, 0.0))
        self.var_tol = float(max(var_tol, 0.0))
        self.conv_patience = int(max(1, conv_patience))
        self.E = int(max(1, network_cfg.get("n_MEC", 1)))

        self.v_nom = vehicle_cfg["v_nominal"]
        self.rho_down = vehicle_cfg["rho_down"]
        self.rho_up = vehicle_cfg["rho_up"]

        lambdas = task_cfg.get("lambdas", [10, 30, 50])
        self.lambda_avg = float(np.mean(lambdas))

        alpha_mid = (task_cfg["comm_size"][0] + task_cfg["comm_size"][1]) / 2.0 * 8000.0
        beta_mid = (task_cfg["comp_size"][0] + task_cfg["comp_size"][1]) / 2.0
        self.alpha_mid = float(alpha_mid)
        self.beta_mid = float(beta_mid)

        self.avg_rate = avg_rate_bps if avg_rate_bps is not None else 33e6
        if avg_cpu is not None:
            self.avg_cpu = avg_cpu
        else:
            lo, hi = network_cfg["CPU_freq"][0], network_cfg["CPU_freq"][1]
            self.avg_cpu = (lo + hi) / 2.0

        self.avg_d_task = alpha_mid / max(self.avg_rate, 1e-9) + beta_mid / max(self.avg_cpu, 1e-9)

        self.f_local = float(task_cfg.get("local_cpu_freq", 0.5))
        self.f_mec = np.full(self.E, self.avg_cpu, dtype=float)
        self.f_cloud = float(network_cfg.get("CPU_satelite", self.avg_cpu))

        self.n_rb = float(network_cfg.get("n_rb", 10))
        self.cloud_unit_cost = float(task_cfg.get("cloud_cost_coefi", task_cfg.get("cost_coefi", 5e-5)))
        self.dmax_wait = float(task_cfg.get("dmax_wait_sec", np.inf))
        if self.dmax_wait <= 0:
            self.dmax_wait = np.inf

        self.road_k = np.array([float(m.get_long()[0]) for m in missions], dtype=float)
        self.utility_k = np.array([float(m.get_profit()) for m in missions], dtype=float)
        self.n_tasks_k = np.array(
            [_n_tasks_estimate(self.road_k[k], self.v_nom, self.lambda_avg) for k in range(self.K)],
            dtype=float,
        )
        self.h0_k = np.array(
            [
                _mission_time_from_delay(
                    self.road_k[k],
                    0.0,
                    int(self.n_tasks_k[k]),
                    self.v_nom,
                    self.rho_down,
                    self.rho_up,
                )
                for k in range(self.K)
            ],
            dtype=float,
        )
        self.initial_positions = initial_positions
        self.t_start_to_mission_kv = np.zeros((self.K, self.V), dtype=float)
        if self.initial_positions is not None and len(self.initial_positions) == self.V:
            for k in range(self.K):
                for v in range(self.V):
                    try:
                        _, d = missions[k].get_infor_to_mission(self.initial_positions[v])
                    except Exception:
                        d = 0.0
                    self.t_start_to_mission_kv[k, v] = float(d) / max(self.v_nom, 1e-9)

        # Base per-(mission,vehicle) execution time with zero computational delay.
        self.h0_kv = self.t_start_to_mission_kv + self.h0_k[:, None]
        self.base_feasible_k = (self.h0_kv.min(axis=1) <= self.tau_sec + 1e-9).astype(float)

        unit_cost = float(task_cfg.get("cost_coefi", 5e-5))
        self.unit_cost_mec = unit_cost

        mid_to_idx = {m.get_mid(): i for i, m in enumerate(missions)}
        self._dep_pairs = []
        for i, m in enumerate(missions):
            for dep_mid in m.get_depends():
                if dep_mid in mid_to_idx:
                    self._dep_pairs.append((i, mid_to_idx[dep_mid]))
        self._deps_by_mission = {k: [] for k in range(self.K)}
        for k, dep in self._dep_pairs:
            self._deps_by_mission[k].append(dep)
        self._dep_level = self._compute_dependency_levels()

        # Nominal travel time between consecutive missions (destination l -> start m).
        self.t_move_lm = np.zeros((self.K, self.K), dtype=float)
        for l in range(self.K):
            p_l = missions[l].get_mission_destination()
            for m in range(self.K):
                if l == m:
                    continue
                try:
                    _, d_lm = missions[m].get_infor_to_mission(p_l)
                except Exception:
                    d_lm = 0.0
                self.t_move_lm[l, m] = float(d_lm) / max(self.v_nom, 1e-9)

        self.big_m_time = float(self.tau_sec)
        self.big_m_assign = float(self.tau_sec)
        self.big_m_order = float(self.tau_sec + np.max(self.t_move_lm))

        self._n_x = self.K * self.V
        self._n_y = self.K
        self._n_z_loc = self.K * self.V
        self._n_z_mec = self.K * self.V * self.E
        self._n_z_cld = self.K * self.V * self.E
        self._n_t_start = self.K
        self._n_t_finish = self.K
        self._n_order = self.K * self.K * self.V
        self._off_x = 0
        self._off_y = self._off_x + self._n_x
        self._off_z_loc = self._off_y + self._n_y
        self._off_z_mec = self._off_z_loc + self._n_z_loc
        self._off_z_cld = self._off_z_mec + self._n_z_mec
        self._off_t_start = self._off_z_cld + self._n_z_cld
        self._off_t_finish = self._off_t_start + self._n_t_start
        self._off_order = self._off_t_finish + self._n_t_finish
        self._n_var = self._off_order + self._n_order

        self._d_queue_local = np.zeros(self.V, dtype=float)
        self._d_queue_mec = np.zeros(self.E, dtype=float)
        self._d_queue_cloud = 0.0

    def _idx_x(self, k: int, v: int) -> int:
        return self._off_x + k * self.V + v

    def _idx_y(self, k: int) -> int:
        return self._off_y + k

    def _idx_z_loc(self, k: int, v: int) -> int:
        return self._off_z_loc + k * self.V + v

    def _idx_z_mec(self, k: int, v: int, e: int) -> int:
        return self._off_z_mec + ((k * self.V + v) * self.E + e)

    def _idx_z_cld(self, k: int, v: int, e: int) -> int:
        return self._off_z_cld + ((k * self.V + v) * self.E + e)

    def _idx_t_start(self, k: int) -> int:
        return self._off_t_start + k

    def _idx_t_finish(self, k: int) -> int:
        return self._off_t_finish + k

    def _idx_order(self, l: int, m: int, v: int) -> int:
        return self._off_order + ((l * self.K + m) * self.V + v)

    def _compute_dependency_levels(self):
        """Compute a topological depth per mission from dependency edges."""
        level = np.zeros(self.K, dtype=int)
        state = np.zeros(self.K, dtype=int)  # 0=unseen, 1=visiting, 2=done

        def dfs(k):
            if state[k] == 2:
                return level[k]
            if state[k] == 1:
                # Cycle guard: keep current estimate to avoid recursion loop.
                return level[k]
            state[k] = 1
            if self._deps_by_mission[k]:
                level[k] = 1 + max(dfs(dep) for dep in self._deps_by_mission[k])
            else:
                level[k] = 0
            state[k] = 2
            return level[k]

        for k in range(self.K):
            dfs(k)
        return level

    def _delay_local(self, v: int) -> float:
        return self._d_queue_local[v] + self.beta_mid / max(self.f_local, 1e-9)

    def _delay_mec(self, e: int) -> float:
        d_com = self.alpha_mid / max(self.avg_rate, 1e-9)
        d_cmp = self._d_queue_mec[e] + self.beta_mid / max(self.f_mec[e], 1e-9)
        return d_com + d_cmp

    def _delay_cloud(self, e: int) -> float:
        d_com = self.alpha_mid / max(self.avg_rate, 1e-9)
        d_cmp = self._d_queue_cloud + self.beta_mid / max(self.f_cloud, 1e-9)
        return d_com + d_cmp

    def _is_pair_feasible_base(self, k: int) -> bool:
        h0 = _mission_time_from_delay(
            self.road_k[k],
            0.0,
            int(self.n_tasks_k[k]),
            self.v_nom,
            self.rho_down,
            self.rho_up,
        )
        return h0 <= self.tau_sec + 1e-9

    def _compute_d_exe(self, z_loc: np.ndarray, z_mec: np.ndarray, z_cld: np.ndarray) -> np.ndarray:
        d_exe = np.zeros((self.K, self.V), dtype=float)
        for k in range(self.K):
            for v in range(self.V):
                d = z_loc[k, v] * self._delay_local(v)
                for e in range(self.E):
                    d += z_mec[k, v, e] * self._delay_mec(e)
                    d += z_cld[k, v, e] * self._delay_cloud(e)
                d_exe[k, v] = d
        return d_exe

    def _update_queue_delays(self, z_loc: np.ndarray, z_mec: np.ndarray, z_cld: np.ndarray):
        load_local = np.zeros(self.V, dtype=float)
        load_mec = np.zeros(self.E, dtype=float)
        load_cloud = 0.0

        for k in range(self.K):
            for v in range(self.V):
                load_local[v] += self.n_tasks_k[k] * z_loc[k, v]
                for e in range(self.E):
                    load_mec[e] += self.n_tasks_k[k] * z_mec[k, v, e]
                    load_cloud += self.n_tasks_k[k] * z_cld[k, v, e]

        for v in range(self.V):
            self._d_queue_local[v] = self.queue_alpha * queue_delay(load_local[v], self.beta_mid, self.f_local)
        for e in range(self.E):
            self._d_queue_mec[e] = self.queue_alpha * queue_delay(load_mec[e], self.beta_mid, self.f_mec[e])
        self._d_queue_cloud = self.queue_alpha * queue_delay(load_cloud, self.beta_mid, self.f_cloud)

    def _linearized_c5_for_pair(
        self,
        k: int,
        v: int,
        x_ref: np.ndarray,
        y_ref: np.ndarray,
        z_loc_ref: np.ndarray,
        z_mec_ref: np.ndarray,
        z_cld_ref: np.ndarray,
    ):
        d_loc = self._delay_local(v)
        d_mec = np.array([self._delay_mec(e) for e in range(self.E)], dtype=float)
        d_cld = np.array([self._delay_cloud(e) for e in range(self.E)], dtype=float)

        d_exe_ref = z_loc_ref[k, v] * d_loc
        d_exe_ref += float(np.dot(z_mec_ref[k, v, :], d_mec))
        d_exe_ref += float(np.dot(z_cld_ref[k, v, :], d_cld))

        d_eff_ref = min(d_exe_ref, self.dmax_wait)
        h_ref_mob = _mission_time_from_delay(
            self.road_k[k],
            d_eff_ref,
            int(self.n_tasks_k[k]),
            self.v_nom,
            self.rho_down,
            self.rho_up,
        )
        h_ref = self.t_start_to_mission_kv[k, v] + h_ref_mob
        dh_dd = _mission_time_grad_delay(
            self.road_k[k],
            d_eff_ref,
            int(self.n_tasks_k[k]),
            self.v_nom,
            self.rho_down,
            self.rho_up,
        )

        if np.isfinite(self.dmax_wait):
            if d_exe_ref < self.dmax_wait - 1e-9:
                dd_ddexe = 1.0
            elif d_exe_ref > self.dmax_wait + 1e-9:
                dd_ddexe = 0.0
            else:
                dd_ddexe = 0.5
        else:
            dd_ddexe = 1.0

        slope = dh_dd * dd_ddexe
        row = np.zeros(self._n_var, dtype=float)

        g_zloc = slope * d_loc
        row[self._idx_z_loc(k, v)] = g_zloc
        g_dot_z_ref = g_zloc * z_loc_ref[k, v]

        for e in range(self.E):
            g_mec = slope * d_mec[e]
            g_cld = slope * d_cld[e]
            row[self._idx_z_mec(k, v, e)] = g_mec
            row[self._idx_z_cld(k, v, e)] = g_cld
            g_dot_z_ref += g_mec * z_mec_ref[k, v, e] + g_cld * z_cld_ref[k, v, e]

        row[self._idx_y(k)] = self.big_m_time
        row[self._idx_x(k, v)] = self.big_m_assign

        rhs = (
            self.tau_sec
            + self.big_m_time
            + self.big_m_assign
            - h_ref
            + g_dot_z_ref
        )
        return row, rhs

    def _pack_solution(self, sol: np.ndarray):
        x = np.zeros((self.K, self.V), dtype=float)
        y = np.zeros(self.K, dtype=float)
        z_loc = np.zeros((self.K, self.V), dtype=float)
        z_mec = np.zeros((self.K, self.V, self.E), dtype=float)
        z_cld = np.zeros((self.K, self.V, self.E), dtype=float)
        t_start = np.zeros(self.K, dtype=float)
        t_finish = np.zeros(self.K, dtype=float)
        order = np.zeros((self.K, self.K, self.V), dtype=float)

        for k in range(self.K):
            y[k] = sol[self._idx_y(k)]
            t_start[k] = sol[self._idx_t_start(k)]
            t_finish[k] = sol[self._idx_t_finish(k)]
            for v in range(self.V):
                x[k, v] = sol[self._idx_x(k, v)]
                z_loc[k, v] = sol[self._idx_z_loc(k, v)]
                for e in range(self.E):
                    z_mec[k, v, e] = sol[self._idx_z_mec(k, v, e)]
                    z_cld[k, v, e] = sol[self._idx_z_cld(k, v, e)]
                for m in range(self.K):
                    if k == m:
                        continue
                    order[k, m, v] = sol[self._idx_order(k, m, v)]
        return x, y, z_loc, z_mec, z_cld, t_start, t_finish, order

    def lp_schedule_to_sol(self, x_lp: np.ndarray, t_start_lp: np.ndarray, y_lp: np.ndarray = None) -> list:
        """Build ITS mission tuple solution (order, vehicle) from LP schedule variables.

        Missions are assigned to the vehicle with highest relaxed x value and
        ordered by LP start time on that vehicle.
        """
        assigned_by_vehicle = {v: [] for v in range(self.V)}
        for k in range(self.K):
            if y_lp is not None and y_lp[k] <= 1e-8:
                continue
            row = x_lp[k]
            if np.max(row) <= 1e-8:
                continue
            v = int(np.argmax(row))
            assigned_by_vehicle[v].append((k, float(t_start_lp[k]), float(row[v])))

        sol = [None] * self.K
        used = np.zeros(self.K, dtype=bool)
        load = np.zeros(self.V, dtype=int)

        for v in range(self.V):
            # Stable sorting: earliest t_start first, then stronger assignment score.
            missions_v = sorted(
                assigned_by_vehicle[v],
                key=lambda x: (self._dep_level[x[0]], x[1], -x[2], x[0]),
            )
            for order_idx, (k, _, _) in enumerate(missions_v):
                sol[k] = (order_idx, v)
                used[k] = True
                load[v] += 1

        # Keep unassigned missions out of all vehicle queues.
        for k in range(self.K):
            if used[k]:
                continue
            sol[k] = (-1, -1)

        return sol

    def _round_assignment_dependency_aware(self, x_lp: np.ndarray, y_lp: np.ndarray) -> np.ndarray:
        """Greedy rounding that enforces capacity and dependency closure."""
        x_int = np.zeros((self.K, self.V), dtype=int)
        capacity = np.zeros(self.V, dtype=int)

        selected = set()
        unselected = set(range(self.K))
        # Prefer missions with higher completion confidence and stronger assignment mass.
        score = y_lp + x_lp.max(axis=1)

        while True:
            candidates = []
            for k in list(unselected):
                deps = self._deps_by_mission.get(k, [])
                if all(dep in selected for dep in deps):
                    candidates.append(k)
            if not candidates:
                break

            # Favor dependency-ready, high-score missions first.
            candidates.sort(key=lambda k: (-score[k], self._dep_level[k], k))

            progress = False
            for k in candidates:
                v_order = np.argsort(-x_lp[k])
                placed = False
                for v in v_order:
                    if capacity[v] >= self.M:
                        continue
                    if x_lp[k, v] <= 1e-8:
                        continue
                    x_int[k, v] = 1
                    capacity[v] += 1
                    selected.add(k)
                    unselected.remove(k)
                    placed = True
                    progress = True
                    break
                if placed:
                    break

            if not progress:
                # No candidate can be placed under current capacities / scores.
                break

        return x_int

    def _solve_linearized_lp(
        self,
        x_ref: np.ndarray,
        y_ref: np.ndarray,
        z_loc_ref: np.ndarray,
        z_mec_ref: np.ndarray,
        z_cld_ref: np.ndarray,
    ):
        c = np.zeros(self._n_var, dtype=float)
        for k in range(self.K):
            c[self._idx_y(k)] = -self.omega1 * self.utility_k[k]
            for v in range(self.V):
                for e in range(self.E):
                    c[self._idx_z_mec(k, v, e)] = (
                        self.omega2 * self.unit_cost_mec * self.n_tasks_k[k] * self._delay_mec(e)
                    )
                    c[self._idx_z_cld(k, v, e)] = (
                        self.omega2 * self.cloud_unit_cost * self.n_tasks_k[k] * self._delay_cloud(e)
                    )

        rows_ub = []
        rhs_ub = []

        rows_eq = []
        rhs_eq = []

        # C1: each mission assigned to at most one vehicle.
        for k in range(self.K):
            row = np.zeros(self._n_var, dtype=float)
            for v in range(self.V):
                row[self._idx_x(k, v)] = 1.0
            rows_ub.append(row)
            rhs_ub.append(1.0)

        # vehicle mission capacity.
        for v in range(self.V):
            row = np.zeros(self._n_var, dtype=float)
            for k in range(self.K):
                row[self._idx_x(k, v)] = 1.0
            rows_ub.append(row)
            rhs_ub.append(float(self.M))

        # y_k <= sum_v x_{k,v}
        for k in range(self.K):
            row = np.zeros(self._n_var, dtype=float)
            row[self._idx_y(k)] = 1.0
            for v in range(self.V):
                row[self._idx_x(k, v)] -= 1.0
            rows_ub.append(row)
            rhs_ub.append(0.0)

        # dependency: y_k <= y_dep
        for (k, k_dep) in self._dep_pairs:
            row = np.zeros(self._n_var, dtype=float)
            row[self._idx_y(k)] = 1.0
            row[self._idx_y(k_dep)] = -1.0
            rows_ub.append(row)
            rhs_ub.append(0.0)

        # Duration activation and completion deadlines.
        for k in range(self.K):
            # t_finish_k >= t_start_k + h0_k * sum_v x_{k,v}
            row = np.zeros(self._n_var, dtype=float)
            row[self._idx_t_start(k)] = 1.0
            row[self._idx_t_finish(k)] = -1.0
            for v in range(self.V):
                row[self._idx_x(k, v)] = self.h0_k[k]
            rows_ub.append(row)
            rhs_ub.append(0.0)

            # t_start_k <= tau * sum_v x_{k,v}
            row = np.zeros(self._n_var, dtype=float)
            row[self._idx_t_start(k)] = 1.0
            for v in range(self.V):
                row[self._idx_x(k, v)] = -self.tau_sec
            rows_ub.append(row)
            rhs_ub.append(0.0)

            # t_finish_k <= tau * sum_v x_{k,v}
            row = np.zeros(self._n_var, dtype=float)
            row[self._idx_t_finish(k)] = 1.0
            for v in range(self.V):
                row[self._idx_x(k, v)] = -self.tau_sec
            rows_ub.append(row)
            rhs_ub.append(0.0)

            # Completed mission must finish within window: t_finish_k <= tau + M*(1 - y_k)
            row = np.zeros(self._n_var, dtype=float)
            row[self._idx_t_finish(k)] = 1.0
            row[self._idx_y(k)] = self.big_m_time
            rows_ub.append(row)
            rhs_ub.append(self.tau_sec + self.big_m_time)

        # Dependency waiting: mission k can start only after each dependency finishes.
        for (k, k_dep) in self._dep_pairs:
            row = np.zeros(self._n_var, dtype=float)
            row[self._idx_t_start(k)] = -1.0
            row[self._idx_t_finish(k_dep)] = 1.0
            row[self._idx_y(k)] = self.big_m_time
            rows_ub.append(row)
            rhs_ub.append(self.big_m_time)

        # Base travel-time feasibility: if h0_k > tau then y_k must be 0.
        for k in range(self.K):
            row = np.zeros(self._n_var, dtype=float)
            row[self._idx_y(k)] = 1.0
            rows_ub.append(row)
            rhs_ub.append(self.base_feasible_k[k])

        # Per-vehicle time budget in one decision window (tightening cut).
        for v in range(self.V):
            row = np.zeros(self._n_var, dtype=float)
            for k in range(self.K):
                row[self._idx_x(k, v)] = self.h0_kv[k, v]
            rows_ub.append(row)
            rhs_ub.append(self.tau_sec)

        # C7-like MEC wireless admission: sum_{k,j,v} z_mec <= n_e.
        for e in range(self.E):
            row = np.zeros(self._n_var, dtype=float)
            for k in range(self.K):
                for v in range(self.V):
                    row[self._idx_z_mec(k, v, e)] = self.n_tasks_k[k]
            rows_ub.append(row)
            rhs_ub.append(self.n_rb)

        # Linearized C5 with assignment activation big-M.
        for k in range(self.K):
            for v in range(self.V):
                row, rhs = self._linearized_c5_for_pair(
                    k, v, x_ref, y_ref, z_loc_ref, z_mec_ref, z_cld_ref
                )
                rows_ub.append(row)
                rhs_ub.append(rhs)

        # Queue-order constraints on each vehicle with pairwise order variables.
        # If two missions are assigned to the same vehicle, one must precede the other,
        # and their execution intervals cannot overlap.
        for v in range(self.V):
            for l in range(self.K):
                for m in range(l + 1, self.K):
                    o_lm = self._idx_order(l, m, v)
                    o_ml = self._idx_order(m, l, v)

                    # o_lm + o_ml <= 1
                    row = np.zeros(self._n_var, dtype=float)
                    row[o_lm] = 1.0
                    row[o_ml] = 1.0
                    rows_ub.append(row)
                    rhs_ub.append(1.0)

                    # x_lv + x_mv - o_lm - o_ml <= 1
                    row = np.zeros(self._n_var, dtype=float)
                    row[self._idx_x(l, v)] = 1.0
                    row[self._idx_x(m, v)] = 1.0
                    row[o_lm] = -1.0
                    row[o_ml] = -1.0
                    rows_ub.append(row)
                    rhs_ub.append(1.0)

                    # o_lm <= x_lv and o_lm <= x_mv
                    row = np.zeros(self._n_var, dtype=float)
                    row[o_lm] = 1.0
                    row[self._idx_x(l, v)] = -1.0
                    rows_ub.append(row)
                    rhs_ub.append(0.0)

                    row = np.zeros(self._n_var, dtype=float)
                    row[o_lm] = 1.0
                    row[self._idx_x(m, v)] = -1.0
                    rows_ub.append(row)
                    rhs_ub.append(0.0)

                    # o_ml <= x_lv and o_ml <= x_mv
                    row = np.zeros(self._n_var, dtype=float)
                    row[o_ml] = 1.0
                    row[self._idx_x(l, v)] = -1.0
                    rows_ub.append(row)
                    rhs_ub.append(0.0)

                    row = np.zeros(self._n_var, dtype=float)
                    row[o_ml] = 1.0
                    row[self._idx_x(m, v)] = -1.0
                    rows_ub.append(row)
                    rhs_ub.append(0.0)

                    # If l is before m on vehicle v: t_start_m >= t_finish_l - M*(1 - o_lm)
                    row = np.zeros(self._n_var, dtype=float)
                    row[self._idx_t_start(m)] = -1.0
                    row[self._idx_t_finish(l)] = 1.0
                    row[o_lm] = self.big_m_order
                    rows_ub.append(row)
                    rhs_ub.append(self.big_m_order - self.t_move_lm[l, m])

                    # If m is before l on vehicle v: t_start_l >= t_finish_m - M*(1 - o_ml)
                    row = np.zeros(self._n_var, dtype=float)
                    row[self._idx_t_start(l)] = -1.0
                    row[self._idx_t_finish(m)] = 1.0
                    row[o_ml] = self.big_m_order
                    rows_ub.append(row)
                    rhs_ub.append(self.big_m_order - self.t_move_lm[m, l])

        # C2: execution destination consistency per mission-vehicle pair.
        for k in range(self.K):
            for v in range(self.V):
                row = np.zeros(self._n_var, dtype=float)
                row[self._idx_z_loc(k, v)] = 1.0
                row[self._idx_x(k, v)] = -1.0
                for e in range(self.E):
                    row[self._idx_z_mec(k, v, e)] += 1.0
                    row[self._idx_z_cld(k, v, e)] += 1.0
                rows_eq.append(row)
                rhs_eq.append(0.0)

        bounds = [(0.0, 1.0)] * self._n_var

        # Time variables live in [0, tau].
        for k in range(self.K):
            bounds[self._idx_t_start(k)] = (0.0, self.tau_sec)
            bounds[self._idx_t_finish(k)] = (0.0, self.tau_sec)

        A_ub = np.array(rows_ub, dtype=float) if rows_ub else None
        b_ub = np.array(rhs_ub, dtype=float) if rhs_ub else None
        A_eq = np.array(rows_eq, dtype=float) if rows_eq else None
        b_eq = np.array(rhs_eq, dtype=float) if rhs_eq else None
        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if not result.success:
            zeros = np.zeros(self._n_var, dtype=float)
            x_star, y_star, z_loc_star, z_mec_star, z_cld_star, t_start_star, t_finish_star, order_star = self._pack_solution(zeros)
            return (
                x_star,
                y_star,
                z_loc_star,
                z_mec_star,
                z_cld_star,
                t_start_star,
                t_finish_star,
                order_star,
                0.0,
                False,
                int(result.status),
                str(result.message),
            )

        x_star, y_star, z_loc_star, z_mec_star, z_cld_star, t_start_star, t_finish_star, order_star = self._pack_solution(result.x)
        obj = float(-result.fun)
        return (
            x_star,
            y_star,
            z_loc_star,
            z_mec_star,
            z_cld_star,
            t_start_star,
            t_finish_star,
            order_star,
            obj,
            True,
            int(result.status),
            str(result.message),
        )

    def solve(self, verbose: bool = True) -> tuple:
        x_ref = np.zeros((self.K, self.V), dtype=float)
        y_ref = np.zeros(self.K, dtype=float)
        z_loc_ref = np.zeros((self.K, self.V), dtype=float)
        z_mec_ref = np.zeros((self.K, self.V, self.E), dtype=float)
        z_cld_ref = np.zeros((self.K, self.V, self.E), dtype=float)

        ub_history = []
        converged = False
        final_obj_rel = np.inf
        final_var_delta = np.inf
        stall_count = 0
        x_star = np.zeros_like(x_ref)
        y_star = np.zeros_like(y_ref)
        z_loc_star = np.zeros_like(z_loc_ref)
        z_mec_star = np.zeros_like(z_mec_ref)
        z_cld_star = np.zeros_like(z_cld_ref)
        t_start_star = np.zeros(self.K, dtype=float)
        t_finish_star = np.zeros(self.K, dtype=float)
        order_star = np.zeros((self.K, self.K, self.V), dtype=float)
        lp_failures = 0
        lp_status_last = 0
        lp_message_last = ""

        if verbose:
            print(
                f"SCA Upper Bound | K={self.K}, V={self.V}, M={self.M}, tau={self.tau_sec}s, "
                f"E={self.E}, gamma={self.gamma:.2f}, avg_d_task={self.avg_d_task:.6f}s, "
                f"min_iter={self.min_iter}, obj_rel_tol={self.obj_rel_tol:.1e}, var_tol={self.var_tol:.1e}"
            )

        for t in range(self.max_iter):
            (
                x_star,
                y_star,
                z_loc_star,
                z_mec_star,
                z_cld_star,
                t_start_star,
                t_finish_star,
                order_star,
                ub,
                lp_ok,
                lp_status,
                lp_message,
            ) = self._solve_linearized_lp(
                x_ref,
                y_ref,
                z_loc_ref,
                z_mec_ref,
                z_cld_ref,
            )

            lp_status_last = int(lp_status)
            lp_message_last = str(lp_message)

            if not lp_ok:
                lp_failures += 1
                ub = ub_history[-1] if ub_history else 0.0
                ub_history.append(ub)
                if verbose:
                    n_assigned = int((x_ref.sum(axis=1) > 1e-6).sum())
                    n_completed = int((y_ref > 1e-6).sum())
                    print(
                        f"  iter {t:3d} | UB={ub:10.4f} | assigned={n_assigned}/{self.K} | "
                        f"completed={n_completed}/{self.K} | LP status={lp_status}: kept previous iterate"
                    )
                continue

            ub_history.append(ub)

            if verbose:
                n_assigned = int((x_star.sum(axis=1) > 1e-6).sum())
                n_completed = int((y_star > 1e-6).sum())
                obj_rel = np.inf
                if t > 0:
                    denom = max(1.0, abs(ub_history[-2]))
                    obj_rel = abs(ub_history[-1] - ub_history[-2]) / denom
                print(
                    f"  iter {t:3d} | UB={ub:10.4f} | assigned={n_assigned}/{self.K} | "
                    f"completed={n_completed}/{self.K} | d_obj_rel={obj_rel:.2e}"
                )

            x_next = x_ref + self.gamma * (x_star - x_ref)
            y_next = y_ref + self.gamma * (y_star - y_ref)
            z_loc_next = z_loc_ref + self.gamma * (z_loc_star - z_loc_ref)
            z_mec_next = z_mec_ref + self.gamma * (z_mec_star - z_mec_ref)
            z_cld_next = z_cld_ref + self.gamma * (z_cld_star - z_cld_ref)

            final_var_delta = max(
                float(np.max(np.abs(x_next - x_ref))),
                float(np.max(np.abs(y_next - y_ref))),
                float(np.max(np.abs(z_loc_next - z_loc_ref))),
                float(np.max(np.abs(z_mec_next - z_mec_ref))),
                float(np.max(np.abs(z_cld_next - z_cld_ref))),
            )

            x_ref = x_next
            y_ref = y_next
            z_loc_ref = z_loc_next
            z_mec_ref = z_mec_next
            z_cld_ref = z_cld_next
            self._update_queue_delays(z_loc_ref, z_mec_ref, z_cld_ref)

            if t > 0:
                denom = max(1.0, abs(ub_history[-2]))
                final_obj_rel = abs(ub_history[-1] - ub_history[-2]) / denom
            else:
                final_obj_rel = np.inf

            if t + 1 >= self.min_iter:
                obj_stable = final_obj_rel < self.obj_rel_tol
                var_stable = final_var_delta < self.var_tol
                if obj_stable and var_stable:
                    stall_count += 1
                else:
                    stall_count = 0

                if stall_count >= self.conv_patience:
                    converged = True
                    if verbose:
                        print(
                            f"  Converged at iter {t} "
                            f"(d_obj_rel={final_obj_rel:.2e}, d_var={final_var_delta:.2e}, "
                            f"patience={self.conv_patience})"
                        )
                    break

        x_lp = x_ref.copy()
        d_exe_final = self._compute_d_exe(z_loc_ref, z_mec_ref, z_cld_ref)
        x_int = self._round_assignment_dependency_aware(x_lp, y_ref)

        # Build ITS queue order from rounded assignment (capacity/dependency-safe)
        # and LP start-time priorities.
        sol_lp = self.lp_schedule_to_sol(x_int.astype(float), t_start_star, y_star)

        info = {
            "converged": converged,
            "n_iter": len(ub_history),
            "lp_failures": int(lp_failures),
            "lp_status_last": int(lp_status_last),
            "lp_message_last": lp_message_last,
            "obj_rel_final": float(final_obj_rel),
            "var_delta_final": float(final_var_delta),
            "d_task_final": d_exe_final,
            "dep_pairs": self._dep_pairs,
            "y_relaxed": y_ref,
            "z_loc_relaxed": z_loc_ref,
            "z_mec_relaxed": z_mec_ref,
            "z_cld_relaxed": z_cld_ref,
            "t_start_lp": t_start_star,
            "t_finish_lp": t_finish_star,
            "order_lp": order_star,
            "sol_lp": sol_lp,
        }
        return ub_history, x_lp, x_int, info

    def assignment_to_sol(self, x_int: np.ndarray) -> list:
        """
        Convert integer assignment matrix (K×V) to the (order, vehicle_id)
        tuple list expected by Vehicle.set_mission(..., mtuple=True).

        sol[k] = (order_within_vehicle, vehicle_id)
        Unassigned missions are marked as (-1, -1).
        """
        sol = [None] * self.K
        order_counter = np.zeros(self.V, dtype=int)

        for k in range(self.K):
            assigned_v = int(np.argmax(x_int[k]))
            if x_int[k, assigned_v] == 1:
                sol[k] = (int(order_counter[assigned_v]), assigned_v)
                order_counter[assigned_v] += 1
            else:
                sol[k] = (-1, -1)

        return sol

    def evaluate_rounded(self, x_int: np.ndarray, decoded_data: list,
                        segments: list, graph, lmap,
                        verbose: bool = False,
                        sol: list = None,
                        initial_positions: list = None) -> dict:
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

        if sol is None:
            sol = self.assignment_to_sol(x_int)

        # Build vehicles
        vehicles = []
        for v in range(self.V):
            if initial_positions is not None and len(initial_positions) == self.V:
                init_pos = initial_positions[v]
            else:
                seg = self.rng.choice(segments)
                init_pos = seg.get_endpoints()[0]
            veh = Vehicle(0.5, init_pos, lmap,
                        task_cfg['tau'], verbose=verbose)
            vehicles.append(veh)

        # Build missions
        missions = []
        for item in decoded_data:
            m = Mission(item['depart_p'], item['depart_s'], 1,
                        graph=graph, verbose=verbose)
            m.set_depends(item['depends'])
            m.set_observers(vehicles)
            m.set_mid(item['i'])
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
