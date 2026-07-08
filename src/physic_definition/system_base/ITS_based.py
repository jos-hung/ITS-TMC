import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')) 
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')) 
sys.path.append(".")
from physic_definition.map.decas import Point, Line, Segment
import matplotlib.pyplot as plt
from physic_definition.map.graph import Graph
import json
import abc
from configs.systemcfg import (
    mission_cfg,
    task_cfg,
    map_cfg,
    vehicle_cfg,
    network_cfg,
    apply_early_stopping,
    apply_offloading_based_max_allow_average_delay,
)
from physic_definition.map.map import *
from physic_definition.network.rate import *
import time
import threading

# -----------------------------------------------------------------------
# Mobility Physics Functions (Paper Section II.C, Eqs. 13–20)
# These model the parallel execution of robot movement and computation.
# -----------------------------------------------------------------------

def mobility_v_safe(u, v_kv, rho_down):
    """
    Safe speed of robot v during waiting interval of task j at elapsed time u (Eq. 13).

    v^safe_{k,v,j}(u) = max(0, v_{k,v} - rho^down_v * u)

    Args:
        u       : elapsed waiting time since task j started (s)
        v_kv    : nominal traveling speed during mission k (m/s)
        rho_down: safe deceleration rate rho^down_v (m/s^2)
    Returns:
        current safe speed (m/s); 0 when robot has fully stopped
    """
    return max(0.0, v_kv - rho_down * u)


def mobility_d_up(v_kv, d_kj, rho_down, rho_up):
    """
    Required speed recovery time after receiving result of control task j (Eq. 14).

    d^up_{k,v,j} = (v_{k,v} - v^safe_{k,v,j}(d_{k,j})) / rho^up_v

    Args:
        v_kv    : nominal traveling speed (m/s)
        d_kj    : effective waiting time d_{k,j} = min(exe_delay, d^max_{k,j}) (Eq. 12)
        rho_down: deceleration rate (m/s^2)
        rho_up  : acceleration rate rho^up_v (m/s^2)
    Returns:
        recovery time (s)
    """
    v_safe = mobility_v_safe(d_kj, v_kv, rho_down)
    return (v_kv - v_safe) / rho_up if rho_up > 0 else 0.0


def mobility_v_wait_avg(d_kj, v_kv, rho_down):
    """
    Average speed of robot v during the waiting (deceleration) phase of task j (Eq. 18).

    v^wait_{k,v,j} = v_{k,v} - 0.5*rho^down*d_{k,j}         if d_{k,j} <= v_{k,v}/rho^down
                   = v_{k,v}^2 / (2*rho^down*d_{k,j})        if d_{k,j} > v_{k,v}/rho^down

    Derived from Eq. (15): integral of v^safe_{k,v,j}(u) over [0, d_{k,j}].

    Args:
        d_kj    : effective waiting time (s)
        v_kv    : nominal traveling speed (m/s)
        rho_down: deceleration rate (m/s^2)
    Returns:
        average speed during deceleration/stop phase (m/s)
    """
    if d_kj <= 0:
        return v_kv
    if rho_down <= 0:
        return v_kv
    threshold = v_kv / rho_down
    if d_kj <= threshold:
        return v_kv - 0.5 * rho_down * d_kj
    else:
        return (v_kv ** 2) / (2.0 * rho_down * d_kj)

def mobility_v_up_avg(v_kv, d_kj, rho_down):
    """
    Average speed of robot v during the speed recovery phase (Eq. 17).

    v^up_{k,v,j} = (v^safe_{k,v,j}(d_{k,j}) + v_{k,v}) / 2

    Args:
        v_kv    : nominal traveling speed (m/s)
        d_kj    : effective waiting time (s)
        rho_down: deceleration rate (m/s^2)
    Returns:
        average speed during acceleration recovery phase (m/s)
    """
    v_safe = mobility_v_safe(d_kj, v_kv, rho_down)
    return (v_safe + v_kv) / 2.0


def mobility_S(d_nom, task_delays, v_kv, rho_down, rho_up):
    """
    Total distance traveled by robot v within the nominal time budget d_nom (Eq. 19).

    S_{k,v} = (d_nom - sum_j(d_{k,j} + d^up_{k,v,j})) * v_{k,v}
            + sum_j(d_{k,j} * v^wait_{k,v,j} + d^up_{k,v,j} * v^up_{k,v,j})

    Key insight: d_nom is a FIXED TIME WINDOW, not the actual travel time.
    S_{k,v} <= road_length when there are delays, so the robot is short on
    distance and needs extra time (d_nom + (road - S)/v_kv) to finish.

    Assumption (non-overlap): each task j is assumed to arrive after the robot
    has fully recovered to v_kv from task j-1.  If tasks overlap, the formula
    underestimates the total slowdown.

    Args:
        d_nom      : nominal driving time (s) = road_length / v_kv
        task_delays: list of effective waiting times d_{k,j} (s) per task j
        v_kv       : nominal traveling speed (m/s)
        rho_down   : deceleration rate (m/s^2)
        rho_up     : acceleration rate (m/s^2)
    Returns:
        S (m): distance covered in d_nom seconds under computation-induced slowdown
    """
    if d_nom <= 0 or v_kv <= 0:
        return d_nom * v_kv  # = road_length, no delays

    sum_slow = 0.0
    sum_weighted = 0.0
    for d_kj in task_delays:
        d_up = mobility_d_up(v_kv, d_kj, rho_down, rho_up)
        vw   = mobility_v_wait_avg(d_kj, v_kv, rho_down)
        vu   = mobility_v_up_avg(v_kv, d_kj, rho_down)
        sum_slow     += d_kj + d_up
        sum_weighted += d_kj * vw + d_up * vu

    if sum_slow >= d_nom:
        # All of d_nom is consumed by slow phases (tasks overlap).
        # The paper's non-overlap term (d_nom - sum_slow)*v_kv is negative and
        # physically meaningless here.  Instead, scale the weighted distance
        # proportionally to the fraction of slow-time that fits in d_nom:
        #   S = sum_weighted * (d_nom / sum_slow)
        #   v_eff = S / d_nom = sum_weighted / sum_slow <= v_kv 
        # Proof: vw_j <= v_kv and vu_j <= v_kv
        #   => sum_weighted <= v_kv * sum_slow
        #   => sum_weighted / sum_slow <= v_kv 
        return sum_weighted * (d_nom / sum_slow)

    # Normal case (non-overlap holds): nominal-speed portion + slow phases.
    return (d_nom - sum_slow) * v_kv + sum_weighted


def mobility_v_eff(d_nom, task_delays, v_kv, rho_down, rho_up):
    """
    Effective average speed v^eff_{k,v} = S_{k,v} / d_nom (Eq. 20).

    Used in paper analysis:
      - Theorem 1 feasibility check (Eq. 26): v_eff >= |r_k| / tau
      - Maximum tolerable delay bound (Eq. 23)
      - Constraint C5 deadline feasibility in the optimization problem (P1)

    NOTE: for simulation travel-time computation use mobility_seg_time() instead,
    which gives the exact value t = d_nom + (road - S) / v rather than this
    conservative approximation road / v_eff = road^2 / (v * S).
    """
    if d_nom <= 0 or v_kv <= 0:
        return v_kv
    S = mobility_S(d_nom, task_delays, v_kv, rho_down, rho_up)
    v_eff = S / d_nom
    # Strict physical bound: 0 <= v_eff <= v_kv.
    # Only allow snap when deviation is tiny floating-point noise.
    eps = 1e-9 * max(1.0, abs(v_kv))
    if v_eff > v_kv:
        if np.isclose(v_eff, v_kv, rtol=0.0, atol=eps):
            v_eff = v_kv
        else:
            raise ValueError(
                f"Calculated effective speed {v_eff:.6f} exceeds nominal speed {v_kv:.6f}. "
                f"d_nom={d_nom:.6f}, S={S:.6f}"
            )
    elif v_eff < 0.0:
        if np.isclose(v_eff, 0.0, rtol=0.0, atol=eps):
            v_eff = 0.0
        else:
            raise ValueError(
                f"Calculated effective speed {v_eff:.6f} is negative. "
                f"d_nom={d_nom:.6f}, S={S:.6f}"
            )
    
    if os.environ.get("ITS_VERBOSE_MOBILITY", "0") == "1":
        print(
            f"mobility_v_eff: d_nom={d_nom:.2f}s, S={S:.2f}m, "
            f"v_kv={v_kv:.2f}m/s => v_eff={v_eff:.2f}m/s"
        )
    return v_eff


def mobility_seg_time(road_length, task_delays, v_kv, rho_down, rho_up):
    """
    Exact actual travel time for a road segment under computation-induced slowdown.

    Derivation:
        d_nom = road_length / v_kv          (nominal time at full speed)
        S     = mobility_S(d_nom, ...)      (distance covered in d_nom seconds)
        extra = (road_length - S) / v_kv    (extra time to cover the gap at v_kv)
        t_actual = d_nom + extra
            = road_length/v_kv + (road_length - S)/v_kv
            = (2*road_length - S) / v_kv

    Why this differs from road_length / v_eff (the paper's Eq. 21 approximation):
        road_length / v_eff  = road_length * d_nom / S  = road_length^2 / (v_kv * S)
        t_actual             = (2*road_length - S) / v_kv
    They agree only when S = road_length (no delays).  The paper's form is a
    conservative approximation (slightly overestimates travel time).

    Args:
        road_length: segment or mission route length (m)
        task_delays: list of d_{k,j} values (s)
        v_kv       : nominal speed (m/s)
        rho_down / rho_up: decel / accel rates (m/s^2)
    Returns:
        actual travel time (s) >= road_length / v_kv
    """
    if road_length <= 0 or v_kv <= 0:
        return 0.0
    d_nom = road_length / v_kv
    S = mobility_S(d_nom, task_delays, v_kv, rho_down, rho_up)
    # Remaining distance after d_nom seconds, covered at nominal speed
    return d_nom + max(0.0, road_length - S) / v_kv


def max_allowable_average_delay(
    road_length,
    tau_budget,
    v_nom,
    rho_down,
    rho_up,
    n_tasks,
    d_hi_init=0.5,
    d_hi_max=60.0,
    max_iter=30,
):
    """Compute Theorem-1-inspired max average delay threshold for one route.

    This helper numerically finds the largest per-task average delay d_bar such
    that the route can still be completed inside tau_budget under the mobility
    degradation model (Eqs. 19-21).
    """
    if road_length <= 0 or v_nom <= 0 or tau_budget <= 0:
        return 0.0
    if n_tasks <= 0:
        return float("inf")

    d_nom = road_length / v_nom
    if d_nom > tau_budget:
        return 0.0

    def _travel_time(d_bar):
        v_eff = mobility_v_eff(d_nom, [d_bar] * n_tasks, v_nom, rho_down, rho_up)
        return road_length / max(v_eff, 1e-9)

    if _travel_time(0.0) > tau_budget:
        return 0.0

    lo = 0.0
    hi = max(d_hi_init, 1e-6)
    while hi < d_hi_max and _travel_time(hi) <= tau_budget:
        lo = hi
        hi *= 2.0

    if hi >= d_hi_max and _travel_time(hi) <= tau_budget:
        return hi

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if _travel_time(mid) <= tau_budget:
            lo = mid
        else:
            hi = mid
    return lo


def choose_delay_with_threshold(remote_delay, max_avg_delay, local_delay=None):
    """Select effective delay under max allowable average delay threshold.

    Priority: keep the fastest feasible destination under threshold screening.
    If no remote option passes the screening, fall back to local execution.

    Returns:
        (selected_delay, use_remote)
    """
    if not np.isfinite(max_avg_delay):
        if local_delay is None or remote_delay <= local_delay:
            return remote_delay, True
        return local_delay, False

    remote_ok = remote_delay <= max_avg_delay
    local_ok = local_delay is not None and local_delay <= max_avg_delay

    if remote_ok and (local_delay is None or not local_ok or remote_delay <= local_delay):
        return remote_delay, True
    if local_delay is not None:
        return local_delay, False
    return remote_delay, True

'''
the below is observer pattern to notify the finish task
'''
class Observer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, mission):
        #when the mission is finished the update will notify for the whole cars
        #that a mission already finish, then the cars check that there task ready to process or not
        pass
    
class Subject(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def register_observer(self, observer_obj):
        pass
    @abc.abstractmethod
    def remove_observer(self, observer_obj):
        pass
    @abc.abstractmethod
    def notify_observer(self):
        pass
    

class Mission(Subject):
    """A class to represent a mission in the ITS-based system."""    
    missionID = 0
    status = ["created", "inprocess", "done"]
    def __init__(self, dpart, desti, tslot, graph, verbose=False):
        self.__taskid = Mission.missionID # the task id of the mission
        self.__status = 0 #the status
        self.__dpart = dpart
        self.__desti = desti
        self.__tslot = tslot
        Mission.missionID += 1
        self.__depend_l = []
        
        self.__obervers = []
        self.__graph = graph
        
        self.__best_road = self.__graph.dijkstra(self.__dpart, self.__desti)
        self.__long = self.__graph.get_min_long()
        
        self.__profit = 50 #temp fix, we can change later
        self.verbose = verbose
        self.__original_plan = None
        
    def add_original_plan(self, plan):
        self.__original_plan = plan
        return True
    
    def get_original_plan(self):
        return self.__original_plan
    
    def get_mission_destination(self):
        return self.__desti
    
    def set_mid(self, val):
        self.__taskid = val
    def get_infor_to_mission(self,p):
        #từ 1 vị trí đến vị trí bắt đầu nhiệm vụ
        best_road = self.__graph.dijkstra(p, self.__dpart)
        #giá trị min long thay đổi sau mỗi lần chạy dijkstra.
        distance = self.__graph.get_min_long()[0]
        return best_road, distance
    
    def reset(self):
        Mission.missionID = 0
    
    def get_long(self):
        return self.__long
    
    def __lt__(self, other):
        return self.__long[0] < other.get_long()[0]
    
    def __eq__(self, other):
        if isinstance(other, int) or isinstance(other, np.int64):
            return other == self.__taskid
        if isinstance(other, Mission):
            return other.get_mid() == self.__taskid
        if isinstance(other, Point):
            return other == self.__desti
            
    def get_profit(self):
        return self.__profit
    
    def set_profit(self, v):
        self.__profit = v
    
    def update_profit(self, v):
        self.__profit += v
        
    def get_best_road(self):
        return self.__best_road
    
    def in_other_road(self,mission):
        #this function help us to know that the current mission can be in a road of another mission or not.
        other_best_road = mission.get_best_road()
        for idx, p in enumerate(self.__best_road):
            if(idx > len(other_best_road)-1):
                return False
            if self.__best_road[idx]!=other_best_road[idx]:
                return False
        return True
    
    def get_mid(self):
        return self.__taskid
    
    #notify function
    def register_observer(self, observer_obj):
        if isinstance(observer_obj, list):
            for obs in observer_obj:
                if obs not in self.__obervers:
                    self.__obervers.append(obs)
        else:
            if observer_obj not in self.__obervers:
                self.__obervers.append(observer_obj)
    
    def remove_observer(self, observer_obj):
        self.__obervers.remove(observer_obj)
        
    def set_observers(self, observer_list):
        if isinstance(observer_list, list):
            self.__obervers = observer_list
    
    def notify_observer(self, time):
        n_remove_depends = 0
        for obs in self.__obervers:
            n_remove_depends += obs.update(self, time)
        return n_remove_depends
    def update_depend(self, id):
        self.__depend_l.append(id)
        if len(self.__depend_l) == 0:
            self.__status = 1
        else:
            self.__status = 0
    
    def set_depends(self, missions):
        self.__depend_l = missions 
        if len(self.__depend_l) ==0:
            self.__status = 1
    def remove_depend(self, id):
        self.__depend_l.remove(id)   
        if len(self.__depend_l) == 0:
            self.__status = 1
        else:
            self.__status = 0
    def get_depends(self):
        return self.__depend_l
    
    def update_status(self, value, missions = None, time = 0):
        """
        Updates the status of the mission and notifies dependent missions and observers.
        Args:
            value (int): The new status value to be set for the mission.
            missions (list, optional): A list of Mission objects that depend on this mission. Defaults to None.
            time (int, optional): The current time to be used for notifying observers. Defaults to 0.
        Returns:
            int: The number of dependencies removed and observers notified.
        Example:
            mission = Mission()
            dependent_mission = Mission()
            mission.update_status(1, [dependent_mission], 10)
        """
        
        
        self.__status = value
        n_remove_depends = 0
        if Mission.status[value] == "done" and missions!=None:
            if self.verbose:
                print("Nhiệm vụ {} bắt đầu thông báo cho các phương tiện khác rằng nó đã được hoàn thành".format(self.get_mid()))
            for mis in missions:
                if self.get_mid() in mis.get_depends():
                    mis.__depend_l.remove(self.get_mid())
                    n_remove_depends += 1
        if Mission.status[value] == "done":
            n_remove_depends += self.notify_observer(time)
            if self.verbose:
                print("Nhiệm vụ {} bắt đầu thông báo cho các phương tiện khác rằng nó đã được hoàn thành".format(self.get_mid()))
        return n_remove_depends
    
    def get_status(self):
        return self.__status
    
    def print_status(self):
        print(Mission.status[self.__status])
    
    def get_dpart(self):
        return self.__dpart
    
    def get_desti(self):
        return self.__desti

    def get_trajectory(self):
        return (self.__dpart, self.__desti)
    
    def get_tslot(self):
        return self.__tslot
    
class Vehicle(Observer):
    """
    A class to represent a vehicle in the ITS-based system.
    The class manages the vehicle's position, missions, and interactions with the map.
    It allows setting missions, processing them, and updating the vehicle's state based on the missions' status.
    It also provides methods to accept missions, check readiness, and process tasks.
    """
    
    id = 0
    def __init__(self, cpu_freqz, cur_pos, map, tau = 120, verbose=False, sts = 0, non_priority_orders=False,
                rho_down=None, rho_up=None, v_nominal=None):
        #sts: là giá trị thể hiện rằng liệu phương tiện có chọn task by task ko
        self.__cpu_freqz = cpu_freqz
        self.__cur_pos = cur_pos
        self.__missions = []
        self.__ready_mis = []
        self.__acceptance_mis = []
        self.__map = map
        self.__vehicle_prof = 0
        self.__profit_by_missions_id = {}
        self.__late = 0
        self.__vid = Vehicle.id
        Vehicle.id += 1
        self.__ctrl_time = 0
        self.__tau = tau*60
        self.__n_completes = 0
        self.__profit = 0
        self.verbose = verbose
        self.sts = sts
        self.__intime = True
        self.__mec = random_bs_num(map.get_intersections())
        self.__non_priority_orders = non_priority_orders
        self.__order = []

        # Mobility parameters for the parallel mobility-computation model (Paper Sec. II.C)
        # Falls back to vehicle_cfg defaults when not explicitly supplied.
        self.__v_nominal = v_nominal if v_nominal is not None else vehicle_cfg['v_nominal']
        self.__rho_down  = rho_down  if rho_down  is not None else vehicle_cfg['rho_down']
        self.__rho_up    = rho_up    if rho_up    is not None else vehicle_cfg['rho_up']

        # Proposed switches from configs.systemcfg.
        self.__apply_early_stopping =  apply_early_stopping
        self.__apply_offloading_delay_threshold = apply_offloading_based_max_allow_average_delay

        # Nominal average task delay used by early-stop feasibility estimation.
        alpha_mid = 0.5 * (task_cfg['comm_size'][0] + task_cfg['comm_size'][1]) * 8000.0
        beta_mid = 0.5 * (task_cfg['comp_size'][0] + task_cfg['comp_size'][1])
        avg_rate = 33e6
        avg_cpu = 0.5 * (network_cfg['CPU_freq'][0] + network_cfg['CPU_freq'][1])
        self.__avg_task_delay = alpha_mid / max(avg_rate, 1e-9) + beta_mid / max(avg_cpu, 1e-9)
        dmax_cfg = task_cfg.get("dmax_wait_sec", float("inf"))
        self.__dmax_wait = float(dmax_cfg) if dmax_cfg is not None else float("inf")
        if self.__dmax_wait <= 0:
            self.__dmax_wait = float("inf")
    
    def compute_original_plan(self):
        start_time = 0
        end_time = 0        
        for mis in self.__missions[:]:
            start_time = end_time
            end_time += mis.get_long()[0]/self.__v_nominal
            if end_time <= self.__tau:
                mis.add_original_plan((start_time, end_time))
            else:
                self.__missions.remove(mis)

    def set_plan_by_step(self, mission):
        start_time = self.__ctrl_time
        end_time = start_time + mission.get_long()[0]/self.__v_nominal
        if end_time <= self.__tau:
            return mission.add_original_plan((start_time, end_time))
        else:
            self.__intime = False
        return False

    def get_vid(self):
        return self.__vid
    
    def get_pos(self):
        return self.__cur_pos
    
    def set_pos(self,pos):
        self.__cur_pos=pos    
    
    def reset(self):
        Vehicle.id = 0
    
    def set_mission(self, sol, missions=[], mtuple = False):
        """
        Sets the mission for the vehicle based on the provided solution and missions.
        Parameters:
        sol (list, np.ndarray, int, np.int64): The solution which can be a list, numpy array, or an integer.
        missions (list): A list of mission objects.
        mtuple (bool): A flag indicating if the solution is a tuple containing order and vehicle information.
        Returns:
        bool: Returns True if there are ready missions, False otherwise.
        Raises:
        ValueError: If the input solution type is incorrect.
        Notes:
        - If `sol` is a list or numpy array and `mtuple` is False, it iterates through the solution to find the vehicle ID and sets the mission.
        - If `sol` is an integer or numpy int64, it directly sets the mission based on the index.
        - If `sol` is a list or numpy array and `mtuple` is True, it processes the solution as a tuple containing order and vehicle information.
        - The method updates the mission status and verifies readiness based on the mission dependencies and priority orders.
        """
        if self.__intime ==False:
            return
        if (isinstance(sol, list) or isinstance(sol, np.ndarray)) and not mtuple:
            for idx, val in enumerate(sol):
                #val = vehicle id
                if val == self.__vid:
                    i = missions.index(int(idx))
                    mis = missions[i]
                    self.accept_mission(miss=mis)
                    if len(mis.get_depends()) == 0:
                        mis.update_status(1)
                    if self.__non_priority_orders and mis.get_status() == 1:
                        self.__ready_mis.append(mis)
                        self.__missions.remove(mis)
                    else:
                        self.verify_ready()
        elif isinstance(sol, int) or isinstance(sol, np.int64):
            i = missions.index(sol) #index of mission
            mis = missions[i]
            if len(mis.get_depends()) == 0:
                mis.update_status(1)
            self.accept_mission(miss=mis)
            if self.__non_priority_orders and mis.get_status() == 1:
                self.__ready_mis.append(mis)
                self.__missions.remove(mis)
            else:
                self.verify_ready()
            if len(self.__ready_mis) == 0:
                return False
            return True
        elif (isinstance(sol, list) or isinstance(sol, np.ndarray)) and mtuple:
            completed_set = []
            for idx, val in enumerate(sol):
                #val = order and vehicle
                if val[1] == self.__vid and val not in completed_set:
                    try:
                        i = missions.index(int(idx))
                        mis = missions[i]
                    except Exception as e:
                        print(f"Error finding mission with index {idx}: {e}")
                        print(f"list of missions: {[m.get_mid() for m in missions]}")
                        exit(1)
                    if len(mis.get_depends()) == 0:
                        mis.update_status(1)                     
                    if self.__non_priority_orders and mis.get_status() == 1:
                        self.__ready_mis.append(mis)
                        self.__missions.remove(mis)
                    # else:
                    #     self.verify_ready()
                    self.accept_mission(miss=mis, order = val[0])
                    completed_set.append(val)
                
        
        else:
            raise ValueError("wrong inputs")
            
    def fit_order(self):
        """
        Sorts and processes orders and missions based on their status.
        This method first sorts the orders by their second element. It then iterates through the sorted orders and 
        matches them with missions. If a mission's status is 1 and it is ready, the mission is moved to the ready 
        missions list and removed from the missions and acceptance missions lists. If the mission is not ready, 
        it is added to the acceptance list. Finally, the missions and acceptance missions lists are updated with 
        the accepted missions.
        Returns:
            None
        """
        
        if len(self.__order) ==0:
            return
        self.__order.sort(key = lambda x:x[1])
        accept = []
        ready = True
        for order in self.__order:
            for mis in self.__missions:
                if mis.get_mid() == order[0]:
                    if mis.get_status() == 1 and ready:
                        self.__ready_mis.append(mis)
                        self.__missions.remove(mis)
                        self.__acceptance_mis.remove(mis)
                    else:
                        ready = False
                        accept.append(mis)             
        self.__missions= accept.copy()
        self.__acceptance_mis =  accept.copy()
        self.compute_original_plan()
    
    
    def accept_mission(self, miss, order = None):
        """
        Accepts a mission and optionally assigns an order to it.
        This method registers the current object as an observer of the mission,
        appends the mission to the list of missions, and optionally appends the 
        mission and its order to the list of orders.
        Args:
            miss (Mission): The mission to be accepted.
            order (Optional[Any]): The order associated with the mission. Defaults to None.
        Returns:
            None
        """
        
        if order != None:
            self.__order.append((miss.get_mid(), order)) 
        miss.register_observer(self)
        self.__missions.append(miss)
        self.__acceptance_mis.append(miss)
    
    def get_ctrl_time(self):
        return self.__ctrl_time
    
    def update(self, mission, time = 0):
        """
        Update the system state based on the given mission and time.
        Args:
            mission: The mission object that contains the details of the mission to be updated.
            time (int, optional): The current time. Defaults to 0.
        Returns:
            int: The number of dependencies removed after checking readiness.
        """
        
        is_depends = False
        if len(self.__ready_mis)==0 :
            is_depends = True
        
        n_remove_depends = self.check_ready(mission=mission)
        if self.__ctrl_time < time and is_depends:
            self.__ctrl_time = time
        return n_remove_depends
    def check_ready(self, mission):
        """
        Check and update the readiness of missions based on dependencies.
        This method checks the current list of missions and updates their status 
        based on the dependencies of the given mission. If a mission has no more 
        dependencies, its status is updated to ready. The method also handles 
        the removal of dependencies and updates the list of ready missions.
        Args:
            mission: The mission object whose dependencies are to be checked.
        Returns:
            int: The number of dependencies removed.
        """
        
        if len(self.__missions) == 0:
            return 0
        id = mission.get_mid()
        n_remove_depends = 0
        for mis in self.__missions:
            deps = mis.get_depends()
            if id in deps:
                mis.remove_depend(id)
                n_remove_depends += 1
                if self.verbose:
                    print("Xe ô tô {} đã cập nhật việc xoá dependence {} của nhiệm vụ {}".format(self.__vid,id, mis.get_mid()))
            if len(deps)==0:
                mis.update_status(1, mission,self.__ctrl_time)
            if mis.get_status() == 1 and self.__non_priority_orders:
                self.__ready_mis.append(mis)
                self.__missions.remove(mis)
        
        if len(self.__missions[0].get_depends())==0 and not self.__non_priority_orders:
            mis = self.__missions.pop(0)
            mis.update_status(1, mission, self.__ctrl_time)
            if mis not in self.__ready_mis:
                self.__ready_mis.append(mis)
        return n_remove_depends      
    def verify_ready(self):
        if len(self.__missions) < 1:
            return
        if len(self.__missions[0].get_depends()) ==0 and len(self.__ready_mis)==0 and not self.__non_priority_orders:
            self.__ready_mis.append(self.__missions[0])
            if self.__missions[0] in self.__acceptance_mis:
                self.__acceptance_mis.remove(self.__missions[0])
            self.__missions.remove(self.__missions[0])
            
    
    def inprocess(self):
        return len(self.__ready_mis)>0    
    
    def check_time(self):
        return self.__intime

    def _get_pending_missions(self):
        pending = []
        seen = set()
        for m in self.__ready_mis + self.__missions:
            mid = m.get_mid()
            if mid in seen:
                continue
            seen.add(mid)
            pending.append(m)
        return pending

    def _estimate_route_task_count(self, mission):
        route = mission.get_best_road()
        if route is None or len(route) < 2:
            return 0
        segments = self.__map.get_segments()
        n_tasks = 0
        for idx in range(len(route) - 1):
            points = (route[idx], route[idx + 1])
            try:
                seg_idx = segments.index(points)
            except ValueError:
                continue
            n_tasks += len(segments[seg_idx].get_offloading_tasks())
        return n_tasks

    def _estimate_mission_time(self, mission):
        _, distance_to_mission = mission.get_infor_to_mission(self.__cur_pos)
        t_to_start = distance_to_mission / max(self.__v_nominal, 1e-9)

        route_len = mission.get_long()[0]
        if route_len <= 0:
            return t_to_start
        d_nom = route_len / max(self.__v_nominal, 1e-9)
        n_tasks = self._estimate_route_task_count(mission)
        if n_tasks <= 0:
            return t_to_start + d_nom

        v_eff = mobility_v_eff(
            d_nom,
            [self.__avg_task_delay] * n_tasks,
            self.__v_nominal,
            self.__rho_down,
            self.__rho_up,
        )
        return t_to_start + route_len / max(v_eff, 1e-9)

    def _build_affected_set(self, violated_mid):
        pending = self._get_pending_missions()
        pending_ids = [m.get_mid() for m in pending]
        if violated_mid not in pending_ids:
            return set()

        edges = {mid: set() for mid in pending_ids}

        # Queue-order edges.
        for i in range(len(pending_ids) - 1):
            edges[pending_ids[i]].add(pending_ids[i + 1])

        # Dependency edges: dep -> mission (mission depends on dep).
        pending_id_set = set(pending_ids)
        for m in pending:
            m_id = m.get_mid()
            for dep in m.get_depends():
                if dep in pending_id_set:
                    edges[dep].add(m_id)

        affected = set()
        stack = [violated_mid]
        while stack:
            cur = stack.pop()
            if cur in affected:
                continue
            affected.add(cur)
            stack.extend(list(edges.get(cur, [])))
        return affected

    def _drop_pending_missions(self, mids_to_drop):
        if not mids_to_drop:
            return 0

        removed = set()

        def _filter_list(missions):
            kept = []
            for mis in missions:
                if mis.get_mid() in mids_to_drop:
                    removed.add(mis.get_mid())
                else:
                    kept.append(mis)
            return kept

        self.__ready_mis = _filter_list(self.__ready_mis)
        self.__missions = _filter_list(self.__missions)
        self.__acceptance_mis = _filter_list(self.__acceptance_mis)
        self.__order = [o for o in self.__order if o[0] not in mids_to_drop]
        self.__late += len(removed)
        return len(removed)

    def _select_retained_affected_set(self, affected_ids, remain_tau):
        """Greedy keep-set selection over affected missions under queue/dependency feasibility."""
        pending = self._get_pending_missions()
        pending_map = {m.get_mid(): m for m in pending}

        kept = set()
        used_time = 0.0

        for mis in pending:
            mid = mis.get_mid()
            if mid not in affected_ids:
                continue

            deps = [d for d in mis.get_depends() if d in affected_ids]
            if any(dep not in kept for dep in deps):
                continue

            est_time = self._estimate_mission_time(pending_map[mid])
            if used_time + est_time <= remain_tau + 1e-9:
                kept.add(mid)
                used_time += est_time

        return kept

    def _effective_wait_delay(self, exe_delay):
        """Eq. (12): d_{k,j,v} = min(d^{exe}_{k,j,v}, d^{max}_{k,j})."""
        return min(float(exe_delay), self.__dmax_wait)

    def _compute_acture_effection_by_delay(self, effective_wait, actual_delay):
        
        v_safety = max(0, self.__v_nominal - self.__rho_down * effective_wait)
        #the time with v_safety = 0
        time_with_v_safety_0 = max(0, actual_delay - effective_wait) #time with v_safety = 0
        if time_with_v_safety_0 > 0:
            v_safety = 0
        d_up = (self.__v_nominal - v_safety) / self.__rho_up
        
        total_effection_time = actual_delay + d_up
        
        #total travel distance in total_effection_time by norminal speed
        total_travel_distance = self.__v_nominal * total_effection_time
        
        #effection speed during slow down time
        if effective_wait <= self.__v_nominal / self.__rho_down:
            v_eff_down = (self.__v_nominal - 0.5 * self.__rho_down * effective_wait)*(effective_wait/actual_delay)
        else:
            v_eff_down = (self.__v_nominal ** 2) / (2 * self.__rho_down * effective_wait)*(effective_wait/actual_delay)
        
        v_eff_up = v_safety + 0.5 * self.__rho_up * d_up
        #actual travel distance in total_effection_time by v_safety
        actual_travel_distance = v_eff_up*d_up + v_eff_down*effective_wait
        
        #gap between total travel distance and actual travel distance
        gap_distance = total_travel_distance - actual_travel_distance
        if gap_distance <= 0:
            raise ValueError(f"The gap distance is negative, something is wrong with the calculation, gap_distance: {gap_distance}, actual_travel_distance: {actual_travel_distance}, total_travel_distance: {total_travel_distance}, effective_wait: {effective_wait}, actual_delay: {actual_delay}, v_safety: {v_safety}")
        
        affection_delay = gap_distance / self.__v_nominal
        
        return affection_delay
    
    def _apply_early_stopping_policy(self):
        """Proposed 1: selective keep/remove over the affected set when delay propagates."""
        if not self.__apply_early_stopping:
            return 0

        dropped_total = 0
        while len(self.__ready_mis) > 0:
            candidate = self.__ready_mis[0]
            remain_tau = max(0.0, self.__tau - self.__ctrl_time)
            est_time = self._estimate_mission_time(candidate)
            if est_time <= remain_tau + 1e-9:
                break

            affected = self._build_affected_set(candidate.get_mid())
            if not affected:
                affected = {candidate.get_mid()}

            retained = self._select_retained_affected_set(affected, remain_tau)
            to_drop = affected - retained
            if not to_drop:
                break

            dropped = self._drop_pending_missions(to_drop)
            dropped_total += dropped

            if self.verbose:
                print(f"Vehicle {self.__vid} early-stop pruned {dropped} missions "
                    f"(affected={len(affected)}, retained={len(retained)})")

            self.verify_ready()
            if remain_tau <= 0:
                break

        return dropped_total
    
    def handle_offloading_with_vehicles_speed(self, offload_task, current_line_of_road, aver_speed, cur_point, trajectory):
        pass
    
    def recompute_task_inter_arrival_rate(self, run_time, n_tasks):
        if run_time <= 0 or n_tasks <= 0:
            return float("inf")
        return n_tasks / run_time

    def update_profit_by_mission_id(self, mission_id, total_length, B_remain):
        self.__profit_by_missions_id[mission_id] = total_length*0.025 + B_remain 
    
    def process_mission(self, missions = None):
        #process priority is the shortest mission first
        #check event two mission in the same parth or not
        #we shoud take care the case that, is there manupulate the parth of two or three  task to reduce the fee
        if len(self.__ready_mis) ==0:
            return
        elif self.__intime == False:
            return

        if self.__apply_early_stopping:
            self._apply_early_stopping_policy()
            if len(self.__ready_mis) == 0:
                return
        # sorted(self.__ready_mis) #sort các nhiệm vụ theo quãng đường thực hiện
        
        cur_mis = self.__ready_mis.pop(0)
        
        if not self.set_plan_by_step(mission=cur_mis):
            return
        if len(cur_mis.get_depends())!=0:
            raise ValueError("The mission is not ready to compute ...")
        
        
        #bước 1, từ vị trí hiện tại của vehicle tính thời gian đến vị trí của mission.
        #để tính được nó, chúng ta có 2 bước:
        # ->bước 1, tìm vị trí hiện tại củ vehicle.
        # ->bước 2, tìm vị trí hiện tại củ mission.
        # ->bước 3, gọi hàm tính toán quảng đường, sau đó tính thời gian di chuyển tới nhiệm vụ.
        veh_position = self.__cur_pos
        best_trajectrory_to_mission, distance_to_mission = cur_mis.get_infor_to_mission(veh_position)

        original_plan = cur_mis.get_original_plan()
        
        processing_time_allow = (original_plan[1] - original_plan[0])*1.2 # because of offloading, the allow time higher than 1.2 time compare to expected time
        
        '''
        khi có các nhiệm vụ trên cùng 1 cung đường,
        chúng ta chỉ cần lấy cung đường dài nhất.
        tính tổng của các profit và gán cho cung đường đó.
        '''
        same_longest_road_mis = cur_mis
        max_point = -float('inf')
        prof = cur_mis.get_profit()
        
        endpoint_to_missions = {cur_mis.get_best_road()[-1]: [cur_mis]}
        
        total_length = cur_mis.get_long()[0]
        trajectory = same_longest_road_mis.get_best_road() 
        segments = self.__map.get_segments()
        completed_cnt = 0
        n_remove_depends = 0
        total_delay = distance_to_mission / self.__v_nominal  # Eq. (21): travel time to mission start
        if len(best_trajectrory_to_mission)>0:
            best_trajectrory_to_mission.remove(best_trajectrory_to_mission[-1]) #loaị bỏ điểm cuối cùng là điểm bắt đầu nhiệm vụ
        trajectory = best_trajectrory_to_mission + trajectory
        
        while len(trajectory) > 1:
            cur_point = trajectory.pop(0)
            points = (cur_point, trajectory[0])
            idx= segments.index(points)
            
            cur_seg = segments[idx]
            _, aver_speed = cur_seg.get_infor() #m/s
            #offloadingtask is a dict key = seg_id, val = list of offoadling task
            offload_task = cur_seg.get_offloading_tasks()
            
            current_road_long = cur_point.get_dis_to_point(trajectory[0]) #m
            # Nominal driving time for this segment (denominator of Eq. 20)
            d_nom_seg = current_road_long / aver_speed if aver_speed > 0 else 0.0 # second

            cur_offloading_delay = 0
            max_avg_delay_seg = float("inf")
            if self.__apply_offloading_delay_threshold and len(offload_task) > 0:
                remain_tau_seg = max(0.0, self.__tau - (self.__ctrl_time + total_delay))
                max_avg_delay_seg = max_allowable_average_delay(
                    road_length=current_road_long,
                    tau_budget=remain_tau_seg,
                    v_nom=aver_speed,
                    rho_down=self.__rho_down,
                    rho_up=self.__rho_up,
                    n_tasks=len(offload_task),
                )
            # Collect per-task effective waiting delays for the parallel mobility model
            task_delays_seg = []
            if self.verbose:
                print(
                    f"Vehicle {self.__vid} processing segment {cur_point} -> {trajectory[0]} "
                    f"with {len(offload_task)} offloading tasks, segment length "
                    f"{current_road_long:.2f}m, nominal time {d_nom_seg:.2f}s"
                )
            
            inter_arrival_time = self.recompute_task_inter_arrival_rate(
                run_time=current_road_long / aver_speed,
                n_tasks=len(offload_task),
            )
            
            for offt in offload_task:
                '''
                Trong thực tế chúng ta sẽ không biết đc khoảng thời gian giữa hai lần offloading.
                vì thế dựa vào tỷ số lambda khi task xuất hiện chúng ta có thể tính được liệu rằng sau bn lâu sẽ có 1 offloading task xuất hiện.
                từ đó có thể tính được vị trí mà xe đang offloading:
                vị trí thời gian cũ + vị trí thời gian mới
                vị trí ở thời gian mới chính bằng khoảng tgian trung bình generate task* tốc độ
                đó chính là inter_arrival_time
                '''
                current_line_of_road = Line(pnt1=cur_point, pnt2=trajectory[0])
                #update position by time
                
                if cur_offloading_delay == 0:
                    new_points = cur_point
                else:
                    current_pointxy = cur_point.get_point()
                    increase = 1
                    if current_pointxy[0] > trajectory[0].get_point()[0]:
                        increase = -1
                    l = aver_speed*cur_offloading_delay
                    slope = current_line_of_road.get_slop()
                    new_x = current_pointxy[0] + increase*(l/(sqrt(l/(1+slope**2))))
                    new_y = current_line_of_road.calculate_point(new_x)
                    new_points = Point(x = new_x, y=new_y)
                
                #approximation arrival time
                arrival_time = self.__ctrl_time + inter_arrival_time
                # rate, cpu_freq, and the shared ComputeNode for this MEC server
                rate, cpu_freq, compute_node = get_rate_and_mec_cpu(new_points, self.__mec)
                task_ifor = offt.get_task()
                # Uplink transmission delay: Eq. (7)  d^com = alpha_{k,j} / R_{v,e}
                communtion_delay = task_ifor[0] * 8000 / rate
                d_queue_peek = compute_node.get_queue_delay()
                remote_delay = communtion_delay + d_queue_peek + task_ifor[1] / cpu_freq

                local_delay = task_ifor[1] / max(self.__cpu_freqz, 1e-9)
                exe_delay = remote_delay
                use_remote = True
                if self.__apply_offloading_delay_threshold:
                    selected_delay, use_remote = choose_delay_with_threshold(
                        remote_delay=remote_delay,
                        max_avg_delay=max_avg_delay_seg,
                        local_delay=local_delay,
                    )
                    exe_delay = selected_delay
                if use_remote:
                    d_queue = compute_node.task_arrive()
                    computing_delay = d_queue + task_ifor[1] / cpu_freq
                    exe_delay = communtion_delay + computing_delay
                    compute_node.task_depart()
                    same_longest_road_mis.update_profit(-task_cfg['cost_coefi'])

                    if self.verbose:
                        print(
                            f"remote delay={exe_delay:.4f}s "
                            f"(rate={rate:.2f}, cpu={cpu_freq:.2f}, q={compute_node.queue_size})"
                        )
                elif self.verbose:
                    print(
                        f"switch to local compute delay={exe_delay:.4f}s "
                        f"(remote={remote_delay:.4f}s, local={local_delay:.4f}s, "
                        f"threshold={max_avg_delay_seg:.4f}s)"
                    )

                effective_wait = min(exe_delay, offt.get_task()[3])
                task_delays_seg.append(effective_wait)
                affection_delay = self._compute_acture_effection_by_delay(
                    effective_wait=effective_wait,
                    actual_delay=exe_delay)
                cur_offloading_delay += affection_delay
                
            seg_time = current_road_long / self.__v_nominal
            total_delay += (seg_time + cur_offloading_delay)
            arrived_point = trajectory[0]
            if processing_time_allow < total_delay and self.__apply_early_stopping:
                #call early stopping policy to remove some mission
                self._apply_early_stopping_policy()
            
            while arrived_point in endpoint_to_missions and len(endpoint_to_missions[arrived_point]) > 0:
                if self.__ctrl_time + total_delay< self.__tau:
                    completed_cnt += 1
                else:
                    break
                mis = endpoint_to_missions[arrived_point].pop(0)
                if mis in self.__missions:
                    self.__missions.remove(mis)
                if mis in self.__ready_mis:
                    self.__ready_mis.remove(mis)
                if mis in self.__acceptance_mis:
                    self.__acceptance_mis.remove(mis)
                n_remove_depends = mis.update_status(2, missions, time = self.__ctrl_time+total_delay)
                self.set_pos(mis.get_desti())
                if self.verbose:
                    print("Nhiệm vụ {} hoàn thành tính toán".format(mis.get_mid()))
                    
        self.__ctrl_time += total_delay
    
        while len(trajectory) > 0 and trajectory[0] in endpoint_to_missions and len(endpoint_to_missions[trajectory[0]]) > 0:
            if self.__ctrl_time < self.__tau:
                completed_cnt += 1
            else:
                break
            mis = endpoint_to_missions[trajectory[0]].pop(0)
            if mis in self.__missions:
                self.__missions.remove(mis)
            if mis in self.__ready_mis:
                self.__ready_mis.remove(mis)
            if mis in self.__acceptance_mis:
                self.__acceptance_mis.remove(mis)
            n_remove_depends = mis.update_status(2, missions, time = self.__ctrl_time)
            if self.verbose:
                print("Nhiệm vụ {} hoàn thành tính toán".format(mis.get_mid()))

        if self.__ctrl_time > self.__tau:
            self.__vehicle_prof -= (len(self.__missions) + len(self.__ready_mis))*50
            self.__missions = []
            self.__ready_mis = []
            self.__intime = False
            if self.verbose:
                print("Het thoi gian thuc hien nhiem vu") 
            return
        
        #nếu end_points >0 nghĩa là có 1 vài nhiệm vụ chưa hoàn thành,
        #việc của chúng ta là trace và giảm total profit đi
        total_mis_prof = 0
        remaining_missions = []
        for v in endpoint_to_missions.values():
            remaining_missions.extend(v)
        for mis in remaining_missions:
            total_mis_prof += mis.get_profit()
            self.__late += 1
            
        B_remain = same_longest_road_mis.get_profit()-total_mis_prof
        if (B_remain<=0):
            B_remain = 0
        
        profit = total_length*0.025 + B_remain
        self.__profit += B_remain
        self.__n_completes += completed_cnt
                
        self.__vehicle_prof += total_length*0.025 + B_remain 
        self.update_profit_by_mission_id(same_longest_road_mis.get_mid(), total_length, B_remain)
                
        if self.__ctrl_time > self.__tau:
            self.__missions += self.__ready_mis
            self.__vehicle_prof -= self.get_profits(self.__missions)
            self.__missions = []
            self.__ready_mis = []
            self.__intime = False
            if self.verbose:
                print("Het thoi gian thuc hien nhiem vu")
        return self.__vid, cur_mis.get_mid(), n_remove_depends, len(self.__missions), profit

    def get_profit_by_forall_missions(self):
        return self.__profit_by_missions_id
    
    def clear_total_reward(self):
        self.__vehicle_prof = 0
        self.__n_completes = 0
        self.__profit = 0
        self.__profit_by_missions_id = {}
    
    def get_profits(self, mission):
        profit = 0
        if isinstance(mission, list):
            for mis in mission:
                profit += mis.get_profit()
        elif isinstance(mission, Mission):
            profit += mission.get_profit()
        return profit

    def get_accepted_missions(self):
        return self.__missions        
            
    def get_vhicle_prof(self):
        return self.__vehicle_prof
    
    def get_earn_profit(self):
        return self.__profit
    
    def get_earn_completes(self):
        return self.__n_completes

class Task:
    id = 0
    def __init__(self, datasize, compsize, id = None, max_delay = None):
        self.__ds = datasize
        self.__cs = compsize
        self.__id = Task.id
        self.__max_delay = max_delay
        if id!=None:
            self.__id = id
        Task.id += 1

    def get_datasize(self):
        return self.__ds
    
    def get_compsize(self):
        return self.__cs 

    def get_task(self):
        return (self.__ds, self.__cs, self.__id, self.__max_delay)
    
    def __lt__(self, other):
        return self.__ds < other.__ds and self.__cs < other.__cs
    
    def __eq__(self, other):
        return self.__id == other.__id    
    
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (list, Task)):
            return list(obj.get_task())
        if isinstance(obj, (list, Point)):
            return list(obj.get_point())
            
        return super(NpEncoder, self).default(obj)



class TaskGenerator:
    def __init__(self, tau, map,
                mindatasize = 100, #kbytes
                maxdatasize = 500, #kbytes
                mincompsize = 1, #mcycles
                maxcompsize = 3, #mcycles
                ):
        self.__map = map
        self.__map.draw_segments()
        self.__map.draw_map()
        # self.__map.save_map()
        self.__tau = tau
        self.__mindatasize = mindatasize
        self.__maxdatasize = maxdatasize
        self.__mincompsize = mincompsize
        self.__maxcompsize = maxcompsize
        self.generator = np.random.default_rng(42)
        segments = self.__map.get_segments()
        self.graph = Graph(segments)
    def update_graph(self, segments):
        self.graph = Graph(segments)
        
    def gen_tasks(self):
        '''
            suppose that the vehicle run with maximum speed, it have maximum number of offloading tasks.
            With maximum speed we assum that the offloading data is higher with lower speed.
            
            The below is a way to calculate the real number of task with current simulation speed.
            
            In this simulation, the vehicle provide the current average speed, then we can use the below equation to get the number of offloading tasks and its list.
            numtasks= (realsimulation/mximum)*number_of_task_with_maximum_speeds
            tasks = tasks[0:numtasks]
        '''
        segments = self.__map.get_segments()
        f= open("./task/task_information.json", "w")
        f.write("[")
        f.close()
        for seg in segments:
            filld = {}
            sta_seg = seg.get_state()
            longs = seg.get_long()
            lam, speed = seg.get_infor()
            runtime = longs/(speed/((sta_seg+1)*0.2))
            numtasks = int(lam*runtime)
            tasks = {}
            filld['seg_id'] = seg.get_sid()
            filld['seg_status'] = sta_seg
            filld['seg_longs'] = longs
            filld['seg_real_speed'] = (speed/((sta_seg+1)*0.2))
            filld['seg_max_numtask'] = numtasks
            vehicles_cpu_capcity = vehicle_cfg['cpu_freqz']
            for t in range(numtasks):
                datasize = self.generator.integers(self.__mindatasize, self.__maxdatasize)
                compsize = self.generator.integers(self.__mincompsize, self.__maxcompsize)
                deadline = compsize/vehicles_cpu_capcity*np.random.uniform(0.8, 1.2) #second
                task = Task(datasize=datasize, compsize=compsize, max_delay=deadline)
                tasks[t] = task
            filld['seg_tasksl'] = tasks
            json_object = json.dumps(filld, indent=5, cls=NpEncoder)
            
            with open("./task/task_information.json", "a") as outfile:
                outfile.write(json_object)
                if seg!=segments[-1]:
                    outfile.write(',\n')
        f= open("./task/task_information.json", "a")
        f.write("]")
        f.close()
        Task.id = 0
        
    def gen_mission_non_file(self, nummissions):
        t = time.perf_counter()
        if map_cfg['real_map']:
            self.__map.update_hmap()
        get_vertexes_p = list(set(self.graph.get_vertexes()))
        avoi_cyc_dep={}
        missions = []
        max_pair_attempts = max(100, len(get_vertexes_p) * 2)
        
        task_depends = 0
        for i in range(nummissions):        
            attempts = 0
            while True:
                seg_dep = self.generator.choice(get_vertexes_p)
                seg_des = self.generator.choice(get_vertexes_p)

                # Keep the original geometric constraint used by this generator.
                straight_path = seg_dep.get_dis_to_point(seg_des)
                times = straight_path / task_cfg['max_speed']

                while times > task_cfg['tau'] * 60 and times < 50 / task_cfg['max_speed']:
                    seg_dep = self.generator.choice(get_vertexes_p)
                    seg_des = self.generator.choice(get_vertexes_p)
                    straight_path = seg_dep.get_dis_to_point(seg_des)
                    times = straight_path / task_cfg['max_speed']

                try:
                    self.graph.dijkstra(seg_dep, seg_des)
                    break
                except ValueError:
                    attempts += 1
                    if attempts >= max_pair_attempts:
                        # Fallback to a trivially reachable mission to keep generation stable.
                        seg_des = seg_dep
                        break
            mem_de = []
            if self.generator.uniform() >0.4:
                nums_depends = self.generator.integers(1,3)
                for k in range(nums_depends):
                    val = self.generator.integers(0, nummissions)
                    while (val == i):
                        val = self.generator.integers(0, nummissions)
                    if (val not in mem_de):
                        if val in avoi_cyc_dep.keys():
                            if i not in avoi_cyc_dep[val]:
                                mem_de.append(val)
                        else:
                            mem_de.append(val)
                task_depends += 1
            avoi_cyc_dep[i] = mem_de 
            depends = mem_de
            profit = self.generator.integers(mission_cfg['benifits'][0], mission_cfg['benifits'][1])
            m = Mission(seg_dep, seg_des, tslot=0, graph=self.graph)
            m.set_depends(depends)
            m.set_profit(profit)
            m.set_mid(i)      
            missions.append(m)
        Mission.missionID = 0
        print("generator time {}".format(time.perf_counter()-t))
        print("-----------> depend = ", task_depends, task_depends/mission_cfg['n_mission'])
        
        return missions, self.graph, self.__map
    def gen_mission(self, nummissions, file='mission_information.json'):
        t = time.perf_counter()
        if map_cfg['real_map']:
            self.__map.update_hmap()
        get_vertexes_p = list(set(self.graph.get_vertexes()))
        # missions = [] 
        f = open("./task/"+file, "w")
        f.write("[")
        f.close()
        
        #because of no cycle dependence, if A depend on B, then B cannot depend on A
        #to avoid this condition during generation, we use a map to handle it.
        
        avoi_cyc_dep={}
        
        for i in range(nummissions):
            filld = {}
            filld['i'] = i            
            seg_dep = self.generator.choice(get_vertexes_p)
            seg_des = self.generator.choice(get_vertexes_p)

            # Attempt to find a valid path
            # using straight path to avoid using  dijkstra, when the number of vertex to large, then the time increasing
            straight_path = seg_dep.get_dis_to_point(seg_des)
            times = straight_path / task_cfg['max_speed']
            while times > task_cfg['tau'] * 60 and times < 50 / task_cfg['max_speed']:
                seg_dep = self.generator.choice(get_vertexes_p)
                seg_des = self.generator.choice(get_vertexes_p)
                straight_path = seg_dep.get_dis_to_point(seg_des)
                times = straight_path / task_cfg['max_speed']
                
            
            mem_de = []
            if self.generator.uniform() >0.4:
                nums_depends = self.generator.integers(1,3)
                for k in range(nums_depends):
                    val = self.generator.integers(0, nummissions)
                    while (val == i):
                        val = self.generator.integers(0, nummissions)
                    if (val not in mem_de):
                        if val in avoi_cyc_dep.keys():
                            if i not in avoi_cyc_dep[val]:
                                mem_de.append(val)
                        else:
                            mem_de.append(val)
            avoi_cyc_dep[i] = mem_de 
            depends = mem_de
            profit = self.generator.integers(mission_cfg['benifits'][0], mission_cfg['benifits'][1])
            filld["depart_p"] = seg_dep
            filld["depart_s"] = seg_des
            filld["depends"] = depends
            filld["profit"] = profit
            json_object = json.dumps(filld, indent=4, cls=NpEncoder)
          
            # missions.append(m)
            
            with open("./task/"+file, "a") as outfile:
                outfile.write(json_object)
                if i!=nummissions-1:
                    outfile.write(',\n')
        f = open("./task/"+file, "a")
        f.write("]")
        f.close()
        e = time.perf_counter()
        print("generated time {}".format(e-t))
    def real_update(self, mission_f_name = "mission_information.json"):
        self.__map.update_hmap()
        self.gen_tasks()
        self.gen_mission(mission_cfg['n_mission'], file = mission_f_name)
def generate(mission_f_name = "mission_information.json"):
    map = Map(map_cfg['n_lines'], busy=map_cfg['busy'], fromfile=map_cfg['fromfile'])
    tg = TaskGenerator(tau = task_cfg['tau'], map=map,
                    mincompsize=task_cfg['comp_size'][0],
                    maxcompsize=task_cfg['comp_size'][1],
                    mindatasize=task_cfg['comm_size'][0],
                    maxdatasize=task_cfg['comm_size'][1],
                    )
    tg.gen_tasks()
    tg.gen_mission(mission_cfg['n_mission'], file = mission_f_name)
if __name__ == "__main__":
    
    generate()
    
