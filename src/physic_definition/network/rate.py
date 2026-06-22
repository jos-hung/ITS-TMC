import random
from configs.systemcfg import network_cfg, map_cfg, task_cfg
from configs.config import ParaConfig
import numpy as np
from physic_definition.map.map import Map
from threading import Lock as _Lock

SEED = ParaConfig.SEED_GLOBAL
rd_generator = np.random.default_rng(SEED)
import sys

max_ = sys.maxsize
# -----------------------------------------------------------------------
# Queueing Model for Computation Nodes (Paper Section II.B, Eq. 9)
# -----------------------------------------------------------------------

def queue_delay(n_queue, avg_beta, F_x):
    """
    Expected queueing delay given current queue length (Eq. 9).

    d^queue_{k,j,x} = E[D^queue_{k,j,x} | |q^cmp_x|] ≈ |q^cmp_x| * avg_beta / F_x

    Derivation: under any work-conserving scheduling policy, the expected wait
    for a newly arriving task equals the expected remaining work in the queue.
    Conditioned only on queue length |q^cmp_x| (not individual task sizes),
    this reduces to n * E[service_time] = n * avg_beta / F_x.

    Args:
        n_queue  : current queue occupancy |q^cmp_x| (number of tasks waiting)
        avg_beta : expected CPU cycles per task (mean of beta_{k,j} distribution)
        F_x      : processing speed of node x (cycles/s)
    Returns:
        expected waiting delay (s); 0.0 when queue is empty
    """
    if F_x <= 0 or n_queue <= 0:
        return 0.0
    return n_queue * avg_beta / F_x


class ComputeNode:
    """
    Stateful computation node modelling q^cmp_x from Paper Section II.B.

    Tracks the number of tasks currently occupying the node's queue so that
    Eq. (9) can be evaluated at the moment each new task arrives.  The class
    is thread-safe and can be shared across Vehicle instances to capture
    cross-vehicle MEC contention.

    Usage in simulation:
        d_q = node.task_arrive()          # snapshot E[wait] then increment counter
        d_cmp = d_q + beta_kj / node.F_x  # Eq. (8)
        ...                               # vehicle accumulates the delay
        node.task_depart()                # decrement when task finishes
    """

    def __init__(self, F_x, avg_beta):
        """
        Args:
            F_x      : processing speed (cycles/s); same unit as beta_{k,j}
            avg_beta : mean CPU cycles per task — used in Eq. (9)
        """
        self.F_x = float(F_x)
        self.avg_beta = float(avg_beta)
        self._queue_size = 0
        self._lock = _Lock()

    @property
    def queue_size(self):
        return self._queue_size

    def get_queue_delay(self):
        """Return E[wait] for a hypothetical arriving task without modifying state."""
        return queue_delay(self._queue_size, self.avg_beta, self.F_x)

    def task_arrive(self):
        """
        Register task arrival at this node.

        Returns d_queue = E[D^queue | current |q^cmp_x|] (Eq. 9) computed
        *before* incrementing the counter — i.e., the wait the arriving task
        will experience given the queue it finds on arrival.
        Thread-safe: snapshot and increment are atomic.
        """
        with self._lock:
            d_q = queue_delay(self._queue_size, self.avg_beta, self.F_x)
            self._queue_size += 1
        return d_q

    def task_depart(self):
        """Register task departure after its processing completes."""
        with self._lock:
            self._queue_size = max(0, self._queue_size - 1)


def random_bs_num(interection):
    """
    Generate a list of tuples representing the positions, CPU frequencies, and
    ComputeNode objects of MEC servers.
    Args:
        interection (list): A list of positions where MECs can be placed.
    Returns:
        list: A list of 3-tuples (position, cpu_freq, ComputeNode).
    Raises:
        ValueError: If the provided `interection` list is empty.
    """

    if len(interection) == 0:
        raise ValueError ("Map is not build")
    position_of_mec = rd_generator.choice(interection, network_cfg['n_MEC'])
    cpu_freq_of_mec = rd_generator.integers(network_cfg['CPU_freq'][0], network_cfg['CPU_freq'][1], network_cfg['n_MEC'])
    # avg_beta: mean CPU cycles per task, used in Eq. (9) to estimate service time
    avg_beta = (task_cfg['comp_size'][0] + task_cfg['comp_size'][1]) / 2.0
    return [
        (pos, int(freq), ComputeNode(F_x=float(freq), avg_beta=avg_beta))
        for pos, freq in zip(position_of_mec, cpu_freq_of_mec)
    ]

def chann_rates(distance):
    """
    Calculate the channel rate based on the given distance.
    Parameters:
    distance (float): The distance between the transmitter and receiver.
    Returns:
    float: The calculated channel rate.
    Notes:
    - The function uses a random generator to create a complex channel gain `h`.
    - The transmit power `Pt` is set to 199.526 mW.
    - The noise power spectral density `No` is set to 3.98e-21.
    - The number of channels `m` is set to 10.
    - The total bandwidth `bandwidth` is set to 20 MHz.
    - The path loss exponent `path_loss` is set to 3.
    - The channel gain `h` is normalized and adjusted based on the distance and path loss.
    - The rate is calculated using the Shannon-Hartley theorem.
    """
    
    rate=0
    # h=complex(rd_generator.standard_normal(size=(5,1)),rd_generator.standard_normal(size=(5,1)))
    h = rd_generator.standard_normal(size=(16,)) + 1j * rd_generator.standard_normal(size=(16,))

    h/=np.sqrt(2)
    Pt=199.526*(10**-3)
    No=3.98*(10**-21)
    m=10
    bandwidth=20*(10**6)
    # h=(np.abs(h))**2
    # print(h)
    path_loss=3
    h = np.linalg.norm(h, ord=2)
    
    if round(distance,2) != 0.0:
        h=h/(distance)**path_loss
    else:
        h = max_

    bandwidth_per_channel=bandwidth/m
    rate=bandwidth_per_channel*np.log2(1+((Pt*h)/(bandwidth_per_channel*No)))   
    return rate

def get_rate_and_mec_cpu(v_pos, mec):
    candidate_mec = []
    min_distance = max_
    mec_min = None
    for m in mec:
        current_mec_point = v_pos.get_dis_to_point(m[0]) #m[0]= a point, m[1]= an integer
        if current_mec_point < network_cfg['best_rate_radius']:
            candidate_mec.append(m)
        if current_mec_point < min_distance:
            min_distance = current_mec_point
            mec_min = m
    select_mec = 0
    distance = 0
    if len(candidate_mec)==0:
        select_mec = mec_min
        distance = min_distance
    else:
        max_cpu = -max_
        for m in candidate_mec:
            if m[1]>max_cpu:
                max_cpu = m[1]
                distance = v_pos.get_dis_to_point(m[0])
                select_mec = m
    # Return (uplink_rate, cpu_freq, ComputeNode) so callers can apply Eq. (9)
    return chann_rates(distance), select_mec[1], select_mec[2]


# -----------------------------------------------------------------------
# Communication and Computation Delay Models (Paper Section II.B)
# -----------------------------------------------------------------------

def uplink_rate(W_ve, p_v, h_ve_sq, N0):
    """
    Uplink transmission rate between robot v and MEC server e (Eq. 6).

    R_{v,e}(tau) = W_{v,e} * log2(1 + p_v * |h_{v,e}|^2 / (N0 * W_{v,e}))

    Args:
        W_ve  : allocated uplink bandwidth (Hz)
        p_v   : transmit power of robot v (W)
        h_ve_sq : squared channel gain |h_{v,e}|^2 (includes path loss)
        N0    : noise power spectral density (W/Hz)
    Returns:
        uplink rate (bps)
    """
    return W_ve * np.log2(1.0 + p_v * h_ve_sq / (N0 * W_ve))


def comm_delay(alpha_kj, R_ve):
    """
    MEC uplink transmission delay for task (k,j) from robot v to MEC e (Eq. 7).

    d^com_{k,j,v,e} = alpha_{k,j} / R_{v,e}

    Args:
        alpha_kj : input data size of task (k,j) in bits
        R_ve     : uplink rate (bps) from uplink_rate()
    Returns:
        transmission delay (s)
    """
    return alpha_kj / R_ve


def comp_delay(beta_kj, F_x, d_queue=0.0):
    """
    Total computation delay for task (k,j) at node x (Eq. 8).

    d^cmp_{k,j,x} = d^queue_{k,j,x} + beta_{k,j} / F_x

    Args:
        beta_kj : required CPU cycles for task (k,j)
        F_x     : processing speed of node x (cycles/s)
        d_queue : expected queueing delay E[D^queue_{k,j,x} | |q^cmp_x|] (Eq. 9),
                defaults to 0 when queue is empty or ignored
    Returns:
        total computation delay (s)
    """
    return d_queue + beta_kj / F_x
