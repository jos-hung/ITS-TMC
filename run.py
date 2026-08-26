"""Command-line entrypoint for running training, evaluation, and analysis jobs."""

import argparse
import os
import sys
from datetime import datetime

# Keep star import because dispatch below supports multiple entry functions
# exposed across sub-packages (e.g. ppo, mppo, ddqn_ma).
from src import *
from src.DRL.rl_eval import eval_ddqn
from src.DRL.rl_run import A2C, a2c, ddpg, ddqn, ppo
from src.meta_heuristic.script_many_metaheuristics import many_metaheuristics
from src.meta_heuristic.script_statistic import get_statistic_results
from src.meta_heuristic.script_visualize import (
    csv_compared_greedy_random_meta,
    draw_compared_greedy_random_drl,
    draw_eval_results,
    draw_results,
)
from configs.systemcfg import log_configs

class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        if self.log:
            self.log.close()
            self.log = None

def setup_stdout_logger():
    """Redirect stdout to a timestamped log file and return logger object."""
    os.makedirs(log_configs["log_dir"], exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_configs["log_dir"], f"run_log_{timestamp}.log")
    logger = Logger(log_filename)
    sys.stdout = logger
    return logger


def parse_arguments():
    """Parse command-line options for execution and result analysis."""
    parser = argparse.ArgumentParser(description='Simulation for ITS Joint task handling and missiong processing paper')

    parser.add_argument('-i', '--input', type=str,choices=['ppo', 'mppo', 'ppo_test', 'A2C', 'a2c', 'ddpg', 'ddqn', 'ddqn_ma', 'eval_ddqn', 'many_metaheuristics', 'run_single_agent_ddqn', 'meta_heuristic_proposal', 'None'], required=True, help='Simulation type (DRL or metaheuristic).')
    parser.add_argument('-c', '--compare', type=str, choices=['drls', 'drl_and_meta_heuristic_proposal'], help='Compare simulation btw DRL and metaheuristic.')
    parser.add_argument('-a', '--analysis', type = int, help='Analysis result from many meta_heuristics')
    parser.add_argument('-device', '--cuda', type = int, default=-1)
    parser.add_argument('--verbose', action='store_true', help='Display information during run simulation')

    return parser.parse_args()


def run_selected_input(args):
    """Dispatch selected entry function by name on current module."""
    current_module = sys.modules[__name__]
    if args.input == 'None':
        print("No function is selected")
        return

    selected = getattr(current_module, args.input, None)
    if selected is None:
        raise ValueError(f"Unsupported input function: {args.input}")

    selected(**vars(args))


def run_analysis(analysis_id):
    """Run post-processing analysis based on the selected analysis id."""
    if analysis_id == 0:
        get_statistic_results()
        draw_results()
    elif analysis_id:
        draw_eval_results(analysis_id)
        draw_compared_greedy_random_drl(analysis_id)
        csv_compared_greedy_random_meta(analysis_id)


def main():
    args = parse_arguments()
    run_selected_input(args)
    run_analysis(args.analysis)
    

if __name__ == "__main__":
    logger = setup_stdout_logger()
    try:
        main()
    finally:
        logger.close()
        sys.stdout = logger.terminal
        