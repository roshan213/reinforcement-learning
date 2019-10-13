import argparse
from datetime import datetime
import logging

import random as rand
import numpy as np

import environments
import experiments

from experiments import plotting

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_experiment(experiment_detals, experiment, timing_key, verbose, timings):
    t = datetime.now()
    for details in experiment_detals:
        logger.info("Running {} experiment: {}".format(timing_key, details.env_readable_name))
        exp = experiment(details, verbose=verbose)
        exp.perform()
    t_d = datetime.now() - t
    timings[timing_key] = t_d.seconds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MDP experiments')
    parser.add_argument('--threads', type=int, default=-1, help='Number of threads (defaults to -1 auto, debugging should use 1)')
    parser.add_argument('--seed', type=int, help='A random seed to set, if desired')
    parser.add_argument('--policy', action='store_true', help='Run the policy iteration experiment')
    parser.add_argument('--value', action='store_true', help='Run the value iteration experiment')
    parser.add_argument('--q', action='store_true', help='Run the Q-Learner experiment')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--plot', action='store_true', help='Plot data results')
    parser.add_argument('--verbose', action='store_true', help='If true, provide verbose output')
    args = parser.parse_args()
    verbose = args.verbose
    threads = args.threads

    seed = args.seed
    if seed is None:
        # TODO: fix with casting
        #seed = np.random.randint(0, (2 ** 32) - 1)
        #seed = np.random.randint(0, (2 ** 32) - 1, dtype=np.int64)
        seed = 0
        logger.info("Using seed {}".format(seed))
        np.random.seed(seed)
        rand.seed(seed)

    logger.info("Creating MDPs")
    logger.info("----------")

    envs = [
        # {
        #     'env': environments.get_small_taxi(),
        #     'name': 'small_taxi',
        #     'readable_name': 'Taxi (5x5)',
        # },

        # # Simple Mazeworlds (with rewards)
        # {
        #     'env': environments.get_mazeworld_environment(),
        #     'name': 'tiny_mazeworld',
        #     'readable_name': 'Mazeworld (4x4)',
        # },
        # {
        #     'env': environments.get_medium3_mazeworld_environment(),
        #     'name': 'medium3_mazeworld',
        #     'readable_name': 'Mazeworld (8x8)',
        # },
        # {
        #     'env': environments.get_medium2_mazeworld_environment(),
        #     'name': 'medium2_mazeworld',
        #     'readable_name': 'Mazeworld (9x9)',
        # },
        # {
        #     'env': environments.get_small_mazeworld_environment(),
        #     'name': 'small_mazeworld',
        #     'readable_name': 'Mazeworld (5x5)',
        # },
        # {
        #     'env': environments.get_medium_mazeworld_environment(),
        #     'name': 'medium_mazeworld',
        #     'readable_name': 'Mazeworld (11x11)',
        # },
        # {
        #     'env': environments.get_large_mazeworld_environment(),
        #     'name': 'large_mazeworld',
        #     'readable_name': 'Mazeworld (15x15)',
        # },

        # Simple Mazeworlds (no rewards)
        # {
        #     'env': environments.get_mazeworld_no_reward_environment(),
        #     'name': 'tiny_mazeworld',
        #     'readable_name': 'Mazeworld (4x4)',
        # },
        # {
        #     'env': environments.get_small_no_reward_mazeworld_environment(),
        #     'name': 'small_mazeworld',
        #     'readable_name': 'Mazeworld (5x5)',
        # },
        # {
        #     'env': environments.get_medium_no_reward_mazeworld_environment(),
        #     'name': 'medium_mazeworld',
        #     'readable_name': 'Mazeworld (11x11)',
        # },
        # {
        #     'env': environments.get_medium3_no_reward_mazeworld_environment(),
        #     'name': 'medium3_mazeworld',
        #     'readable_name': 'Mazeworld (8x8)',
        # },
        # {
        #     'env': environments.get_medium2_no_reward_mazeworld_environment(),
        #     'name': 'medium2_mazeworld',
        #     'readable_name': 'Mazeworld (9x9)',
        # },
        # {
        #     'env': environments.get_large_no_reward_mazeworld_environment(),
        #     'name': 'large_mazeworld',
        #     'readable_name': 'Mazeworld (15x15)',
        # },

        # # These are not really a rewarding frozen lake env, but the custom class has extra functionality
        {
            'env': environments.get_small_frozen_lake_environment(),
            'name': 'small_frozen_lake',
            'readable_name': 'Frozen Lake (4x4)',
        },
        # {
        #     'env': environments.get_medium_frozen_lake_environment(),
        #     'name': 'medium_frozen_lake',
        #     'readable_name': 'Frozen Lake (11x11)',
        # },
        {
            'env': environments.get_large_frozen_lake_environment(),
            'name': 'large_frozen_lake',
            'readable_name': 'Frozen Lake (15x15)',
        },


        # # These are not really a rewarding frozen lake env, but the custom class has extra functionality
        # {
        #     'env': environments.get_small_no_reward_frozen_lake_environment(),
        #     'name': 'no_reward_small_frozen_lake',
        #     'readable_name': 'Frozen Lake (4x4)',
        # },
        # {
        #     'env': environments.get_medium_no_reward_frozen_lake_environment(),
        #     'name': 'medium_frozen_lake',
        #     'readable_name': 'Frozen Lake (11x11)',
        # },
        # {
        #     'env': environments.get_large_no_reward_frozen_lake_environment(),
        #     'name': 'no_reward_large_frozen_lake',
        #     'readable_name': 'Frozen Lake (15x15)',
        # },

        # # No wind Cliff Walking
        # {
        #     'env': environments.get_small_cliff_walking_environment(),
        #     'name': 'small_cliff_walking',
        #     'readable_name': 'Small Cliff Walking (4x4)',
        # },
        # {
        #     'env': environments.get_medium_cliff_walking_environment(),
        #     'name': 'medium_cliff_walking',
        #     'readable_name': 'Medium Cliff Walking (4x12)',
        # },
        # {
        #     'env': environments.get_large_cliff_walking_environment(),
        #     'name': 'large_cliff_walking',
        #     'readable_name': 'Large Cliff Walking (6x12)',
        # },

        # # Windy Cliff Walking
        # {
        #     'env': environments.get_small_windy_cliff_walking_environment(),
        #     'name': 'small_windy_cliff_walking',
        #     'readable_name': 'Small Windy Cliff Walking (4x4)',
        # },
        # {
        #     'env': environments.get_medium_windy_cliff_walking_environment(),
        #     'name': 'medium_windy_cliff_walking',
        #     'readable_name': 'Medium Windy Cliff Walking (4x12)',
        # },
        # {
        #     'env': environments.get_large_windy_cliff_walking_environment(),
        #     'name': 'large_windy_cliff_walking',
        #     'readable_name': 'Large Windy Cliff Walking (6x12)',
        # },

    ]

    experiment_details = []
    for env in envs:
        env['env'].seed(seed)
        logger.info('{}: State space: {}, Action space: {}'.format(env['readable_name'], env['env'].unwrapped.nS,
                                                                   env['env'].unwrapped.nA))
        experiment_details.append(experiments.ExperimentDetails(
            env['env'], env['name'], env['readable_name'],
            threads=threads,
            seed=seed
        ))

    if verbose:
        logger.info("----------")
    logger.info("Running experiments")

    timings = {}

    if args.policy or args.all:
        run_experiment(experiment_details, experiments.PolicyIterationExperiment, 'PI', verbose, timings)

    if args.value or args.all:
        run_experiment(experiment_details, experiments.ValueIterationExperiment, 'VI', verbose, timings)

    if args.q or args.all:
        run_experiment(experiment_details, experiments.QLearnerExperiment, 'Q', verbose, timings)

    logger.info(timings)

    if args.plot:
        if verbose:
            logger.info("----------")

        logger.info("Plotting results")
        plotting.plot_results(envs)
