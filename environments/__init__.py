import gym
from gym.envs.registration import register

from .taxi import *
from .mazeworld import *
from .windy_cliff_walking import *
from .frozen_lake import *

__all__ = ['TaxiEnv', 'MazeworldEnv', 'RewardingFrozenLakeEnv', 'WindyCliffWalkingEnv']

register(
    id='Taxi-v0',
    entry_point='environments:TaxiEnv',
)

register(
    id='Mazeworld4x4-v0',
    entry_point='environments:MazeworldEnv',
    kwargs={'map_name': '4x4'},
)

register(
    id='MazeworldNoRewards4x4-v0',
    entry_point='environments:MazeworldEnv',
    kwargs={'map_name': '4x4', 'rewarding': False},
)

register(
    id='Mazeworld5x5-v0',
    entry_point='environments:MazeworldEnv',
    kwargs={'map_name': '5x5'},
)

register(
    id='MazeworldNoRewards5x5-v0',
    entry_point='environments:MazeworldEnv',
    kwargs={'map_name': '5x5', 'rewarding': False},
)

register(
    id='Mazeworld8x8-v0',
    entry_point='environments:MazeworldEnv',
    kwargs={'map_name': '8x8'}
)

register(
    id='MazeworldNoRewards8x8-v0',
    entry_point='environments:MazeworldEnv',
    kwargs={'map_name': '8x8', 'rewarding': False}
)

register(
    id='Mazeworld9x9-v0',
    entry_point='environments:MazeworldEnv',
    kwargs={'map_name': '9x9'}
)

register(
    id='MazeworldNoRewards9x9-v0',
    entry_point='environments:MazeworldEnv',
    kwargs={'map_name': '9x9', 'rewarding': False}
)

register(
    id='Mazeworld11x11-v0',
    entry_point='environments:MazeworldEnv',
    kwargs={'map_name': '11x11'}
)

register(
    id='MazeworldNoRewards11x11-v0',
    entry_point='environments:MazeworldEnv',
    kwargs={'map_name': '11x11', 'rewarding': False}
)

register(
    id='Mazeworld15x15-v0',
    entry_point='environments:MazeworldEnv',
    kwargs={'map_name': '15x15'}
)

register(
    id='MazeworldNoRewards15x15-v0',
    entry_point='environments:MazeworldEnv',
    kwargs={'map_name': '15x15', 'rewarding': False}
)

register(
    id='Mazeworld21x21-v0',
    entry_point='environments:MazeworldEnv',
    kwargs={'map_name': '21x21'}
)

register(
    id='MazeworldNoRewards21x21-v0',
    entry_point='environments:MazeworldEnv',
    kwargs={'map_name': '21x21', 'rewarding': False}
)

register(
    id='FrozenLake4x4-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '4x4'},
)

register(
    id='FrozenLakeNoRewards4x4-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '4x4'},
)

register(
    id='FrozenLake11x11-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '11x11'}
)

register(
    id='FrozenLakeNoRewards11x11-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '11x11', 'rewarding': False}
)

register(
    id='FrozenLake15x15-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '15x15'}
)

register(
    id='FrozenLakeNoRewards15x15-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '15x15', 'rewarding': False}
)

register(
    id='CliffWalking4x4-v0',
    entry_point='environments:WindyCliffWalkingEnv',
    kwargs={'map_name': '4x4', 'wind_prob': 0.0}
)


register(
    id='CliffWalking4x12-v0',
    entry_point='environments:WindyCliffWalkingEnv',
    kwargs={'map_name': '4x12', 'wind_prob': 0.0}
)


register(
    id='CliffWalking6x12-v0',
    entry_point='environments:WindyCliffWalkingEnv',
    kwargs={'map_name': '6x12', 'wind_prob': 0.0}
)


register(
    id='WindyCliffWalking4x4-v0',
    entry_point='environments:WindyCliffWalkingEnv',
    kwargs={'map_name': '4x4', 'wind_prob': 0.1}
)


register(
    id='WindyCliffWalking4x12-v0',
    entry_point='environments:WindyCliffWalkingEnv',
    kwargs={'map_name': '4x12', 'wind_prob': 0.1}
)


register(
    id='WindyCliffWalking6x12-v0',
    entry_point='environments:WindyCliffWalkingEnv',
    kwargs={'map_name': '6x12', 'wind_prob': 0.1}
)


def get_small_taxi():
    return gym.make('Taxi-v0')


def get_mazeworld_environment():
    return gym.make('Mazeworld4x4-v0')


def get_mazeworld_no_reward_environment():
    return gym.make('MazeworldNoRewards4x4-v0')


def get_small_mazeworld_environment():
    return gym.make('Mazeworld5x5-v0')


def get_small_no_reward_mazeworld_environment():
    return gym.make('MazeworldNoRewards5x5-v0')


def get_medium3_mazeworld_environment():
    return gym.make('Mazeworld8x8-v0')


def get_medium3_no_reward_mazeworld_environment():
    return gym.make('MazeworldNoRewards8x8-v0')


def get_medium2_mazeworld_environment():
    return gym.make('Mazeworld9x9-v0')


def get_medium2_no_reward_mazeworld_environment():
    return gym.make('MazeworldNoRewards9x9-v0')


def get_medium_mazeworld_environment():
    return gym.make('Mazeworld11x11-v0')


def get_medium_no_reward_mazeworld_environment():
    return gym.make('MazeworldNoRewards11x11-v0')


def get_large_mazeworld_environment():
    return gym.make('Mazeworld15x15-v0')


def get_large_no_reward_mazeworld_environment():
    return gym.make('MazeworldNoRewards15x15-v0')


def get_huge_mazeworld_environment():
    return gym.make('Mazeworld21x21-v0')


def get_huge_no_reward_mazeworld_environment():
    return gym.make('MazeworldNoRewards21x21-v0')


def get_small_frozen_lake_environment():
    return gym.make('FrozenLake4x4-v0')


def get_small_no_reward_frozen_lake_environment():
    return gym.make('FrozenLakeNoRewards4x4-v0')


def get_medium_frozen_lake_environment():
    return gym.make('FrozenLake11x11-v0')


def get_medium_no_reward_frozen_lake_environment():
    return gym.make('FrozenLakeNoRewards11x11-v0')


def get_large_frozen_lake_environment():
    return gym.make('FrozenLake15x15-v0')


def get_large_no_reward_frozen_lake_environment():
    return gym.make('FrozenLakeNoRewards15x15-v0')


def get_cliff_walking_environment():
    return gym.make('CliffWalking-v0')


def get_small_cliff_walking_environment():
    return gym.make('CliffWalking4x4-v0')


def get_medium_cliff_walking_environment():
    return gym.make('CliffWalking4x12-v0')


def get_large_cliff_walking_environment():
    return gym.make('CliffWalking6x12-v0')


def get_small_windy_cliff_walking_environment():
    return gym.make('WindyCliffWalking4x4-v0')


def get_medium_windy_cliff_walking_environment():
    return gym.make('WindyCliffWalking4x12-v0')


def get_large_windy_cliff_walking_environment():
    return gym.make('WindyCliffWalking6x12-v0')