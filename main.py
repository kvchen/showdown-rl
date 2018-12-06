#!/usr/bin/env python3

import gym
import click
import random
import tensorflow as tf
from functools import partial
from gym_showdown.envs import ShowdownEnv

from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo


def random_battle(env):
    num_actions = len(env.ALL_ACTIONS)

    is_terminal = False
    while not is_terminal:
        _state, _reward, is_terminal = env.step(
            [random.randint(0, num_actions - 1), random.randint(0, num_actions - 1)]
        )

    num_moves = len(env.current_battle["data"]["inputLog"])
    print(f"Terminated in {num_moves} moves")

    print(env.current_battle["data"])


@click.command()
@click.option(
    "--format",
    type=str,
    default="gen1randombattle",
    help="The format of the battle to use.",
)
@click.option("--cpu", type=int, default=4)
def main(format: str, cpu: int):
    env_fn = partial(ShowdownEnv, {"formatid": format})
    ac_kwargs = dict(hidden_sizes=(64, 64))

    with tf.Graph().as_default():
        ppo(env_fn, epochs=100, ac_kwargs=ac_kwargs)


if __name__ == "__main__":
    main()  # pylint: disable=E1120
