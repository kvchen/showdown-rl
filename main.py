#!/usr/bin/env python3

import gym
import click
import random
import tensorflow as tf
from functools import partial
from gym_showdown.envs import ShowdownEnv
from ppo.ppo import ppo


@click.command()
@click.option(
    "--format",
    type=str,
    default="gen1randombattle",
    help="The format of the battle to use.",
)
def main(format: str, *args, **kwargs):
    env_fn = partial(ShowdownEnv, {"formatid": format})
    ac_kwargs = dict(hidden_sizes=(64, 64))

    with tf.Graph().as_default():
        ppo(env_fn, ac_kwargs=ac_kwargs)


if __name__ == "__main__":
    main()  # pylint: disable=E1120
