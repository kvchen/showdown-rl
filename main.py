#!/usr/bin/env python3

import gym
import click
import tensorflow as tf
from functools import partial
from gym_showdown.envs import ShowdownEnv
from importlib import import_module
from ppo.ppo import ppo


@click.command()
@click.option(
    "--format",
    type=str,
    default="gen1randombattle",
    help="The format of the battle to use.",
)
@click.option("--opponent", type=str, default="random")
@click.option("--epochs", type=int, default=250)
@click.option("--steps", type=int, default=4000)
def main(format: str, opponent: str, epochs: int, steps: int, *args, **kwargs):
    agent_module = import_module("." + opponent, "agents")
    print(dir(agent_module))

    env_fn = partial(ShowdownEnv, agent_module.agent, {"formatid": format})
    ac_kwargs = {"hidden_sizes": (256, 256)}

    with tf.Graph().as_default():
        ppo(env_fn, epochs=epochs, steps_per_epoch=steps, ac_kwargs=ac_kwargs)


if __name__ == "__main__":
    main()  # pylint: disable=E1120
