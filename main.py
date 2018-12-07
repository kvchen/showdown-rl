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
@click.option("--agent", type=str, default="random")
def main(format: str, agent: str, *args, **kwargs):
    agent_module = import_module("." + agent, "agents")
    print(dir(agent_module))

    env_fn = partial(ShowdownEnv, agent_module.agent, {"formatid": format})
    ac_kwargs = {"hidden_sizes": (128, 128)}

    with tf.Graph().as_default():
        ppo(env_fn, epochs=100, ac_kwargs=ac_kwargs)


if __name__ == "__main__":
    main()  # pylint: disable=E1120
