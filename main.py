#!/usr/bin/env python3

from typing import Optional

import time
import os
import gym
import click
import tensorflow as tf
from functools import partial
from gym_showdown.envs import ShowdownEnv
from spinup.utils.logx import restore_tf_graph
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
@click.option("--checkpoint", type=click.Path(exists=True))
@click.option(
    "--logdir",
    type=click.Path(),
    default=os.path.join(os.getcwd(), "experiments", int(time.time())),
)
def main(
    format: str,
    opponent: str,
    epochs: int,
    steps: int,
    checkpoint: Optional[str],
    *args,
    **kwargs
):
    agent_module = import_module("." + opponent, "agents")
    env_fn = partial(ShowdownEnv, agent_module.agent, {"formatid": format})
    ac_kwargs = {"hidden_sizes": (512, 512)}

    graph = tf.Graph()
    with graph.as_default():
        if checkpoint:
            sess = tf.Session(graph=graph)
            restore_tf_graph(sess, checkpoint)

        ppo(env_fn, epochs=epochs, steps_per_epoch=steps, ac_kwargs=ac_kwargs)


if __name__ == "__main__":
    main()  # pylint: disable=E1120
