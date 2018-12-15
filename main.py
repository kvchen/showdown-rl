#!/usr/bin/env python3

from typing import Optional

import time
import os
import gym
import click
import tensorflow as tf
from functools import partial
from gym_showdown.envs import ShowdownEnv
from importlib import import_module
from ppo.ppo import ppo
from ppo.test_policy import load_policy, run_policy


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
@click.option("--logdir", type=click.Path())
def main(
    format: str,
    opponent: str,
    epochs: int,
    steps: int,
    checkpoint: Optional[str],
    logdir: Optional[str],
    *args,
    **kwargs,
):
    if logdir is None:
        current_time = int(time.time())
        logdir = os.path.join(
            "experiments", f"{format}_{opponent}_e{epochs}_s{steps}_{current_time}"
        )

    agent_module = import_module("." + opponent, "agents")
    env_fn = partial(ShowdownEnv, agent_module.agent, {"formatid": format})
    ac_kwargs = {"hidden_sizes": (2048, 1024, 512)}

    if checkpoint:
        _, get_action = load_policy(checkpoint)
        run_policy(env_fn(), get_action)
    else:
        graph = tf.Graph()
        with graph.as_default():
            ppo(
                env_fn,
                epochs=epochs,
                steps_per_epoch=steps,
                ac_kwargs=ac_kwargs,
                logger_kwargs={"output_dir": logdir},
            )


if __name__ == "__main__":
    main()  # pylint: disable=E1120
