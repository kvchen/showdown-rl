#!/usr/bin/env python3

import random


def agent(env):
    """Agent that selects an available move at random."""
    moves = env.current_battle["actions"][1]
    return random.choice(moves)
