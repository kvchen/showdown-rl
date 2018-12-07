#!/usr/bin/env python3


def is_terminal(battle_data) -> bool:
    return battle_data["ended"]


def get_side_value(side) -> float:
    return sum(pokemon["hp"] / pokemon["maxhp"] for pokemon in side["pokemon"])


def get_heuristic_value(battle_data):
    sides = battle_data["sides"]
    return get_side_value(sides[1]) - get_side_value(sides[0])


def alpha_beta(env, battle, depth, alpha, beta, player_idx, last_move):
    client = env.client
    battle_id = battle["id"]
    battle_data = battle["data"]

    next_player_idx = (player_idx + 1) % 2

    if depth == 0 or is_terminal(battle_data):
        return get_heuristic_value(battle_data), None

    best_move_idx = None

    if player_idx == 0:
        value = -float("inf")
        for move_idx in battle["actions"][1]:
            successor_value, _ = alpha_beta(
                env, battle, depth, alpha, beta, next_player_idx, env.get_move(move_idx)
            )

            if successor_value > value:
                value = successor_value
                best_move_idx = move_idx

            alpha = max(alpha, value)
            if alpha >= beta:
                break

        return value, best_move_idx
    else:
        value = float("inf")
        for move_idx in battle["actions"][0]:
            successor = client.do_move(battle_id, env.get_move(move_idx), last_move)
            successor_value, _ = alpha_beta(
                env, successor, depth - 1, alpha, beta, next_player_idx, None
            )

            if successor_value < value:
                value = successor_value
                best_move_idx = move_idx

            beta = min(beta, value)
            if alpha >= beta:
                break

        return value, best_move_idx


def agent(env, depth=1):
    best_value, best_move_idx = alpha_beta(
        env, env.current_battle, depth, -float("inf"), float("inf"), 0, None
    )

    print(best_value, best_move_idx)
    return best_move_idx
