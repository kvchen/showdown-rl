import click
from gym_showdown.envs import ShowdownEnv

# from spinup import ppo


def main():
    env = ShowdownEnv()
    client = env.client
    battle = client.start_battle({"formatid": "gen1randombattle"})

    # States are carried out simultaneously
    # while not battle["data"]["ended"]:
    battle = client.do_move(battle["id"], "default", "default")
    print(battle["data"])


if __name__ == "__main__":
    main()
