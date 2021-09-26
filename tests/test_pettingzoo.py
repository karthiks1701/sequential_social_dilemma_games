import unittest

from config.default_args import add_default_args
import argparse
import numpy as np
from social_dilemmas.envs.env_creator import get_env_creator
from social_dilemmas.envs.pettingzoo_env import MAX_CYCLES, parallel_env, env as aec_env

parser = argparse.ArgumentParser()
add_default_args(parser)
args = parser.parse_args()
# env = get_env_creator(args.env, args.num_agents, args)(args.num_agents)

from pettingzoo.test import api_test, parallel_api_test


class PettingZooTest(unittest.TestCase):
    def test_parallel(self):
        env = parallel_env(max_cycles=MAX_CYCLES, ssd_args=args)
        env.seed(0)
        env.reset()
        n_act = env.action_spaces["agent-0"].n
        dones = [False] * env.num_agents
        for _ in range(MAX_CYCLES):
            agents = env.agents
            actions = {agent: np.random.randint(n_act) for agent in agents}
            obss, rewss, dones, infos = env.step(actions)
            if not env.agents:
                env.reset()
        # parallel_api_test(env, MAX_CYCLES)

    # def test_aec(self):
    #     env = aec_env(max_cycles=MAX_CYCLES, ssd_args=args)
    #     env.seed(0)
    #     env.reset()
    #     n_act = env.action_spaces["agent-0"].n
    #     for _ in range(MAX_CYCLES):
    #         for agent in env.agent_iter():
    #             observation, reward, done, info = env.last()
    #             action = np.random.randint(n_act)
    #             env.step(action)
    #             if not env.agents:
    #                 env.reset()
    #     # api_test(env, MAX_CYCLES)


if __name__ == "__main__":
    unittest.main()
