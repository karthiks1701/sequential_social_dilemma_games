from tests.test_envs import TestMapEnv
import numpy as np
from icecream import ic
from social_dilemmas.envs.harvest import HarvestEnv


env = HarvestEnv(num_agents=4)

agents = list(env.agents.values())

action_dim = env.action_space.n

ic(action_dim)


env.reset()
for timestep in range(100):
    # ic(timestep)
    agent_action_dict = dict()
    # ic(env.ret_num_agents())
    agents = list(env.ret_agents())
    for agent in agents:
        rand_action = np.random.randint(action_dim)
        agent_action_dict[agent.agent_id] = rand_action
    
    # ic(agent_action_dict)
    obs,rew,dones,info = env.step(agent_action_dict)
    ic(rew)
    env.render(time=0.1)

