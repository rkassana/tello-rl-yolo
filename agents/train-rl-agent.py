from agents.rl_drone import RLAgent
from agents.drone_sim_env import drone_sim

env = drone_sim()
agent = RLAgent(env)
ENV_NAME = 'drone'
agent.agent.fit(env, nb_steps=100000, visualize=True, verbose=1, nb_max_episode_steps=10)

#After training is done, we save the final weights.
agent.agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
#agent.agent.load_weights('ddpg_{}_weights.h5f'.format(ENV_NAME))

# Finally, evaluate our algorithm for 5 episodes.