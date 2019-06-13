import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, Lambda
from keras.optimizers import Adam
from keras.initializers import RandomUniform
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess


class RLAgent:

    def __init__(self, env):
        ENV_NAME = 'drone'
        # Get the environment and extract the number of actions.
        #env = gym.make(ENV_NAME)
        self.env = env
        np.random.seed(123)
        self.env.seed(123)
        assert len(self.env.action_space.shape) == 1
        nb_actions = self.env.action_space.shape[0]

        # Next, we build a very simple model.
        self.actor = Sequential()
        self.actor.add(Flatten(input_shape=(1,) + self.env.observation_space.shape))
        self.actor.add(Dense(16))
        self.actor.add(Activation('relu'))
        self.actor.add(Dense(16))
        self.actor.add(Activation('relu'))
        self.actor.add(Dense(16))
        self.actor.add(Activation('relu'))
        self.actor.add(Dense(nb_actions,activation='tanh',kernel_initializer=RandomUniform()))
        self.actor.add(Lambda(lambda x: x * 60.0))
        print(self.actor.summary())

        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=(1,) + self.env.observation_space.shape, name='observation_input')
        flattened_observation = Flatten()(observation_input)
        x = Concatenate()([action_input, flattened_observation])
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)
        print(critic.summary())

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        memory = SequentialMemory(limit=100000, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
        self.agent = DDPGAgent(nb_actions=nb_actions, actor=self.actor, critic=critic, critic_action_input=action_input,
                          memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                          random_process=random_process, gamma=.99, target_model_update=1e-3)
        self.agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

