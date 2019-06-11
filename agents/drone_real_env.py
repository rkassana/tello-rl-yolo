# -*- coding: utf-8 -*-

import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

class drone_real(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.min_action = -60.0
        self.max_action = 60.0
        self.min_position = 0
        self.max_positionx = 960
        self.max_positiony = 720
        self.goal_position = np.array([480,360])

        self.low_state = np.array([self.min_position,self.min_position])
        self.high_state = np.array([self.max_positionx,self.max_positiony])

        self.viewer = None

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
                                       shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        done=False
        reward=0
        dist = self.get_dist()
        reward -= dist*0.25 if dist>100 else -(100- dist)

        return self.state, reward, done, {}

    def update_state(self,posx,posy):
        self.state[0] = posx
        self.state[1] = posy

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=0, high=960),self.np_random.uniform(low=0, high=720)])
        self.init_dist = np.sqrt((np.square(self.state - self.goal_position)).sum())
        return np.array(self.state)


    def get_dist(self):
        dist = np.sqrt((np.square(self.state - self.goal_position)).sum())
        return dist


    def render(self, mode='human'):
        screen_width = 960
        screen_height = 720
        scale = 1
        carwidth=20
        carheight=20


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position[0]-self.min_position)*scale
            flagy1 = (self.goal_position[1]-self.min_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

        posx = self.state[0]
        posy = self.state[1]
        self.cartrans.set_translation((posx-self.min_position)*scale, (posy-self.min_position)*scale)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None