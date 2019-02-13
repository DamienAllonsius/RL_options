import sys
import gym
import numpy as np
import time
import cv2
from gym.envs.classic_control import rendering
from variables import *
sys.path.append('gridenvs')
import gridenvs.examples  # load example gridworld environments


class ObservationZoneWrapper(gym.ObservationWrapper):

    def __init__(self, env, zone_size_x, zone_size_y, blurred, number_gray_colors = 0):

        super(gym.ObservationWrapper, self).__init__(env)
        self.blurred = blurred
        self.zone_size_x = zone_size_x
        self.zone_size_y = zone_size_y
        self.number_gray_colors = number_gray_colors
    
    def observation(self, observation):
        len_y = len(observation) # with MontezumaRevenge-v4 : 160
        len_x = len(observation[0]) # with MontezumaRevenge-v4 : 210
        downsampled_size = (len_x // self.zone_size_x , len_y // self.zone_size_y)
        img_blurred = cv2.resize(observation, downsampled_size, interpolation=cv2.INTER_AREA) # vector of size "downsampled_size"
        img_blurred_resized = cv2.resize(img_blurred, (512, 512), interpolation=cv2.INTER_NEAREST) # vector of size "512x512"
        return img_blurred_resized
"""
env = ObservationZoneWrapper(gym.make(ENV_NAME),  zone_size_x = 1, zone_size_y = 1, blurred = True)
obs = env.reset()

viewer = rendering.SimpleImageViewer()
viewer.imshow(obs)

time.sleep(5)
"""
