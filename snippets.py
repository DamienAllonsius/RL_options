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
    """
    to be used with class ZonesEnv
    """
    def __init__(self, env, zone_size_x, zone_size_y, blurred):
        super(gym.ObservationWrapper, self).__init__(env)
        self.zone_size_x = zone_size_x
        self.zone_size_y = zone_size_y
        self.blurred = blurred
        
    def observation(self, observation):
        if self.blurred:
            len_y = len(observation) # with MontezumaRevenge-v4 : 160
            len_x = len(observation[0]) # with MontezumaRevenge-v4 : 210
            if (len_x % self.zone_size_x == 0) and (len_y % self.zone_size_y == 0):
                downsampled_size = (len_x // self.zone_size_x , len_y // self.zone_size_y)
                img_blurred = cv2.resize(observation, downsampled_size, interpolation=cv2.INTER_AREA) # vector of size "downsampled_size"
                return img_blurred
            
            else:
                raise Exception("The gridworld can not be fragmented into zones")
        
        else:
            return observation

        # gray scale ?
        #             if self.number_gray_colors:
        #                 for j in range(size_x_image_blurred):
        #                     for i in range(size_y_image_blurred):
        #                         rgb = image_blurred[i][j]
        #                         gray_level = (255 * 3) // self.number_gray_colors
        #                         sum_rgb = (sum(rgb) // gray_level) * gray_level
        #                         image_blurred[i][j] = [sum_rgb] * 3
        #             return image_blurred

"""
env = ObservationZoneWrapper(gym.make(ENV_NAME),  zone_size_x = 1, zone_size_y = 1, blurred = True)
obs = env.reset()

viewer = rendering.SimpleImageViewer()
viewer.imshow(obs)

time.sleep(5)
"""
