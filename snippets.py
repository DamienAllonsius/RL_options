import sys
import gym
import numpy as np
import time
import cv2
from gym.envs.classic_control import rendering
from variables import *
sys.path.append('gridenvs')
import gridenvs.examples  # load example gridworld environments
from gym.envs.classic_control import rendering

class ObservationZoneWrapper(gym.ObservationWrapper):
    """
    to be used with class ZonesEnv
    """
    def __init__(self, env, zone_size_x, zone_size_y, blurred):
        super(gym.ObservationWrapper, self).__init__(env)
        self.zone_size_x = zone_size_x
        self.zone_size_y = zone_size_y
        self.blurred = blurred
        
    def render(self, size = (512, 512), mode = 'human', close = False):
        if hasattr(self.env.__class__, 'render_scaled'): # we call render_scaled function from gridenvs
            return self.env.render_scaled(size, mode, close)
         
        else: # we scale the image from other environment (like Atari)
            img = self.env.env.ale.getScreenRGB2()
            if self.blurred:
                img = self.make_downsampled_image(img)
            img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
            if mode == 'rgb_array':
                return img
            elif mode == 'human':
                from gym.envs.classic_control import rendering
                if self.env.env.viewer is None:
                    self.env.env.viewer = rendering.SimpleImageViewer()
                    self.env.env.viewer.imshow(img)
                return self.env.env.viewer.isopen

    def make_downsampled_image(self, image):
        len_y = len(image) # with MontezumaRevenge-v4 : 160
        len_x = len(image[0]) # with MontezumaRevenge-v4 : 210
        if (len_x % self.zone_size_x == 0) and (len_y % self.zone_size_y == 0):
            downsampled_size = (len_x // self.zone_size_x , len_y // self.zone_size_y)
            img_blurred = cv2.resize(image, downsampled_size, interpolation=cv2.INTER_AREA) # vector of size "downsampled_size"
            return img_blurred
        else:
            raise Exception("The gridworld " + str(len_x) + "x" + str(len_y) +  " can not be fragmented into zones " + str(self.zone_size_x) + "x" + str(self.zone_size_y))
            
    def observation(self, observation):
        #instead of returning a nested array, returnes a *blurred*, *nested* *tuple*
        if self.blurred:
            img_blurred = self.make_downsampled_image(observation)
            # transform the observation in tuple
            img_blurred_tuple = tuple(tuple(tuple(color) for color in lig) for lig in img_blurred)
            observation_tuple = tuple(tuple(tuple(color) for color in lig) for lig in observation)
            return observation_tuple, img_blurred_tuple
        
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
