import gym
import numpy as np
import time
import cv2
from gym.envs.classic_control import rendering

class ObservationZoneWrapper(gym.ObservationWrapper):

    def __init__(self, env, downsampled_size_x, downsampled_size_y, blurred, number_gray_colors=0):

        super(gym.ObservationWrapper, self).__init__(env)
        self.blurred = blurred
        self.downsampled_size = (downsampled_size_x, downsampled_size_y)
        self.number_gray_colors = number_gray_colors
    
    def observation(self, observation):
        img_blurred = cv2.resize(observation, self.downsampled_size, interpolation=cv2.INTER_AREA)
        img_blurred_resized = cv2.resize(img_blurred, (512, 512), interpolation=cv2.INTER_NEAREST)
        return img_blurred_resized

env = ObservationZoneWrapper(gym.make("MontezumaRevenge-v4"),  downsampled_size_x = 70, downsampled_size_y = 70, blurred = True)
obs = env.reset()

viewer = rendering.SimpleImageViewer()
viewer.imshow(obs)

time.sleep(100)
