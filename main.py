import numpy as np
import os
import sys
import time
import gym
from tqdm import tqdm
from agent.agent import AgentOption
from variables import *
from gridenvs.wrappers.obs import ObservationZoneWrapper
from multiprocessing import Pool
sys.path.append('gridenvs')


class Experiment(object):
    """
    This class makes experiments between an agent and its environment
    TODO : agent.reset()
    """
    
    def __init__(self, experiment_name):
        self.experiment_data = return_data(experiment_name)
        assert self.experiment_data is not None

        self.seed = None

        # environment variables
        self.display_learning = False
        self.blurred_render = False
        self.gray_scale_render = False
        self.agent_view = True
        self.env = self.get_environment()

        # agent variables
        self.total_reward = 0
        self.agent = self.get_agent(self.env.reset())

        self.ATARI_state = self.save_state(self.agent.initial_state)
        self.result_file_name = self.write_setting()

    def get_agent(self, initial_state):
        if self.experiment_data["AGENT"] == "AgentOption":
            return AgentOption(initial_state=initial_state,
                               current_state=initial_state,
                               number_actions=self.env.action_space.n,
                               type_exploration="OptionExplore",
                               play=False,
                               experiment_data=self.experiment_data)

        else:
            raise NotImplementedError()

    def get_environment(self, wrapper_obs=True):
        if wrapper_obs:
            env = gym.make(self.experiment_data["ENV_NAME"]).env  # to remove wrapper TimeLimit
            env = ObservationZoneWrapper(env,
                                         zone_size_option_x=self.experiment_data["ZONE_SIZE_OPTION_X"],
                                         zone_size_option_y=self.experiment_data["ZONE_SIZE_OPTION_Y"],
                                         zone_size_agent_x=self.experiment_data["ZONE_SIZE_AGENT_X"],
                                         zone_size_agent_y=self.experiment_data["ZONE_SIZE_AGENT_Y"],
                                         blurred=self.experiment_data["BLURRED"],
                                         thresh_binary_option=self.experiment_data["THRESH_BINARY_OPTION"],
                                         thresh_binary_agent=self.experiment_data["THRESH_BINARY_AGENT"],
                                         gray_scale=self.experiment_data["GRAY_SCALE"])

            env.render(blurred_render=self.blurred_render,
                       gray_scale_render=self.gray_scale_render,
                       agent_render=self.agent_view)

            env.unwrapped.viewer.window.on_key_press = self.key_press
            env.unwrapped.viewer.window.on_key_release = self.key_release

            return env

        else:
            raise NotImplementedError()

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(seed)
        self.env.seed(seed)

    def save_state(self, obs):
        if self.experiment_data["SAVE_STATE"]:
            self.agent.initial_state = obs
            return self.env.unwrapped.clone_full_state()

    def restore_state(self):
        if self.experiment_data["SAVE_STATE"]:
            self.agent.reset()
            self.env.unwrapped.restore_full_state(self.ATARI_state)

        else:
            self.agent.reset()
            self.env.reset()

    def write_reward(self, t):
        f = open(self.result_file_name, "a")
        f.write("t = " + str(t) + " reward = " + str(self.total_reward) + "\n")
        f.close()

    def write_setting(self):
        dir_name = "results/" + self.experiment_data["NAME"]
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        
        dir_name += "/" + time.asctime(time.localtime(time.time())).replace(" ","_")
        os.mkdir(dir_name)

        f = open(dir_name + "/" + "setting", "a")
        for key in self.experiment_data:
            f.write(key + " : " + str(self.experiment_data[key]) + "\n")
        f.write("\n" * 3)
        f.close()

        return dir_name

    def write_message(self, message):
        f = open(self.result_file_name, "a")
        f.write(message)
        f.close()

    def learn(self):
        self.result_file_name = self.result_file_name + "/" + "seed_" + str(self.seed)
        full_lives = {'ale.lives': 6}
        
        for t in tqdm(range(1, self.experiment_data["ITERATION_LEARNING"] + 1)):                

            self.restore_state()
            done = False
            option = None

            while not done:

                if self.display_learning:
                    self.env.render(blurred_render=self.blurred_render,
                                    gray_scale_render=self.gray_scale_render,
                                    agent_render=self.agent_view)

                else:
                    self.env.unwrapped.viewer.window.dispatch_events()

                if option is None:
                    option = self.agent.choose_option(t)

                action = option.act()
                obs, reward, done, info = self.env.step(action)
                end_option = option.update_option(reward, obs, action, info['ale.lives'])

                if reward > 0:
                    self.total_reward += reward
                    self.write_reward(t)
                    self.ATARI_state = self.save_state(obs)
                    break
                
                if end_option:
                    option = None
                    self.agent.update_agent(obs, option, action)

                done = (info != full_lives)                
                
        experiment.write_message("Experiment complete.")

    def key_press(self, key, mod):
        if key == ord("d"):
            self.display_learning = not self.display_learning

        if key == ord("b"):
            self.blurred_render = not self.blurred_render

        if key == ord("g"):
            self.gray_scale_render = not self.gray_scale_render

        if key == ord("a"):
            self.agent_view = not self.agent_view

    def key_release(self, key, mod):
        pass


if __name__ == '__main__':
    
    experiment = Experiment("First_good_results")
    parallel = False

    if parallel:
        number_cores = 6

        def distribute_seed_and_learn(seed):
            experiment.set_seed(seed)
            experiment.learn()

        p = Pool()
        p.map(distribute_seed_and_learn, range(number_cores))
        p.close()
        p.join()

    else:
        experiment.learn()
