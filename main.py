import numpy as np
import os, sys, time
sys.path.append('gridenvs')
import gridenvs.examples 
import gym
import subprocess
import cv2
from tqdm import tqdm
from agent.agent import KeyboardAgent, AgentOption, QAgent
from gridenvs.utils import Point
from variables import *
from wrappers.obs import ObservationZoneWrapper 
from multiprocessing import Pool
from datetime import datetime

class Experiment(object):
    """
    This class makes experiments between an agent and its environment
    TODO : agent.reset()
    """
    
    def __init__(self, experiment_name):
        self.experiment_data = return_data(experiment_name)
        if self.experiment_data is None:
            raise Exception("This experiment does not exist")

        self.seed = None
        self.set_environment()
        self.make_agent(self.env.reset())
        self.save_state(self.agent.initial_state)
        self.write_setting()
        
    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(seed)
        self.env.seed(seed)

    def set_environment(self, wrapper_obs = True):
        if wrapper_obs:
            self.display_learning = False
            self.blurred_render = False
            self.gray_scale_render = False
            self.agent_view = True
            self.env = gym.make(self.experiment_data["ENV_NAME"]).env # to remove wrapper TimeLimit
            self.env = ObservationZoneWrapper(self.env,
                                         zone_size_option_x = self.experiment_data["ZONE_SIZE_OPTION_X"],
                                         zone_size_option_y = self.experiment_data["ZONE_SIZE_OPTION_Y"],
                                         zone_size_agent_x = self.experiment_data["ZONE_SIZE_AGENT_X"],
                                         zone_size_agent_y = self.experiment_data["ZONE_SIZE_AGENT_Y"],
                                         blurred = self.experiment_data["BLURRED"],
                                         thresh_binary_option = self.experiment_data["THRESH_BINARY_OPTION"],
                                         thresh_binary_agent = self.experiment_data["THRESH_BINARY_AGENT"],
                                         gray_scale = self.experiment_data["GRAY_SCALE"])

            self.env.render(blurred_render = self.blurred_render, gray_scale_render = self.gray_scale_render, agent = self.agent_view)
            self.env.unwrapped.viewer.window.on_key_press = self.key_press
            self.env.unwrapped.viewer.window.on_key_release = self.key_release

        else:
            raise Exception("not implemented yet")

    def save_state(self, obs):
        if self.experiment_data["SAVE_STATE"]:
            self.agent.initial_state = obs
            self.ATARI_state = self.env.unwrapped.clone_full_state()

    def restore_state(self):
        if self.experiment_data["SAVE_STATE"]:
            self.agent.reset()
            self.env.unwrapped.restore_full_state(self.ATARI_state)

        else:
            self.agent.reset()
            self.env.reset()

    def make_agent(self, initial_state):
        self.total_reward = 0
        if self.experiment_data["AGENT"] == "AgentOption":
            self.agent = AgentOption(initial_state = initial_state,
                                     current_state = initial_state,
                                     number_actions = self.env.action_space.n,
                                     type_exploration = "OptionExplore",
                                     play = False,
                                     experiment_data = self.experiment_data)
            
        else:
            raise Exception("Not implemented yet")

    def write_reward(self, t):
        f = open(self.result_file_name, "a")
        f.write("t = " + str(t) + " reward = " + str(self.total_reward) + "\n")
        f.close()

    def write_setting(self):
        dir_name = "results/" + self.experiment_data["NAME"]
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        
        dir_name += "/" + time.asctime( time.localtime(time.time())).replace(" ","_")
        self.result_dir_name = dir_name
        os.mkdir(dir_name)

        f = open(dir_name + "/" + "setting", "a")
        for key in self.experiment_data:
            f.write(key + " : " + str(self.experiment_data[key]) + "\n")
        f.write("\n" * 3)
        f.close()

    def write_message(self, message):
        f = open(self.result_file_name, "a")
        f.write(message)
        f.close()

    def learn(self):
        self.result_file_name = self.result_file_name + "/" + "seed_" + str(self.seed)
        full_lives = {'ale.lives': 6}
        
        for t in tqdm(range(1, self.experiment_data["ITERATION_LEARNING"] + 1)):                

            self.restore_state()
            
            running_option = False
            done = False
            positive_reward = False
            while not(done):

                if self.display_learning:
                    self.env.render(blurred_render = self.blurred_render, gray_scale_render = self.gray_scale_render, agent = self.agent_view)

                else:
                    self.env.unwrapped.viewer.window.dispatch_events()

                if not(running_option):
                    option = self.agent.choose_option(t)
                    running_option = True

                action = option.act()
                obs, reward, done, info = self.env.step(action)
                end_option = option.update_option(reward, obs, action, info['ale.lives'])

                if reward > 0:
                    self.total_reward += reward
                    self.write_reward(t)
                    self.save_state(obs)
                    done = True
                
                if end_option:
                    running_option = False
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


# def make_environment_agent(env_name, type_agent, seed):
#     """
#     type_agent parameter should be "AgentOption" or "QAgent"
#     """
#     np.random.seed(seed)
#     if type_agent == "AgentOption":
#         env = ObservationZoneWrapper(gym.make(ENV_NAME), zone_size_option_x = ZONE_SIZE_OPTION_X, zone_size_option_y = ZONE_SIZE_OPTION_Y, zone_size_agent_x = ZONE_SIZE_AGENT_X, zone_size_agent_y = ZONE_SIZE_AGENT_Y, blurred = BLURRED, thresh_binary_option = THRESH_BINARY_OPTION, thresh_binary_agent = THRESH_BINARY_AGENT, gray_scale = GRAY_SCALE)
#         env.seed(seed) # There is randomness in ATARI !
#         obs = env.reset() #first output : observation, second output : blurred observation
        
#         type_exploration = "OptionExplore"
#         number_actions = env.action_space.n
#         agent = AgentOption(current_state = obs, number_actions = number_actions, type_exploration = type_exploration, play = False)
#         return env, agent, obs
        
#     elif type_agent == "QAgent":
#         raise Exception("Not Implemented yet")
#         env = gym.make(ENV_NAME)
#         env.reset()
#         initial_agent_position = env.get_hero_position()
#         grid_size = env.world.grid_size
#         agent = QAgent(initial_agent_position, grid_size, play = False)
#         return env, agent, initial_agent_position
        
#     else:
#         raise Exception("agent name does not exist")
    
# def act_options(env, t, initial_setting):
#     """
#     0/ The agent chooses an option
#     1/ The option makes the action
#     2/ The environment gives the feedback
#     3/ We update the option's parameters and we get end_option which is True if only if the option is done.
#     4/ The agent update his info about the option
#     """
#     agent.reset(initial_setting)
#     running_option = False
#     #start the loop
#     done = False
#     full_lives = {'ale.lives': 6}
#     display_learning = False#t>800 and t<1400
#     positive_reward = False
#     while not(done):
#         if display_learning:
#             env.render(blurred_render = False, gray_scale_render = False, agent = True)
#         # if no option acting, choose an option
#         if not(running_option):
#             option = agent.choose_option(t)
#             running_option = True
            
#         # else, let the current option act
#         action = option.act()
#         obs, reward, done, info = env.step(action)
#         end_option = option.update_option(reward, obs, action, info['ale.lives'])
#         # if the option ended then update the agent's data
#         # In Montezuma : done = dead, reward when you pick a key or open a door, info : number of lifes
#         if end_option:
#             #agent.update_agent(new_position, new_agent_state, option, action)
#             # In this case the option ended normally and the process continues
#             running_option = False
#             positive_reward = agent.update_agent(obs, option, action)
#             if positive_reward:
#                 f = open("results","a")
#                 f.write("t = " + str(t) + " reward = " + str(agent.personal_reward) + "\n")
#                 f.close()
#                 #subprocess.Popen(['notify-send', "got a posive reward ! at t = " + str(
#                 print(yellow + " got a posive reward !" + white) 
                
#         done = (info != full_lives)

#     return positive_reward

# def act(env, t, initial_setting):
#     agent.reset(initial_setting)
#     done = False
#     display_learning = True
#     #start the loop
#     while not(done):
#         if display_learning or play:
#             if play:
#                 time.sleep(.2)
                
#             env.render_scaled()
                
#         action = agent.act(t)
#         _, reward, done, info = env.step(action)
#         new_position = info['position']
#         new_state_id = info['state_id']
#         agent.update(reward, new_position, action, new_state_id)
    
# def learn_or_play(env, agent, play, initial_setting, iteration = ITERATION_LEARNING):
#     agent.play = play
#     #agent.make_save_data(seed)
#     ATARI_state = env.unwrapped.clone_full_state()
#     if play:
#         iteration = 1
#         env.reset()
#         env.render()
#         wait = input("PRESS ENTER TO PLAY.")
        
#     for t in tqdm(range(1, iteration + 1)):
#         # reset the parameters
#         #env.reset()
#         env.unwrapped.restore_full_state(ATARI_state)
#         if type(agent).__name__ == "AgentOption":
#             positive_reward = act_options(env, t, initial_setting)
#             if positive_reward:
#                 ATARI_state = env.unwrapped.clone_full_state()
#                 initial_setting = #env.get_observation()
            
#         elif type(agent).__name__ == "QAgent":
#             act(env, t, initial_setting)
      
#         # if(not(play)):
#         #     agent.record_reward(t)
#     if play:
#         env.render_scaled()
#         time.sleep(1)
        
#     env.close()
#     if not(play):
#         return agent

# env_name = ENV_NAME if len(sys.argv)<2 else sys.argv[1] #default environment or input from command line 'GE_Montezuma-v1'
# type_agent = "AgentOption"

# for seed in range(NUMBER_SEEDS):
#     env, agent, initial_setting = make_environment_agent(env_name, type_agent, seed)
#     agent_learned = learn_or_play(env, agent, iteration = ITERATION_LEARNING, play = False, initial_setting = initial_setting)

# #agent_learned.save_data.plot_data()
