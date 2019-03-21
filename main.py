import numpy as np
import sys, time
sys.path.append('gridenvs')
import gridenvs.examples  # load example gridworld environments
import gym
import subprocess
import cv2
from tqdm import tqdm
from agent.agent import KeyboardAgent, AgentOption, QAgent
from gridenvs.utils import Point
from variables import *
from wrappers.obs import ObservationZoneWrapper # TODO : make a proper file dedicated to wrappers

def make_environment_agent(env_name, type_agent, seed):
    """
    type_agent parameter should be "AgentOption" or "QAgent"
    """
    np.random.seed(seed)
    if type_agent == "AgentOption":
        env = ObservationZoneWrapper(gym.make(ENV_NAME), zone_size_option_x = ZONE_SIZE_OPTION_X, zone_size_option_y = ZONE_SIZE_OPTION_Y, zone_size_agent_x = ZONE_SIZE_AGENT_X, zone_size_agent_y = ZONE_SIZE_AGENT_Y, blurred = BLURRED, thresh_binary_option = THRESH_BINARY_OPTION, thresh_binary_agent = THRESH_BINARY_AGENT, gray_scale = GRAY_SCALE)
        env.seed(seed) # There is randomness in ATARI !
        obs = env.reset() #first output : observation, second output : blurred observation
        #ATARI_state = env.unwrapped.clone_full_state()
        type_exploration = "OptionExplore"
        number_actions = env.action_space.n
        agent = AgentOption(current_state = obs, number_actions = number_actions, type_exploration = type_exploration, play = False)
        return env, agent, obs
        
    elif type_agent == "QAgent":
        raise Exception("Not Implemented yet")
        env = gym.make(ENV_NAME)
        env.reset()
        initial_agent_position = env.get_hero_position()
        grid_size = env.world.grid_size
        agent = QAgent(initial_agent_position, grid_size, play = False)
        return env, agent, initial_agent_position
        
    else:
        raise Exception("agent name does not exist")
    
def act_options(env, t, initial_setting):
    """
    0/ The agent chooses an option
    1/ The option makes the action
    2/ The environment gives the feedback
    3/ We update the option's parameters and we get end_option which is True if only if the option is done.
    4/ The agent update his info about the option
    """
    agent.reset(initial_setting)
    running_option = False
    #start the loop
    done = False
    full_lives = {'ale.lives': 6}
    display_learning = t>15000 and t<20000
    while not(done):
        if display_learning:
            env.render(blurred_render = True, gray_scale_render = True, agent = True)
        # if no option acting, choose an option
        if not(running_option):
            option = agent.choose_option(t)
            running_option = True
            
        # else, let the current option act
        action = option.act()
        obs, reward, done, info = env.step(action)
        end_option = option.update_option(reward, obs, action, info['ale.lives'])
        # if the option ended then update the agent's data
        # In Montezuma : done = dead, reward when you pick a key or open a door, info : number of lifes
        if end_option:
            #agent.update_agent(new_position, new_agent_state, option, action)
            # In this case the option ended normally and the process continues
            running_option = False
            positive_reward = agent.update_agent(obs, option, action)
            if positive_reward:
                subprocess.Popen(['notify-send', "got a posive reward !"])
                print("\033[93m got a posive reward !")
                #ATARI_state = env.unwrapped.clone_full_state()

        done = (info != full_lives)

def act(env, t, initial_setting):
    agent.reset(initial_setting)
    done = False
    display_learning = True
    #start the loop
    while not(done):
        if display_learning or play:
            if play:
                time.sleep(.2)
                
            env.render_scaled()
                
        action = agent.act(t)
        _, reward, done, info = env.step(action)
        new_position = info['position']
        new_state_id = info['state_id']
        agent.update(reward, new_position, action, new_state_id)
    
def learn_or_play(env, agent, play, initial_setting, iteration = ITERATION_LEARNING):
    agent.play = play
    #agent.make_save_data(seed)
    if play:
        iteration = 1
        env.reset()
        env.render()
        wait = input("PRESS ENTER TO PLAY.")
        
    for t in tqdm(range(1, iteration + 1)):
        # reset the parameters
        env.reset()
        #env.unwrapped.restore_full_state(ATARI_state)
        if type(agent).__name__ == "AgentOption":
            act_options(env, t, initial_setting)
            
        elif type(agent).__name__ == "QAgent":
            act(env, t, initial_setting)
      
        # if(not(play)):
        #     agent.record_reward(t)
    if play:
        env.render_scaled()
        time.sleep(1)
        
    env.close()
    if not(play):
        return agent

env_name = ENV_NAME if len(sys.argv)<2 else sys.argv[1] #default environment or input from command line 'GE_Montezuma-v1'
type_agent = "AgentOption"

for seed in range(NUMBER_SEEDS):
    env, agent, initial_setting = make_environment_agent(env_name, type_agent, seed)
    agent_learned = learn_or_play(env, agent, iteration = ITERATION_LEARNING, play = False, initial_setting = initial_setting)
    
#agent_learned.save_data.plot_data()
