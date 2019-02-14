import sys, time
sys.path.append('gridenvs')
import gridenvs.examples  # load example gridworld environments
import gym
import numpy as np
import time
from tqdm import tqdm
from agent.agent import KeyboardAgent, AgentOption, QAgent
from gridenvs.utils import Point
from variables import *
from snippets import ObservationZoneWrapper # TODO : make a proper file dedicated to wrappers

def make_environment_agent(env_name, type_agent):
    """
    type_agent parameter should be "AgentOption" or "QAgent"
    """
    
    if type_agent == "AgentOption":
        env = ObservationZoneWrapper(gym.make(ENV_NAME), zone_size_x = ZONE_SIZE_X, zone_size_y = ZONE_SIZE_Y, blurred = BLURRED)
        _, initial_agent_state = env.reset() #first output : observation, second output : blurred observation
        type_exploration = "OptionExplore"
        agent = AgentOption(state = initial_agent_state, number_actions = env.action_space.n, type_exploration = type_exploration, play = False)
        return env, agent, initial_agent_state
        
    elif type_agent == "QAgent":
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
    display_learning = True
    while not(done):
        if display_learning:
            env.render()
            #time.sleep(1)
        # if no option acting, choose an option
        if not(running_option):
            option = agent.choose_option(t)
            #print(agent.q)
            running_option = True
                
        # else, let the current option act
        action = option.act()
        new_obs, new_obs_blurred, reward, done, _ = env.step(action)
        end_option = option.update_option(reward, new_obs, new_obs_blurred, action)
        # if the option ended then update the agent's data
        # In Montezuma : done = dead, reward when you pick a key or open a door, info : number of lifes
        if end_option:
            #agent.update_agent(new_position, new_agent_state, option, action)
            # In this case the option ended normally and the process continues
            running_option = False
            agent.update_agent(new_obs_blurred, option, action)

def act(env, t, initial_setting):
    agent.reset(initial_setting)
    done = False
    display_learning = True
    #start the loop
    while not(done):
        if display_learning:
            #time.sleep(.2)
            env.render_scaled()
                
        action = agent.act(t)
        _, reward, done, info = env.step(action)
        new_position = info['position']
        new_state_id = info['state_id']
        agent.update(reward, new_position, action, new_state_id)
    

def learn_or_play(env, agent, play, initial_setting, iteration = ITERATION_LEARNING, seed = 0):
    
    np.random.seed(seed)
    agent.play = play
    agent.make_save_data(seed)
    if play:
        iteration = 1
        env.reset()
        env.render()
        wait = input("PRESS ENTER TO PLAY.")
        
    for t in tqdm(range(1, iteration + 1)):
        # reset the parameters
        env.reset()
        if type(agent).__name__ == "AgentOption":
            act_options(env, t, initial_setting)
            
        elif type(agent).__name__ == "QAgent":
            act(env, t, initial_setting)
      
        if(not(play)):
            agent.record_reward(t)
    if play:
        env.render_scaled()
        time.sleep(1)
        
    env.close()
    if not(play):
        return agent



env_name = ENV_NAME if len(sys.argv)<2 else sys.argv[1] #default environment or input from command line 'GE_Montezuma-v1'
type_agent = "AgentOption"

for seed in range(NUMBER_SEEDS):
    env, agent, initial_setting = make_environment_agent(env_name, type_agent = type_agent)
    agent_learned = learn_or_play(env, agent, iteration = ITERATION_LEARNING, play = False, seed = seed, initial_setting = initial_setting)
    
agent_learned.save_data.plot_data()
