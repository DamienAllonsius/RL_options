import sys, time
import gridenvs.examples  # load example gridworld environments
import gym
import numpy as np
import time
from tqdm import tqdm
from agent.agent import KeyboardAgent, AgentOption, QAgent
from gridenvs.utils import Point
from variables import * 

def make_environment_agent(env_name, blurred_bool = False, type_agent = "KeyboardAgent", number_gray_colors = NUMBER_GRAY_COLORS):
    env = gym.make(env_name)
    env.set_zone_size(ZONE_SIZE_X, ZONE_SIZE_Y)
    env.reset()
    env.blurred = blurred_bool
    env.number_gray_colors = number_gray_colors
    agent_position = env.get_hero_position()
    agent_state = (env.get_hero_zone(), 0)
    grid_size = env.world.grid_size
    
    if type_agent == "KeyboardAgent":
        if not hasattr(env.action_space, 'n'):
            raise Exception('Keyboard agent only supports discrete action spaces')
    
        from gridenvs.keyboard_controller import Controls
        agent = KeyboardAgent(env, controls={**Controls.Arrows, **Controls.KeyPad})

    elif type_agent == "AgentOption":
        grid_size_option = Point(ZONE_SIZE_X, ZONE_SIZE_Y)
        agent = AgentOption(agent_position, agent_state, False, grid_size_option)
        
    elif type_agent == "QAgent":
        agent = QAgent(agent_position, grid_size, False)
        
    else:
        raise Exception("agent name does not exist")
    
    return env, agent

def learn_or_play_options(env, agent, play, iteration = ITERATION_LEARNING, seed = 0):
    """
    0/ The agent chooses an option
    1/ The option makes the action
    TOFIX : I change the info in the env render. Info contains observations for the moment : zone and position of the agent
    2/ The environment gives the feedback
    3/ We update the option's parameters and we get end_option which is True if only if the option is done.
    4/ The agent update his info about the option
    """
    np.random.seed(seed)
    agent.play = play
    agent.make_save_data(seed)
    if play:
        iteration = 1
        env.reset()
        env.render_scaled()
        wait = input("PRESS ENTER TO PLAY.")
        
    for t in tqdm(range(1, iteration + 1)):
        # reset the parameters
        env.reset()
        agent.reset(INITIAL_AGENT_POSITION, INITIAL_AGENT_STATE)
        done = False
        running_option = False
        #start the loop
        while not(done):
            if False:
                env.render_scaled()
                #time.sleep(1)
            # if no option acting, choose an option
            if not(running_option):
                option = agent.choose_option(t)
                #print(agent.q)
                running_option = True
            
            # else, let the current option act
            action = option.act()
            _, reward, done, info = env.step(action)
            new_position, new_state = info['position'], (info['zone'], info['state_id'])
            end_option = option.update_option(reward, new_position, new_state, action)
            # if the option ended then update the agent's data
            if done:
                # The agent found the door or hit a wall
                if new_state[1] == 2:
                    # In this case the agent found the door
                    running_option = False
                    agent.update_agent(new_position, new_state, option, action)

            else:
                if end_option:
                    # In this case the option ended normally and the process continues
                    running_option = False
                    agent.update_agent(new_position, new_state, option, action)
            """
            if done:
                running_option = False
                
            if end_option and (not(done) or new_state[1] == 2):
                running_option = False
                agent.update_agent(new_position, new_state, option)
            """

        if(not(play)):
            agent.record_reward(t)
    if play:
        env.render_scaled()
        time.sleep(1)
    env.close()
    if not(play):
        return agent


def learn_or_play(env, agent, play, iteration = ITERATION_LEARNING, seed = 0):
    """
    play Q learning
    """
    np.random.seed(seed)
    initial_agent_position = INITIAL_AGENT_POSITION
    agent.play = play
    agent.make_save_data(seed)
    if play:
        iteration = 1
        env.reset()
        env.render_scaled()
        wait = input("PRESS ENTER TO PLAY.")
    for t in tqdm(range(1, iteration + 1)):
        # reset the parameters
        env.reset()
        agent.reset(initial_agent_position)
        done = False
        #start the loop
        while not(done):
            if play:
                #time.sleep(.2)
                env.render_scaled()
                
            action = agent.act(t)
            _, reward, done, info = env.step(action)
            new_position = info['position']
            new_state_id = info['state_id']
            agent.update(reward, new_position, action, new_state_id)

        if(not(play)):
            agent.record_reward(t)

    if play:
        env.render_scaled()
        time.sleep(1)
        
    env.close()
    if not(play):
        return agent

#play_keyboard(env, agent)
def play_keyboard(env, agent):
    """
    play with the Keyboard agent
    """
    
    #env_blurred, agent_blurred = make_environment_agent(env_name, type_agent = type_agent, blurred_bool = True)
    done = False
    total_reward = 0
    shut_down = agent.human_wants_shut_down
        
    while(not(done) and not(shut_down)):
        shut_down = agent.human_wants_shut_down
        #env_blurred.render_scaled()
        env.render_scaled()
        action = agent.act()
        if action != None:
            _, reward, done, info = env.step(action)
            total_reward += reward
            print('zone = ' + repr(info['zone']))
            #env_blurred.close()
            
    env.close()
    print('End of the episode')
    print('reward = ' + str(total_reward))

type_agent_list = ["KeyboardAgent", "AgentOption", "QAgent"]
env_name = 'GE_MazeOptions-v0' if len(sys.argv)<2 else sys.argv[1] #default environment or input from command line 'GE_Montezuma-v1'
type_agent = type_agent_list[1]

for seed in range(NUMBER_SEEDS):
    env, agent = make_environment_agent(env_name, blurred_bool = False, type_agent = type_agent)
    INITIAL_AGENT_POSITION = agent.position
    INITIAL_AGENT_STATE = agent.state
    
    if type_agent == type_agent_list[0]:
        play_keyboard(env, agent)

    elif type_agent == type_agent_list[1]:
        agent_learned = learn_or_play_options(env, agent, iteration = ITERATION_LEARNING, play = False, seed = seed)
        #learn_or_play_options(env, agent_learned, play = True)
        
    elif type_agent == type_agent_list[2]:
        agent_learned = learn_or_play(env, agent, iteration = ITERATION_LEARNING, play = False, seed = seed)
    #learn_or_play(env, agent_learned, play = True)
    
agent_learned.save_data.plot_data()
