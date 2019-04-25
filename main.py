"""RL_options version 1
RL_options designed for solving Montezuma's Revenge (ATARI)

Usage:
    main.py [options]

Options:
    -h                          Display this help.
    -a [type of Agent]          Select the agent type among: "AgentOption", "AgentQ". "AgentOption" chosen by default.
    --test                      Run the tests and exit.
"""

import sys
import time
import gym
from agent.agent import AgentOption, AgentQ, AgentOneOption
import variables
from wrappers.obs import ObservationZoneWrapper
from multiprocessing import Pool
from docopt import docopt
sys.path.append('gridenvs')


class Experiment(object):
    """
    This class makes experiments in a chosen environment and agent
    """
    
    def __init__(self, experiment_name, agent_name):
        self.agent_name = agent_name
        self.experiment_data = variables.return_data(experiment_name)

        # environment variables
        self.env = self.get_environment()

        # agent variables
        self.agent = self.get_agent(self.env.reset())

        # self.ATARI_state = self.save_state(self.agent.initial_state)

    def get_agent(self, initial_state):
        if self.agent_name == "AgentOption":
            return AgentOption(initial_state=initial_state,
                               current_state=initial_state,
                               number_actions=self.env.action_space.n,
                               type_exploration="OptionExplore",
                               play=False,
                               experiment_data=self.experiment_data)

        if self.agent_name == "AgentQ":
            return AgentQ()

        if self.agent_name == "AgentOneOption":
            return AgentOneOption(initial_state=initial_state,
                                  current_state=initial_state,
                                  number_actions=self.env.action_space.n,
                                  experiment_data=self.experiment_data)

        raise Exception(str(self.experiment_data["AGENT"]) +
                        " is not implemented")

    def get_environment(self, wrapper_obs=True):
        if wrapper_obs:
            # to remove wrapper TimeLimit
            env = gym.make(self.experiment_data["ENV_NAME"]).env
            env = ObservationZoneWrapper(env,
                                         zone_size_option_x=self.experiment_data["ZONE_SIZE_OPTION_X"],
                                         zone_size_option_y=self.experiment_data["ZONE_SIZE_OPTION_Y"],
                                         zone_size_agent_x=self.experiment_data["ZONE_SIZE_AGENT_X"],
                                         zone_size_agent_y=self.experiment_data["ZONE_SIZE_AGENT_Y"],
                                         blurred=self.experiment_data["BLURRED"],
                                         thresh_binary_option=self.experiment_data["THRESH_BINARY_OPTION"],
                                         thresh_binary_agent=self.experiment_data["THRESH_BINARY_AGENT"],
                                         gray_scale=self.experiment_data["GRAY_SCALE"])

            return env

        else:
            raise NotImplementedError()

    # def save_state(self, obs):
    #     self.agent.initial_state = obs
    #     return self.env.unwrapped.clone_full_state()
    #
    # def restore_state(self):
    #     self.agent.reset()
    #     self.env.unwrapped.restore_full_state(self.ATARI_state)


if __name__ == '__main__':
    args = docopt(__doc__)

    # set the agent's name
    if args['-a']:
        agent_chosen = args['-a']

    else:
        agent_chosen = "AgentOption"
        print("AgentOption chosen by default")
        time.sleep(1)

    # run the tests if necessary
    if args['--test']:
        from tests.test_Node import *
        from tests.test_Tree import *
        import unittest
        del sys.argv[1:]
        unittest.main()

    else:  # run the proper experiment
        experiment = Experiment("refactored", agent_chosen)
        parallel = False

        if parallel:  # parallel computations with different seeds
            number_cores = 6
            p = Pool()
            # set the seeds for each experiment
            p.map(lambda s: experiment.agent.learn(experiment.env, s), range(number_cores))

            p.close()
            p.join()

        else:
            experiment.agent.learn(experiment.env)
