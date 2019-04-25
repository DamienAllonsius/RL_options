from agent.option import Option, OptionExplore
from agent.q import QTree
from abc import ABCMeta, abstractmethod
from tqdm import tqdm
from utils import SaveResults, ShowRender
import numpy as np


class AbstractAgent(object):
    """
    Abstract option class that barely only needs update, reset and act functions
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def choose_option(self):
        raise NotImplementedError()

    @abstractmethod
    def update_agent(self, *kwargs):
        raise NotImplementedError()

    @abstractmethod
    def compute_total_reward(self, *kwargs):
        raise NotImplementedError()

    @abstractmethod
    def learn(self, *kwargs):
        raise NotImplementedError()


class AgentOption(AbstractAgent):
    """
    option_list[0] will always be an exploration option.
    """

    def __init__(self,
                 initial_state,  # {"state": int, "blurred_state", int}
                 current_state,  # {"state": int, "blurred_state", int}
                 number_actions,
                 type_exploration,
                 play,
                 experiment_data):

        self.initial_state = initial_state  # The starting point of the agent
        self.current_state = current_state

        self.experiment_data = experiment_data
        self.number_actions = number_actions
        self.play = play

        self.q = QTree(current_state["blurred_state"])
        self.type_exploration = type_exploration

        self.option_list = []
        self.total_reward = 0

        if not play:
            if type_exploration == "OptionExplore":
                self.option_list.append(OptionExplore(number_actions, experiment_data))

            else:
                raise Exception("type_exploration unknown")

    def __len__(self):
        return len(self.option_list) - 1

    def display_tree(self):
        """
        Displays the QTree
        """
        print(self.q)

    def reset(self):
        self.total_reward = 0
        self.current_state = self.initial_state
        self.q.reset()

    def choose_option(self):
        """
        if no option : explore
        else flip a coin, then take the best or explore
        """
        if self.play:
            raise NotImplementedError()
            # best_option_index, terminal_state = self.q.find_best_action()
            # best_option_index += 1
            # best_option = self.option_list[best_option_index]
            # best_option.reset(self.current_state["blurred_state"], self.current_state["state"], terminal_state)
            # best_option.play = True
            # return best_option

        else:
            best_option_index, terminal_state = self.q.find_best_action()
            best_option_index += 1  # because the first option is always the exploring option

            if self.q.get_number_visits() < self.experiment_data["BUDGET_EXPLORATION"] or \
                    terminal_state is None:  # in this case : explore
                self.option_list[0].reset(initial_state=self.current_state["blurred_state"],
                                          current_state=None,
                                          terminal_state=None)

                return 0  # the explore option index

            else:  # in this case, play an option from the list self.option_set
                self.option_list[best_option_index].reset(self.current_state["blurred_state"],
                                                          self.current_state["state"],
                                                          terminal_state)

                return best_option_index

    def update_agent(self, new_state, reward, option, remaining_lives):
        # self.display_tree(new_state["blurred_state"])
        if self.play:
            self.current_state = new_state

        else:
            # update the q value only if the option is not the explore_option
            total_reward = self.compute_total_reward(option, reward, remaining_lives)
            if type(option).__name__ != self.type_exploration:
                self.q.update_q_value(option.terminal_state,
                                      total_reward,
                                      new_state["blurred_state"],
                                      self.experiment_data["LEARNING_RATE"])

            # add the new state to q and add a new option to agent if necessary
            self.q.add_state(new_state["blurred_state"])
            if self.q.number_options > len(self):
                self.option_list.append(Option(self.number_actions, self.play, self.experiment_data))

            # update the current state
            self.current_state = new_state

    def compute_total_reward(self, option, reward, remaining_lives):
        total_reward = reward
        total_reward += self.experiment_data["PENALTY_AGENT_ACTION"]  # each action can give a penalty

        return total_reward

    def learn(self, env, seed=0):
        # set the seeds
        np.random.seed(seed)
        env.seed(seed)

        # prepare the file for the results
        save_results = SaveResults(self.experiment_data)
        save_results.write_setting()
        save_results.set_file_results_name(seed)

        # prepare the renders
        show_render = ShowRender(env)

        for t in tqdm(range(1, self.experiment_data["ITERATION_LEARNING"] + 1)):

            # reset the parameters
            self.reset()
            env.reset()
            option_index = None
            done = False

            # render the first image
            show_render.display()

            while not done:
                if option_index is None:
                    option_index = self.choose_option()

                action = self.option_list[option_index].act()
                obs, reward, done, info = env.step(action)
                end_option = self.option_list[option_index].update_option(reward, obs, action, info['ale.lives'])

                if end_option:
                    self.update_agent(obs, reward, self.option_list[option_index], info['ale.lives'])
                    print("number of options: " + str(len(self.option_list)))
                    option_index = None

                if reward > 0:
                    self.total_reward += reward
                    save_results.write_reward(t, self.total_reward)
                    # self.ATARI_state = self.save_state(obs)
                    break

                show_render.display()
                # done = (info != full_lives)

        # write that the experiment went well
        save_results.write_message("Experiment complete.")


class AgentQ(AbstractAgent):

    def __init__(self):
        pass

    def reset(self):
        pass

    def choose_option(self):
        pass

    def update_agent(self, *kwargs):
        pass

    def compute_total_reward(self, *kwargs):
        pass

    def learn(self, *kwargs):
        pass


class AgentOneOption(AbstractAgent):
    """
    Show how the option learns the first option
    """

    def __init__(self, experiment_data, number_actions, initial_state, current_state):
        self.initial_state = initial_state  # {"state": int, "blurred_state", int}
        self.current_state = current_state # {"state": int, "blurred_state", int}
        self.terminal_state = None
        self.number_actions = number_actions
        self.experiment_data = experiment_data

        self.experiment_data["PROBABILITY_EXPLORE_IN_OPTION"] = 0
        self.option_list = [OptionExplore(number_actions, experiment_data)]

    def reset(self):
        self.current_state = self.initial_state

    def choose_option(self):
        len_option_list = len(self.option_list)
        if len_option_list == 1:  # in this case, only 1 option explore

            self.option_list[-1].reset(initial_state=self.current_state["blurred_state"],
                                       current_state=None,
                                       terminal_state=None)

        elif len_option_list == 2:  # in this case, 1 option explore and 1 normal option
            self.option_list[-1].reset(self.current_state["blurred_state"],
                                       self.current_state["state"],
                                       self.terminal_state)

        else:
            raise Exception("too many options : " + str(len_option_list))

        return -1

    def update_agent(self, obs):
        self.current_state = obs
        self.terminal_state = obs["blurred_state"]
        if len(self.option_list) == 1:
            self.option_list.append(Option(self.number_actions, False, self.experiment_data))

    def compute_total_reward(self, *kwargs):
        pass

    def learn(self, env, seed=0):
        # set the seeds
        np.random.seed(seed)
        env.seed(seed)

        # prepare the renders
        show_render = ShowRender(env)

        for t in tqdm(range(1, self.experiment_data["ITERATION_LEARNING"] + 1)):

            # reset the parameters
            env.reset()
            option_index = None
            done = False

            # render the first image
            show_render.display()

            print(t)

            while not done:

                if option_index is None:
                    option_index = self.choose_option()

                action = self.option_list[option_index].act()
                try:
                    print(self.option_list[option_index].q)

                except:
                    pass

                obs, reward, done, info = env.step(action)
                end_option = self.option_list[option_index].update_option(reward, obs, action, info['ale.lives'])

                if end_option:
                    self.update_agent(obs)
                    print("number of options: " + str(len(self.option_list)))
                    option_index = None
                    done = True

                show_render.display()
