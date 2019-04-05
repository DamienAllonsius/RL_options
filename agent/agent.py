from agent.option import Option, OptionExplore
from agent.q import QTree


class AgentOption(object):

    def __init__(self,
                 initial_state,  # blurred observation
                 current_state,  # untransformed observation (High resolution image, for option)
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

        if not play:
            if type_exploration == "OptionExplore":
                self.explore_option = \
                    OptionExplore(self.number_actions)

            else:
                raise Exception("type_exploration unknown")

    def __len__(self):
        return len(self.option_list)

    def display_tree(self):
        """
        Displays the QTree
        """
        print(self.q)

    def reset(self):
        self.current_state = self.initial_state

    def choose_option(self):
        """
        if no option : explore
        else flip a coin, then take the best or explore
        """
        if self.play:
            best_option_index, terminal_state = self.q.find_best_action(self.current_state["blurred_state"])
            best_option = self.option_list[best_option_index]
            best_option.reset(self.current_state["blurred_state"], self.current_state["state"], terminal_state)
            best_option.play = True
            return best_option

        else:
            if self.q.get_number_visits(self.current_state["blurred_state"]) < \
                    self.experiment_data["BUDGET_EXPLORATION"]:  # in this case : explore

                self.explore_option.reset(initial_state=self.current_state["blurred_state"],
                                          current_state=None,
                                          terminal_state=None)

                return self.explore_option

            else:  # in this case, play an option from the list self.option_set.
                best_option_index, terminal_state = self.q.find_best_action(self.current_state["blurred_state"])
                best_option = self.option_list[best_option_index]
                best_option.reset(self.current_state["blurred_state"], self.current_state["state"], terminal_state)

                return best_option

    def update_agent(self, new_state, option, remaining_lives):
        # self.display_tree(new_state["blurred_state"])
        if self.play:
            self.current_state = new_state

        else:
            total_reward = self.compute_total_reward(option, remaining_lives)
            self.update_q_function(new_state,
                                   option.terminal_state,
                                   total_reward,
                                   type(option).__name__ != self.type_exploration)

            if self.q.get_number_actions(self.current_state["blurred_state"]) > len(self):
                if self.q.get_number_actions(self.current_state["blurred_state"]) == len(self) + 1:

                    self.option_list.append(Option(self.number_actions, self.play, self.experiment_data))

                else:
                    raise Exception("strange number of actions !")

            self.current_state = new_state

    def compute_total_reward(self, option, remaining_lives):
        total_reward = 0
        total_reward += self.experiment_data["PENALTY_AGENT_ACTION"]  # each action can give a penalty
        total_reward += option.reward_for_agent  # the option may have found a reward
        total_reward += (option.lives > remaining_lives) * self.experiment_data["PENALTY_LOST_LIFE_FOR_AGENT"]  # agent
        # can get a penalty if the agent dies (Not recommended)

        return total_reward

    def update_q_function(self, new_state, action, total_reward, bool_update_value):
        self.q.add_state(self.current_state["blurred_state"], new_state["blurred_state"])

        if bool_update_value:
            self.q.update_q_value(self.current_state["blurred_state"],
                                  action,
                                  total_reward,
                                  new_state["blurred_state"],
                                  self.experiment_data["LEARNING_RATE"])

        print("number of options: " + str(len(self)))
