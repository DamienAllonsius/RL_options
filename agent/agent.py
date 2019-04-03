from agent.option import Option, OptionExplore
from agent.q import QTree


class AgentOption(object): 

    def __init__(self, initial_state, current_state, number_actions, type_exploration, play, experiment_data):
        self.initial_state = initial_state
        self.current_state = current_state

        self.experiment_data = experiment_data
        self.number_actions = number_actions
        self.play = play

        self.q = QTree(current_state["blurred_state"], experiment_data["BUDGET_EXPLORATION"])
        self.type_exploration = type_exploration

        if not play:
            if type_exploration == "OptionExplore":
                self.explore_option = OptionExplore(current_state["blurred_state"], self.number_actions)

            else:
                raise Exception("type_exploration unknown")

    def display_tree(self, next_node_data):
        """
        for QTree
        """
        print(self.q.str_tree(next_node_data))

    def reset_explore_option(self):
        self.explore_option.reward_for_agent = 0
        self.explore_option.initial_state = self.current_state["blurred_state"]
        
    def reset(self):
        """
        Same as __init__ but the q function is preserved
        """
        self.q.reset(self.initial_state["blurred_state"])
        self.current_state = self.initial_state
        self.reset_explore_option()

    def choose_option(self, t):
        """
        if no option : explore
        else flip a coin, then take the best or explore
        """
        if self.play: 
            _, best_option = self.q.find_best_action(self.current_state["blurred_state"])
            best_option.play = True
            return best_option

        else:
            if not self.q.is_actions(self.current_state["blurred_state"]):
                self.reset_explore_option()
                return self.explore_option

            else:

                # self.current_node.number_visits < budget

                best_reward, best_option = self.q.find_best_action(self.current_state["blurred_state"])
                if best_reward == 0:
                    next_terminal_state = self.q.get_tree_advices()
                    for opt in self.q.get_actions(self.current_state["blurred_state"]):
                        if opt.terminal_state == next_terminal_state:
                            best_option = opt
                    
                best_option.reward_for_agent = 0
                best_option.lost_life = False
                best_option.set_current_state(self.current_state["state"])
                
                return best_option
            
    def update_agent(self, new_state, option, action):
        """
        In this order
        _update last action done
        _update reward
        _add an option if a new state has just been discovered
        _update the q function value
        _update the state
        """
        # self.display_Qtree(new_state["blurred_state"])
        if self.play:
            self.current_state = new_state
            
        else:
            total_reward = self.experiment_data["PENALTY_AGENT_ACTION"] + \
                           option.reward_for_agent + \
                           option.lost_life * self.experiment_data["PENALTY_LOST_LIFE_FOR_AGENT"]

            self.update_q_function_options(new_state, option, total_reward)
            self.current_state = new_state
            
    def update_q_function_options(self, new_state, option, reward):
        if self.q.no_return_update(self.current_state["blurred_state"], new_state["blurred_state"]):

            action = Option(self.number_actions,
                            self.current_state["blurred_state"],
                            new_state["state"],
                            new_state["blurred_state"],
                            self.play,
                            self.experiment_data)
            
            self.q.update_q_action_state(self.current_state["blurred_state"], new_state["blurred_state"], action)
            # print("number of options: " + str(len(self.q)))
            if option != self.explore_option:
                self.q.update_q_value(self.current_state["blurred_state"],
                                      option,
                                      reward,
                                      new_state["blurred_state"],
                                      self.experiment_data["LEARNING_RATE"])

        else:
            self.q.update_current_node(new_state["blurred_state"])
