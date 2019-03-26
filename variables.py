# GLOBAL VARIABLES
red = '\033[91m' 
green = '\033[92m'
yellow = '\033[93m'
white = '\033[0m'

def return_data(name):
    data = None
    if name == "reload_ATARI":
        data = {"NAME" : "reload_ATARI",

                "ENV_NAME" : 'MontezumaRevenge-v0',
                "AGENT" : "AgentOption",
                
                "ITERATION_LEARNING" : 30000,
                "LEARNING_RATE" : 0.1,
                
                "PROBABILITY_EXPLORE_FOR_AGENTOPTION" : 0.0, # useless with OptionExploreQ
                "PROBABILITY_EXPLORE_IN_OPTION" : 0.1,

                # Zones setting
                "NUMBER_ZONES_MONTEZUMA_X" : (2**5)*5,
                "NUMBER_ZONES_MONTEZUMA_Y" : 2*3*5*7,
                
                "NUMBER_ZONES_OPTION_X" : (2**3)*5,
                "NUMBER_ZONES_OPTION_Y" : 3*7,
                "THRESH_BINARY_OPTION" : 0,
     
                "NUMBER_ZONES_AGENT_X" : 2**3,
                "NUMBER_ZONES_AGENT_Y" : 7,
                "THRESH_BINARY_AGENT" : 40,
     
                "BLURRED" : True,
                "GRAY_SCALE" : True,
                
                "REWARD_END_OPTION" : 100,
                "PENALTY_END_OPTION" : - 100,
                "PENALTY_OPTION_ACTION" : -1,
     
                "PENALTY_LOST_LIFE" : - 1000,
                "PENALTY_AGENT_ACTION" : 0, # should stay 0 for the moment
        }
        
        data.update({"ZONE_SIZE_OPTION_X" : data["NUMBER_ZONES_MONTEZUMA_X"] // data["NUMBER_ZONES_OPTION_X"],
                     "ZONE_SIZE_OPTION_Y" : data["NUMBER_ZONES_MONTEZUMA_Y"] // data["NUMBER_ZONES_OPTION_Y"],
                     "ZONE_SIZE_AGENT_X" : data["NUMBER_ZONES_MONTEZUMA_X"] // data["NUMBER_ZONES_AGENT_X"], 
                     "ZONE_SIZE_AGENT_Y" : data["NUMBER_ZONES_MONTEZUMA_Y"] // data["NUMBER_ZONES_AGENT_Y"],
        })

    if name == "reload_ATARI_more_zones_for_agent":
        data = {"NAME" : "reload_ATARI_more_zones_for_agent",

                "ENV_NAME" : 'MontezumaRevenge-v0',
                "AGENT" : "AgentOption",
                
                "ITERATION_LEARNING" : 30000,
                "LEARNING_RATE" : 0.1,
                
                "PROBABILITY_EXPLORE_FOR_AGENTOPTION" : 0.0, # useless with OptionExploreQ
                "PROBABILITY_EXPLORE_IN_OPTION" : 0.1,

                # Zones setting
                "NUMBER_ZONES_MONTEZUMA_X" : (2**5)*5,
                "NUMBER_ZONES_MONTEZUMA_Y" : 2*3*5*7,
                
                "NUMBER_ZONES_OPTION_X" : (2**3)*5,
                "NUMBER_ZONES_OPTION_Y" : 3*7,
                "THRESH_BINARY_OPTION" : 0,
     
                "NUMBER_ZONES_AGENT_X" : 2**4,
                "NUMBER_ZONES_AGENT_Y" : 2*7,
                "THRESH_BINARY_AGENT" : 40,
     
                "BLURRED" : True,
                "GRAY_SCALE" : True,
                
                "REWARD_END_OPTION" : 100,
                "PENALTY_END_OPTION" : - 100,
                "PENALTY_OPTION_ACTION" : -1,
     
                "PENALTY_LOST_LIFE" : - 1000,
                "PENALTY_AGENT_ACTION" : 0, # should stay 0 for the moment
        }
        
        data.update({"ZONE_SIZE_OPTION_X" : data["NUMBER_ZONES_MONTEZUMA_X"] // data["NUMBER_ZONES_OPTION_X"],
                     "ZONE_SIZE_OPTION_Y" : data["NUMBER_ZONES_MONTEZUMA_Y"] // data["NUMBER_ZONES_OPTION_Y"],
                     "ZONE_SIZE_AGENT_X" : data["NUMBER_ZONES_MONTEZUMA_X"] // data["NUMBER_ZONES_AGENT_X"], 
                     "ZONE_SIZE_AGENT_Y" : data["NUMBER_ZONES_MONTEZUMA_Y"] // data["NUMBER_ZONES_AGENT_Y"],
        })
    return data

                


