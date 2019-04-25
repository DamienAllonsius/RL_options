# GLOBAL VARIABLES
red = '\033[91m' 
green = '\033[92m'
yellow = '\033[93m'
white = '\033[0m'
tab = '   '

enter = 65293
space = 32


def return_data(name):
    if name == "refactored":
        data = {"ENV_NAME": 'MontezumaRevenge-v0',

                "ITERATION_LEARNING": 10000,
                "LEARNING_RATE": 0.1,

                "PROBABILITY_EXPLORE_FOR_AGENTOPTION": 0.0,  # useless with OptionExploreQ
                "PROBABILITY_EXPLORE_IN_OPTION": 0.1,
                "BUDGET_EXPLORATION": 20,

                # Zones setting
                "NUMBER_ZONES_MONTEZUMA_X": (2 ** 5) * 5,
                "NUMBER_ZONES_MONTEZUMA_Y": 2 * 3 * 5 * 7,

                "NUMBER_ZONES_OPTION_X": (2 ** 3) * 5,
                "NUMBER_ZONES_OPTION_Y": 3 * 7,
                "THRESH_BINARY_OPTION": 0,

                "NUMBER_ZONES_AGENT_X": 2 ** 3,
                "NUMBER_ZONES_AGENT_Y": 7,
                "THRESH_BINARY_AGENT": 40,

                "BLURRED": True,
                "GRAY_SCALE": True,

                "REWARD_END_OPTION": 100,
                "PENALTY_END_OPTION": - 100,
                "PENALTY_OPTION_ACTION": -1,

                "PENALTY_LOST_LIFE_FOR_OPTIONS": - 1000,
                "PENALTY_LOST_LIFE_FOR_AGENT": 0,
                "PENALTY_AGENT_ACTION": 0,  # should stay 0 for the moment

                "SAVE_STATE": False}

        data.update({"ZONE_SIZE_OPTION_X": data["NUMBER_ZONES_MONTEZUMA_X"] // data["NUMBER_ZONES_OPTION_X"],
                     "ZONE_SIZE_OPTION_Y": data["NUMBER_ZONES_MONTEZUMA_Y"] // data["NUMBER_ZONES_OPTION_Y"],
                     "ZONE_SIZE_AGENT_X": data["NUMBER_ZONES_MONTEZUMA_X"] // data["NUMBER_ZONES_AGENT_X"],
                     "ZONE_SIZE_AGENT_Y": data["NUMBER_ZONES_MONTEZUMA_Y"] // data["NUMBER_ZONES_AGENT_Y"],
                     "NAME": name})

    elif name == "First_good_results":
        data = {"ENV_NAME": 'MontezumaRevenge-v0',
                "AGENT": "AgentOption",

                "ITERATION_LEARNING": 20000,
                "LEARNING_RATE": 0.1,

                "PROBABILITY_EXPLORE_FOR_AGENTOPTION": 0.0,  # useless with OptionExploreQ
                "PROBABILITY_EXPLORE_IN_OPTION": 0.1,

                # Zones setting
                "NUMBER_ZONES_MONTEZUMA_X": (2**5)*5,
                "NUMBER_ZONES_MONTEZUMA_Y": 2*3*5*7,

                "NUMBER_ZONES_OPTION_X": (2**3)*5,
                "NUMBER_ZONES_OPTION_Y": 3*5*7,
                "THRESH_BINARY_OPTION": 0,

                "NUMBER_ZONES_AGENT_X": 2**3,
                "NUMBER_ZONES_AGENT_Y": 7,
                "THRESH_BINARY_AGENT": 40,

                "BLURRED": True,
                "GRAY_SCALE": True,

                "REWARD_END_OPTION": 100,
                "PENALTY_END_OPTION": - 100,
                "PENALTY_OPTION_ACTION": 0,

                "PENALTY_LOST_LIFE_FOR_OPTIONS": - 1000,
                "PENALTY_LOST_LIFE_FOR_AGENT": - 1000,
                "PENALTY_AGENT_ACTION": 0,  # should stay 0 for the moment

                "SAVE_STATE": False}

        data.update({"ZONE_SIZE_OPTION_X": data["NUMBER_ZONES_MONTEZUMA_X"] // data["NUMBER_ZONES_OPTION_X"],
                     "ZONE_SIZE_OPTION_Y": data["NUMBER_ZONES_MONTEZUMA_Y"] // data["NUMBER_ZONES_OPTION_Y"],
                     "ZONE_SIZE_AGENT_X": data["NUMBER_ZONES_MONTEZUMA_X"] // data["NUMBER_ZONES_AGENT_X"],
                     "ZONE_SIZE_AGENT_Y": data["NUMBER_ZONES_MONTEZUMA_Y"] // data["NUMBER_ZONES_AGENT_Y"],
                     "NAME": name})

    elif name == "reload_ATARI_more_zones_for_agent":
        data = {"ENV_NAME": 'MontezumaRevenge-v0',
                "AGENT": "AgentOption",

                "ITERATION_LEARNING": 30000,
                "LEARNING_RATE": 0.1,

                "PROBABILITY_EXPLORE_FOR_AGENTOPTION": 0.0,  # useless with OptionExploreQ
                "PROBABILITY_EXPLORE_IN_OPTION": 0.1,

                # Zones setting
                "NUMBER_ZONES_MONTEZUMA_X": (2**5)*5,
                "NUMBER_ZONES_MONTEZUMA_Y": 2*3*5*7,

                "NUMBER_ZONES_OPTION_X": (2**3)*5,
                "NUMBER_ZONES_OPTION_Y": 3*7,
                "THRESH_BINARY_OPTION": 0,

                "NUMBER_ZONES_AGENT_X": 2**4,
                "NUMBER_ZONES_AGENT_Y": 2*7,
                "THRESH_BINARY_AGENT": 40,

                "BLURRED": True,
                "GRAY_SCALE": True,

                "REWARD_END_OPTION": 100,
                "PENALTY_END_OPTION": - 100,
                "PENALTY_OPTION_ACTION": -1,

                "PENALTY_LOST_LIFE_FOR_OPTIONS": - 1000,
                "PENALTY_AGENT_ACTION": 0,  # should stay 0 for the moment
                "PENALTY_LOST_LIFE_FOR_AGENT": - 10}

        data.update({"ZONE_SIZE_OPTION_X": data["NUMBER_ZONES_MONTEZUMA_X"] // data["NUMBER_ZONES_OPTION_X"],
                     "ZONE_SIZE_OPTION_Y": data["NUMBER_ZONES_MONTEZUMA_Y"] // data["NUMBER_ZONES_OPTION_Y"],
                     "ZONE_SIZE_AGENT_X": data["NUMBER_ZONES_MONTEZUMA_X"] // data["NUMBER_ZONES_AGENT_X"],
                     "ZONE_SIZE_AGENT_Y": data["NUMBER_ZONES_MONTEZUMA_Y"] // data["NUMBER_ZONES_AGENT_Y"],
                     "NAME": name})

    else:
        raise Exception("data name does not exist")

    return data
