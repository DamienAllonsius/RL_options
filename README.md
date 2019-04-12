# Reinforcement Learning with options 

- We train an agent to solve Montezuma's Revenge using the option framework of 
Sutton-Precup-Singh (see the original article [here](http://www-anw.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf)).  
The code is written with `Python 3.6` and uses the ATARI environment from [gym](https://github.com/openai/gym). 

- To run the script, first install the libraries of `requirements.txt` and execute `python3 main.py`.

- To run the experiment on a gridworld environment, clone this repo and, in RL_options folder, clone the repo [gridenvs](https://github.com/aig-upf/gridenvs) 
(this gridworld environment is developed by AI-ML team of [Universitat Pompeu Fabra](https://www.upf.edu/web/ai-ml) (Barcelona)).
You can change the shape of the gridworld in gridenvs/example/.

## Learning phase
![result](/animations/learning_phase.gif)

You can change the render of the game by selecting the window and typing: `b` (Blurred) to switch between a downsampled image and the original image, `g` (Grayscale) to activate the grayscaling, `a` (Agent) to switch between the option and the agent view, `d` (Display) to activate/deactivate the display (of course, the display activation slows down the performances).


[![made-with-OpenAIGym](https://img.shields.io/badge/Made%20with-OpenAI%20Gym-1f425f.svg)](https://gym.openai.com/)
[![made-with-Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-OpenCV](https://img.shields.io/badge/Made%20with-OpenCV-1f425f.svg)](https://opencv.org/)
