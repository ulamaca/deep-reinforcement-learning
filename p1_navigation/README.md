[//]: # (Image References)

[image1]:results/animation/dddqn-demo.gif "Trained Agent"

# Project 1: Navigation

### Introduction

In this project, Deep Q-Network (DQN) and its variants (Dueling DQN, Double DQN) are implemented and trained to navigate (and collect bananas!) in a large, square world. 

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Prerequisites
1. Please first setup a Python3 [Anaconda](https://www.anaconda.com/download) environment. 
2. Then install the requirements for the project through:
```
pip install -r requirement.txt
```

3. Follow the instructions to download the environment from the [Getting Started](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation) section in Udacity DRLND repo.

### Instructions
1. To train a DQN agent from scratch, execute in the command line:
```
python run.py  
```
Setting customized options is possible by specified ```-arg_name value``` right after run.py. For example,
```
python run.py -name exp-name -dd -du -seed 453
```
After trained, two files will be saved in ./data/exp-name: progress.txt and checkpoint.pth. progress.txt saves the training score traces and checkpoint.pth is the model parameters of the trained agent.
More detailed instructions can be found using:
```
python run.py -h
```
2. To get statistics plots after training, execute:
```
python plot.py -l model1,model2...
```
*Note that if you would like to test a model with mutliple training runs, please name your experiment follwoing the convention: model-name-i (i is the run no.) so that the program can compute the average over all the trials. 
3. To see how your favorite agent plays, use
```
python play.py -p path/to/model-params 
```
if the model is a dueling DQN, please add another argument -du because it will need to setup different model architecture
