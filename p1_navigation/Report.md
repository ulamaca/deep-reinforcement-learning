### Algorithms
    In this project, I implement three deep Q-network based agents and demonstrate their abilty to sucessfully learn how to navigate and collect bananas in a 3D environment from scratch [ref]. The most important feature of DQN is that it extends the classical tabular Q-learning agent to work in high-dimension state space using neural network approximation and advanced deep learning models such as CNN, RNN can thus easily come into play. Technically speaking, there are two main tricks to make DQN trainable: fitted Q-target and experience replay. 
    When using neural network approximation, Q-learning will be translated from direct value update into parameter tuning. The fitting here is to minimize the difference between bootstrap value and the current value such that the Q-network satisfies Bellman equation. It is essentially a supervised learning problem but both targets and predictions are being tuned. Such objectives might not be learnable in that it is totally possible to end up at bad soutions, for example, setting both networks to be exactly zero. To solve this problem, fitted Q-target was proposed to use a fixed target network (and update it periodiclly) to evaluate the bootstraped values, serving as the learning targets. Such approach works well in practice despite that it introduces another source of bias, . 
    Experience replay is another trick that mitigates data-hunger and correlated data problems in RL. A replay buffer is constructed to store the most recent N experiences, which will then be sampled ocassionally to train the Q-network. Off-policy characteristic of Q-learning algorithm enables it to learn from old experiences in the buffer.
    I also include two variants of DQN. First, Double DQN uses the actions of the local Q-network to determine the boostrap value in the target Q-network and is shown to reduce overestimation problem of Q values in DQN. Second, Dueling DQN represents state-value V(s) and the advantage A(s,a) separately in the network architecutre. Accurate estimation for the two would result in better overall performance.
     
### Learning 
    - Deep-Q Network (DQN)
    - Dueling DQN 
    - Double DQN
    - hyperparams setup (table, todo: to add)
        - Optimizer
        - Soft-Update
        - Agent Model Type
        - Agent Model Arch
        - Update Frequency every 4 steps
        - 


### Results  
    - statistics
**Result-1: Statistics
** Figure 1: Average training progress over the five runs.
[average scores](results/dqns2020_avg.png)

[best scores](results/dqns2020_best.png)
- The task can be solved in around 400 episodes, which is about 500 seconds in wall time.   
- are quite stable from the avg plot
- no dicernable difference for DQN variants in this case

- A trained agent is available in video  ![trained agent](./results/animation/dddqn-demo.mp4)

### Future Work
I list some possible future directions to enhance the current work.
    - Adopt and test two possible additional modules: auxiliary tasks (e.g. reward prediction, state prediction) and recurrent networks. 
    - Implement RAINBOW agent
    - Work on hyperparameter optimization to get possible better results. 
    - Work on the challenge task (learning from pixel) with convolutional agents
    - Benchmark the task together with other RL approaches (policy-based, ES, AC-based, DDPG etc.)

### Reference
    - [Mnih 2015]
    - [Dueling DQN]
    - [Double DQN]
    
### Appendix
** Key equations and the corresponding lines of codes in the project 