from unityagents import UnityEnvironment
from utils import dqn
from dqn_agent import Agent
import os
import argparse

EXPS_ROOT_PATH = './data'

parser=argparse.ArgumentParser(description="train a RL agent in Unity Banana Navigation Environment")
parser.add_argument('-n', '--name', type=str, metavar='', default='no-name-exp', help="name of the training run")
parser.add_argument('-s', '--save_trace', type=bool, metavar='', default=True, help='whether to save the training trace')
parser.add_argument('-M', '--max_score', type=float, metavar='', default=13.0, help="the pass score a trained agent should achieve")
parser.add_argument('-ra', '--seed', type=int, metavar='', default=0, help='random seed of the agent')
parser.add_argument('-dd', '--double_dqn', action='store_true', help='whether to use double dqn training')
parser.add_argument('-du', '--dueling_dqn', action='store_true', help='whether to use dueling dqn arch')
# todo, more args:
    # ENV_SEED ?, todo: discuss in forum
    # hyperparams
    # soft or hard update
args=parser.parse_args()

if __name__ == "__main__":
    # define and check environment information:
    env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")
    brain_name = env.brain_names[0] # todo, check what this 0 means for
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)
    print('Number of agents:', len(env_info.agents))
    print('Number of actions:', action_size)
    print('States have length:', state_size)

    # define our agent
    agent = Agent(state_size=state_size, action_size=action_size,
                  seed=args.seed,
                  double_dqn=args.double_dqn,
                  dueling_dqn=args.dueling_dqn,
                  dqn_way_update=False)

    # create exp_dir for saving
    exp_dir = os.path.join(EXPS_ROOT_PATH, args.name)
    os.makedirs(exp_dir, exist_ok=True)
    scores = dqn(agent, env,
                 max_score=args.max_score,
                 eps_end=0.001, # todo, check
                 eps_decay=0.97,
                 save_dir=exp_dir)

    # save training trace
    if args.save_trace:
        with open(os.path.join(exp_dir, 'progress.txt'), 'w') as myfile:
            myfile.write(str(scores))
        myfile.close()