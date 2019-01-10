from utils import play, get_env_spec
from unityagents import UnityEnvironment
from dqn_agents import Agent
import argparse

parser=argparse.ArgumentParser(description="Play an agent acting in the environment")
parser.add_argument('-p', '--params_path', type=str, metavar='', default='data/dudqn-1/checkpoint.pth',
                                           help="path to the model parameters")
parser.add_argument('-du', '--dueling_dqn', action='store_true', help='whether the model is a dueling dqn')
args=parser.parse_args()

if __name__ == "__main__":
    env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")
    env_spec = get_env_spec(env)
    agent = Agent(state_size=env_spec['state_size'],
                  action_size=env_spec['action_size'],
                  seed=100,
                  dueling_dqn=args.dueling_dqn,
                  double_dqn=False,
                  dqn_way_update=args.dueling_dqn)
    play(env, agent, params_path=args.params_path)