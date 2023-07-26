import torch
import argparse
import numpy as np

from agent import PPOAgent
from collections import deque
from controller import Controller
from replaymemory import ReplayMemory

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--episode', type=int, default=50000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--memory_size', type=int, default=10000)
args = parser.parse_args()

if __name__ == '__main__':

    agent = PPOAgent()
    buffer = ReplayMemory(args.memory_size)
    dq = deque(maxlen=10000)
    T = torch.tensor

    steps = 0
    
    for i in range(args.episode):
        action, pis, _ = agent.get_action()
        reward = agent.get_reward(action)

        sample = [pis.detach(), T(reward).unsqueeze(0)]
        buffer.push(sample)

        if len(buffer) >= args.batch_size:
            sample = buffer.sample(len(buffer))
            sample = list(zip(*sample))
            sample = list(map(torch.cat, sample))

            agent.update(*sample)
            print(f'{i} episode update done: score {np.mean(dq)}')
        
        dq.append(reward)
        steps += 1

    print(agent.con.result())
            
