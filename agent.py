import torch
import torch.nn as nn
import numpy as np

from controller import Controller
from typing import Tuple
from typing import Callable

class PPOAgent:
    def __init__(self):

        self.con = Controller()
        self.opt = torch.optim.Adam(self.con.parameters(), lr=1e-5)
        
    def get_action(self) -> \
        Tuple[Callable[[float], float], torch.tensor, torch.tensor]:
        """
        action: generated function
        """
        indice, pis, value = self.con.search()
        action = self.con.generate(indice)
        return action, pis, value
    
    def get_reward(self, action: Callable[[float], float]) -> float:
        """
        Reward: mse between target func and action func
        """
        target_func = lambda x: (x-1) * (x+1)**2 

        points = np.linspace(-1.0, 1.0, 1000)
        pred = action(points)
        target = target_func(points)
        reward = np.mean((pred - target)**2) / 10
        reward = np.array([reward])

        return -reward
        
    def update(self, pi_o, r):
        """
        PPO style update
        """
        eps_clip = 0.2
        
        _, pi, _ = self.con.search() 
        ratio = pi / pi_o
        surr1 = ratio * r
        surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * r

        actor_loss = -torch.min(surr1, surr2)
        actor_loss = actor_loss.mean()

        self.opt.zero_grad()
        actor_loss.backward()
        self.opt.step()
    

