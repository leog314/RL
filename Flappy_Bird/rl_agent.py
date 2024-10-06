import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
import flappy_bird_env  # noqa

class Memory:
    def __init__(self, mem_size: int, batch_size: int = 64):
        self.mem = []
        self.mem_size = mem_size
        self.batch_size = batch_size

    def push(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, nstate: torch.Tensor, done: torch.Tensor):
        self.mem.append((state, action, reward, nstate, done))
        if len(self.mem) > self.mem_size:
            self.mem.pop(0)

    def random_sample(self):
        if len(self.mem) < self.batch_size:
            return False
        return random.sample(self.mem, self.batch_size)
class Agent(nn.Module):
    def __init__(self, mem_size: int = 8192, batch_size: int = 16, gamma: float = 0.999, eps_start: float = 0.95, eps_end: float = 0.05, steps: int = 3000):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 10, 5),
            nn.MaxPool2d(7),
            nn.SiLU(),
            nn.Conv2d(10, 1, 5),
            nn.MaxPool2d(7),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(165, 40),
            nn.SiLU(),
            nn.Linear(40, 2)
        )
        self.mem = Memory(mem_size, batch_size)

        self.decay: float = 1.
        self.gamma: float = gamma
        self.batch_size = batch_size

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.steps = steps
        self.step = 0

        self.loss = nn.SmoothL1Loss()
        self.optim = optim.SGD(self.parameters(), lr=0.1)

    def forward(self, x_ob: torch.Tensor):
        return self.conv_net(x_ob)

    def select_action(self, image: torch.Tensor) -> int:
        self.update_decay()
        t = np.random.random()

        if t < self.decay:
            return np.random.randint(0, 2)
        self.eval()
        return int(torch.argmax(self.conv_net(image)))

    def update_decay(self):
        self.decay = self.eps_end + (self.eps_start-self.eps_end)*np.exp(-self.step/self.steps)
        self.step += 1

    def training_(self) -> float:
        self.train()
        self.optim.zero_grad(True)

        sample = self.mem.random_sample()

        if isinstance(sample, bool):
            return 0.

        state_ten, action_ten, reward_ten, nstate_ten, done_ten = zip(*sample)

        state_ten = torch.stack(state_ten)
        action_ten = torch.stack(action_ten)
        rew_ten = torch.stack(reward_ten)
        nstate_ten = torch.stack(nstate_ten)

        out = self(state_ten)
        target = out.clone()

        for i in range(self.batch_size):
            target[i, int(action_ten[i])] = rew_ten[i] + self.gamma*torch.max(self(nstate_ten[i]), dim=1).values.unsqueeze(0)*(1-done_ten[i])

        loss = self.loss(out, target)
        loss.backward()

        self.optim.step()

        return loss.item()
