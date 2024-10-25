import gymnasium as gym
import torch
import torch.nn.functional as f
import numpy as np
from torchvision.transforms.functional import rgb_to_grayscale
import matplotlib.pyplot as plt
# import flappy_bird_env  # noqa

class Env:
    def __init__(self, build: str, state_shape: int):
        self.env = gym.make(build, render_mode = "human")
        self.shape = 0

    @staticmethod
    def obs_to_tensor(obs) -> torch.Tensor:
        return torch.Tensor(np.array(obs)).unsqueeze(0)

    def start_mdp(self):
        init_obs = self.env.reset()
        self.env.render()

        res = self.obs_to_tensor(init_obs[0])
        self.shape = res.shape

        return res

    def step(self, action: int):
        obs, rew, done, _, _ = self.env.step(action)

        if not self.linear:
            state = rgb_to_grayscale(self.tobs_to_tensor(obs))
            self.state_hyst.append(state)
            self.state_hyst.pop(0)
            return torch.cat(self.state_hyst).unsqueeze(0), torch.Tensor([float(rew)]), done

        return self.oobs_to_tensor(obs), torch.Tensor([float(rew)]), done

