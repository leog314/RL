import gymnasium as gym
import torch
import torch.nn.functional as f
import numpy as np
from torchvision.transforms.functional import rgb_to_grayscale
import matplotlib.pyplot as plt
# import flappy_bird_env  # noqa

class Env:
    def __init__(self, build: str, state_hst_size: int = 4, state_shape: tuple | torch.Tensor | int = (4, 84, 84)):
        self.env = gym.make(build, render_mode = "human", continuous=False)
        print(self.env.action_space)
        self.linear = True if isinstance(state_shape, int) else False
        self.shape = ()
        self.state_hy_size = state_hst_size if not self.linear else None
        self.state_hyst = []
        self.img_shape = state_shape[1:] if not self.linear else None

    def tobs_to_tensor(self, obs) -> torch.Tensor:
        full_inf_state = torch.Tensor(np.array(obs)/127.5-1.).permute((2, 1, 0)).unsqueeze(0)
        return f.interpolate(full_inf_state, self.img_shape).squeeze_(0)

    def oobs_to_tensor(self, obs) -> torch.Tensor:
        return torch.Tensor(np.array(obs)).unsqueeze(0)

    def start_mdp(self):
        init_obs = self.env.reset()

        self.env.render()

        if not self.linear:
            init_state = rgb_to_grayscale(self.tobs_to_tensor(init_obs[0]))
            self.state_hyst = [torch.zeros(tuple(init_state.shape)) for _ in range(self.state_hy_size-1)] + [init_state]
            res = torch.cat(self.state_hyst)

            self.shape = res.shape
            return res.unsqueeze(0)

        res = self.oobs_to_tensor(init_obs[0])
        self.shape = int(res.shape[-1])
        return res

    def step(self, action: int):
        obs, rew, done, _, _ = self.env.step(action)

        if not self.linear:
            state = rgb_to_grayscale(self.tobs_to_tensor(obs))
            self.state_hyst.append(state)
            self.state_hyst.pop(0)
            return torch.cat(self.state_hyst).unsqueeze(0), torch.Tensor([float(rew)]), done

        return self.oobs_to_tensor(obs), torch.Tensor([float(rew)]), done
