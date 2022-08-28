import gym
import torch
import numpy as np
from copy import deepcopy
from torchvision.transforms import Resize


def to_torch_img(np_img: np.ndarray, size: tuple=(64, 64)):
    """
    Transforms an np img into a torch img, with the right size
    
    :param np_img: numpy array of shape (H, W, 3)
    :param size: output size of the image, in the format (h, w)

    :return torch_img: torch tensor of shape (3, h, w)
    """
    x = torch.from_numpy(np_img.copy())
    y = x.permute(2, 0, 1)
    torch_img = Resize(size)(y)
    return torch_img


def process_img(img: torch.Tensor, depth: int=5):
    """
    Returns a centered/normalized version of img, with reduced bit depth

    :param img: img to modify between [0, 255]
    :param depth: integer, amount of bits to reduce the img to

    :return: the modified img between [-1, 1], with reduced bit depth
    """
    return img.div(2 ** (8 - depth)).floor().div(2 ** depth).sub(.5).mul(2)


class TorchImageEnvWrapper:
    def __init__(self, env: str, bit_depth: int=5, observation_size: tuple=None) -> None:
        self.env = gym.make(env, render_mode='rgb_array')
        
        self.bit_depth = bit_depth
        self.observation_size = observation_size if observation_size is not None else (64, 64)
    
    def reset(self, return_obs=False):
        self.env.reset()
        obs = self.env.render()[0]
        tmp = to_torch_img(obs)
        if return_obs:
            return process_img(tmp, self.bit_depth), obs
        else:
            return process_img(tmp, self.bit_depth)
    
    def step(self, action: torch.Tensor, return_obs: bool=False):
        action = action.cpu().detach().numpy()
        _, reward, done, info = self.env.step(action)
        img = self.env.render()[0]
        obs = to_torch_img(img)
        obs = process_img(obs, self.bit_depth)
        if return_obs:
            return obs, reward, done, info, img
        else:
            return obs, reward, done, info
    
    def render(self):
        obs = self.env.render()[0]
        return obs

    def sample(self):
        return torch.tensor(self.env.action_space.sample())
    
    def close(self):
        self.env.close()
    
    @property
    def observation_space(self):
        return (3, *self.observation_size)
    
    @property
    def action_space(self):
        return self.env.action_space.shape[0]
    
    @property
    def action_bounds(self):
        return (self.env.action_space.low[0], self.env.action_space.high[0])
