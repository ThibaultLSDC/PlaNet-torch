from env_wrapper import TorchImageEnvWrapper
from models.rssm import RSSM
from policies.cem2 import CEM
from memory.buffer import Buffer, Episode
from utils import bottle, Tracker

import torch
import torch.distributions as tfd
import numpy as np

import torch.nn.functional as F

from tqdm import tqdm

from PIL import Image

import wandb

from adan_pytorch import Adan


class Trainer:
    def __init__(self,
        env,
        buffer_size: int,
        seed_episodes: int,
        training_iterations: int,
        batch_size: int,
        chunk_length: int,
        repeat_action: int,
        exploration_time: int,
        kl_weight: float,
        free_nats: float,
        device: str,
        cem_conf: dict,
        optim: str='Adam',
        output_path: str='data',
        track_wandb: bool=False,
        ) -> None:

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.env = TorchImageEnvWrapper(env)
        self.model = RSSM(self.env.action_space).to(self.device)
        self.policy = CEM(
            self.model,
            **cem_conf,
            action_bounds=self.env.action_bounds,
            device=self.device
        )
        self.buffer = Buffer(buffer_size)

        # optimization
        self.seed_episodes = seed_episodes
        self.training_iterations = training_iterations
        self.batch_size = batch_size
        self.chunk_length = chunk_length
        self.kl_weight = kl_weight
        self.free_nats = torch.Tensor([free_nats]).float().to(self.device)

        # exploration
        self.repeat_action = repeat_action
        self.exploration_time = exploration_time

        self.trained_steps = 0

        if optim == 'Adam':
            self.optim = torch.optim.Adam(self.model.parameters(), 1e-3, eps=1e-4)
        elif optim == 'Adan':
            self.optim = Adan(self.model.parameters(), lr=1e-3, eps=1e-4)

        self.output_path = output_path

        self.tracker = Tracker(['loss_obs', 'loss_reward', 'loss_kl', 'reward', 'grad_norm'], ['animation'])

        self.track_wandb = track_wandb
        if track_wandb:
            wandb.init(project=f"PlaNet_{env}")


    def train_on_batch(self):
        # pick a batch and extract
        batch = self.buffer.sample(self.batch_size, self.chunk_length)
        # B x T x dim --> T x B x dim
        obs = batch[0].to(self.device).transpose(0, 1)
        action = batch[1].to(self.device).transpose(0, 1)
        reward = batch[2].to(self.device).transpose(0, 1)
        done = batch[3].to(self.device).transpose(0, 1)

        assert obs.shape[0:2] == (self.chunk_length + 1, self.batch_size), \
            f"Obs isn't 'time then batch' shape, {obs.shape[0:2]} instead of {(self.chunk_length + 1, self.batch_size)}"
        assert action.shape[0:2] == (self.chunk_length, self.batch_size), \
            f"Action isn't 'time then batch' shape {action.shape[0:2]} instead of {(self.chunk_length, self.batch_size)}"
        assert reward.shape[0:2] == (self.chunk_length, self.batch_size), \
            f"Reward isn't 'time then batch' shape {reward.shape[0:2]} instead of {(self.chunk_length, self.batch_size)}"
        assert done.shape[0:2] == (self.chunk_length, self.batch_size), \
            f"Done isn't 'time then batch' shape {done.shape[0:2]} instead of {(self.chunk_length, self.batch_size)}"

        # encode and initialize states
        encoded = bottle(self.model.encoder, obs)
        s, h = self.model.init_deterministic_state(encoded[0])
        states, priors, posteriors, posterior_samples = [], [], [], []

        # infer
        for a, e in zip(action.unbind(), encoded.unbind()[1:]):
            h = self.model.forward_deterministic_state(h, s, a)
            states.append(h)
            priors.append(self.model.forward_prior(h))
            posteriors.append(self.model.forward_posterior(h, e))
            posterior_samples.append(tfd.Normal(*posteriors[-1]).rsample())
            s = posterior_samples[-1]
        
        prior_dist = tfd.Normal(*map(torch.stack, zip(*priors)))
        posterior_dist = tfd.Normal(*map(torch.stack, zip(*posteriors)))

        states = torch.stack(states)
        posterior_samples = torch.stack(posterior_samples)
        
        pred_obs = bottle(self.model.decoder, states, posterior_samples)
        assert pred_obs.shape == obs[1:].shape, f"Shape mismatch on observations, got {pred_obs.shape} instead of {obs.shape}"
        loss_obs = F.mse_loss(pred_obs, obs[1:], reduction='none').sum((2, 3, 4)).mean()

        pred_reward = bottle(self.model.forward_reward, states, posterior_samples)
        assert pred_reward.shape == reward.shape
        loss_reward = F.mse_loss(pred_reward, reward.float()).mean()

        loss_kl = torch.max(tfd.kl_divergence(posterior_dist, prior_dist).sum(-1), self.free_nats).mean()

        self.optim.zero_grad()
        (loss_obs + 10 * loss_reward + self.kl_weight*loss_kl).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1000., norm_type=2)
        self.optim.step()

        metrics = {
            'loss_obs': loss_obs.detach().cpu().numpy(),
            'loss_reward': loss_reward.detach().cpu().numpy(),
            'loss_kl': loss_kl.detach().cpu().numpy(),
            'grad_norm': grad_norm.detach().cpu().numpy()
        }
        return metrics

    def explore(self, epoch: int, save_gif: bool=False):
        # exploration loop
        obs = self.env.reset(save_gif)

        explore = epoch%20 == 0
        if explore:
            print(f"Not exploring for epoch {epoch}")

        if save_gif:
            obs, img = self.env.reset(True)
            list_img = [Image.fromarray(img)]
        else:
            obs = self.env.reset()

        episode = Episode(self.exploration_time)
        counter = tqdm(range(self.exploration_time), desc=f"Exp Epoch {epoch}")
        total_reward = 0
        ep_done = 0
        ep_reward = 0

        with torch.no_grad():
            self.policy.reset()
            done = False
            for t in counter:
                action = self.policy.act(obs.to(self.device), explore=explore)

                reward = 0
                for i in range(self.repeat_action):
                    if not done:
                        # skipping some frames
                        if save_gif and i%4 == 0:
                            next_obs, r, done, _, img = self.env.step(action, return_obs=True)
                            list_img.append(Image.fromarray(img))
                        else:
                            next_obs, r, done, _ = self.env.step(action)
                    else:
                        ep_done += 1
                        total_reward += ep_reward
                        ep_reward = 0
                        done = False
                        # skipping some frames
                        if save_gif and i%4 == 0:
                            next_obs, img = self.env.reset(return_obs=True)
                            list_img.append(Image.fromarray(img))
                        else:
                            next_obs = self.env.reset()
                        break

                    reward += r
                episode.append(obs, action, reward, done)
                obs = next_obs

                ep_reward += reward

            if ep_done == 0:
                ep_done = 1

            if save_gif:
                list_img[0].save(f"{self.output_path}/ep_{epoch}.gif", save_all=True, append_images=list_img[1:])

            episode.end()
            self.buffer.store(episode)
        return {
            'reward': total_reward / ep_done,
            'animation': f"{self.output_path}/ep_{epoch}.gif"
        }

    def train(self, epochs: int):

        while len(self.buffer.episodes) < self.seed_episodes:
            episode = Episode(self.exploration_time)
            obs = self.env.reset()
            for _ in range(self.exploration_time):
                action = self.env.sample()
                reward = 0
                for _ in range(self.repeat_action):
                    next_obs, r, done, _ = self.env.step(action)
                    reward += r
                    if done:
                        break
                episode.append(obs, action, reward, done)
                obs = next_obs
            
            episode.end()
            self.buffer.store(episode)

        for epoch in range(0, epochs):
            # optimization loop
            counter = tqdm(range(self.training_iterations), desc=f"Opt Epoch {epoch}, loss_obs = NaN, loss_rew = NaN, loss_kl = NaN")
            for step in counter:
                metrics = self.train_on_batch()

                self.tracker.step(metrics)
                metrics = self.tracker.showcase(['loss_obs', 'loss_reward', 'loss_kl'])

                counter.set_description(
                    f"Opt Epoch {epoch}, loss_obs = {metrics['loss_obs']:.4f}, loss_rew = {metrics['loss_reward']:.4f}, loss_kl = {metrics['loss_kl']:.4f}")

            metrics = self.explore(epoch, save_gif=True)
            self.tracker.step(metrics)

            metrics = self.tracker.terminate()
            if self.track_wandb:
                wandb.log(metrics)