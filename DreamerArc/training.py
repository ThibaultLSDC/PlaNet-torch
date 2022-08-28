from env_wrapper import TorchImageEnvWrapper
from models.rssm import RSSM
from policies.cem import CEM
from memory.buffer import Buffer, Episode
from utils import bottle, Tracker

import torch
import torch.distributions as tfd

import torch.nn.functional as F

from tqdm import tqdm

from PIL import Image

import wandb


class Trainer:
    def __init__(self,
        env,
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
        output_path: str='data',
        track_wandb: bool=False,
        ) -> None:

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.env = TorchImageEnvWrapper(env)
        self.model = RSSM(self.env.action_space).to(self.device)
        self.policy = CEM(
            self.model,
            **cem_conf,
            action_bounds=self.env.action_bounds
        )
        self.buffer = Buffer(100)

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

        self.optim = torch.optim.Adam(self.model.parameters(), 1e-3, eps=1e-4)

        self.output_path = output_path

        self.tracker = Tracker(['loss_obs', 'loss_reward', 'loss_kl', 'reward'], ['animation'])

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
        mask = batch[4].to(self.device).transpose(0, 1)

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
        loss_obs = F.mse_loss(pred_obs, obs[1:], reduction='none').sum((2, 3, 4))
        loss_obs = (loss_obs * mask).sum() / (mask.sum() + 1e-9)

        pred_reward = bottle(self.model.forward_reward, states, posterior_samples)
        assert pred_reward.shape == reward.shape
        loss_reward = F.mse_loss(pred_reward, reward.float()).squeeze()
        loss_reward = (loss_reward * mask).sum() / (mask.sum() + 1e-9)

        loss_kl = torch.max(tfd.kl_divergence(posterior_dist, prior_dist).sum(-1), self.free_nats)
        loss_kl = (loss_kl * mask).sum() / (mask.sum() + 1e-6)

        self.optim.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1000., norm_type=2)
        (loss_obs + loss_reward + self.kl_weight*loss_kl).backward()
        self.optim.step()

        metrics = {
            'loss_obs': loss_obs.detach().cpu().numpy(),
            'loss_reward': loss_reward.detach().cpu().numpy(),
            'loss_kl': loss_kl.detach().cpu().numpy()
        }
        return metrics

    def explore(self, epoch: int, save_gif: bool=False):
        # exploration loop
        obs = self.env.reset(save_gif)

        if save_gif:
            obs, img = self.env.reset(True)
            list_img = [Image.fromarray(img)]
        else:
            obs = self.env.reset()

        episode = Episode(100)
        counter = tqdm(range(self.exploration_time), desc=f"Exp Epoch {epoch}")
        total_reward = 0

        with torch.no_grad():
            state = None
            for t in counter:
                action, state = self.policy.act(obs.to(self.device), self.env.sample(), state=state)
                action = action + torch.randn_like(action) * 0.3

                reward = 0
                for i in range(self.repeat_action):
                    if save_gif and i%2 == 0:
                        next_obs, r, done, _, img = self.env.step(action, return_obs=True)
                        list_img.append(Image.fromarray(img))
                    else:
                        next_obs, r, done, _ = self.env.step(action)

                    reward += r
                    if done:
                        break
                episode.append(obs, action, reward, done)
                obs = next_obs

                total_reward += reward

                if done:
                    # print(f"Total reward for the episode was {total_reward}")
                    total_reward = 0
                    episode.end()
                    self.buffer.store(episode)
                    episode = Episode(100)
                    if save_gif:
                        list_img[0].save(f"{self.output_path}/ep_{epoch}.gif", save_all=True, append_images=list_img[1:])
                        obs, img = self.env.reset(True)
                        list_img = [Image.fromarray(img)]
                    else:
                        obs = self.env.reset()
        return {
            'reward': total_reward,
            'animation': f"{self.output_path}/ep_{epoch}.gif"
        }

    def train(self, epochs: int):

        while len(self.buffer.episodes) < self.seed_episodes:
            episode = Episode(100)
            obs = self.env.reset()
            done = False
            while not done:
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

            wandb.log(metrics)