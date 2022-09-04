from copy import deepcopy
from models.rssm import RSSM
import torch
import torch.distributions as tfd
from utils import bottle


class CEM:
    def __init__(self,
        world_model: RSSM,
        planning_horizon: int,
        optimization_iteration: int,
        candidates_per_iteration: int,
        sorted_candidates: int,
        action_bounds,
        device
        ) -> None:
        
        self.world_model = world_model
        self.planning_horizon = planning_horizon
        self.optimization_iteration = optimization_iteration
        self.candidates_per_iteration = candidates_per_iteration
        self.sorted_candidates = sorted_candidates
        self.action_bounds = action_bounds

        self.noise_scale = torch.abs(torch.tensor(action_bounds[1] - action_bounds[0], device=device)) / 2

        self.device = device

    def reset(self):
        self.h = torch.zeros((1, self.world_model.state_dim)).to(self.device)
        self.s = torch.zeros((1, self.world_model.latent_dim)).to(self.device)
        self.a = torch.zeros((1, self.world_model.action_dim)).to(self.device)


    def _act(self, obs: torch.Tensor) -> torch.Tensor:
        shape = (self.planning_horizon, self.world_model.action_dim)

        mu = torch.zeros(shape).to(self.device)
        std = torch.ones(shape).to(self.device)

        assert len(obs.shape) == 3, f"obs isn't an image, shape is {obs.shape}"

        encoded = self.world_model.encoder(obs)

        self.s, self.h = self.world_model.init_deterministic_state(
            encoded, self.h, self.s, self.a
        )

        for i in range(self.optimization_iteration):
            # batch_size x time x action_dim --> time x batch_size x action_dim
            actions = tfd.Normal(mu, std).sample((self.candidates_per_iteration,)).permute(1, 0, 2)

            state = self.h.clone().repeat(self.candidates_per_iteration, 1)
            stoch_state = self.s.clone().repeat(self.candidates_per_iteration, 1)

            # time x bs x dim
            states, stoch_states = self.world_model.rollout_prior(actions, state, stoch_state)

            # time x bs x 1
            rewards = bottle(self.world_model.forward_reward, states, stoch_states)
            assert rewards.shape == (self.planning_horizon, self.candidates_per_iteration, 1), \
                f"shape mismatch, has shape {rewards.shape} instead of {(self.planning_horizon, self.candidates_per_iteration, 1)}"
            
            # bs
            rewards = rewards.sum(dim=0).squeeze()
        
            _, k_best = torch.topk(rewards, k=self.sorted_candidates, largest=True)

            mu = actions.permute(1, 0, 2)[k_best].mean(0)
            std = actions.permute(1, 0, 2)[k_best].std(0, unbiased=False)

        self.a = mu[0:1]

    def act(self, obs, explore):
        self._act(obs)
        if explore:
            self.a += torch.randn_like(self.a) * 0.3 * self.noise_scale
        return self.a.squeeze(0).cpu()