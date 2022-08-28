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
        sorted_candidates: int
        ) -> None:
        
        self.world_model = world_model
        self.planning_horizon = planning_horizon
        self.optimization_iteration = optimization_iteration
        self.candidates_per_iteration = candidates_per_iteration
        self.sorted_candidates = sorted_candidates

    def act(self, obs: torch.Tensor, example_action: torch.Tensor, state: torch.Tensor=None) -> torch.Tensor:
        dev = obs.device
        shape = (self.planning_horizon,) + example_action.shape
        dist = tfd.Normal(torch.zeros(shape, device=dev), torch.ones(shape, device=dev))

        encoded = self.world_model.encoder(obs)
        for i in range(self.optimization_iteration):
            # batch_size x time x action_dim --> time x batch_size x action_dim
            actions = dist.sample((self.candidates_per_iteration,)).transpose(0, 1)

            if state == None:
                stoch_state, state = self.world_model.init_deterministic_state(encoded)
            else:
                loc, variance = self.world_model.forward_prior(state)
                stoch_state = tfd.Normal(loc, variance).sample()

            reward = torch.zeros((actions.shape[1],), device=dev)

            current_state = deepcopy(state).repeat((1, 1000, 1)).squeeze(0)
            stoch_state = stoch_state.repeat((1, 1000, 1)).squeeze(0)

            states, stoch_states = self.world_model.rollout_prior(actions, current_state, stoch_state)

            assert states.shape == (self.planning_horizon,) + current_state.shape, \
                f"States mismatch, has dim {states.shape} instead of {(self.planning_horizon,) + current_state.shape}"
            assert stoch_states.shape == (self.planning_horizon,) + stoch_state.shape, \
                f"Stochastic states mismatch, has dim {stoch_states.shape} instead of {(self.planning_horizon,) + stoch_state.shape}"

            reward = bottle(self.world_model.forward_reward, states, stoch_states) # time x bs x 1
            
            reward = reward.sum(0).squeeze() # bs

            k_best = torch.argsort(reward, dim=0, descending=True)[:self.sorted_candidates]

            # time x batch_size x action_dim --> k_best x time x action_dim
            actions = actions.transpose(0, 1)[k_best]

            # time x action_dim
            mu = torch.mean(actions, dim=0, keepdim=False)
            std = torch.sum(torch.abs(actions - mu), dim=0) / (self.sorted_candidates - 1)
            
            assert mu.shape == (self.planning_horizon,) + example_action.shape, \
                f"Shape mismatch, has shape {mu.shape} instead of {(self.planning_horizon,) + example_action.shape}"

            dist = tfd.Normal(mu, std.square() + 1e-9)
        
        assert mu.shape == shape, f"Output mismatch, has shape {mu.shape} instead of {shape}"

        action = mu[0].unsqueeze(0)

        loc, variance = self.world_model.forward_prior(state)
        stoch_state = tfd.Normal(loc, variance).sample()

        next_state = self.world_model.forward_deterministic_state(state, stoch_state, action)

        return action.cpu().squeeze(0), next_state