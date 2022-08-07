import torch

import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.act = F.relu
        self.embedding_dim = embedding_dim
        self.conv1 = nn.Conv2d(3, 64, 3, 2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, padding=1)
        self.conv4 = nn.Conv2d(256, embedding_dim, 3, 2, padding=1)

        self.globalavg = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, input):
        batch_size, _, _, _ = input.shape
        x = self.act(self.conv1(input))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.globalavg(x)
        x = x.view((-1, self.embedding_dim))
        assert x.shape[0] == batch_size
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, state_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.input_dim = latent_dim + state_dim
        self.embedding_dim = embedding_dim
        self.act = F.relu
        self.fc = nn.Linear(self.input_dim, embedding_dim)
        # 1x1
        self.conv1 = nn.ConvTranspose2d(embedding_dim, 256, 5)
        # 5x5
        self.conv2 = nn.ConvTranspose2d(256, 128, 5, 2)
        # 13x13
        self.conv3 = nn.ConvTranspose2d(128, 64, 6, 2)
        # 30x30
        self.conv4 = nn.ConvTranspose2d(64, 32, 6, 2)
        # 64x64
    
    def forward(self, input):
        assert input.shape[-1] == self.input_dim
        x = self.act(self.fc(input))
        x = x.view((-1, self.embedding_dim, 1, 1))
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        assert x.shape[2:] == [64, 64]
        return x


class NormalLinear(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features) -> None:
        super().__init__()
        self.act = F.relu

        self.fc = nn.Linear(in_features, hidden_dim)

        self.mu = nn.Linear(hidden_dim, out_features)
        self.var = nn.Linear(hidden_dim, out_features)

    def forward(self, input):
        h = self.act(self.fc(input))
        mu = self.mu(h)
        var = F.softplus(self.var(h)) + 1e-1
        return mu, var


class RSSM(nn.Module):
    def __init__(self,
        action_dim: int,
        state_dim: int=200,
        latent_dim: int=30,
        hidden_dim: int=200,
        embedding_dim: int=1024,
        activation=F.relu
        ) -> None:
        super().__init__()

        # Dimensions
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.act = activation
        
        # Encoder and decoder, from and to img shape (N, 3, 64, 64,) <-> (N, embedding_dim). #TODO change img sizes
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(latent_dim, state_dim, embedding_dim)

        # Deterministic path through the model
        self.grucell = nn.GRUCell(state_dim + action_dim, state_dim)
        self.deterministic_fc = nn.Linear(latent_dim + action_dim, state_dim)

        # Latent prior and posterior estimations of the stochastic state
        self.prior_latent = NormalLinear(state_dim, hidden_dim, latent_dim)
        self.posterior_latent = NormalLinear(state_dim + embedding_dim, hidden_dim, latent_dim)

        # Reward estimation
        self.reward_model = nn.Sequential(
            nn.Linear(latent_dim + state_dim, hidden_dim),
            nn.ReLU,
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    

    def init_deterministic_state(self, e1, h0=None, s0=None, a0=None):
        batch_size = e1.shape[0]
        device = e1.device
        
        h0 = torch.zeros((batch_size, self.state_dim), device=device) if h0 is None else h0
        s0 = torch.zeros((batch_size, self.latent_dim), device=device) if s0 is None else s0
        a0 = torch.zeros((batch_size, self.action_dim), device=device) if a0 is None else a0

        h1 = self.forward_deterministic_state(h0, s0, a0)
        mu, var = self.forward_prior(h1, e1)

        s1 = torch.randn_like(mu) * var.sqrt() + mu

        return s1, h1

    def forward_deterministic_state(self, h, s, a):
        s_a = (torch.cat((s, a), dim=-1))
        input = self.deterministic_fc(s_a)
        return self.grucell(input, h)

    def forward_prior(self, h):
        mu, var = self.prior_latent(h)
        return mu, var
    
    def forward_posterior(self, h, e):
        h_e = torch.cat((h, e), dim=-1)
        mu, var = self.posterior_latent(h_e)
        return mu, var
    
    def forward_reward(self, h, s):
        h_s = torch.cat((h, s), dim=-1)
        return self.reward_model(h_s).squeeze()
    

    