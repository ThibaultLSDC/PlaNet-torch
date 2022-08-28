import torch
from collections import deque


class Episode:
    def __init__(self, max_length: int) -> None:
        self.max_length = max_length

        self.observation = deque([], maxlen=max_length)
        self.action = deque([], maxlen=max_length)
        self.reward = deque([], maxlen=max_length)
        self.done = deque([], maxlen=max_length)

        self.len = 0

    def append(self, observation: torch.Tensor, action: torch.Tensor, reward: float, done: bool):
        self.observation.append(observation)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)

        self.len += 1
    
    def __len__(self):
        return self.len

    def reset(self):
        self.__init__(self.max_length)
    
    def end(self):
        self.observation = torch.stack(tuple(self.observation), dim=0)
        self.action = torch.stack(tuple(self.action), dim=0)
        self.reward = torch.tensor(self.reward).unsqueeze(-1)
        self.done = torch.tensor(self.done).unsqueeze(-1)

        assert self.observation.shape[0] == self.len, f"Lost an obs"
        assert self.action.shape[0] == self.len, f"Lost an action"
        assert self.reward.shape == (self.len, 1), f"Reward isn't of right shape"
        assert self.done.shape == (self.len, 1), f"Done isn't of right shape"


class Buffer:
    def __init__(self, max_episodes: int) -> None:
        self.max_episodes = max_episodes

        self.episodes = deque([], maxlen=max_episodes)
        self.ep_lengths = deque([], maxlen=max_episodes)

    def store(self, episode: Episode):
        self.episodes.append(episode)
        self.ep_lengths.append(episode.len)
    
    def reset(self):
        self.__init__(self.max_episodes)
    
    def sample(self, batch_size: int, chunk_size: int):
        ep_idx = torch.randint(0, len(self.episodes), (batch_size,))

        ep_max_lengths = [torch.randint(self.ep_lengths[i] - chunk_size - 1, (1,)) for i in ep_idx]

        obs, act, rew, done = [], [], [], []
        for i, l in zip(ep_idx, ep_max_lengths):
            obs.append(self.episodes[i].observation[l:l+chunk_size+1])
            act.append(self.episodes[i].action[l+1:l+chunk_size+1])
            rew.append(self.episodes[i].reward[l+1:l+chunk_size+1])
            done.append(self.episodes[i].done[l+1:l+chunk_size+1])
        
        return [torch.stack(i, dim=0) for i in (obs, act, rew, done)]