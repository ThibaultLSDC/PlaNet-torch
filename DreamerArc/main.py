from env_wrapper import TorchImageEnvWrapper
from training import Trainer

import torch


if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cem_conf = {
        "planning_horizon": 12,
        "optimization_iteration": 10,
        "candidates_per_iteration": 1000,
        "sorted_candidates": 100
    }

    trainer = Trainer(
        'Pendulum-v1',
        seed_episodes=5,
        training_iterations=100,
        batch_size=50,
        chunk_length=50,
        repeat_action=2,
        exploration_time=100,
        kl_weight=1.,
        free_nats=3.,
        device=device,
        cem_conf=cem_conf
    )
    
    trainer.train(100)