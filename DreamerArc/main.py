from training import Trainer

import torch


if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cem_conf = {
        "planning_horizon": 10,
        "optimization_iteration": 12,
        "candidates_per_iteration": 1000,
        "sorted_candidates": 100
    }

    trainer = Trainer(
        'Walker2d-v4',
        buffer_size=1000,
        seed_episodes=5,
        training_iterations=100,
        batch_size=50,
        chunk_length=50,
        repeat_action=4,
        exploration_time=200,
        kl_weight=1.,
        free_nats=3.,
        device=device,
        optim='Adam',
        cem_conf=cem_conf,
        track_wandb=True
    )
    
    trainer.train(1000)