import torch
import numpy as np
import gymnasium as gym
import wandb
import cleanrl.vanilla_policy_gradient
import cleanrl.common

if __name__ == "__main__":
    wandb_logger = wandb.init(project="cleanrl_pendulum")
    env = gym.make("InvertedPendulum-v5")
    T = 1000
    gamma = 0.99
    actor = cleanrl.common.MlpGaussianActor(
        env, [4, 8, 16], lower=torch.Tensor([-3.0]), upper=torch.Tensor([3.0])
    )
    actor.logstd.data = torch.tensor([-0.2])
    lr = 0.001
    num_epoch = 1000
    cleanrl.vanilla_policy_gradient.train(
        env, T, gamma, actor, lr, num_epoch, wandb_logger
    )
    import pdb

    pdb.set_trace()
    pass
