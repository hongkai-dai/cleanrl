"""
Vanilla policy gradient
dJ = E[∇ log pi(a|s) * rtg]
where rtg is the reward-to-go
"""

from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn
import wandb

import cleanrl.common


def reward_to_go(r_traj: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute reward_to_go = sum_{t'} r(s_t', a_t') * gamma^(t')
    """
    T = r_traj.size if r_traj.ndim == 1 else r_traj.shape[1]
    gamma_power = np.power(gamma, np.arange(T))
    return np.flip(np.flip(r_traj * gamma_power, axis=-1).cumsum(axis=-1), axis=-1)


def calc_loss(r_traj: np.ndarray, gamma: float, logprob_traj: torch.Tensor):
    """
    Compute the loss = -sum_t log pi(a_t|s_t) * rtg
    """
    assert r_traj.shape == logprob_traj.shape
    rtg = reward_to_go(r_traj, gamma).copy()
    return -torch.sum(
        torch.from_numpy(rtg).to(logprob_traj.device).to(logprob_traj.dtype)
        * logprob_traj
    )

def calc_J(r_traj: np.ndarray, gamma: float):
    """
    Compute J = sum_t r(t) * gamma**t
    """
    return np.sum(r_traj * np.power(gamma, np.arange(r_traj.size)))



def collect_rollout(
    env: gym.Env, T: int, actor: cleanrl.common.StochasticActor
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, np.ndarray]:
    """
    Collect a rollout (obs_traj, act_traj, logprob_traj, reward_traj)
    """
    obs_traj = np.empty((T + 1,) + env.observation_space.shape, dtype=np.float32)
    act_traj = np.empty((T,) + env.action_space.shape, dtype=np.float32)
    logprob_traj = torch.empty((T,))
    reward_traj = np.empty((T,), np.float32)
    obs_traj[0], _ = env.reset()
    for i in range(T):
        act, logprob_traj[i] = actor.sample(torch.from_numpy(obs_traj[i]))
        act_traj[i] = act.detach().numpy()
        obs_traj[i + 1], reward_traj[i], terminated, truncated, info = env.step(
            act_traj[i]
        )
        T_final = i + 1
        if terminated or truncated:
            break
    return (
        obs_traj[:T_final],
        act_traj[:T_final],
        logprob_traj[:T_final],
        reward_traj[:T_final],
    )


def train(
    env: gym.Env,
    T: int,
    gamma: float,
    actor: cleanrl.common.StochasticActor,
    lr: float,
    num_epoch: int,
    wandb_logger ,
):
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    for i in range(num_epoch):
        # Collect trajectories.
        num_rollouts = 10
        costs = torch.empty((num_rollouts,))
        Js = np.empty((num_rollouts,))
        for rollout_count in range(num_rollouts):
            obs_traj, act_traj, logprob_traj, reward_traj = collect_rollout(env, T, actor)
            costs[rollout_count] = calc_loss(reward_traj, gamma, logprob_traj)
            Js[rollout_count] = calc_J(reward_traj, gamma)
        optimizer.zero_grad()
        cost = torch.mean(costs)
        cost.backward()
        optimizer.step()
        J = Js.mean()
        wandb_logger.log({"J": J, "epoch": i})
