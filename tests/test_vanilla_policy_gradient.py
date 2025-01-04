import cleanrl.vanilla_policy_gradient as mut

import pytest
import torch
import numpy as np
import gymnasium as gym

import cleanrl.common


def test_reward_to_go():
    # A single reward traj
    r_traj = np.array([0.5, 0.4, 0.3])
    gamma = 0.99
    rtg = mut.reward_to_go(r_traj, gamma)
    assert rtg.shape == (3,)
    np.testing.assert_allclose(rtg[-1], r_traj[-1] * gamma**2)
    for i in range(2):
        np.testing.assert_allclose(rtg[i], rtg[i + 1] + r_traj[i] * gamma**i)

    # A batch of r_traj
    r_traj = np.array([[-0.5, 0.2, 0.1], [1.2, -0.3, 0.5]])
    rtg = mut.reward_to_go(r_traj, gamma)
    assert rtg.shape == (2, 3)
    for i in range(2):
        rtg_i = mut.reward_to_go(r_traj[i], gamma)
        np.testing.assert_allclose(rtg[i], rtg_i)


def test_calc_J():
    # Test a single r_traj
    r_traj = np.array([0.5, 1, 3])
    gamma = 0.9
    logprob_traj = torch.tensor([0.3, 1.5, 2])
    loss = mut.calc_loss(r_traj, gamma, logprob_traj)
    rtg = mut.reward_to_go(r_traj, gamma)
    np.testing.assert_allclose(loss.item(), -np.sum(rtg * logprob_traj.detach().numpy()))

    # Test a batch of r_traj
    r_traj = np.array([[0.5, 1, 1.2], [0.2, 3, -0.1]])
    logprob_traj = torch.tensor([[-0.1, 0.4, 0.3], [0.2, 0.5, 0.4]])
    loss = mut.calc_loss(r_traj, gamma, logprob_traj)
    loss_single_traj = [mut.calc_loss(r_traj[i], gamma, logprob_traj[i]) for i in range(2)]
    np.testing.assert_allclose(loss.item(), (loss_single_traj[0] + loss_single_traj[1]).item())

def test_collect_rollout():
    env = gym.make("MountainCarContinuous-v0")
    T = 5
    actor = cleanrl.common.MlpGaussianActor(env, [4, 8])

    obs_traj, act_traj, logprob_traj, reward_traj = mut.collect_rollout(env, T, actor)

    assert obs_traj.shape == (T, 2)
    assert act_traj.shape == (T, 1)
    assert logprob_traj.shape == (T,)
    assert reward_traj.shape == (T,)

