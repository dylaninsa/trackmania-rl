import numpy as np
import pytest
import torch

from algos.ppo import PPOAgent, RolloutBuffer, RunningMeanStd
from networks.ppo import ContinuousActor, Critic

OBS_DIM = 3
ACT_DIM = 2
BATCH_SIZE = 16
GAMMA = 0.9
CONFIG = {
    "hidden": 32,
    "lr_actor": 1e-3,
    "lr_critic": 2e-3,
    "gamma": GAMMA,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "entropy_coef": 0.0,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "ppo_epochs": 2,
    "batch_size": 8,
    "buffer_size": 32,
    "normalize_states": True,
    "reward_scale": 0.1,
    "device": "cpu",
}
CONTRACT_KEYS = ["actor_loss", "critic_loss", "entropy", "approx_kl", "clip_fraction", "policy_std"]


def make_agent(seed=0):
    """Creates a small agent with a fixed seed.

    Args:
        seed (int): Seed of the network initialization.

    Returns:
        PPOAgent: The agent.
    """
    torch.manual_seed(seed)
    return PPOAgent(OBS_DIM, ACT_DIM, CONFIG)


def fill_rollout(agent, seed=0):
    """Fills the rollout buffer of the agent with random transitions.

    Args:
        agent (PPOAgent): Agent whose buffer is filled.
        seed (int): Seed of the random observations and rewards.
    """
    rng = np.random.default_rng(seed)
    for _ in range(CONFIG["buffer_size"]):
        obs = rng.normal(size=OBS_DIM).astype(np.float32)
        _, raw_action, log_prob, value = agent.select_action(obs, update_stats=True)
        next_obs = rng.normal(size=OBS_DIM).astype(np.float32)
        agent.store(obs, raw_action, log_prob, float(rng.normal()), value, next_obs, False, False)


def make_buffer(transitions):
    """Builds a rollout buffer holding the given transitions.

    Args:
        transitions (list[tuple]): ``(reward, value, next_value, terminated, truncated)`` of each transition.

    Returns:
        RolloutBuffer: The buffer, with its advantages and returns computed with ``GAMMA`` and lambda = 1.
    """
    buffer = RolloutBuffer(len(transitions), OBS_DIM, ACT_DIM, torch.device("cpu"))
    state, action = np.zeros(OBS_DIM), np.zeros(ACT_DIM)
    for reward, value, next_value, terminated, truncated in transitions:
        buffer.add(state, action, 0.0, reward, value, next_value, terminated, truncated)
    buffer.compute_returns_and_advantages(gamma=GAMMA, gae_lambda=1.0)
    return buffer


def test_running_mean_std_matches_numpy():
    data = np.random.default_rng(0).normal(loc=3.0, scale=2.0, size=(1000, OBS_DIM))
    rms = RunningMeanStd(shape=(OBS_DIM,))

    for batch in np.split(data, 10):
        rms.update(batch)

    np.testing.assert_allclose(rms.mean, data.mean(axis=0), rtol=1e-3)
    np.testing.assert_allclose(rms.var, data.var(axis=0), rtol=1e-3)


def test_normalize_keeps_zero_and_clips():
    rms = RunningMeanStd(shape=(2,))
    rms.update(np.array([[5.0, 0.0], [7.0, 0.002]]))

    normalized = rms.normalize(np.array([0.0, 1.0]))

    assert normalized[0] == 0.0
    assert normalized[1] == 10.0
    assert normalized.dtype == np.float32


def test_actor_output_shapes_and_mean_bounds():
    actor = ContinuousActor(OBS_DIM, ACT_DIM, max_action=1.0, hidden_dim=32)

    action, log_prob = actor.get_action_and_log_prob(torch.randn(BATCH_SIZE, OBS_DIM))
    mu, std = actor(100 * torch.randn(BATCH_SIZE, OBS_DIM))

    assert action.shape == (BATCH_SIZE, ACT_DIM)
    assert log_prob.shape == (BATCH_SIZE, 1)
    assert torch.all(mu.abs() <= 1.0)
    assert torch.all(std > 0)


def test_actor_evaluate_actions_matches_sampling():
    actor = ContinuousActor(OBS_DIM, ACT_DIM, max_action=1.0, hidden_dim=32)
    states = torch.randn(BATCH_SIZE, OBS_DIM)

    actions, log_prob = actor.get_action_and_log_prob(states)
    evaluated, entropy = actor.evaluate_actions(states, actions)

    assert torch.allclose(evaluated, log_prob)
    assert entropy.shape == (BATCH_SIZE, 1)


def test_critic_output_shape():
    assert Critic(OBS_DIM, hidden_dim=32)(torch.randn(BATCH_SIZE, OBS_DIM)).shape == (BATCH_SIZE, 1)


def test_act_returns_bounded_action_without_updating_stats():
    agent = make_agent()

    action = agent.act(np.full(OBS_DIM, 100.0, dtype=np.float32))

    assert isinstance(action, np.ndarray)
    assert action.shape == (ACT_DIM,)
    assert action.dtype == np.float32
    assert np.all((action >= -1.0) & (action <= 1.0))
    assert agent.state_rms.count == pytest.approx(1e-4)


def test_act_deterministic_is_repeatable():
    agent = make_agent()
    obs = np.ones(OBS_DIM, dtype=np.float32)

    np.testing.assert_array_equal(agent.act(obs, deterministic=True), agent.act(obs, deterministic=True))


def test_select_action_updates_stats_only_when_asked():
    agent = make_agent()
    obs = np.ones(OBS_DIM, dtype=np.float32)

    agent.select_action(obs)
    agent.select_action(obs, update_stats=True)

    assert agent.state_rms.count == pytest.approx(1 + 1e-4)


def test_gae_cuts_the_future_after_termination():
    buffer = make_buffer([(1.0, 0.5, 2.0, True, False)])

    assert buffer.advantages[0, 0] == pytest.approx(1.0 - 0.5)
    assert buffer.returns[0, 0] == pytest.approx(1.0)


def test_gae_bootstraps_after_truncation():
    buffer = make_buffer([(1.0, 0.5, 2.0, False, True)])

    assert buffer.advantages[0, 0] == pytest.approx(1.0 + GAMMA * 2.0 - 0.5)


def test_gae_does_not_mix_episodes():
    buffer = make_buffer([(1.0, 0.5, 2.0, False, True), (3.0, 0.0, 0.0, True, False)])

    assert buffer.advantages[0, 0] == pytest.approx(1.0 + GAMMA * 2.0 - 0.5)
    assert buffer.advantages[1, 0] == pytest.approx(3.0)


def test_gae_accumulates_within_an_episode():
    buffer = make_buffer([(1.0, 0.0, 0.0, False, False), (1.0, 0.0, 0.0, True, False)])

    assert buffer.advantages[0, 0] == pytest.approx(1.0 + GAMMA * 1.0)


def test_buffer_raises_when_full():
    buffer = RolloutBuffer(1, OBS_DIM, ACT_DIM, torch.device("cpu"))
    buffer.add(np.zeros(OBS_DIM), np.zeros(ACT_DIM), 0.0, 0.0, 0.0, 0.0, False, False)

    with pytest.raises(RuntimeError):
        buffer.add(np.zeros(OBS_DIM), np.zeros(ACT_DIM), 0.0, 0.0, 0.0, 0.0, False, False)


def test_store_scales_reward_and_keeps_raw_action():
    agent = make_agent()
    obs = np.ones(OBS_DIM, dtype=np.float32)
    next_obs = np.zeros(OBS_DIM, dtype=np.float32)
    raw_action = np.array([3.0, -3.0], dtype=np.float32)

    agent.store(obs, raw_action, -1.0, 5.0, 0.2, next_obs, False, True)

    assert agent.buffer.rewards[0, 0] == pytest.approx(5.0 * CONFIG["reward_scale"])
    np.testing.assert_array_equal(agent.buffer.actions[0], raw_action)
    assert agent.buffer.next_values[0, 0] == pytest.approx(agent.get_value(next_obs))
    assert (agent.buffer.terminateds[0, 0], agent.buffer.dones[0, 0]) == (0.0, 1.0)


def test_update_returns_contract_keys_and_empties_buffer():
    agent = make_agent()
    fill_rollout(agent)
    weights = agent.actor.mu_head.weight.clone()

    logs = agent.update()

    assert sorted(logs) == sorted(CONTRACT_KEYS)
    assert all(isinstance(value, float) and np.isfinite(value) for value in logs.values())
    assert agent.buffer.ptr == 0
    assert not torch.equal(weights, agent.actor.mu_head.weight)


def test_set_learning_rate_fraction():
    agent = make_agent()

    agent.set_learning_rate_fraction(0.5)

    assert agent.actor_optimizer.param_groups[0]["lr"] == pytest.approx(0.5 * CONFIG["lr_actor"])
    assert agent.critic_optimizer.param_groups[0]["lr"] == pytest.approx(0.5 * CONFIG["lr_critic"])


def test_save_load_round_trip(tmp_path):
    agent = make_agent()
    fill_rollout(agent)
    agent.update()
    path = tmp_path / "ppo.pt"
    agent.save(path)

    restored = make_agent(seed=1)
    restored.load(path)

    obs = np.ones(OBS_DIM, dtype=np.float32)
    np.testing.assert_array_equal(agent.act(obs, deterministic=True), restored.act(obs, deterministic=True))
    np.testing.assert_array_equal(agent.state_rms.var, restored.state_rms.var)
    assert agent.get_value(obs) == restored.get_value(obs)
