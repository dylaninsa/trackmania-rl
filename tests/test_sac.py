import numpy as np
import torch

from algos.sac import SACAgent
from networks.sac import Actor, Critic

OBS_DIM = 3
ACT_DIM = 1
BATCH_SIZE = 64
CONFIG = {"hidden": 64, "lr": 3e-4, "gamma": 0.99, "tau": 0.005, "device": "cpu"}
CONTRACT_KEYS = ["critic_loss", "actor_loss", "alpha_loss", "alpha", "q_mean", "entropy"]


def make_agent(seed=0):
    """Creates a small agent with a fixed seed.

    Args:
        seed (int): Seed of the network initialization.

    Returns:
        SACAgent: The agent.
    """
    torch.manual_seed(seed)
    return SACAgent(OBS_DIM, ACT_DIM, CONFIG)


def make_batch(terminated=0.0, seed=0):
    """Builds a random batch in the format returned by ``ReplayBuffer.sample``.

    Args:
        terminated (float): Value of every ``terminated`` flag.
        seed (int): Seed of the random batch.

    Returns:
        dict: Float32 tensors ``obs``, ``act``, ``rew``, ``next_obs`` and ``terminated``.
    """
    generator = torch.Generator().manual_seed(seed)
    return {
        "obs": torch.randn(BATCH_SIZE, OBS_DIM, generator=generator),
        "act": torch.rand(BATCH_SIZE, ACT_DIM, generator=generator) * 2 - 1,
        "rew": torch.randn(BATCH_SIZE, generator=generator),
        "next_obs": torch.randn(BATCH_SIZE, OBS_DIM, generator=generator),
        "terminated": torch.full((BATCH_SIZE,), terminated),
    }


def test_actor_output_shapes_and_bounds():
    actor = Actor(OBS_DIM, ACT_DIM, 64)

    action, log_prob = actor(torch.randn(BATCH_SIZE, OBS_DIM))

    assert action.shape == (BATCH_SIZE, ACT_DIM)
    assert log_prob.shape == (BATCH_SIZE,)
    assert torch.all((action >= -1.0) & (action <= 1.0))


def test_actor_log_prob_is_finite_when_saturated():
    actor = Actor(OBS_DIM, ACT_DIM, 64)
    with torch.no_grad():
        actor.net[-1].weight.zero_()
        actor.net[-1].bias[:ACT_DIM] = 50.0
        actor.net[-1].bias[ACT_DIM:] = -5.0

    action, log_prob = actor(torch.randn(BATCH_SIZE, OBS_DIM))

    assert torch.all(action == 1.0)
    assert torch.all(torch.isfinite(log_prob))


def test_actor_deterministic_is_repeatable():
    actor = Actor(OBS_DIM, ACT_DIM, 64)
    obs = torch.randn(BATCH_SIZE, OBS_DIM)

    first, _ = actor(obs, deterministic=True)
    second, _ = actor(obs, deterministic=True)

    assert torch.equal(first, second)


def test_critic_output_shape():
    critic = Critic(OBS_DIM, ACT_DIM, 64)

    q = critic(torch.randn(BATCH_SIZE, OBS_DIM), torch.randn(BATCH_SIZE, ACT_DIM))

    assert q.shape == (BATCH_SIZE,)


def test_act_returns_numpy_action():
    action = make_agent().act(np.zeros(OBS_DIM, dtype=np.float32))

    assert isinstance(action, np.ndarray)
    assert action.shape == (ACT_DIM,)
    assert action.dtype == np.float32
    assert np.all((action >= -1.0) & (action <= 1.0))


def test_target_is_reward_when_terminated():
    batch = make_batch(terminated=1.0)

    target = make_agent().compute_target(batch)

    assert target.shape == (BATCH_SIZE,)
    assert torch.equal(target, batch["rew"])


def test_update_returns_contract_keys():
    logs = make_agent().update(make_batch())

    assert sorted(logs) == sorted(CONTRACT_KEYS)
    assert all(isinstance(value, float) for value in logs.values())


def test_update_targets_is_polyak_average():
    agent = make_agent()
    with torch.no_grad():
        for p in agent.q1.parameters():
            p.add_(1.0)
    old_target = agent.q1_target.net[0].weight.clone()

    agent.update_targets()

    expected = (1 - CONFIG["tau"]) * old_target + CONFIG["tau"] * agent.q1.net[0].weight
    assert torch.allclose(agent.q1_target.net[0].weight, expected)


def test_critic_loss_decreases_on_fixed_batch():
    agent = make_agent()
    batch = make_batch()

    first_loss, _ = agent.update_critics(batch)
    for _ in range(200):
        last_loss, _ = agent.update_critics(batch)

    assert last_loss < first_loss


def test_save_load_round_trip(tmp_path):
    agent = make_agent()
    batch = make_batch()
    for _ in range(10):
        agent.update(batch)
    path = tmp_path / "sac.pt"
    agent.save(path)

    restored = make_agent(seed=1)
    restored.load(path)

    obs = np.zeros(OBS_DIM, dtype=np.float32)
    np.testing.assert_array_equal(agent.act(obs, deterministic=True), restored.act(obs, deterministic=True))
    assert agent.log_alpha.item() == restored.log_alpha.item()
    torch.manual_seed(2)
    expected = agent.update(batch)
    torch.manual_seed(2)
    assert restored.update(batch) == expected
