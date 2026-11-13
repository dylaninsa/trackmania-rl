"""
Module 4 : Boucle d'entraînement et d'orchestration PPO (Pendulum-v1 -> TMRL).

Ce script :
1. Initialise l'environnement Gymnasium (Pendulum-v1 par défaut) et l'agent PPO.
2. Évalue une politique aléatoire (baseline de référence) pour comparer la progression.
3. Alterne entre :
   - Collecte de trajectoires On-Policy de taille `buffer_size` (ex: 2048 pas)
   - Mise à jour PPO sur K époques par mini-batchs
   - Évaluation périodique sans bruit d'exploration (politique déterministe)
4. Enregistre les logs dans TensorBoard (`runs/`), sauvegarde les checkpoints (`checkpoints/`)
   et génère automatiquement le graphique de la courbe d'apprentissage (`results/`).
"""

import argparse
import os
import random
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.environment import get_env_specs, make_env
from src.ppo import PPOAgent


def set_seed(seed: int) -> None:
    """Fixe toutes les graines aléatoires pour la reproductibilité scientifique."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_random_baseline(env_id: str, episodes: int = 10, seed: int = 100) -> Tuple[float, float]:
    """
    Évalue une politique purement aléatoire (Baseline 1 du projet) pour disposer
    d'un point de repère objectif sur la courbe d'apprentissage.
    """
    env = make_env(env_id, seed=seed)
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            done = terminated or truncated
        returns.append(ep_return)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def evaluate_agent(
    agent: PPOAgent, env_id: str, episodes: int = 5, seed: int = 1000
) -> Tuple[float, float]:
    """
    Évalue l'agent PPO en mode déterministe (sans bruit d'exploration) sur `episodes` épisodes.
    """
    eval_env = make_env(env_id, seed=seed)
    returns = []
    for ep in range(episodes):
        state, _ = eval_env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        while not done:
            env_action, _, _, _ = agent.select_action(
                state, deterministic=True, update_stats=False
            )
            state, reward, terminated, truncated, _ = eval_env.step(env_action)
            ep_return += float(reward)
            done = terminated or truncated
        returns.append(ep_return)
    eval_env.close()
    return float(np.mean(returns)), float(np.std(returns))


def plot_learning_curve(
    train_steps: List[int],
    train_returns: List[float],
    eval_steps: List[int],
    eval_returns: List[float],
    eval_stds: List[float],
    random_baseline: float,
    save_path: str,
    env_id: str,
) -> None:
    """Génère et sauvegarde la courbe d'apprentissage PPO vs Baseline aléatoire."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))

    # Courbe des épisodes d'entraînement (brute + moyenne glissante)
    if len(train_returns) > 0:
        plt.plot(
            train_steps,
            train_returns,
            color="steelblue",
            alpha=0.25,
            label="Récompense d'entraînement (épisode)",
        )
        if len(train_returns) >= 10:
            window = min(20, len(train_returns))
            smoothed = np.convolve(train_returns, np.ones(window) / window, mode="valid")
            plt.plot(
                train_steps[window - 1 :],
                smoothed,
                color="royalblue",
                linewidth=2.0,
                label=f"Moyenne glissante ({window} épisodes)",
            )

    # Courbe d'évaluation déterministe avec écart-type
    if len(eval_returns) > 0:
        eval_steps_arr = np.array(eval_steps)
        eval_means_arr = np.array(eval_returns)
        eval_stds_arr = np.array(eval_stds)
        plt.plot(
            eval_steps_arr,
            eval_means_arr,
            color="darkorange",
            marker="o",
            linewidth=2.2,
            label="Évaluation déterministe (moyenne ± std)",
        )
        plt.fill_between(
            eval_steps_arr,
            eval_means_arr - eval_stds_arr,
            eval_means_arr + eval_stds_arr,
            color="darkorange",
            alpha=0.2,
        )

    # Ligne de la politique aléatoire (Baseline)
    plt.axhline(
        y=random_baseline,
        color="crimson",
        linestyle="--",
        linewidth=1.8,
        label=f"Baseline Aléatoire ({random_baseline:.1f})",
    )

    plt.title(f"Courbe d'apprentissage PPO sur {env_id} (MTI881 - Groupe 4)", fontsize=13)
    plt.xlabel("Pas de temps d'environnement (Timesteps)", fontsize=11)
    plt.ylabel("Récompense cumulée par épisode (Return)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entraînement PPO (Validation Pendulum-v1 & Préparation TMRL)"
    )
    parser.add_argument("--env-id", type=str, default="Pendulum-v1", help="Nom de l'environnement Gymnasium")
    parser.add_argument("--total-timesteps", type=int, default=150000, help="Nombre total de pas d'entraînement")
    parser.add_argument("--buffer-size", type=int, default=2048, help="Taille du RolloutBuffer avant chaque update PPO")
    parser.add_argument("--batch-size", type=int, default=64, help="Taille des mini-batchs PPO")
    parser.add_argument("--ppo-epochs", type=int, default=10, help="Nombre d'époques PPO par rollout")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Taille des couches cachées Acteur/Critique")
    parser.add_argument("--lr-actor", type=float, default=1e-3, help="Learning rate initial de l'Acteur")
    parser.add_argument("--lr-critic", type=float, default=2e-3, help="Learning rate initial du Critique")
    parser.add_argument("--gamma", type=float, default=0.95, help="Facteur d'actualisation gamma (0.95 sur Pendulum, 0.99 sur TMRL)")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="Paramètre lambda pour GAE")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="Epsilon de clipping PPO")
    parser.add_argument("--entropy-coef", type=float, default=0.0, help="Coefficient du bonus d'entropie")
    parser.add_argument("--eval-freq", type=int, default=6144, help="Fréquence d'évaluation (en pas de temps)")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire")
    parser.add_argument("--device", type=str, default="cpu", help="Périphérique PyTorch ('cpu' ou 'cuda')")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    run_name = f"PPO_{args.env_id}_seed{args.seed}_{int(time.time())}"
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

    # 1. Création de l'environnement (Module 1)
    env = make_env(args.env_id, seed=args.seed)
    state_dim, action_dim, action_low, action_high = get_env_specs(env)

    print("=" * 72)
    print(f" MTI881 - Groupe 4 | Entraînement PPO sur {args.env_id}")
    print("=" * 72)
    print(f" - Dimension de l'état (state_dim)   : {state_dim}")
    print(f" - Dimension de l'action (action_dim): {action_dim} | Bornes: [{action_low}, {action_high}]")
    print(f" - Périphérique PyTorch              : {args.device}")
    print(f" - Total timesteps                   : {args.total_timesteps}")
    print(f" - Horizon de rollout (buffer_size)  : {args.buffer_size} | Mini-batch: {args.batch_size} | Époques: {args.ppo_epochs}")

    # 2. Évaluation de la politique aléatoire (Baseline)
    random_mean, random_std = evaluate_random_baseline(args.env_id, episodes=10, seed=args.seed)
    print(f" - Baseline Aléatoire (10 épisodes)  : {random_mean:.1f} ± {random_std:.1f}")
    print("=" * 72)

    # 3. Initialisation de l'agent PPO (Modules 2 & 3)
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        hidden_dim=args.hidden_dim,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        normalize_states=True,
        reward_scale=0.1,  # Ramène les récompenses de [-16, 0] vers [-1.6, 0] pour stabiliser le Critique
        device=args.device,
    )

    # Suivi des métriques
    train_steps_history: List[int] = []
    train_returns_history: List[float] = []
    eval_steps_history: List[int] = []
    eval_returns_history: List[float] = []
    eval_stds_history: List[float] = []

    best_eval_return = -float("inf")
    state, _ = env.reset(seed=args.seed)
    episode_return = 0.0
    episode_count = 0
    start_time = time.time()

    # Évaluation initiale à t = 0
    init_mean, init_std = evaluate_agent(agent, args.env_id, episodes=5, seed=args.seed + 1000)
    eval_steps_history.append(0)
    eval_returns_history.append(init_mean)
    eval_stds_history.append(init_std)
    writer.add_scalar("eval/mean_return", init_mean, 0)

    # 4. Boucle principale d'interaction avec l'environnement
    for step in range(1, args.total_timesteps + 1):
        # Sélection d'action selon la politique courante pi_theta(a|s)
        env_action, raw_action, log_prob, value = agent.select_action(
            state, deterministic=False, update_stats=True
        )

        # Exécution dans l'environnement Gymnasium
        next_state, reward, terminated, truncated, _ = env.step(env_action)
        done = terminated or truncated
        episode_return += float(reward)

        # Valeur de l'état suivant V(s_{t+1}) pour calculer le résidu TD dans GAE
        next_value = agent.get_value(next_state)

        # Stockage de la transition dans le RolloutBuffer (avec mise à l'échelle de la récompense)
        norm_state = agent.preprocess_state(state, update_stats=False)
        agent.buffer.add(
            state=norm_state,
            action=raw_action,
            log_prob=log_prob,
            reward=float(reward) * agent.reward_scale,
            value=value,
            next_value=next_value,
            terminated=terminated,
            truncated=truncated,
        )

        state = next_state

        # Gestion de la fin d'épisode
        if done:
            episode_count += 1
            train_steps_history.append(step)
            train_returns_history.append(episode_return)
            writer.add_scalar("train/episode_return", episode_return, step)

            state, _ = env.reset()
            episode_return = 0.0

        # Mise à jour PPO dès que le RolloutBuffer est plein (`buffer_size` pas collectés)
        if step % args.buffer_size == 0:
            # Décroissance linéaire du learning rate (permet d'affiner l'angle à 0.0° exact)
            frac = max(1.0 - (step / args.total_timesteps), 0.05)
            for pg in agent.actor_optimizer.param_groups:
                pg["lr"] = args.lr_actor * frac
            for pg in agent.critic_optimizer.param_groups:
                pg["lr"] = args.lr_critic * frac

            metrics = agent.update()
            writer.add_scalar("loss/actor_loss", metrics["actor_loss"], step)
            writer.add_scalar("loss/critic_loss", metrics["critic_loss"], step)
            writer.add_scalar("policy/entropy", metrics["entropy"], step)
            writer.add_scalar("policy/approx_kl", metrics["approx_kl"], step)
            writer.add_scalar("policy/std", metrics["policy_std"], step)

        # Évaluation périodique et sauvegarde des checkpoints
        if step % args.eval_freq == 0 or step == args.total_timesteps:
            eval_mean, eval_std = evaluate_agent(
                agent, args.env_id, episodes=5, seed=args.seed + 1000
            )
            eval_steps_history.append(step)
            eval_returns_history.append(eval_mean)
            eval_stds_history.append(eval_std)
            writer.add_scalar("eval/mean_return", eval_mean, step)
            writer.add_scalar("eval/std_return", eval_std, step)

            recent_train = (
                np.mean(train_returns_history[-10:])
                if len(train_returns_history) > 0
                else float("nan")
            )
            elapsed = time.time() - start_time
            fps = int(step / max(elapsed, 1e-6))

            save_marker = ""
            if eval_mean > best_eval_return:
                best_eval_return = eval_mean
                agent.save("checkpoints/ppo_pendulum_best.pt")
                save_marker = " ★ [Nouveau meilleur modèle]"

            agent.save("checkpoints/ppo_pendulum_latest.pt")

            print(
                f"Step {step:7d}/{args.total_timesteps} | "
                f"Train (10 ép.): {recent_train:7.1f} | "
                f"Eval: {eval_mean:7.1f} ± {eval_std:5.1f} | "
                f"Vitesse: {fps:4d} steps/s{save_marker}"
            )

    env.close()
    writer.close()

    # 5. Génération de la courbe d'apprentissage finale
    curve_path = "results/ppo_pendulum_learning_curve.png"
    plot_learning_curve(
        train_steps=train_steps_history,
        train_returns=train_returns_history,
        eval_steps=eval_steps_history,
        eval_returns=eval_returns_history,
        eval_stds=eval_stds_history,
        random_baseline=random_mean,
        save_path=curve_path,
        env_id=args.env_id,
    )

    print("=" * 72)
    print(" Entraînement terminé avec succès !")
    print(f" - Baseline Aléatoire          : {random_mean:.1f} ± {random_std:.1f}")
    print(f" - Meilleure Évaluation PPO    : {best_eval_return:.1f}")
    print(" - Meilleur modèle sauvegardé  : checkpoints/ppo_pendulum_best.pt")
    print(f" - Courbe d'apprentissage      : {curve_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
