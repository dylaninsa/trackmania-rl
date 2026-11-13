"""
Script d'évaluation et de démonstration visuelle de l'agent PPO entraîné.

Usage :
  - Évaluation statistique (sans fenêtre graphique) :
      python evaluate_ppo.py --episodes 10
  - Générer un extrait visuel (séquence d'images + télémétrie + GIF animé) :
      python evaluate_ppo.py --save-visual
  - Visualisation en temps réel dans une fenêtre :
      python evaluate_ppo.py --render --episodes 3
"""

import argparse
import os
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.environment import get_env_specs, make_env
from src.ppo import PPOAgent
from train_ppo import evaluate_random_baseline


def save_visual_extract(
    agent: PPOAgent,
    env_id: str = "Pendulum-v1",
    seed: int = 2028,
    frames_path: str = "results/ppo_pendulum_visual_extract.png",
    gif_path: str = "results/ppo_pendulum_demo.gif",
) -> None:
    """
    Capture un épisode complet en mode 'rgb_array' et génère :
    1. Une planche visuelle montrant les images du pendule à différents instants clés
       ainsi que les courbes physiques (angle theta, vitesse angulaire, couple appliqué, récompense).
    2. Une animation GIF de l'épisode complet.
    """
    os.makedirs(os.path.dirname(frames_path), exist_ok=True)
    env = make_env(env_id, render_mode="rgb_array", seed=seed)

    state, _ = env.reset(seed=seed)
    frames: List[np.ndarray] = []
    angles_deg: List[float] = []
    ang_vels: List[float] = []
    torques: List[float] = []
    rewards: List[float] = []

    done = False
    while not done:
        frame = env.render()
        frames.append(frame)

        # state = [cos(theta), sin(theta), theta_dot]
        cos_th, sin_th, th_dot = state[0], state[1], state[2]
        angle = float(np.degrees(np.arctan2(sin_th, cos_th)))
        angles_deg.append(angle)
        ang_vels.append(float(th_dot))

        env_action, _, _, _ = agent.select_action(state, deterministic=True, update_stats=False)
        torques.append(float(env_action[0]))

        state, reward, terminated, truncated, _ = env.step(env_action)
        rewards.append(float(reward))
        done = terminated or truncated

    env.close()

    # 1. Sauvegarde du GIF animé (1 image sur 2 pour fluidité, 25 fps)
    pil_frames = [Image.fromarray(f) for f in frames[::2]]
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=40,
        loop=0,
    )

    # 2. Création de la planche d'analyse visuelle (8 instants clés + 3 courbes physiques)
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(3, 8, height_ratios=[1.2, 1.0, 1.0], hspace=0.45, wspace=0.2)

    key_steps = [0, 10, 20, 30, 45, 70, 120, 199]
    for idx, step_i in enumerate(key_steps):
        ax_img = fig.add_subplot(gs[0, idx])
        ax_img.imshow(frames[step_i])
        ax_img.set_title(
            f"t = {step_i} ({step_i*0.05:.1f}s)\nθ = {angles_deg[step_i]:+.0f}°",
            fontsize=9.5,
            fontweight="bold" if step_i >= 45 else "normal",
        )
        ax_img.axis("off")

    steps_x = np.arange(len(angles_deg))

    # Sous-graphique 1 : Angle du pendule (0° = équilibre haut vertical)
    ax_angle = fig.add_subplot(gs[1, :4])
    ax_angle.plot(steps_x, angles_deg, color="royalblue", linewidth=2.0, label="Angle θ (°)")
    ax_angle.axhline(0, color="forestgreen", linestyle="--", linewidth=1.5, label="Cible verticale (0°)")
    ax_angle.set_ylabel("Angle θ (degrés)")
    ax_angle.set_title("Stabilisation de l'angle à la verticale (0° = haut, ±180° = bas)", fontsize=10.5)
    ax_angle.set_ylim(-190, 190)
    ax_angle.grid(True, linestyle=":", alpha=0.6)
    ax_angle.legend(loc="upper right", fontsize=9)

    # Sous-graphique 2 : Couple moteur envoyé par l'Acteur PPO
    ax_torque = fig.add_subplot(gs[1, 4:])
    ax_torque.plot(steps_x, torques, color="darkorange", linewidth=1.8, label="Action PPO : Couple u (N·m)")
    ax_torque.axhline(0, color="gray", linestyle=":", linewidth=1.0)
    ax_torque.set_ylabel("Couple u ∈ [-2, +2]")
    ax_torque.set_title("Commandes continues choisies par l'Acteur PPO", fontsize=10.5)
    ax_torque.set_ylim(-2.2, 2.2)
    ax_torque.grid(True, linestyle=":", alpha=0.6)
    ax_torque.legend(loc="upper right", fontsize=9)

    # Sous-graphique 3 : Récompense instantanée à chaque pas de temps
    ax_rew = fig.add_subplot(gs[2, :])
    ax_rew.plot(steps_x, rewards, color="seagreen", linewidth=2.0, label="Récompense instantanée r_t (0 = optimal)")
    ax_rew.fill_between(steps_x, rewards, 0, color="seagreen", alpha=0.15)
    ax_rew.set_xlabel("Pas de temps de l'épisode (200 pas = 10 secondes)")
    ax_rew.set_ylabel("Récompense r_t")
    ax_rew.set_title(
        f"Évolution de la récompense sur l'épisode (Total cumulé = {sum(rewards):.1f})",
        fontsize=10.5,
    )
    ax_rew.set_ylim(-17, 0.5)
    ax_rew.grid(True, linestyle=":", alpha=0.6)
    ax_rew.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        "Démonstration Visuelle & Télémétrie de l'Agent PPO sur Pendulum-v1 (MTI881 - Groupe 4)",
        fontsize=13,
        fontweight="bold",
    )
    plt.savefig(frames_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f" [Visuel sauvegardé] Planche complète : {frames_path}")
    print(f" [Animation GIF]     Démonstration    : {gif_path}")


def save_before_after_comparison(
    trained_agent: PPOAgent,
    env_id: str = "Pendulum-v1",
    seed: int = 2028,
    save_path: str = "results/comparaison_avant_apres_ppo.png",
) -> None:
    """
    Prouve visuellement l'effet de l'entraînement en comparant sur la MÊME graine :
    - Un agent PPO neuf (Step 0, poids aléatoires avant entraînement)
    - L'agent PPO entraîné (chargé depuis checkpoints/ppo_pendulum_best.pt)
    """
    env = make_env(env_id, render_mode="rgb_array", seed=seed)
    state_dim, action_dim, action_low, action_high = get_env_specs(env)
    untrained_agent = PPOAgent(state_dim, action_dim, action_low, action_high, device="cpu")

    def run_episode(ag: PPOAgent):
        s, _ = env.reset(seed=seed)
        imgs, angs, rews = [], [], []
        done = False
        while not done:
            imgs.append(env.render())
            angs.append(float(np.degrees(np.arctan2(s[1], s[0]))))
            act, _, _, _ = ag.select_action(s, deterministic=True, update_stats=False)
            s, r, term, trunc, _ = env.step(act)
            rews.append(float(r))
            done = term or trunc
        return imgs, angs, rews

    imgs_untrained, angs_untrained, rews_untrained = run_episode(untrained_agent)
    imgs_trained, angs_trained, rews_trained = run_episode(trained_agent)
    env.close()

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.1, 1.1, 1.1], hspace=0.45, wspace=0.2)
    key_steps = [0, 20, 45, 80, 140, 199]

    for i, st in enumerate(key_steps):
        ax1 = fig.add_subplot(gs[0, i])
        ax1.imshow(imgs_untrained[st])
        ax1.set_title(f"Avant (t={st})\nθ = {angs_untrained[st]:+.0f}°", fontsize=9, color="crimson")
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[1, i])
        ax2.imshow(imgs_trained[st])
        ax2.set_title(f"Après PPO (t={st})\nθ = {angs_trained[st]:+.0f}°", fontsize=9, color="forestgreen", fontweight="bold")
        ax2.axis("off")

    ax_curve = fig.add_subplot(gs[2, :])
    ax_curve.plot(angs_untrained, color="crimson", linestyle="--", alpha=0.8, label=f"Avant entraînement (Step 0, poids initiaux) — Score: {sum(rews_untrained):.0f}")
    ax_curve.plot(angs_trained, color="forestgreen", linewidth=2.2, label=f"Après entraînement PPO (150 000 pas) — Score: {sum(rews_trained):.0f}")
    ax_curve.axhline(0, color="black", linestyle=":", linewidth=1.2, label="Verticale exacte (0°)")
    ax_curve.set_ylabel("Angle θ (°)")
    ax_curve.set_xlabel("Pas de temps de l'épisode (0 à 200)")
    ax_curve.set_ylim(-190, 190)
    ax_curve.grid(True, linestyle=":", alpha=0.6)
    ax_curve.legend(loc="lower right", fontsize=9.5)
    fig.suptitle("Preuve de l'apprentissage : Réseau avant entraînement (Step 0) vs Après entraînement PPO", fontsize=12.5, fontweight="bold")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f" [Preuve visuelle]    Avant / Après    : {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Évaluer et visualiser l'agent PPO")
    parser.add_argument("--env-id", type=str, default="Pendulum-v1", help="Environnement Gymnasium")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/ppo_pendulum_best.pt",
        help="Chemin du checkpoint PyTorch à charger",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Nombre d'épisodes d'évaluation")
    parser.add_argument("--render", action="store_true", help="Afficher la fenêtre graphique du pendule")
    parser.add_argument("--save-visual", action="store_true", help="Générer une planche visuelle PNG + GIF")
    parser.add_argument("--seed", type=int, default=2026, help="Graine aléatoire d'évaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    render_mode = "human" if args.render else None
    env = make_env(args.env_id, render_mode=render_mode, seed=args.seed)
    state_dim, action_dim, action_low, action_high = get_env_specs(env)

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        device="cpu",
    )
    agent.load(args.checkpoint)

    print("=" * 68)
    print(f" Évaluation de l'agent PPO ({args.checkpoint}) sur {args.env_id}")
    print("=" * 68)

    ppo_returns = []
    for ep in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + ep)
        done = False
        ep_return = 0.0
        steps = 0

        while not done:
            env_action, _, _, _ = agent.select_action(
                state, deterministic=True, update_stats=False
            )
            state, reward, terminated, truncated, _ = env.step(env_action)
            ep_return += float(reward)
            steps += 1
            done = terminated or truncated

        ppo_returns.append(ep_return)
        print(f" - Épisode {ep:2d}/{args.episodes} : Récompense = {ep_return:7.2f} ({steps} pas)")

    env.close()

    random_mean, random_std = evaluate_random_baseline(
        args.env_id, episodes=args.episodes, seed=args.seed
    )
    ppo_mean = float(np.mean(ppo_returns))
    ppo_std = float(np.std(ppo_returns))

    print("-" * 68)
    print(f" Résultat Baseline Aléatoire : {random_mean:7.2f} ± {random_std:6.2f}")
    print(f" Résultat Agent PPO          : {ppo_mean:7.2f} ± {ppo_std:6.2f}")
    print(f" Gain net d'apprentissage    : +{ppo_mean - random_mean:7.2f} points")
    print("=" * 68)

    if args.save_visual:
        save_visual_extract(agent, env_id=args.env_id, seed=args.seed + 2)
        save_before_after_comparison(agent, env_id=args.env_id, seed=args.seed + 2)


if __name__ == "__main__":
    main()

