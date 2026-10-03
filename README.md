# TrackMania RL — MTI881, Groupe 4

Agent d'apprentissage par renforcement codé de zéro (PPO et SAC) pour boucler un
circuit TrackMania 2020 à partir d'observations LIDAR. [TMRL](https://github.com/trackmania-rl/tmrl)
n'est utilisé que pour son interface Gymnasium avec le jeu.

## Structure

| Dossier      | Contenu                                                            | Référent |
|--------------|--------------------------------------------------------------------|----------|
| `envs/`      | Lien avec TrackMania via TMRL, aplatissement des observations, récompense | Membre 1 |
| `baselines/` | Agent aléatoire et heuristique PID sur LIDAR                       | Membre 1 |
| `networks/`  | Réseaux Acteur / Critique (PyTorch)                                | Membre 2 |
| `algos/`     | PPO et SAC                                                         | Membre 2 |
| `training/`  | Boucle d'entraînement, logs, checkpoints                           | Membre 3 |
| `scripts/`   | Points d'entrée : entraînement, évaluation                         | Membres 3 et 4 |
| `configs/`   | Hyperparamètres et graines des expériences                         | Membre 4 |
| `data/reward/` | Trajectoire de démonstration utilisée par la récompense          | Membre 1 |
| `tests/`     | Tests unitaires (exécutables sans le jeu)                          | Tous |

## Installation

Le projet est géré avec [uv](https://docs.astral.sh/uv/). L'environnement virtuel
est créé dans `.venv/` à la racine du dépôt ; il n'est pas versionné, mais
`uv.lock` garantit que tout le monde a les mêmes versions.

```bash
uv sync                       # développement hors jeu (Pendulum-v1)
uv sync --extra trackmania    # machine qui lance TrackMania (ajoute TMRL)
```

Pour ajouter une dépendance : `uv add <paquet>`, puis commiter `pyproject.toml`
et `uv.lock`.

L'installation de TrackMania 2020 et d'OpenPlanet suit le
[guide TMRL](https://github.com/trackmania-rl/tmrl/blob/master/readme/Install.md).
Après le premier `uv sync --extra trackmania`, lancer `uv run python -m tmrl --install`.

## Exécution et tests

```bash
uv run python scripts/<script>.py
uv run pytest
```

## Contrat de l'environnement

À compléter par le membre 1 : forme de l'observation, sémantique des actions,
échelle de la récompense, différence `terminated` / `truncated`.

## Méthode de travail

- Une branche par fonctionnalité, fusionnée dans `main` par Pull Request.
- Chaque PR est relue par au moins un autre membre.
- Les configurations et graines des expériences rapportées sont versionnées dans `configs/`.
