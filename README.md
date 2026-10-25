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

```python
from envs.trackmania import make_env
env = make_env()  # TrackMania doit être ouvert sur la piste tmrl-test
```

Vérification avec le jeu : `uv run python scripts/check_env.py` (options : `--help`).

**Observation** : vecteur `float32` de dimension 83. Les tranches sont définies
dans `envs/wrappers.py` (`SPEED`, `LIDAR`, `LAST_LIDAR`, `ACTIONS`) ; utilisez-les
plutôt que des indices en dur.

| Indices | Contenu | Normalisation |
|---------|---------|---------------|
| `0` | vitesse (m/s) | ÷ `SPEED_SCALE` (100) |
| `1–76` | 4 balayages LIDAR de 19 faisceaux, du plus ancien au plus récent | ÷ `LIDAR_SCALE` (500) |
| `77–82` | 2 dernières actions envoyées, la plus ancienne en premier | aucune |

Dans un balayage, le faisceau 0 pointe à droite, le 9 droit devant, le 18 à gauche.
Au `reset()`, les 4 balayages sont identiques et les actions valent `[0, 0, 0]`.

**Action** : `[gaz, frein, direction]`, chacune dans [-1, 1] (écrêtée avant
l'envoi au jeu). Seules les valeurs positives du gaz et du frein ont un effet ;
direction > 0 tourne à droite. Un pas dure 0,05 s (20 Hz).

**Récompense** (provisoire : celle de TMRL, remplacée à l'étape 5) : nombre de
points de la trajectoire franchis depuis le pas précédent ÷ 100, soit 1 unité
pour 10 m ; +100 à l'arrivée.

**Fin d'épisode**
- `terminated=True` : arrivée, ou aucune progression pendant 10 pas consécutifs
  (après 70 pas de grâce au départ). Ne pas bootstrapper.
- `truncated=True` : `ep_max_length` atteint (1000 pas = 50 s par défaut).
  Bootstrapper normalement.

**`info`** : `pos`, `speed`, `end_of_track`, `traj_index`, `progress` (0 → 1),
`distance_m`, `dist_to_traj`.

**Contraintes temps réel**
- Le jeu ne se met pas en pause entre deux `step()` : si le calcul dépasse
  0,05 s, la voiture continue avec l'action précédente. Faire les mises à jour
  de gradient entre les épisodes.
- En fin d'épisode, l'environnement se met en pause tout seul jusqu'au
  prochain `reset()`.
- `reset(seed=...)` ne rend pas le jeu déterministe : la graine ne contrôle
  que la politique et les réseaux.

La trajectoire `data/reward/tmrl-test.pkl` est celle fournie par TMRL pour la
piste `tmrl-test` (30 273 points espacés de 0,1 m, environ 3 km).

## Méthode de travail

- Une branche par fonctionnalité, fusionnée dans `main` par Pull Request.
- Chaque PR est relue par au moins un autre membre.
- Les configurations et graines des expériences rapportées sont versionnées dans `configs/`.
