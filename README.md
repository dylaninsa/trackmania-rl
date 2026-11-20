# TrackMania RL — MTI881, Groupe 4

Agents d'apprentissage par renforcement codés de zéro (PPO et SAC) pour boucler un circuit TrackMania 2020 à
partir d'observations LIDAR. [TMRL](https://github.com/trackmania-rl/tmrl) n'est utilisé que pour son
interface Gymnasium avec le jeu.

**Où on en est :** sur la piste `tmrl-test` (2 059 m), SAC termine le parcours sur les 3 graines dès
60 000 pas, alors que PPO ne le termine jamais. SAC est en revanche instable : deux graines sur trois
s'effondrent en fin de run. La phase 3 cherche à le stabiliser (voir
[`experiments/journal.md`](experiments/journal.md)).

## Structure

| Dossier | Contenu |
|---------|---------|
| `envs/` | Lien avec TrackMania via TMRL, observations, récompense |
| `baselines/` | Politique aléatoire et heuristique PID sur LIDAR |
| `networks/`, `algos/` | Réseaux PyTorch, PPO, SAC et replay buffer |
| `training/` | Éléments communs aux scripts d'entraînement : environnements, évaluation, checkpoints |
| `scripts/` | Entraînement, file d'attente, évaluation, vérifications |
| `configs/` | Hyperparamètres des expériences |
| `experiments/` | Journal, files d'attente et résultats de chaque run (CSV, log, TensorBoard) |
| `docs/` | Documentation détaillée et résultats des validations, pour le rapport |
| `data/reward/` | Trajectoire de démonstration utilisée par la récompense |
| `tests/` | Tests unitaires, exécutables sans le jeu |

## Installation

1. Installer [uv](https://docs.astral.sh/uv/), puis fermer et rouvrir le terminal :
   ```bash
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
2. À la racine du dépôt :
   ```bash
   uv sync
   uv run python -m tmrl --install
   ```
3. Installer TrackMania 2020 et OpenPlanet en suivant le
   [guide TMRL](https://github.com/trackmania-rl/tmrl/blob/master/readme/Install.md).
4. Vérifier : `uv run pytest` (sans le jeu), puis `uv run python scripts/check_env.py` (avec le jeu).

`uv.lock` fixe les versions pour tout le monde. Pour ajouter une dépendance : `uv add <paquet>`, puis commiter
`pyproject.toml` et `uv.lock`.

**Préparer le jeu avant chaque script qui l'utilise :** TrackMania ouvert sur la piste `tmrl-test`, en
fenêtré, vue cockpit, plugin TMRL chargé ; cliquer dans la fenêtre du jeu juste après avoir lancé le script.
Le jeu doit rester au premier plan pendant tout l'entraînement.

## Utilisation

```bash
# Un entraînement
uv run python scripts/train_sac.py --config configs/sac_trackmania.json --seed 0
uv run python scripts/train_ppo.py --config configs/ppo_trackmania.json --seed 0

# Plusieurs entraînements à la suite, sans surveillance (relançable après une coupure)
uv run python scripts/run_queue.py experiments/queue_phase3a_seed0.json

# Évaluer un agent entraîné, ou une politique de référence
uv run python scripts/evaluate.py --policy sac --config configs/sac_trackmania.json --checkpoint checkpoints/sac_trackmania_seed0_best.pt --episodes 10
uv run python scripts/evaluate.py --policy pid --episodes 5

# Suivre les courbes de tous les runs partagés
uv run tensorboard --logdir experiments
```

Chaque run s'appelle `<config>_seed<graine>` et produit :

| Fichier | Contenu | Versionné |
|---------|---------|-----------|
| `results/<run>.csv` | Évaluations pendant l'entraînement : distance, taux d'arrivée, retour | non |
| `checkpoints/<run>.pt` et `<run>_best.pt` | Dernier agent et meilleur agent (en distance) | non, sur le Drive |
| `runs/<run>/` | Logs TensorBoard | non |
| `experiments/<run>/` | Copie du CSV, du log et des logs TensorBoard, faite par `run_queue.py` | oui |

## Documentation

- [Environnement](docs/environnement.md) : observation, action, récompense, fins d'épisode, contraintes temps
  réel, enregistrement d'une trajectoire.
- [Politiques de référence](docs/politiques_reference.md) : politique aléatoire et PID, avec leurs résultats.
- [Validation sur Pendulum-v1](docs/validation_pendulum.md) : PPO, replay buffer et SAC, protocole et
  résultats.
- [Test d'intégration SAC et TrackMania](docs/test_integration_sac.md) : durées mesurées en jeu.
- [Entraînement sur TrackMania](docs/entrainement_trackmania.md) : boucle, checkpoints et résultats de la
  phase 1.
- [Journal d'expériences](experiments/journal.md) : tous les runs, phase par phase.

## Méthode de travail

- Une branche par fonctionnalité, fusionnée dans `main` par Pull Request relue par un autre membre.
- Les configurations et graines des expériences rapportées sont versionnées dans `configs/` et
  `experiments/`.
- Chaque run, même raté, a sa ligne dans `experiments/journal.md`.
