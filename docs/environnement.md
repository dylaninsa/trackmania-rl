# Environnement TrackMania

Contrat de `envs/trackmania.py` : ce que reçoivent et envoient les agents.

```python
from envs.trackmania import make_env
env = make_env()  # TrackMania doit être ouvert sur la piste tmrl-test
```

Vérification avec le jeu : `uv run python scripts/check_env.py` (options : `--help`).

**Observation** : vecteur `float32` de dimension 83. Les tranches sont définies
dans `envs/trackmania.py` (`SPEED`, `LIDAR`, `LAST_LIDAR`, `ACTIONS`) ; utilisez-les
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

**Récompense** : calculée par `TrajectoryReward` (`envs/reward.py`). C'est le
nombre de points de la trajectoire franchis depuis le pas précédent ÷ 100, soit
1 unité pour 10 m, plus un bonus de +100 au pas de l'arrivée. Elle n'est jamais
négative : reculer ne rapporte rien.

**Fin d'épisode**
- `terminated=True` dans trois cas :
  - arrivée franchie, avec le bonus de +100 ;
  - aucune progression pendant 10 pas consécutifs, après 70 pas de grâce au
    départ ;
  - écart de plus de 30 m avec la trajectoire. En longeant le mur sur
    `tmrl-test`, la voiture atteint 20 m d'écart : le seuil garde 10 m de marge.
- `truncated=True` : `ep_max_length` atteint (3000 pas = 150 s par défaut). Cette
  limite laisse au PID, qui met environ 2 330 pas, le temps de finir le parcours ;
  utiliser la même valeur pour toutes les politiques comparées.

Les réglages de la récompense sont les paramètres de `TrajectoryReward`. La
section `REWARD_CONFIG` de `config.json` n'est plus utilisée.

**`info`** : `speed`, `end_of_track`, `progress` (0 → 1),
`distance_m`, `dist_to_traj`.

**Contraintes temps réel**
- Le jeu ne se met pas en pause entre deux `step()` : si le calcul dépasse
  0,05 s, la voiture continue avec l'action précédente. Faire les mises à jour
  de gradient entre les épisodes.
- En fin d'épisode, l'environnement se met en pause tout seul jusqu'au
  prochain `reset()`.
- `reset(seed=...)` ne rend pas le jeu déterministe : la graine ne contrôle
  que la politique et les réseaux.

La trajectoire `data/reward/tmrl-test.pkl` a été enregistrée par nos soins sur
`tmrl-test` (20 589 points espacés de 0,1 m, 2 059 m). Celle fournie par TMRL ne
correspond pas à la piste : l'arrivée tombait à 72,6 % de progression.

**Enregistrer une trajectoire** : `uv run python -m tmrl --record-reward` enregistre
dès son lancement et s'arrête à l'arrivée. Faire un seul parcours sans relancer la
course : un respawn est enregistré comme une ligne droite vers le point de
réapparition, et la progression se bloque ensuite à cet endroit. En cas d'erreur,
arrêter avec Ctrl+C et recommencer.
