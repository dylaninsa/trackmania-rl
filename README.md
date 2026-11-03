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
| `scripts/`   | Points d'entrée : entraînement, évaluation, test d'intégration SAC | Membres 3 et 4 |
| `configs/`   | Hyperparamètres et graines des expériences                         | Membre 4 |
| `data/reward/` | Trajectoire de démonstration utilisée par la récompense          | Membre 1 |
| `tests/`     | Tests unitaires (exécutables sans le jeu)                          | Tous |

## Installation

Le projet est géré avec [uv](https://docs.astral.sh/uv/). L'environnement virtuel
est créé dans `.venv/` à la racine du dépôt ; il n'est pas versionné, mais
`uv.lock` garantit que tout le monde a les mêmes versions.

pour installer uv :
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

(Après l'installation, fermez et rouvrez votre terminal pour que la commande uv soit reconnue).

Une fois le terminal rouvert, et après s'être placé à la racine du projet : 
```bash
uv sync
```

Pour ajouter une dépendance : `uv add <paquet>`, puis commiter `pyproject.toml`
et `uv.lock`.

L'installation de TrackMania 2020 et d'OpenPlanet suit le
[guide TMRL](https://github.com/trackmania-rl/tmrl/blob/master/readme/Install.md).
Après le premier `uv sync`, lancer `uv run python -m tmrl --install`.

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

## Politiques de référence

Deux baselines, dans `baselines/policies.py`, partagent l'interface des futurs
agents : `act(obs)` renvoie l'action, `reset()` est appelée en début d'épisode.

- **`RandomPolicy(seed)`** : chaque action est tirée uniformément dans [-1, 1].
  Gaz et frein étant chacun actifs la moitié du temps, la voiture avance à peine.
  C'est la borne basse.
- **`PIDPolicy`** : suivi du centre de la piste à partir du dernier balayage LIDAR.
  - Direction : PID sur l'erreur « place à droite − place à gauche » (moyenne des
    faisceaux 0 à 8 moins celle des faisceaux 10 à 18).
  - Vitesse visée : de `min_speed` à `max_speed` selon le faisceau avant (9),
    multipliée par `1 − |direction|` pour ralentir en braquant, et jamais sous
    `min_speed`. Gaz sous la vitesse visée, frein au-delà de `brake_margin`.
  - Réglages retenus : `kp=6`, `ki=0`, `kd=0.6`, `min_speed=10`, `max_speed=60`,
    `brake_margin=5`.

**Évaluation** : `scripts/evaluate.py` fait tourner une politique en jeu et
enregistre un CSV par exécution dans `results/` (non versionné).

```bash
uv run python scripts/evaluate.py --policy pid --episodes 5
uv run python scripts/evaluate.py --policy random --episodes 10 --seed 0
```

**Résultats sur `tmrl-test`** (`ep_max_length = 3000`) :

| Politique | Épisodes | Distance moyenne | Parcours terminés | Temps de parcours |
|-----------|----------|------------------|-------------------|-------------------|
| Aléatoire (graines 0, 1, 2) | 3 × 10 | 4,7 m (± 1,2 entre graines) | 0/30 | — |
| PID | 5 | 2 058,8 m | 5/5 | 116,4 s (± 0,4) |

- Aléatoire : moyennes par graine de 6,1, 3,9 et 4,0 m ; au mieux 19,1 m. La
  plupart des épisodes s'arrêtent au pas 80, soit les 70 pas de grâce plus 10
  pas sans progression.
- PID : temps de parcours entre 115,95 et 116,90 s. Il est déterministe, mais le
  jeu ne l'est pas tout à fait, d'où ces petites variations.

Limite connue du PID : il réagit à ce qu'il voit sans anticiper, et frôle le mur
à environ 10 m/s dans le virage serré situé vers 1 659 m.

**Replay buffer**
`algos/replay_buffer.py` stocke les transitions pour SAC. La boucle d'entraînement y ajoute chaque pas et tire des lots qu'elle passe à `SACAgent.update`.

```python
from algos.replay_buffer import ReplayBuffer
buffer = ReplayBuffer(obs_dim=83, act_dim=3, capacity=100_000, device="cpu", seed=0)
buffer.add(obs, act, rew, next_obs, terminated)
batch = buffer.sample(256)  # dict de tenseurs float32 : obs, act, rew, next_obs, terminated
```

Choix de conception :
- **Tampon circulaire** de tableaux numpy préalloués : une fois plein, les nouvelles transitions écrasent les plus anciennes. Une transition occupe environ 0,7 Ko (171 flottants), soit 68 Mo pour 100 000 transitions et 0,7 Go pour 1 million.
- **Seul `terminated` est stocké, pas `truncated`**. Un épisode tronqué à `ep_max_length` aurait pu continuer : sa dernière transition doit rester bootstrapée dans la cible de Bellman. Les épisodes terminés par un échec ou par l'arrivée coupent le futur.
- **`rew` et `terminated` ont la forme `(B,)`, pas `(B, 1)`**. Les critiques renvoient des Q de forme `(B,)` : avec une cible `(B, 1)`, le broadcasting produirait une matrice `(B, B)` sans aucune erreur, et la perte serait fausse.
- **`seed` optionnel** : `torch.manual_seed` ne contrôle pas le tirage des lots (numpy). Sans cette graine, deux entraînements avec la même graine divergeraient, ce qui contredit le protocole « plusieurs graines ».
- `sample` lève une `ValueError` si le tampon est vide.

Tests (sans le jeu) : 
```bash
uv run pytest tests/test_replay_buffer.py.
```

## Test d'intégration SAC et TrackMania
 
`scripts/sac_smoke_test.py` vérifie sur le vrai jeu que la chaîne environnement → replay buffer → agent SAC
fonctionne, avant de lancer de longs entraînements. **Ce n'est pas un script d'entraînement** : l'agent est
beaucoup trop peu entraîné pour apprendre quoi que ce soit.
 
```bash
uv run python scripts/sac_smoke_test.py
uv run python scripts/sac_smoke_test.py --device cuda --episodes 5   # options : --help
```
 
Même préparation que `check_env.py` (jeu ouvert sur `tmrl-test`, fenêtré, vue cockpit, clic dans la
fenêtre). Le script se déroule en quatre phases :
 
1. **Warm-up** : actions aléatoires pour remplir le tampon.
2. **Épisodes SAC** (politique stochastique, 300 pas maximum) avec des mises à jour de gradient **entre**
   les épisodes uniquement, puisque le jeu ne se met pas en pause pendant un épisode.
3. **Évaluation** : un épisode avec la politique déterministe.
4. **Checkpoint** : `save` puis `load` dans un nouvel agent doivent donner la même action.
Il vérifie le contrat de l'environnement (observation de 83 `float32`, 4 balayages identiques et actions
nulles au `reset`, clés de `info`, récompense jamais négative), la validité des actions, la finitude des
métriques de `update`, et mesure la durée de `act()`, d'un pas et d'une mise à jour. Les violations sont
listées en fin d'exécution et le code de sortie vaut 1 s'il y en a.
 
**Résultats** (machine d'Antoine, CPU ; réseaux de 256 unités, lots de 256 ; réglages provisoires fixés
dans le script, à remplacer par `configs/`) :
 
| Mesure | Valeur | Lecture |
|--------|--------|---------|
| Durée d'un pas avec SAC | 50,0 à 50,3 ms en moyenne, pics de 55 à 67 ms | Identique à `check_env.py` (50,0 ms) : `act()` ne fait pas dépasser le budget de 0,05 s |
| `act()` | environ 1,2 ms | Négligeable devant le pas |
| Mise à jour (CPU) | environ 12,6 ms | 3 000 mises à jour, soit un épisode complet de 3 000 pas, coûtent environ 38 s, contre 150 s de jeu |
| Agent non entraîné | 0,5 à 2,5 m (stochastique), 0,0 m (déterministe), fin au pas 80 à 86 | Même comportement que la politique aléatoire (4 à 8 m) : attendu |
| Checkpoint | identique | `save`/`load` conservent la politique |
 
Ce qu'on en retient pour la boucle d'entraînement :
 
- **Ratio mises à jour / données.** Au ratio 1:1, les mises à jour entre les épisodes ajoutent environ 25 %
  de temps mort (38 s pour 150 s de jeu) sur CPU. C'est acceptable, et à remesurer sur la machine GPU.
- **Pics de durée de pas.** Sous SAC, les pas les plus longs (jusqu'à 67 ms) dépassent un peu ceux de
  `check_env.py` (52 à 61 ms). Un pas de plus de 50 ms fait rouler la voiture avec l'action précédente :
  à surveiller pendant les vrais entraînements.
- **Agent déterministe immobile.** Un réseau non entraîné sort une action proche de zéro : le gaz n'est pas
  positif, la voiture ne bouge pas, et l'épisode s'arrête au pas 80 (70 pas de grâce plus 10 sans
  progression).
Limites : le test ne valide pas l'apprentissage. SAC doit encore être validé sur Pendulum-v1 avant de
tirer une conclusion sur les performances. La troncation à `ep_max_length` est surtout couverte par
`check_env.py --ep-max-length 100`, car des épisodes de 300 pas sont presque toujours terminés avant par un
échec.


## Méthode de travail

- Une branche par fonctionnalité, fusionnée dans `main` par Pull Request.
- Chaque PR est relue par au moins un autre membre.
- Les configurations et graines des expériences rapportées sont versionnées dans `configs/`.
