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

Pour installer uv sous Windows :

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Fermez et rouvrez ensuite le terminal pour que la commande `uv` soit reconnue, puis, à la racine du projet :

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

## Validation de PPO sur Pendulum-v1

PPO (`algos/ppo.py`, `networks/ppo.py`) est validé sur `Pendulum-v1` avant d'être branché sur le jeu.
Les hyperparamètres sont dans `configs/ppo_pendulum.json`.

```bash
uv run python scripts/train_ppo.py --config configs/ppo_pendulum.json --seed 0
```

Le script écrit les logs TensorBoard dans `runs/`, les évaluations dans `results/ppo_pendulum_seed<N>.csv`
et l'agent dans `checkpoints/ppo_pendulum_seed<N>.pt` (le dernier) et `checkpoints/ppo_pendulum_seed<N>_best.pt`
(le meilleur). Une évaluation joue 10 épisodes avec la politique déterministe, tous les 5 000 pas.

**Critère de validation :** un retour moyen en évaluation supérieur à −250 sur les 3 graines, soit environ
1 000 points de mieux que l'agent aléatoire. Il est moins strict que celui de SAC (−200) : PPO, on-policy,
est connu pour finir moins précisément que SAC sur Pendulum.

**Résultats** (150 000 pas, environ 5 minutes par graine sur CPU) :

| Graine | Retour à 20 000 pas | Retour final |
|--------|---------------------|--------------|
| 0 | −323 | −226 ± 139 |
| 1 | −335 | −210 ± 88 |
| 2 | −368 | −227 ± 86 |

L'agent aléatoire obtient −1 225 ± 268 sur 10 épisodes. PPO apprend en 20 000 pas environ, puis plafonne
entre −210 et −230. L'agent n'est mis à jour qu'en fin d'épisode, une fois au moins `buffer_size` pas
collectés, soit tous les 2 200 pas sur Pendulum. Le grand écart-type vient de quelques positions de départ depuis
lesquelles l'agent met plus longtemps à redresser le pendule.

## Replay buffer de SAC

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
uv run pytest tests/test_replay_buffer.py
```

## Validation de SAC sur Pendulum-v1

Avant de brancher SAC (`algos/sac.py`, `networks/sac.py`, `algos/replay_buffer.py`) sur TrackMania, on vérifie
qu'il apprend sur un environnement léger où le résultat est connu. Si SAC échoue ici, le bug est dans
l'algorithme et non dans le jeu. Cette étape ne demande pas TrackMania et tourne sur Mac, Windows et Linux.

Deux scripts entraînent le même agent avec les mêmes hyperparamètres (`configs/sac_pendulum.json` : 20 000 pas,
1 000 pas aléatoires de warm-up, réseaux de 256 unités, lr 3e-4, γ = 0,99, τ = 0,005, lots de 256, entropie
cible = -1, α appris à partir de 1). Ils ne diffèrent que par le moment des mises à jour :

| Script | Mises à jour | Rôle |
|--------|--------------|------|
| `scripts/validate_sac_pendulum.py` | une à chaque pas, pendant l'épisode | SAC de référence, tel que dans l'article ; impossible dans TrackMania |
| `scripts/train_sac.py` | autant que de pas joués, groupées en fin d'épisode | La boucle utilisée dans TrackMania, où le jeu ne se met pas en pause |

```bash
uv run python scripts/validate_sac_pendulum.py --seeds 0 1 2   # référence + critères (6 à 15 min par graine sur CPU)
uv run python scripts/plot_sac_pendulum.py                     # courbes de la référence dans results/sac_pendulum_reference.png
uv run python scripts/train_sac.py --config configs/sac_pendulum.json --seed 0   # boucle TrackMania
```

### Protocole

- **Pendulum-v1** : observation de dimension 3, action de dimension 1 dans [-2, 2], épisodes de 200 pas. Il ne
  se termine jamais (`terminated` toujours faux), il est seulement tronqué à 200 pas : c'est un bon test du
  choix de ne stocker que `terminated` dans le replay buffer.
- L'agent sort des actions dans [-1, 1] ; les scripts les multiplient par 2 avant de les envoyer à
  l'environnement.
- Tous les 2 000 pas, la politique **déterministe** est évaluée sur 10 épisodes, dans un environnement
  d'évaluation distinct de celui d'entraînement. Les graines d'évaluation dépendent de la graine du run : un
  écart de quelques dizaines de points entre deux graines n'est pas significatif (l'écart type entre épisodes est
  de 65 à 110).
- `validate_sac_pendulum.py` évalue aussi l'agent non entraîné au pas 0, qui doit être proche d'une politique
  aléatoire. Il écrit `results/sac_pendulum_reference_seed<N>.csv` (retour d'évaluation et son écart type,
  retour d'entraînement, moyennes de `critic_loss`, `actor_loss`, `alpha`, `q_mean` et `entropy`) et sort avec le
  code 1 si un critère échoue.
- `train_sac.py` écrit `results/sac_pendulum_seed<N>.csv`, `checkpoints/sac_pendulum_seed<N>.pt`,
  `checkpoints/sac_pendulum_seed<N>_best.pt` et les logs TensorBoard dans `runs/`, comme pour PPO.

### Critères de réussite

Pour chacune des 3 graines (une seule graine qui réussit ne prouve rien) :

| Critère | Seuil |
|---------|-------|
| Métriques de `update` | toujours finies |
| Apprentissage | retour d'évaluation **> -250** au plus tard à 10 000 pas |
| Stabilité | moyenne des deux dernières évaluations **> -250** (pas d'effondrement) |

Repères : une politique aléatoire fait environ **-1 200** par épisode ; un SAC correct se stabilise entre -100 et
-200.

### Résultats

Runs de 20 000 pas, évaluation tous les 2 000 pas (le « premier pas » a donc cette résolution).

| Graine | Référence : premier pas > -250 | Référence : retour final | Boucle TrackMania : premier pas > -250 | Boucle TrackMania : retour final |
|--------|--------------------------------|--------------------------|----------------------------------------|----------------------------------|
| 0 | 4 000 (-154,7) | -109,4 | 4 000 (-236) | -169 ± 92 |
| 1 | 4 000 (-248,4) | -191,2 | 6 000 (-193) | -192 ± 76 |
| 2 | 4 000 (-198,6) | -145,3 | 4 000 (-249) | -191 ± 75 |

Le retour final de la référence est la moyenne des deux dernières évaluations, celui de la boucle TrackMania la
dernière évaluation. Les deux versions respectent les trois critères. Grouper les mises à jour en fin d'épisode
ralentit à peine l'apprentissage et donne des politiques finales un peu moins précises, dans la marge de
variation entre graines : la contrainte du temps réel ne coûte presque rien à SAC. Il apprend en 4 000 à
8 000 pas, soit environ 5 fois moins que PPO : off-policy, il réutilise chaque transition du buffer de
nombreuses fois, alors que PPO jette son rollout après chaque mise à jour.

Les durées de la référence (388 à 929 s par graine, machine d'Antoine non dédiée au calcul) ne mesurent pas le
coût d'entraînement : voir la mesure de `sac_smoke_test.py`, environ 12,6 ms par mise à jour.

Limite : cette validation montre que SAC apprend sur Pendulum-v1, pas qu'il apprendra sur TrackMania
(observations de dimension 83, récompense creuse, contrainte temps réel).

### Lecture des courbes

- **Retour d'évaluation** : doit monter de -1 200 vers -150 environ, puis rester stable.
- **α** : part de 1 (`init_alpha`) et diminue progressivement.
- **Entropie** : baisse vers la cible (-1 en dimension 1) quand α se stabilise.
- **Q moyen** : doit rester borné. Une explosion signale un problème de cible de Bellman.

### Si SAC n'apprend pas

Vérifier dans cet ordre :

1. Échelle de l'action (×2 oublié ou appliqué deux fois).
2. Correction tanh du log-probabilité dans `networks/sac.py`.
3. Formes des tenseurs : `rew`, `terminated` et Q doivent être de forme `(B,)`, sinon un `(B, B)` silencieux fausse la perte.
4. Cible de Bellman : réseaux cibles, action échantillonnée sur `next_obs`, masque `(1 - terminated)`, sous `torch.no_grad()`.
5. Mise à jour de Polyak (τ = 0,005) et non une copie complète.
6. Gradients : l'acteur ne doit pas modifier les critiques ; α est optimisé via `log_alpha`.
7. Évaluation : `deterministic=True` doit utiliser `tanh(moyenne)`.

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

Limites : le test ne valide pas l'apprentissage (voir la validation sur Pendulum-v1 ci-dessus). La troncation
à `ep_max_length` est surtout couverte par `check_env.py --ep-max-length 100`, car des épisodes de 300 pas
sont presque toujours terminés avant par un échec.

## Entraînement sur TrackMania

Les deux scripts d'entraînement choisissent l'environnement d'après le champ `env` de la configuration :
`"trackmania"` crée l'environnement du jeu avec `make_env`. Les mises à jour et les évaluations ont lieu en fin
d'épisode, pendant que le jeu attend le prochain `reset()`.

```bash
uv run python scripts/train_ppo.py --config configs/ppo_trackmania.json --seed 0
uv run python scripts/train_sac.py --config configs/sac_trackmania.json --seed 0
uv run python scripts/run_queue.py experiments/queue.json     # enchaîne plusieurs runs sans surveillance
uv run python scripts/evaluate.py --policy sac --config configs/sac_trackmania.json --checkpoint checkpoints/sac_trackmania_seed0_best.pt --episodes 10
```

Les évaluations pendant l'entraînement (3 épisodes tous les 20 000 pas) enregistrent le retour, la distance
moyenne et le taux d'arrivée. Le run s'arrête au pas `total_steps` (ou `--max-steps`), même en plein épisode,
puis l'agent est évalué une dernière fois. Chaque évaluation sauvegarde l'agent dans
`checkpoints/<run>.pt`, et dans `checkpoints/<run>_best.pt` s'il bat toutes les évaluations précédentes en
distance moyenne : SAC pouvant s'effondrer en fin de run, c'est ce checkpoint qu'on évalue sur 10 épisodes
avec `evaluate.py`. Son score à l'entraînement, choisi parmi plusieurs évaluations de 3 épisodes, est
optimiste ; seul celui de `evaluate.py` compte. `run_queue.py` lance chaque run dans son propre processus, écrit sa sortie dans
`logs/` et saute les runs déjà terminés : on peut le relancer après une coupure. Le jeu doit rester au premier
plan pendant toute la file.

**Phase 1** (200 000 pas, réglages par défaut, 3 graines, sur `tmrl-test`) :

| | PPO | SAC |
|---|-----|-----|
| Parcours terminé | jamais | les 3 graines, dès environ 60 000 pas |
| Meilleure évaluation (m) | 1 044 / 1 285 / 1 287 | 2 059 / 1 926 / 2 055 |
| Durée d'un run | environ 3,3 h | environ 4,2 h |

SAC apprend environ 3 fois plus vite et termine le parcours de 2 059 m ; PPO bute sur trois virages, vers 97,
835 et 1 285 m. SAC est en revanche instable : deux graines sur trois s'effondrent après avoir terminé le
parcours. Ces runs datent d'avant l'évaluation finale et `_best.pt` : leur dernière évaluation a eu lieu
vers 180 000 pas, et seul l'agent final a été gardé. Le détail des runs, les variantes de la phase 2 et les CSV sont dans `experiments/` (voir
`experiments/journal.md`).

## Méthode de travail

- Une branche par fonctionnalité, fusionnée dans `main` par Pull Request.
- Chaque PR est relue par au moins un autre membre.
- Les configurations et graines des expériences rapportées sont versionnées dans `configs/`.
