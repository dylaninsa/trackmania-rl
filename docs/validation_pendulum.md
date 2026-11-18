# Validation de PPO et SAC sur Pendulum-v1

## PPO

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

## SAC

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
