# Politiques de référence

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
