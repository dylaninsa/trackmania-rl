# Journal d'expériences

Une ligne par run lancé, y compris les runs ratés ou coupés : ils expliquent les choix. Le nom du run est
`<config>_seed<graine>`, comme les fichiers de `results/` et `checkpoints/`.

Les CSV des runs rapportés dans le rapport sont copiés dans `experiments/<nom du run>/`. Les checkpoints
finaux vont sur le Drive du groupe.

## Phase 1 : comparaison avec les réglages par défaut (200 000 pas, 3 graines)

Évaluations de 3 épisodes tous les 20 000 pas. La dernière a lieu vers 180 000 pas : le run s'arrête à
200 000 pas en plein épisode, sans évaluation finale.

| Date | Membre | Run | Machine | Dernière éval. (m) | Meilleure éval. (m) | Première arrivée | Remarque |
|------|--------|-----|---------|--------------------|---------------------|------------------|----------|
| 14/11 | Dylan | `ppo_trackmania_seed0` | PC Dylan | 1 044 | 1 044 (180 k) | — | 3,2 h. Rien avant 60 k, puis oscille entre ~95 et ~1 000 m. |
| 14/11 | Dylan | `ppo_trackmania_seed1` | PC Dylan | 1 283 | 1 285 (120 k) | — | 3,3 h. Plateau à ~1 285 m dès 120 k. |
| 15/11 | Dylan | `ppo_trackmania_seed2` | PC Dylan | 519 | 1 287 (140 k) | — | 3,3 h. Plateau à ~1 285 m, puis rechute. |
| 14/11 | Dylan | `sac_trackmania_seed0` | PC Dylan | 2 053 (3/3 arrivées) | 2 059 (60 k) | 61 k | 4,2 h. Arrivées à 60 k, 100 k et 180 k ; creux à 140 k. |
| 14/11 | Dylan | `sac_trackmania_seed1` | PC Dylan | 594 | 1 926 (140 k) | 60 k | 4,1 h. Jusqu'à 2/3 arrivées, mais rechute à la fin. |
| 15/11 | Dylan | `sac_trackmania_seed2` | PC Dylan | 40 | 2 055 (100 k) | 61 k | 4,4 h. 3/3 arrivées à 60 k et 100 k, puis effondrement dès 120 k. |

## Phase 2 : variantes (60 000 pas, graine 0)

Chaque variante ne change qu'un seul hyperparamètre par rapport à la configuration de référence ; elle se
compare à la référence de même graine, à 20 000, 40 000 et 60 000 pas.

Sans évaluation finale, la dernière évaluation des variantes a eu lieu vers 40 000 pas et non à 60 000 :
on compare donc à 20 000 et 40 000 pas. Une seule graine et deux points : ces résultats orientent la
suite, ils ne prouvent rien.

| Variante | Changement | Membre | 20 k / 40 k (m) | Référence 20 k / 40 k (m) | Remarque |
|----------|------------|--------|-----------------|---------------------------|----------|
| `ppo_trackmania_ent01` | `entropy_coef` 0,001 → 0,01 | Dylan | 0 / 0 | 0 / 0 | Aucun effet visible. |
| `ppo_trackmania_lr1e3` | `lr_actor` 3e-4 → 1e-3 | Dylan | 0 / 0 | 0 / 0 | Aucun effet visible. |
| `ppo_trackmania_buf2048` | `buffer_size` 4 096 → 2 048 | Dylan | 0 / 226 | 0 / 0 | Seule variante PPO qui roule à 40 k. |
| `sac_trackmania_alpha01` | α initial 1 → 0,1 | Dylan | 97 / 1 287 | 97 / 12 | Nettement plus rapide. |
| `sac_trackmania_upd05` | `updates_per_step` 1 → 0,5 | Dylan | 0 / 1 004 | 97 / 12 | Pauses 2 fois plus courtes, sans perte visible. |
| `sac_trackmania_lr1e3` | `lr` 3e-4 → 1e-3 | Dylan | 8 / 14 | 97 / 12 | Bloqué : à écarter. |

## Phase 3 : amélioration de l'algorithme retenu

Depuis cette phase, chaque run se termine par une évaluation au dernier pas et garde aussi son meilleur agent
dans `checkpoints/<run>_best.pt` (meilleure distance moyenne). Les résultats finaux s'obtiennent en évaluant
ce checkpoint sur 10 épisodes avec `scripts/evaluate.py`.
