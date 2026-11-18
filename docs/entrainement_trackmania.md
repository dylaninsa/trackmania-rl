# Entraînement sur TrackMania

## Boucle d'entraînement

Les deux scripts d'entraînement choisissent l'environnement d'après le champ `env` de la configuration :
`"trackmania"` crée l'environnement du jeu avec `make_env`. Les mises à jour et les évaluations ont lieu en fin
d'épisode, pendant que le jeu attend le prochain `reset()`.

Les évaluations pendant l'entraînement (3 épisodes tous les 20 000 pas) enregistrent le retour, la distance
moyenne et le taux d'arrivée. Le run s'arrête au pas `total_steps` (ou `--max-steps`), même en plein épisode,
puis l'agent est évalué une dernière fois.

Chaque évaluation sauvegarde l'agent dans `checkpoints/<run>.pt`, et dans `checkpoints/<run>_best.pt` s'il
bat toutes les évaluations précédentes en distance moyenne. SAC pouvant s'effondrer en fin de run, c'est ce
checkpoint qu'on évalue sur 10 épisodes avec `evaluate.py`. Son score à l'entraînement, choisi parmi plusieurs
évaluations de 3 épisodes, est optimiste : seul celui de `evaluate.py` compte.

## File d'attente

`run_queue.py` lance chaque run dans son propre processus, écrit sa sortie dans `logs/` et saute les runs déjà
terminés : on peut le relancer après une coupure. Quand un run réussit, son CSV, son log et ses logs
TensorBoard sont copiés dans `experiments/<run>/`, prêts à être commités. Le jeu doit rester au premier plan
pendant toute la file.

## Phase 1 : PPO contre SAC

200 000 pas, réglages par défaut, 3 graines, sur `tmrl-test` :

| | PPO | SAC |
|---|-----|-----|
| Parcours terminé | jamais | les 3 graines, dès environ 60 000 pas |
| Meilleure évaluation (m) | 1 044 / 1 285 / 1 287 | 2 059 / 1 926 / 2 055 |
| Durée d'un run | environ 3,3 h | environ 4,2 h |

SAC apprend environ 3 fois plus vite et termine le parcours de 2 059 m ; PPO bute sur trois virages, vers 97,
835 et 1 285 m. SAC est en revanche instable : deux graines sur trois s'effondrent après avoir terminé le
parcours. Ces runs datent d'avant l'évaluation finale et `_best.pt` : leur dernière évaluation a eu lieu vers
180 000 pas, et seul l'agent final a été gardé.

Le détail des runs, les variantes de la phase 2 et la phase 3 sont dans
[`experiments/journal.md`](../experiments/journal.md).
