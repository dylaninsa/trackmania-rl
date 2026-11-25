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

## Phase 2 : variantes

Une graine, une variante par hyperparamètre. Deux réglages de SAC aident : `init_alpha` à 0,1 au lieu de 1
(moins d'exploration au départ) et `updates_per_step` à 0,5 (pauses deux fois plus courtes en fin d'épisode).
Aucune variante de PPO ne change le résultat. Avec une seule graine, ces résultats ont seulement orienté la
phase 3.

## Phase 3 : stabiliser SAC, puis l'accélérer

**Critère de réussite**, fixé avant les runs : sur chaque graine, au moins 2 arrivées sur 3 à chacune des
évaluations à 160 000, 180 000 et 200 000 pas.

**3a, stabilité.** Trois configurations, une graine par machine du groupe :
- `p3_base` combine les deux réglages gagnants de la phase 2 ;
- `p3_lr1e4` y ajoute `lr` 1e-4 ;
- `p3_gamma995` y ajoute `gamma` 0,995.

Sur le PC de Dylan, `p3_base` remplit le critère ; les deux autres atteignent l'arrivée mais restent
instables. Sur les deux autres machines, presque rien n'apprend.

**3b, dépendance à la machine.** Ce plan confondait la machine et la graine. `p3_base` a donc été relancée sur
3 nouvelles graines, toutes sur le PC de Dylan, et elle apprend sur les 3. La même configuration apprend sur
4 graines sur 4 sur ce PC, et sur aucune des 3 ailleurs. Les causes probables :
- le LIDAR, calculé à partir de l'image de la fenêtre, dépend du rendu ;
- certaines machines dépassent souvent le pas de 0,05 s.

Tous les runs suivants ont tourné sur cette seule machine. Comparée aux réglages par défaut, `p3_base` double
le taux d'arrivée en fin de run (67 % contre 33 % aux trois dernières évaluations) sans supprimer
l'instabilité : seule une graine sur quatre remplit le critère strict.

**3c, pénalité de temps.** La récompense de base rapporte environ 306 par tour quel que soit le temps.
`p3_time` retire 0,01 à chaque pas (voir [`environnement.md`](environnement.md)). Sur les mêmes graines, les
3 runs atteignent l'arrivée dès 40 000 pas, et la stabilité ne change pas (59 % contre 56 %).

## Évaluation finale

Le checkpoint `_best.pt` de chaque run est évalué sur 10 épisodes, sur le PC de Dylan. Les temps ne portent
que sur les tours terminés. Les CSV sont dans `experiments/final_eval/`.

| Agent | Arrivées | Temps moyen (s) | Meilleur temps (s) |
|-------|----------|-----------------|--------------------|
| PID | 10/10 | 117,0 | 116,1 |
| `p3_base` graines 0 / 4 / 5 / 6 | 8 / 9 / 10 / 8 | 115,1 / 92,6 / 90,1 / 69,4 | 108,8 / 90,8 / 85,8 / 67,5 |
| `p3_time` graines 4 / 5 / 6 | 10 / 5 / 10 | 61,4 / 74,1 / 97,2 | 57,9 / 69,1 / 93,8 |

- Tous les agents bouclent le tour, et 6 sur 7 vont plus vite que le PID en moyenne. Le meilleur
  (`p3_time`, graine 4) termine 10 tours sur 10, deux fois plus vite que le PID.
- `p3_base` termine 35 épisodes sur 40, et `p3_time` 25 sur 30.
- L'effet de la pénalité de temps n'est pas systématique. Graine par graine, le tour passe de 92,6 à 61,4 s
  et de 90,1 à 74,1 s, mais de 69,4 à 97,2 s sur la graine 6. Trois graines ne suffisent pas à conclure.
- La graine 6 s'explique par le choix du checkpoint. `_best.pt` est choisi sur la distance, qui est la même
  pour tout agent qui boucle le tour. Le checkpoint retenu est donc un agent de 120 000 pas, lent, alors que
  celui de 180 000 pas bouclait en environ 70 s. Avec une récompense qui intègre la vitesse, il faudrait
  choisir le checkpoint sur le retour. Ce critère n'a pas été changé après coup.

## Limites

- **Une seule machine.** Les résultats finaux viennent tous du PC de Dylan, et la même configuration n'apprend
  pas sur deux autres machines du groupe.
- **Peu de graines.** 3 ou 4 graines par configuration montrent des tendances, pas des différences
  statistiquement établies.
- **Instabilité.** SAC reste instable en fin de run sur une graine sur trois ou quatre. Garder le meilleur
  checkpoint contourne le problème sans le résoudre.
- **Une seule piste.** Tout l'entraînement a lieu sur `tmrl-test`.

Le détail de chaque run est dans [`experiments/journal.md`](../experiments/journal.md).
