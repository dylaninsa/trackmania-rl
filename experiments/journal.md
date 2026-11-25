# Journal d'expériences

Une ligne par run lancé, y compris les runs ratés ou coupés : ils expliquent les choix. Le nom du run est
`<config>_seed<graine>`, comme les fichiers de `results/` et `checkpoints/`.

`run_queue.py` copie le CSV, le log et les logs TensorBoard de chaque run réussi dans
`experiments/<nom du run>/` ; on les commite avec la ligne du journal. Les checkpoints `_best.pt` vont sur le
Drive du groupe.

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

**Objectif :** un SAC qui termine le parcours sur toutes les graines et le reste jusqu'à la fin du run.
**Critère de réussite**, fixé avant les runs : sur chaque graine, au moins 2 arrivées sur 3 à chacune des trois
dernières évaluations (160 000, 180 000 et 200 000 pas).

### Phase 3a : stabilité (200 000 pas, graines 0 à 3, une graine par machine)

| Configuration | Changement | Rôle |
|---------------|------------|------|
| `sac_trackmania_p3_base` | `init_alpha` 0,1 et `updates_per_step` 0,5 | les deux gagnants de la phase 2, combinés |
| `sac_trackmania_p3_lr1e4` | base + `lr` 1e-4 | des pas plus petits, pour des critiques plus stables |
| `sac_trackmania_p3_gamma995` | base + `gamma` 0,995 | un horizon plus long (environ 10 s au lieu de 5 s) |

Chaque machine lance `experiments/queue_phase3a_seed<N>.json` : les 3 configurations avec sa graine, dans un
ordre décalé d'une machine à l'autre pour que chaque configuration ait une graine terminée après la première
nuit.

| Date | Membre | Run | Machine | Dernière éval. (m) | Meilleure éval. (m) | Arrivées aux 3 dernières éval. | Remarque |
|------|--------|-----|---------|--------------------|---------------------|--------------------------------|----------|
| 19/11 | Dylan | `sac_trackmania_p3_base_seed0` | PC Dylan | 2 059 | 2 059 (80 k) | 3/3, 3/3, 3/3 | Arrive dès 80 k, 3/3 à chaque évaluation dès 120 k. Seul run qui remplit le critère. |
| 19/11 | Dylan | `sac_trackmania_p3_lr1e4_seed0` | PC Dylan | 338 | 1 825 (100 k) | 1/3, 0/3, — | Jusqu'à 2/3 arrivées, puis s'effondre. Coupé à 180 k, non relancé. |
| 20/11 | Dylan | `sac_trackmania_p3_gamma995_seed0` | PC Dylan | 1 791 | 1 930 (80 k) | 0/3, 2/3, 1/3 | Arrive dès 40 k, mais jamais plus de 2/3. |
| 19/11 | Antoine | `sac_trackmania_p3_base_seed1` | PC Antoine | 587 | 587 (200 k) | 0, 0, 0 | Jamais d'arrivée ; 1 146 épisodes, souvent coincé au premier virage. |
| 19/11 | Antoine | `sac_trackmania_p3_lr1e4_seed1` | PC Antoine | 828 | 828 (200 k) | 0, 0, 0 | Jamais d'arrivée ; 1 079 épisodes. |
| 19/11 | Antoine | `sac_trackmania_p3_gamma995_seed1` | PC Antoine | 329 | 835 (120 k) | 0, 0, 0 | Jamais d'arrivée. |
| 21/11 | Antoine | `sac_trackmania_p3_base_seed2` | PC Antoine | 341 | 343 (160 k) | 0, 0, 0 | Jamais d'arrivée ; 1 294 épisodes. |
| 21/11 | Antoine | `sac_trackmania_p3_lr1e4_seed2` | PC Antoine | 95 | 786 (160 k) | 0, 0, 0 | Jamais d'arrivée. |
| 21/11 | Antoine | `sac_trackmania_p3_gamma995_seed2` | PC Antoine | 584 | 822 (160 k) | 0, 0, 0 | Jamais d'arrivée. |
| 19/11 | Léa | `sac_trackmania_p3_base_seed3` | PC Léa | 88 | 1 645 (140 k) | 1/3, 0/3, 0/3 | 2/3 arrivées à 140 k, puis s'effondre. 194 dépassements du pas de temps. |
| 20/11 | Léa | `sac_trackmania_p3_lr1e4_seed3` | PC Léa | 1 391 | 1 391 (200 k) | 0, 0, 0 | Jamais d'arrivée ; 150 dépassements du pas de temps. |
| 20/11 | Léa | `sac_trackmania_p3_gamma995_seed3` | PC Léa | 104 | 348 (180 k) | 0, 0, 0 | Jamais d'arrivée ; 362 dépassements du pas de temps. |

**Bilan de la phase 3a.** Sur le PC de Dylan, `p3_base` remplit le critère ; `p3_lr1e4` et `p3_gamma995`
arrivent mais restent instables. Sur les deux autres machines, aucune configuration n'apprend vraiment : une
seule série d'arrivées isolée en 9 runs (`p3_base`, graine 3, à 140 k et 160 k), et la plupart des
évaluations restent bloquées vers 95–100 m, au premier virage.

Le plan « une graine par machine » confond la machine et la graine : la phase 3a seule ne permet pas de dire
si ces échecs viennent des graines ou des machines. Plusieurs indices désignent la machine. Le LIDAR est
calculé à partir de l'image de la fenêtre du jeu, qui dépend de la résolution et du rendu. Le PC de Léa
dépasse souvent le pas de temps de 0,05 s (jusqu'à 0,2 s). Le PC d'Antoine joue environ deux fois plus
d'épisodes que celui de Dylan pour le même nombre de pas, donc des épisodes beaucoup plus courts. La phase 3b
lève le doute ; tous les runs suivants ont tourné sur le PC de Dylan.

### Phase 3b : `p3_base` sur de nouvelles graines (PC Dylan)

Graines 4 à 6, pour ne pas écraser les dossiers de la phase 3a. File : `experiments/queue_phase3b.json`.

| Date | Membre | Run | Machine | Dernière éval. (m) | Meilleure éval. (m) | Arrivées aux 3 dernières éval. | Remarque |
|------|--------|-----|---------|--------------------|---------------------|--------------------------------|----------|
| 21/11 | Dylan | `sac_trackmania_p3_base_seed4` | PC Dylan | 2 059 | 2 059 (200 k) | 0/3, 2/3, 3/3 | Apprend lentement : première arrivée à 80 k, régulier seulement à la fin. |
| 22/11 | Dylan | `sac_trackmania_p3_base_seed5` | PC Dylan | 2 059 | 2 059 (80 k) | 1/3, 3/3, 3/3 | Première arrivée à 40 k, 3/3 de 80 k à 140 k ; un creux à 160 k. |
| 22/11 | Dylan | `sac_trackmania_p3_base_seed6` | PC Dylan | 1 512 | 2 057 (60 k) | 2/3, 0/3, 1/3 | 3/3 de 60 k à 100 k, puis oscille. |

**Bilan de la phase 3b.** La même configuration apprend sur les 4 graines lancées sur le PC de Dylan (0, 4,
5, 6) et sur aucune des 3 graines lancées ailleurs : les échecs de la phase 3a viennent de la machine. Le
critère strict n'est rempli que par la graine 0, mais la fin de run est nettement meilleure qu'avec les
réglages par défaut de la phase 1 :

| Sur le PC de Dylan | Défaut (phase 1, graines 0 à 2) | `p3_base` (graines 0, 4, 5, 6) |
|--------------------|---------------------------------|--------------------------------|
| Taux d'arrivée moyen aux 3 dernières évaluations | 33 % | 67 % |
| Graines à 3/3 à la dernière évaluation | 1 sur 3 | 3 sur 4 |

L'instabilité est réduite, pas éliminée. `p3_base` devient la configuration finale.

### Phase 3c : pénalité de temps (PC Dylan)

La récompense de base rapporte environ 306 par tour quel que soit le temps : rien ne pousse la voiture à aller
vite. `sac_trackmania_p3_time` reprend `p3_base` et retire 0,01 à chaque pas (`"reward": {"time_penalty":
0.01}`, voir `docs/environnement.md`). Mêmes graines que la phase 3b, pour comparer graine par graine. File :
`experiments/queue_phase3c_time.json`.

| Date | Membre | Run | Machine | Dernière éval. (m) | Meilleure éval. (m) | Arrivées aux 3 dernières éval. | Remarque |
|------|--------|-----|---------|--------------------|---------------------|--------------------------------|----------|
| 22/11 | Dylan | `sac_trackmania_p3_time_seed4` | PC Dylan | 2 057 | 2 059 (140 k) | 1/3, 2/3, 3/3 | Première arrivée à 40 k. 78 dépassements du pas de temps. |
| 23/11 | Dylan | `sac_trackmania_p3_time_seed5` | PC Dylan | 1 511 | 2 058 (80 k) | 1/3, 0/3, 1/3 | Première arrivée à 40 k, s'effondre en fin de run. |
| 23/11 | Dylan | `sac_trackmania_p3_time_seed6` | PC Dylan | 2 058 | 2 059 (120 k) | 2/3, 3/3, 3/3 | Première arrivée à 40 k ; remplit le critère. |

Les 3 graines arrivent dès 40 k pas, contre 40 à 80 k sans pénalité. La stabilité ne change pas : 59 %
d'arrivées aux 3 dernières évaluations, contre 56 % pour `p3_base` sur les mêmes graines. Aucun signe de
crash volontaire : les épisodes ratés s'arrêtent loin dans le parcours.

## Évaluation finale (24/11, PC Dylan)

Le checkpoint `_best.pt` de chaque run est évalué sur 10 épisodes avec `scripts/evaluate.py`, à côté du PID.
`evaluate.py` n'applique pas la pénalité de temps : on compare la distance, les arrivées et le temps au tour.
Les CSV sont dans `experiments/final_eval/`.

| Agent | Arrivées | Temps moyen (s) | Écart-type (s) | Meilleur temps (s) |
|-------|----------|-----------------|----------------|--------------------|
| PID | 10/10 | 117,0 | 1,2 | 116,1 |
| `p3_base` graine 0 | 8/10 | 115,1 | 3,3 | 108,8 |
| `p3_base` graine 4 | 9/10 | 92,6 | 1,2 | 90,8 |
| `p3_base` graine 5 | 10/10 | 90,1 | 2,3 | 85,8 |
| `p3_base` graine 6 | 8/10 | 69,4 | 1,1 | 67,5 |
| `p3_time` graine 4 | 10/10 | 61,4 | 2,8 | 57,9 |
| `p3_time` graine 5 | 5/10 | 74,1 | 2,8 | 69,1 |
| `p3_time` graine 6 | 10/10 | 97,2 | 2,2 | 93,8 |

Le temps moyen et l'écart-type ne portent que sur les tours terminés. La première évaluation de la graine 0 a
été refaite : son premier épisode était resté à 0 m pendant qu'on cliquait dans la fenêtre du jeu. La seconde
donne aussi 8/10, avec deux vraies sorties de piste vers 1 668 m.

- Tous les agents terminent le tour, et 6 sur 7 vont plus vite que le PID en moyenne. Le meilleur,
  `p3_time` graine 4, boucle 10 tours sur 10 en 61,4 s en moyenne : deux fois plus vite que le PID.
- `p3_base` termine 35 épisodes sur 40, `p3_time` 25 sur 30 (la graine 5 n'en termine que 5).
- Sans pénalité, le temps dépend fortement de la graine (69 à 115 s). Avec pénalité, le gain n'est pas
  systématique : 92,6 → 61,4 s sur la graine 4, 90,1 → 74,1 s sur la graine 5, mais 69,4 → 97,2 s sur la
  graine 6. Trois graines ne suffisent pas à conclure que la pénalité accélère l'agent en général.
- La graine 6 s'explique par le choix du checkpoint. `_best.pt` est choisi sur la distance moyenne, qui vaut
  environ 2 059 m pour tout agent qui boucle le tour : le checkpoint retenu est celui de 120 k pas, dont le
  retour (286) correspond à un tour d'environ 98 s. Celui de 180 k (retour 292, environ 70 s) a fini à
  quelques centimètres de moins. Quand la récompense intègre la vitesse, il faudrait choisir le checkpoint sur
  le retour. Ce critère n'a pas été changé après coup, pour ne pas adapter le protocole aux résultats.
