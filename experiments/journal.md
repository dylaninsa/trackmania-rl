# Journal d'expériences

Une ligne par run lancé, y compris les runs ratés ou coupés : ils expliquent les choix. Le nom du run est
`<config>_seed<graine>`, comme les fichiers de `results/` et `checkpoints/`.

Les CSV des runs rapportés dans le rapport sont copiés dans `experiments/<nom du run>/`. Les checkpoints
finaux vont sur le Drive du groupe.

## Phase 1 : comparaison avec les réglages par défaut (200 000 pas, 3 graines)

| Date | Membre | Run | Machine | Distance finale (m) | Arrivées | Remarque |
|------|--------|-----|---------|---------------------|----------|----------|

## Phase 2 : variantes (60 000 pas, graine 0)

Chaque variante ne change qu'un seul hyperparamètre par rapport à la configuration de référence ; elle se
compare à la référence de même graine, à 20 000, 40 000 et 60 000 pas.

| Variante | Changement | Membre | Distance à 60 000 pas (m) | Référence à 60 000 pas (m) | Remarque |
|----------|------------|--------|---------------------------|----------------------------|----------|
| `ppo_trackmania_ent01` | `entropy_coef` 0,001 → 0,01 | | | | |
| `ppo_trackmania_lr1e3` | `lr_actor` 3e-4 → 1e-3 | | | | |
| `ppo_trackmania_buf2048` | `buffer_size` 4 096 → 2 048 | | | | |
| `sac_trackmania_alpha01` | α initial 1 → 0,1 | | | | |
| `sac_trackmania_upd05` | `updates_per_step` 1 → 0,5 | | | | |
| `sac_trackmania_lr1e3` | `lr` 3e-4 → 1e-3 | | | | |

## Phase 3 : amélioration de l'algorithme retenu
