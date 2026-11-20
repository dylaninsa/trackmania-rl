# Test d'intégration SAC et TrackMania

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

Limites : le test ne valide pas l'apprentissage (voir [la validation sur Pendulum-v1](validation_pendulum.md)). La troncation
à `ep_max_length` est surtout couverte par `check_env.py --ep-max-length 100`, car des épisodes de 300 pas
sont presque toujours terminés avant par un échec.
