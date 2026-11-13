# MTI881 - Projet d'Apprentissage par Renforcement (Groupe 4)
**Apprentissage par renforcement continu (PPO) : de `Pendulum-v1` à `TrackMania (TMRL)`**

Équipe **Groupe 4** : Léa Cavoit, Dylan Sanson, Jean-Baptiste Vanherpen, Antoine Dumont.

---

## 1. Architecture logicielle (les 4 modules du projet)

Conformément à la proposition de projet (Section 4), le code est découpé en **4 modules indépendants** pour permettre de valider 100 % de la logique mathématique sur `Pendulum-v1` avant de brancher `TrackMania 2020` via `TMRL` :

```text
Projet/
├── src/
│   ├── environment.py   # Module 1 : Interface Gymnasium (Pendulum-v1 & wrapper TMRL LIDAR) + Normalisation
│   ├── networks.py      # Module 2 : Réseaux Acteur (Gaussien continu) et Critique en PyTorch
│   └── ppo.py           # Module 3 : Algorithme PPO (RolloutBuffer, avantage GAE, perte L^CLIP)
├── train_ppo.py         # Module 4 : Boucle d'entraînement, logs TensorBoard, checkpoints et courbes
├── evaluate_ppo.py      # Script d'évaluation statistique et de visualisation temps réel (--render)
├── requirements.txt     # Dépendances Python du projet
└── .vscode/             # Configuration VS Code prête à l'emploi (interpréteur MTI881_env + debug)
```

---

## 2. Comprendre la démarche PPO en contrôle continu

### Pourquoi un Acteur Gaussien (`src/networks.py`) ?
Dans `Pendulum-v1` comme dans `TrackMania`, les actions sont **continues** :
- **`Pendulum-v1`** : 1 action continue $a \in [-2.0, +2.0]$ (couple moteur appliqué au pendule).
- **`TMRL` (LIDAR)** : 3 actions continues $a \in [-1.0, +1.0]^3$ (accélération, freinage, direction).

L'Acteur (`ContinuousActor`) ne sort donc pas des probabilités discrètes (Softmax), mais les paramètres d'une **loi normale multidimensionnelle** :
$$\pi_\theta(a \mid s) = \mathcal{N}\!\left(\mu_\theta(s),\, \text{diag}(\sigma_\theta^2)\right)$$
- **En entraînement** : on échantillonne $a \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta^2)$ pour explorer, et on calcule $\log \pi_\theta(a \mid s)$.
- **En évaluation** : on applique directement l'action moyenne déterministe $a = \mu_\theta(s)$.

### Comment fonctionne une itération PPO (`src/ppo.py`) ?
1. **Collecte On-Policy (`RolloutBuffer`)** : l'agent interagit avec l'environnement pendant $T = 2048$ pas et stocke $(s_t, a_t, \log \pi_{\theta_{\text{old}}}(a_t|s_t), r_t, V_\phi(s_t))$.
2. **Calcul des avantages GAE (`compute_returns_and_advantages`)** :
   $$\delta_t = r_t + \gamma (1 - \text{terminated}_t) V_\phi(s_{t+1}) - V_\phi(s_t), \qquad \hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$
   *(Distinction rigoureuse entre un crash réel `terminated=True` où la valeur future vaut $0$, et une fin par limite de temps `truncated=True` où l'on bootstrappe avec $V_\phi(s_{t+1})$).*
3. **Optimisation par mini-batchs sur $K$ époques** :
   - Ratio de probabilité : $r_t(\theta) = \exp\!\left(\log \pi_\theta(a_t|s_t) - \log \pi_{\theta_{\text{old}}}(a_t|s_t)\right)$
   - Perte de l'Acteur (clippée à $[1-\epsilon, 1+\epsilon]$ pour éviter les mises à jour destructrices) :
     $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\!\left( r_t(\theta)\hat{A}_t,\, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$
   - Perte du Critique : erreur quadratique moyenne $\text{MSE}(V_\phi(s_t), R_t)$ avec $R_t = \hat{A}_t + V_{\phi_{\text{old}}}(s_t)$.

---

## 3. Passer de `Pendulum-v1` à `TMRL` (dans un second temps)

L'architecture a été conçue pour que **ni `src/networks.py`, ni `src/ppo.py` n'aient besoin d'être modifiés** lors du passage à TrackMania :

| Caractéristique | Jalon 1 : `Pendulum-v1` | Jalon 2 : `TMRL` (Mode LIDAR) |
| :--- | :--- | :--- |
| **Observation brute** | Vecteur `Box(3,)` : $[\cos\theta, \sin\theta, \dot{\theta}]$ | Tuple `(vitesse[1], lidar[4,19], actions[2,3])` |
| **Sortie de `src/environment.py`** | Vecteur 1D normalisé (`state_dim = 3`) | Vecteur 1D aplati et normalisé (`state_dim = 83`) via `FlattenTupleObservationWrapper` |
| **Espace d'actions** | `Box(-2.0, 2.0, (1,))` (`action_dim = 1`) | `Box(-1.0, 1.0, (3,))` (`action_dim = 3` : gaz, frein, volant) |
| **Commande de lancement** | `python train_ppo.py --env-id Pendulum-v1` | `python train_ppo.py --env-id TMRL --gamma 0.99` |

---

## 4. Commandes rapides (Terminal ou VS Code)

### Activer l'environnement virtuel
```bash
source MTI881_env/bin/activate
```

### 1. Lancer l'entraînement PPO sur `Pendulum-v1`
```bash
python train_ppo.py --total-timesteps 150000 --eval-freq 6144
```
- Sauvegarde automatiquement le meilleur modèle dans `checkpoints/ppo_pendulum_best.pt`.
- Génère la courbe d'apprentissage comparée à la baseline aléatoire dans `results/ppo_pendulum_learning_curve.png`.

### 2. Évaluer et visualiser le pendule en temps réel
```bash
# Évaluation statistique sur 10 épisodes face à la politique aléatoire
python evaluate_ppo.py --episodes 10

# Visualisation graphique en temps réel
python evaluate_ppo.py --render --episodes 3
```

### 3. Suivre les courbes dans TensorBoard
```bash
tensorboard --logdir runs/
```
