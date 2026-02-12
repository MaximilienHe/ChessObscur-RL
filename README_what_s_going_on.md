# Chess Obscur — Documentation Technique Complète du Pipeline d'Entraînement par Reinforcement Learning

---

## Table des matières

1. [Introduction et contexte](#1-introduction-et-contexte)
2. [Logique globale du Reinforcement Learning](#2-logique-globale-du-reinforcement-learning)
3. [Architecture logicielle](#3-architecture-logicielle)
4. [L'environnement GPU vectorisé](#4-lenvironnement-gpu-vectorisé)
5. [L'espace d'observation](#5-lespace-dobservation)
6. [L'espace d'actions](#6-lespace-dactions)
7. [Les mécaniques spécifiques de Chess Obscur](#7-les-mécaniques-spécifiques-de-chess-obscur)
8. [Le réseau de neurones Actor-Critic](#8-le-réseau-de-neurones-actor-critic)
9. [L'algorithme PPO](#9-lalgorithme-ppo)
10. [Le reward shaping](#10-le-reward-shaping)
11. [Le self-play et la collecte de rollouts](#11-le-self-play-et-la-collecte-de-rollouts)
12. [Le curriculum learning](#12-le-curriculum-learning)
13. [Le warmstart par behavioral cloning](#13-le-warmstart-par-behavioral-cloning)
14. [Le serveur d'inférence (ai_server.py)](#14-le-serveur-dinférence)
15. [Paramètres actuels et justifications](#15-paramètres-actuels-et-justifications)
16. [Optimisations de performance](#16-optimisations-de-performance)
17. [Monitoring et diagnostics](#17-monitoring-et-diagnostics)
18. [Historique des itérations (v1 → v3)](#18-historique-des-itérations)

---

## 1. Introduction et contexte

Ce projet implémente un pipeline complet d'entraînement par **Reinforcement Learning (RL)** pour un jeu baptisé **Chess Obscur** — une variante des échecs classiques qui introduit trois mécaniques originales :

- **Les captures stochastiques avec QTE (Quick Time Events)** : quand une pièce attaque une autre, le défenseur peut tenter un blocage ou une parade via un système de timing (QTE). La probabilité de succès dépend des statistiques ATK/DEF des pièces impliquées.
- **La parade (parry)** : si le défenseur réussit une parade, il prend temporairement le contrôle de la pièce attaquante adverse et peut la déplacer (potentiellement pour la sacrifier ou créer une menace).
- **La règle des 3 échecs** : un joueur perd si son roi est mis en échec trois fois au cours de la partie.

L'objectif est d'entraîner un agent capable de jouer à ce jeu de manière compétente uniquement par self-play (l'agent joue contre lui-même), en utilisant l'algorithme **PPO (Proximal Policy Optimization)**. L'intégralité du pipeline — environnement, génération de coups légaux, entraînement — s'exécute **sur GPU** sans jamais transférer de données vers le CPU pendant un rollout, ce qui permet d'atteindre des débits de simulation très élevés (50k+ pas/sec sur RTX 5090).

---

## 2. Logique globale du Reinforcement Learning

### 2.1 Le paradigme RL appliqué aux jeux

Le Reinforcement Learning modélise le problème comme un **Processus Décisionnel de Markov (MDP)** :

- **État (s)** : la position actuelle du plateau, les droits de roque, la phase de jeu, etc.
- **Action (a)** : le coup choisi par l'agent (déplacer une pièce, choisir une défense, contrôler une pièce en parade).
- **Récompense (r)** : un signal numérique indiquant la qualité de l'action (victoire = +1, défaite = -1, capture d'une pièce = bonus proportionnel à la valeur de la pièce, etc.).
- **Politique π(a|s)** : la distribution de probabilité sur les actions que le réseau de neurones apprend à produire.

L'agent cherche à **maximiser la somme cumulée des récompenses futures actualisées** (le "return") :

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
```

où γ = 0.99 est le facteur d'actualisation.

### 2.2 Le cycle d'entraînement

Le cycle se décompose en deux phases alternées :

```
┌──────────────────────────────────────────────────────┐
│                    BOUCLE PRINCIPALE                  │
│                                                       │
│  1. COLLECTE (Self-Play Rollout)                     │
│     ├─ 2048 environnements jouent en parallèle       │
│     ├─ 256 pas par environnement                      │
│     ├─ Le réseau choisit les actions pour les 2 camps│
│     ├─ Stocke : (obs, action, log_prob, reward, done)│
│     └─ Total : 524 288 transitions par rollout        │
│                                                       │
│  2. OPTIMISATION (PPO Update)                        │
│     ├─ Calcul des avantages par GAE                   │
│     ├─ 3 epochs de PPO sur les données collectées    │
│     ├─ 8 minibatches par epoch                        │
│     ├─ Gradient clipping + AMP (float16)             │
│     └─ Mise à jour des poids du réseau               │
│                                                       │
│  Répéter jusqu'à 500M pas total                      │
└──────────────────────────────────────────────────────┘
```

### 2.3 Pourquoi PPO ?

PPO a été choisi pour plusieurs raisons :

- **Stabilité** : le clipping de la politique empêche les mises à jour trop agressives, ce qui est crucial dans un jeu avec un espace d'actions aussi grand (4163 actions possibles).
- **Efficacité d'échantillonnage** : PPO permet de réutiliser les données collectées sur plusieurs epochs (ici 3), contrairement aux méthodes purement on-policy comme REINFORCE.
- **Compatibilité GPU** : l'algorithme se parallélise naturellement sur GPU via le batching des environnements.
- **Antécédents** : PPO est l'algorithme utilisé par OpenAI Five (Dota 2) et d'autres systèmes de jeu à grande échelle, ce qui valide son utilisation pour ce type de problème.

---

## 3. Architecture logicielle

```
CC-RL/
├── config.py                         # Tous les hyperparamètres centralisés
├── ai_server.py                      # Serveur HTTP FastAPI pour l'inférence en production
├── evaluate_model.py                 # Évaluation complète d'un checkpoint
│
├── env/                              # Environnement de jeu (tout sur GPU)
│   ├── chess_obscur_env.py           # Environnement vectorisé principal
│   ├── move_tables.py                # Tables de coups précalculées
│   └── reward.py                     # Fonctions de reward shaping
│
├── model/                            # Réseau de neurones et algorithme
│   ├── network.py                    # Architecture Actor-Critic (ResNet)
│   └── ppo.py                        # Implémentation PPO (clipped, GAE, entropy bonus)
│
├── training/                         # Boucle d'entraînement
│   ├── train.py                      # Boucle principale + CLI
│   ├── self_play.py                  # Collecte des rollouts (self-play)
│   └── import_ndjson.py              # Import de parties humaines pour warmstart
│
└── utils/                            # Utilitaires
    ├── checkpoint.py                 # Sauvegarde/chargement des checkpoints
    └── logger.py                     # Logging TensorBoard + console
```

### Principe de séparation des responsabilités

Chaque module a un rôle unique et bien défini :

- **`config.py`** centralise *tous* les hyperparamètres dans un unique `@dataclass`. Cela permet de modifier n'importe quel paramètre sans toucher au code d'entraînement, et de tracer exactement quelle configuration a produit quel résultat.
- **`env/`** est purement computationnel : il ne connaît ni le réseau ni PPO. Il expose une interface standard `reset() → obs` et `step(actions) → (obs, reward, done, info)`.
- **`model/`** est purement structurel : il définit l'architecture du réseau et la logique de mise à jour PPO, sans connaissance de l'environnement spécifique.
- **`training/`** orchestre les deux : il appelle le réseau pour choisir des actions, les envoie à l'environnement, stocke les transitions, puis lance la mise à jour PPO.

---

## 4. L'environnement GPU vectorisé

### 4.1 Pourquoi vectoriser sur GPU ?

Dans un entraînement RL classique, l'environnement s'exécute sur CPU et les transitions sont transférées au GPU pour l'optimisation du réseau. Ce transfert CPU↔GPU est un goulot d'étranglement majeur.

Ici, **tout l'état du jeu vit sur GPU** sous forme de tenseurs PyTorch :

```python
self.board = torch.zeros(num_envs, 64, dtype=torch.int8, device="cuda")
self.turn_is_white = torch.ones(num_envs, dtype=torch.bool, device="cuda")
self.phase = torch.zeros(num_envs, dtype=torch.int8, device="cuda")
# ... etc.
```

Chaque tenseur a une première dimension `N` (nombre d'environnements parallèles, typiquement 2048). Toutes les opérations sont batchées : quand on appelle `env.step(actions)`, les 2048 parties avancent simultanément en une seule opération tensorielle.

### 4.2 Représentation du plateau

Le plateau est un tenseur `(N, 64)` de type `int8`. Chaque case contient un code :

| Code | Pièce | Code | Pièce |
|------|-------|------|-------|
| 0 | Vide | 7 | Pion noir |
| 1 | Pion blanc | 8 | Cavalier noir |
| 2 | Cavalier blanc | 9 | Fou noir |
| 3 | Fou blanc | 10 | Tour noire |
| 4 | Tour blanche | 11 | Dame noire |
| 5 | Dame blanche | 12 | Roi noir |
| 6 | Roi blanc | | |

L'indexation est `idx = file + rank * 8` (a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63).

### 4.3 Les phases du jeu

L'environnement modélise 4 phases distinctes, encodées dans `self.phase` :

```
PHASE_MOVE (0)      → Le joueur actif choisit un coup normal
PHASE_DEFENSE (1)   → Le défenseur choisit : bloquer, parer, ou accepter la perte
PHASE_PARRY (2)     → Le contrôleur de la parade déplace la pièce attaquante
PHASE_FINISHED (3)  → La partie est terminée
```

La machine à états fonctionne comme suit :

```
MOVE ──capture──→ DEFENSE ──block/fail──→ MOVE (tour suivant)
                          ──parry réussi──→ PARRY ──→ MOVE
     ──non-capture──→ MOVE (tour suivant)
```

### 4.4 Génération de coups légaux

La génération de coups légaux se fait en deux étapes, entièrement batchées :

1. **`_gen_pseudo_legal_batched()`** : génère tous les coups pseudo-légaux (sans vérifier si le roi est en échec après le coup). Cela produit un masque `(N, 64, 64)` → aplati en `(N, 4096)`.

2. **`_filter_king_safety_batched()`** : pour chaque coup pseudo-légal, simule le coup sur une copie du plateau et vérifie que le roi du joueur n'est pas en échec. Les coups qui laissent le roi en échec sont retirés du masque.

Les tables précalculées (`MoveTables`) accélèrent ces opérations :

- **`knight_attack_table[64, 64]`** : pour chaque case, quelles cases un cavalier peut attaquer (masque booléen).
- **`king_attack_table[64, 64]`** : idem pour le roi.
- **`w_pawn_attack_table[64, 64]`** / **`b_pawn_attack_table[64, 64]`** : attaques de pions.
- **`ray_aligned[64, 64]`** : quelles paires de cases sont alignées sur une diagonale ou une rangée/colonne.
- **`ray_type[64, 64]`** : 1 = diagonale (fou/dame), 2 = droite (tour/dame).
- **`between_mask[64, 64, 64]`** : pour chaque paire de cases alignées, quelles cases sont entre elles (pour détecter les pièces bloquantes).

### 4.5 Détection d'échec batchée

`_is_in_check_batched(boards, is_white)` détermine si le roi du camp spécifié est en échec dans chaque environnement. L'algorithme vérifie l'attaque par chaque type de pièce adverse en utilisant les tables précalculées, sans aucune boucle Python sur les cases.

---

## 5. L'espace d'observation

L'observation est un tenseur `(N, 19, 8, 8)` — 19 "plans" (channels) de 8×8, similaire à la représentation utilisée par AlphaZero.

| Plans | Contenu | Justification |
|-------|---------|---------------|
| 0–5 | Pièces du joueur actif (P, N, B, R, Q, K) — masques binaires | Le réseau voit toujours "ses" pièces en premier, indépendamment de la couleur réelle |
| 6–11 | Pièces de l'adversaire | Même logique, symétrisée |
| 12 | Case en passant (1 sur la case cible, 0 partout ailleurs) | Nécessaire pour la légalité des coups |
| 13 | Droits de roque (4 bits broadcast sur les 4 premières rangées) | Encodage compact de 4 booléens |
| 14 | Indicateur de tour (tout 1 = blancs jouent) | Permet au réseau de distinguer les camps |
| 15 | Indicateur d'échec (tout 1 si le joueur actif est en échec) | Information critique pour la survie |
| 16 | Phase de jeu (0=move, 0.5=defense, 1=parry, normalisé /3) | Le réseau doit adapter sa politique selon la phase |
| 17 | Compteur d'échecs (normalisé /3) | Signal pour la règle des 3 échecs |
| 18 | Compteur de demi-coups (normalisé /100) | Signal pour la règle des 50 coups |

**Point clé : la symétrie des couleurs.** Les plans 0–5 contiennent toujours les pièces du joueur actif, quelle que soit sa couleur réelle. Cela signifie que le réseau apprend une seule politique qui fonctionne pour les deux camps — un principe directement hérité d'AlphaZero.

---

## 6. L'espace d'actions

L'espace d'actions comporte **4163 actions discrètes**, structurées en 3 catégories :

```
Actions 0–4095      : Coups sur le plateau (64 cases départ × 64 cases arrivée)
Actions 4096–4159   : Sous-promotions (réservé, non utilisé activement)
Actions 4160–4162   : Actions de défense
  4160 = ATTEMPT_BLOCK   (tenter un blocage)
  4161 = ATTEMPT_PARRY   (tenter une parade)
  4162 = ACCEPT_LOSS     (accepter la capture)
```

### Encodage des coups

Un coup de la case `from` vers la case `to` est encodé : `action = from * 64 + to`.

Inversement : `from = action // 64`, `to = action % 64`.

### Masquage des actions illégales

Avant chaque décision, l'environnement produit un masque binaire `(N, 4163)` indiquant quelles actions sont légales. Le réseau de neurones reçoit ce masque et l'applique aux logits de sa politique : les actions illégales reçoivent un logit de -10000 (ou -1e4 pour la compatibilité float16), ce qui leur donne une probabilité essentiellement nulle après softmax.

### Cas spécial : le "skip" en parade

Pendant la phase de parade, l'agent contrôle une pièce adverse. Il peut choisir de ne pas la déplacer (skip) en jouant l'action `from == to` (la pièce reste sur place). Cette action est toujours légale pendant la parade et donne un petit reward positif, ce qui en fait un choix sûr quand aucun bon coup n'est disponible.

---

## 7. Les mécaniques spécifiques de Chess Obscur

### 7.1 Captures stochastiques et QTE

Quand une pièce attaque une autre, la capture n'est pas automatique. Le processus est :

1. L'attaquant déplace sa pièce vers la case cible → **PHASE_DEFENSE** s'active.
2. Le défenseur choisit parmi 3 options : BLOCK, PARRY, ou ACCEPT_LOSS.
3. Pour BLOCK et PARRY, la probabilité de succès est : **τ = DEF / (ATK + DEF)** où ATK et DEF sont les statistiques de combat des pièces impliquées.

Les statistiques de combat, définies dans `MoveTables` :

| Pièce | ATK | DEF |
|-------|-----|-----|
| Pion | 1 | 1 |
| Cavalier | 3 | 3 |
| Fou | 3 | 3 |
| Tour | 5 | 7 |
| Dame | 9 | 8 |
| Roi | 10 | 10 |

Exemple : si une Dame (ATK=9) attaque une Tour (DEF=7), τ = 7/(9+7) = 43.75% de chance de bloquer/parer.

### 7.2 La parade (parry)

Si le défenseur réussit une parade, il prend le **contrôle temporaire de la pièce attaquante**. Il peut alors :

- **La déplacer vers une case vide** (bon mouvement) → reward positif (+0.08)
- **Capturer une pièce adverse avec** → déclenche une nouvelle phase de défense (reward +0.05)
- **Capturer une de ses propres pièces** (auto-capture, le pire cas) → reward négatif (-0.20 × valeur de la pièce)
- **Ne pas la bouger** (skip) → reward modéré (+0.025)

La hiérarchie des rewards encourage le réseau à apprendre que l'auto-capture est catastrophique et que le skip est toujours préférable à une mauvaise action.

### 7.3 La règle des 3 échecs

Chaque fois qu'un joueur met son adversaire en échec, un compteur est incrémenté pour l'adversaire. Si ce compteur atteint 3, l'adversaire perd immédiatement. Cela crée un objectif stratégique additionnel : donner des échecs est activement récompensé (+0.10), et subir un 3e échec est un game over.

**Bugfix important (v2)** : dans la version initiale, `_enforce_check()` vérifiait si l'*acteur* était en échec au lieu de l'*adversaire*. Ce bug a été corrigé — le code vérifie maintenant correctement si l'adversaire du joueur qui vient de bouger est en échec.

---

## 8. Le réseau de neurones Actor-Critic

### 8.1 Architecture

Le réseau est un **ResNet Actor-Critic** inspiré d'AlphaZero :

```
Entrée : (batch, 19, 8, 8)
         │
    ┌────▼────┐
    │ Conv2d  │  19 → 128 filtres, 3×3, padding=1
    │ BN+ReLU │
    └────┬────┘
         │
    ┌────▼────┐
    │ 10 blocs│  Chaque bloc : Conv→BN→ReLU→Conv→BN + skip connection → ReLU
    │ ResNet  │  128 filtres tout du long
    └────┬────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│Policy │ │Value  │
│Head   │ │Head   │
└───┬───┘ └───┬───┘
    │         │
Conv 128→32  Conv 128→1
BN+ReLU      BN+ReLU
Flatten      Flatten
FC 2048→4163 FC 64→384→1
    │         │
    ▼         ▼
 logits    valeur ∈ [-1, 1]
 (4163)    (scalaire, tanh)
```

### 8.2 Tête de politique (Policy Head)

Produit un vecteur de 4163 logits (un par action possible). Le masque de légalité est appliqué *avant* le softmax :

```python
policy_logits = policy_logits.masked_fill(~legal_mask, -1e4)
```

La valeur -1e4 (et non -inf) est choisie pour la compatibilité avec AMP (float16), où -inf peut causer des NaN.

### 8.3 Tête de valeur (Value Head)

Produit un scalaire dans [-1, 1] via `tanh`. Ce scalaire estime la probabilité de victoire depuis l'état actuel. Il sert de baseline pour le calcul des avantages dans PPO (réduction de variance).

La couche cachée de 384 neurones (augmentée depuis 256 dans la v3) a été agrandie car la value loss restait élevée (~0.46), indiquant que le réseau n'avait pas assez de capacité pour estimer correctement la valeur des positions.

### 8.4 Initialisation des poids

- **Conv2d** : initialisation Kaiming (He) en mode "fan_out" — standard pour les réseaux avec ReLU.
- **BatchNorm** : poids = 1, biais = 0.
- **Linear** : Xavier uniform pour les poids, 0 pour les biais.

### 8.5 Inférence combinée

La méthode `get_action_and_value()` effectue un seul forward pass et retourne simultanément l'action échantillonnée, sa log-probabilité, l'entropie de la distribution, et la valeur estimée :

```python
action, log_prob, entropy, value = network.get_action_and_value(obs, legal_mask)
```

Cela évite de dupliquer le calcul du tronc ResNet entre la politique et la valeur.

---

## 9. L'algorithme PPO

### 9.1 Principe de PPO-Clip

PPO optimise une politique en limitant l'ampleur des mises à jour. La loss de politique est :

```
L_policy = -min(r(θ) · A, clip(r(θ), 1-ε, 1+ε) · A)
```

où :
- `r(θ) = π_new(a|s) / π_old(a|s)` est le ratio de probabilité entre la nouvelle et l'ancienne politique.
- `A` est l'avantage estimé (combien l'action est meilleure que la moyenne).
- `ε = 0.15` est le rayon de clipping.

Si `A > 0` (bonne action), le ratio est clippé à `1+ε` max — on ne veut pas surpondérer une action même bonne.
Si `A < 0` (mauvaise action), le ratio est clippé à `1-ε` min — on ne veut pas trop pénaliser non plus.

### 9.2 Generalized Advantage Estimation (GAE)

L'avantage `A_t` est calculé par GAE(γ, λ) avec γ=0.99 et λ=0.95 :

```
δ_t = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)
A_t = δ_t + γλ · (1 - done_t) · A_{t+1}
```

GAE interpole entre :
- **λ=0** : avantage à 1 pas (biais faible, variance élevée).
- **λ=1** : avantage Monte Carlo (biais potentiel, variance faible).
- **λ=0.95** : compromis qui réduit la variance tout en maintenant un biais acceptable.

### 9.3 Loss totale

La loss combinée est :

```
L = L_policy + c_value · L_value - c_entropy · H(π)
```

- **`L_value`** : MSE entre la valeur prédite et les returns calculés, avec clipping de la valeur (`clip_value = 0.5`).
- **`c_value = 1.0`** : coefficient de la value loss.
- **`c_entropy`** : coefficient d'entropie, décroît linéairement de 0.005 à 0.0015 sur 200M pas.
- **`H(π)`** : entropie de la distribution de la politique — encourage l'exploration.

### 9.4 Decay de l'entropie

L'entropie est un terme de régularisation qui force le réseau à ne pas converger trop vite vers une politique déterministe. Le coefficient décroît linéairement :

```
entropy_coef(t) = 0.005 + (0.0015 - 0.005) × min(t / 200M, 1)
```

- **Phase initiale (0.005)** : forte exploration, le réseau essaie beaucoup de coups différents.
- **Phase finale (0.0015)** : exploration minimale mais non nulle — cruciale pour les situations de parade où le réseau doit continuer à explorer des alternatives.

Le minimum de 0.0015 (augmenté depuis 0.0005 dans la v3) garantit que le réseau ne se fige jamais complètement, surtout pour les situations de parade qui sont rares et complexes.

### 9.5 Micro-batching pour la gestion mémoire

Les minibatches (65 536 transitions) peuvent être trop gros pour un seul forward/backward pass. Le code les découpe en "micro-batches" de 8192 transitions, accumule les gradients, puis fait un seul `optimizer.step()`. En cas d'OOM CUDA, la taille des micro-batches est automatiquement réduite de moitié.

---

## 10. Le reward shaping

Le reward shaping est la partie la plus délicate du système. Il fournit des signaux d'apprentissage intermédiaires pour guider l'agent avant qu'il ne soit capable de gagner des parties entières.

### 10.1 Rewards terminaux

| Événement | Reward | Justification |
|-----------|--------|---------------|
| Victoire | +1.0 | Signal principal |
| Défaite | -1.0 | Signal principal |
| Match nul (stalemate, 50 coups) | -0.3 | Légèrement négatif pour encourager à jouer pour la victoire |
| Match nul (timeout) | -0.3 + 0.2·tanh(matériel/10) | Bonus si l'agent avait un avantage matériel — incite à dominer même si le temps expire |

### 10.2 Rewards intermédiaires

| Signal | Valeur | Rôle |
|--------|--------|------|
| `REWARD_STEP_PENALTY` | -0.002 / pas | Pression temporelle douce — évite les parties interminables sans être trop agressif |
| `REWARD_CAPTURE_SCALE` | +0.10 × valeur pièce | Récompense proportionnelle à la valeur de la pièce capturée |
| `REWARD_LOSE_PIECE_SCALE` | -0.10 × valeur pièce | Pénalité proportionnelle quand l'agent perd une pièce |
| `REWARD_CHECK_GIVEN` | +0.10 | Donner un échec est stratégiquement bon (règle des 3 échecs) |
| `REWARD_BLOCK_SUCCESS` | +0.06 | Récompense un blocage réussi en défense |
| `REWARD_PARRY_SUCCESS` | +0.12 | La parade est plus difficile, donc mieux récompensée |
| `REWARD_DEFENSE_FAIL` | -0.01 | Petite pénalité si la défense échoue (pas trop, c'est stochastique) |
| `REWARD_ACCEPT_LOSS` | -0.02 | Légèrement pire que d'essayer — incite à tenter la défense |
| `REWARD_PARRY_MOVE_GOOD` | +0.08 | Déplacer la pièce parée vers une case vide |
| `REWARD_PARRY_SELF_CAPTURE` | -0.20 × valeur | **Pénalité sévère** pour auto-capture — le problème principal de la v1 |
| `REWARD_PARRY_ENEMY_CAPTURE` | +0.05 | Capturer une pièce adverse pendant la parade |
| `REWARD_PARRY_SKIP` | +0.025 | Skip sûr — meilleur que l'auto-capture |
| `REWARD_CHECK_ATTEMPT_PENALTY` | -0.05 | Pénalité quand l'adversaire donne un échec à l'agent |

### 10.3 Philosophie du reward shaping

Les rewards intermédiaires sont tous **au moins un ordre de grandeur plus petits** que les rewards terminaux (±1.0). Cela garantit que l'objectif principal (gagner la partie) n'est jamais éclipsé par l'accumulation de petits bonus. Le réseau apprend d'abord à gagner, puis affine sa stratégie via les signaux plus fins.

### 10.4 Le reward de match nul basé sur le matériel

Innovation notable : quand une partie se termine par timeout (dépassement du nombre maximum de coups), le reward de match nul est modulé par l'avantage matériel de l'agent :

```python
normalized_advantage = tanh(material_advantage / 10.0)
timeout_reward = -0.3 + 0.2 * normalized_advantage
```

Un agent qui domine au matériel mais n'a pas eu le temps de mater reçoit un reward moins négatif (~-0.1), tandis qu'un agent dominé reçoit un reward plus négatif (~-0.5). Cela crée une pression pour accumuler du matériel même quand la victoire complète n'est pas atteignable dans le temps imparti.

---

## 11. Le self-play et la collecte de rollouts

### 11.1 Self-play symétrique

L'agent joue **les deux camps simultanément**. Dans chaque environnement, `agent_is_white` est tiré aléatoirement au reset. Le réseau voit toujours l'observation du point de vue du joueur actif (grâce à la symétrie des plans 0–5 / 6–11), donc il apprend une seule politique unifiée.

Les rewards sont signés relativement à l'agent : si l'agent joue blanc et blanc capture une pièce noire, le reward est positif. Si l'agent joue noir et blanc capture une pièce noire, le reward est négatif (l'agent a perdu une pièce).

### 11.2 Le RolloutBuffer

Le `RolloutBuffer` stocke `T × N` transitions (256 pas × 2048 envs = 524 288) :

```
obs           : (T, N, 19, 8, 8)   — observations
actions       : (T, N)              — actions choisies
log_probs     : (T, N)              — log π(a|s)
rewards       : (T, N)              — rewards reçus
dones         : (T, N)              — flags de fin de partie
values        : (T, N)              — V(s) estimées par le réseau
legal_masks   : (T, N, 4163)        — masques de légalité
```

Tout est stocké sur GPU et ne quitte jamais la carte graphique.

### 11.3 Métriques collectées

Pendant chaque rollout, le système collecte :

- Nombre de parties terminées, win/draw/loss rates par couleur.
- Rewards moyens par couleur (pour détecter un déséquilibre blanc/noir).
- Distribution des phases (% du temps en move/defense/parry).
- Nombre moyen d'actions légales par pas.
- Longueur des parties.
- **Statistiques de parade** : taux d'auto-capture, de skip, de bons mouvements, de captures adverses.

---

## 12. Le curriculum learning

### 12.1 Principe

Les parties d'échecs peuvent être très longues. Entraîner directement sur des parties de 300 coups est inefficace car l'agent ne reçoit de signal terminal (victoire/défaite) qu'après très longtemps, diluant le gradient.

Le curriculum learning résout ce problème en commençant par des parties courtes et en augmentant progressivement la durée maximale :

```
Pas 0          → max_steps = 120  (parties courtes, feedback rapide)
Pas 5M         → max_steps = 135
Pas 10M        → max_steps = 150
Pas 50M        → max_steps = 270
Pas 100M       → max_steps = 300  (cap atteint)
```

### 12.2 Paramètres

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `curriculum_start_steps` | 120 | Durée initiale des parties |
| `curriculum_step_increase` | 15 | Augmentation par palier |
| `curriculum_every_n_timesteps` | 5 000 000 | Fréquence d'augmentation |
| `curriculum_max_steps_cap` | 300 | Durée maximale (plafond) |

### 12.3 Justification

- **Phase 120 coups** : l'agent apprend les tactiques de base (captures, défenses, parades) sur des parties très courtes où le feedback est dense.
- **Phase 150–200 coups** : l'agent apprend le milieu de partie, la gestion du matériel, les séquences d'échecs.
- **Phase 300 coups** : l'agent apprend les fins de partie et les stratégies à long terme.

---

## 13. Le warmstart par behavioral cloning

### 13.1 Principe

Avant le self-play pur, le réseau peut être pré-entraîné sur des parties humaines jouées sur le site web de Chess Obscur. C'est un apprentissage supervisé classique : on minimise la cross-entropy entre les prédictions du réseau et les coups réellement joués par les humains.

### 13.2 Pipeline d'import

1. **`import_ndjson.py`** lit le fichier NDJSON exporté par le site web.
2. Pour chaque partie, les coups sont rejoués séquentiellement pour reconstruire l'état du plateau à chaque instant.
3. À chaque décision, un triplet `(observation, action, résultat)` est extrait.
4. Les données sont sauvegardées dans un fichier `.pt` (tenseurs PyTorch).

### 13.3 Entraînement supervisé

Le warmstart effectue 5 epochs sur les données humaines avec un learning rate de 1e-3, en optimisant :

```
L = CrossEntropy(policy, action_humaine) + 0.5 · MSE(value, résultat)
```

Cela donne au réseau un point de départ bien meilleur qu'une initialisation aléatoire — il commence déjà par jouer des coups plausibles, ce qui accélère considérablement la phase de self-play.

---

## 14. Le serveur d'inférence

### 14.1 Rôle

`ai_server.py` est une API HTTP (FastAPI) qui sert le modèle entraîné en production. Le serveur Node.js du jeu web envoie des requêtes POST à `/move` avec l'état actuel de la partie, et reçoit en retour le coup choisi par l'IA.

### 14.2 Conversion état JS → observation

L'état du jeu arrive au format JSON (plateau de 64 cases, tour, roque, phase, etc.). La fonction `build_obs_from_request()` le convertit en tenseur `(1, 19, 8, 8)` identique à celui que l'environnement d'entraînement produit.

### 14.3 Gestion des zones QTE

En production, les zones de timing QTE (block et parry) sont transmises par le client. Quand l'IA choisit BLOCK ou PARRY, elle doit fournir un `stopMs` qui tombe dans la bonne zone temporelle. La fonction `compute_stop_ms_for_zone()` place le slider au milieu de la zone appropriée :

```python
# BLOCK → milieu de [blockStartMs, blockEndMs]
# PARRY → milieu de [parryStartMs, parryEndMs]
mid = (zones.blockStartMs + zones.blockEndMs) // 2
```

### 14.4 Température de sampling

Le serveur utilise une température configurable (défaut 0.5) pour contrôler la diversité des coups :

- **T=0** : greedy (toujours le meilleur coup).
- **T=0.3** : faible variabilité, coups forts.
- **T=1.0** : suit fidèlement la distribution apprise.
- **T>1** : coups plus aléatoires.

---

## 15. Paramètres actuels et justifications

### 15.1 Hyperparamètres d'entraînement

| Paramètre | Valeur | Pourquoi cette valeur |
|-----------|--------|----------------------|
| `num_envs` | 2048 | Maximise l'occupation GPU sur RTX 5090 (16 Go VRAM). Plus d'envs = plus de transitions par rollout = gradients plus stables. |
| `rollout_steps` | 256 | Compromis entre longueur de rollout (pour GAE) et fréquence des mises à jour PPO. |
| `batch_size` | 524 288 | = 2048 × 256. Grand batch pour des gradients stables dans un jeu stochastique. |
| `ppo_epochs` | 3 | Réduit de 4 à 3 (v3) : avec un clip_eps plus serré (0.15), plus d'epochs causeraient de l'overshoot. |
| `num_minibatches` | 8 | Augmenté de 4 à 8 (v3) : minibatches plus petits (65 536) = gradients plus stables. |
| `microbatch_size` | 8192 | Maximum qui tient en VRAM pour le forward/backward pass. |
| `lr` | 3e-4 | Standard PPO pour les réseaux de cette taille. Décroît linéairement vers 0 au fil de l'entraînement. |
| `gamma` | 0.99 | Facteur d'actualisation élevé — les parties d'échecs nécessitent de la planification à long terme. |
| `gae_lambda` | 0.95 | Standard PPO — bon compromis biais/variance pour GAE. |
| `clip_eps` | 0.15 | Réduit de 0.2 à 0.15 (v3) : le clipfrac était ~0.25, indiquant trop de mises à jour hors zone de confiance. Un clip plus serré stabilise l'entraînement. |
| `clip_value` | 0.5 | Clipping de la value head pour éviter de trop corriger les estimations de valeur d'un coup. |
| `max_grad_norm` | 0.5 | Gradient clipping standard pour PPO. |
| `entropy_coef` | 0.005 → 0.0015 | Augmenté (v3) : l'agent convergeait trop vite vers des politiques déterministes, surtout en parade. |
| `entropy_coef_decay_steps` | 200M | Doublé (v3) : decay plus lent pour maintenir l'exploration plus longtemps. |
| `total_timesteps` | 500M | Entraînement très long — le self-play aux échecs nécessite des centaines de millions de pas. |

### 15.2 Architecture du réseau

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `num_res_blocks` | 10 | Profondeur modérée — suffisante pour capturer les patterns tactiques sans être trop coûteuse. AlphaZero en utilisait 19–40 pour les échecs classiques, mais Chess Obscur est plus simple (pas besoin de calculer 20 coups à l'avance). |
| `num_filters` | 128 | Largeur du réseau. 128 est un bon compromis entre expressivité et vitesse d'entraînement. |
| `value_head_hidden` | 384 | Augmenté de 256 (v3) : la value loss restait à ~0.46, indiquant un manque de capacité. 384 permet une meilleure estimation de la valeur positionnelle. |
| `policy_head_filters` | 32 | Réduit de 128 à 32 pour la tête de politique — la convolution 1×1 avant le FC n'a pas besoin de beaucoup de filtres. |

### 15.3 Rewards

Les valeurs de reward ont été calibrées itérativement (v1 → v3) en observant les métriques TensorBoard :

- **`REWARD_PARRY_SELF_CAPTURE`** passé de -0.08 à **-0.20** × valeur_pièce : le taux d'auto-capture était encore trop élevé en v2. Multiplier par la valeur de la pièce rend l'auto-capture d'une Dame catastrophique (-1.8) vs celle d'un Pion (-0.2).
- **`REWARD_PARRY_SKIP`** passé de 0.005 à **0.025** : le skip n'était pas assez attractif par rapport aux mouvements aléatoires.
- **`REWARD_STEP_PENALTY`** passé de -0.003 à **-0.002** : l'agent jouait de manière trop agressive/rapide pour minimiser la pénalité temporelle, sacrifiant la qualité positionnelle.

---

## 16. Optimisations de performance

### 16.1 torch.compile()

Le réseau est compilé avec `torch.compile(mode="default")` après chargement du checkpoint. Cela permet au compilateur PyTorch de fusionner les opérations et d'optimiser le graphe de calcul, offrant un gain de vitesse de 10–30% selon le matériel.

**Note importante** : la compilation ajoute un préfixe `_orig_mod.` aux clés du state_dict. Le code de checkpoint gère cette transformation dans les deux sens (ajout/suppression du préfixe).

### 16.2 Automatic Mixed Precision (AMP)

L'entraînement utilise AMP (`torch.amp.autocast('cuda')`) qui exécute les convolutions et les multiplications matricielles en float16 tout en gardant les accumulations en float32. Cela réduit l'utilisation mémoire et accélère les calculs de ~40% sur les GPU récents (architectures Ampere+).

Le `GradScaler` est utilisé pour éviter les underflows de gradients en float16.

### 16.3 Zéro transfert CPU-GPU

Pendant un rollout de 256 pas × 2048 envs, aucune donnée ne transite entre le CPU et le GPU. L'état du jeu, les observations, les masques de légalité, les actions, les rewards — tout reste sur la carte graphique. Seules les métriques scalaires (compteurs de parties, rewards moyens) sont lues sur CPU pour le logging.

### 16.4 Tables précalculées

Les tables d'attaque (`knight_attack_table`, `ray_aligned`, `between_mask`, etc.) sont calculées une seule fois au démarrage et stockées en mémoire GPU. La génération de coups légaux utilise ces tables via des opérations de masquage et d'indexation tensorielles, évitant toute boucle Python sur les 64 cases du plateau.

---

## 17. Monitoring et diagnostics

### 17.1 TensorBoard

Le système logge en continu vers TensorBoard :

**Métriques de jeu** : `game/win_rate`, `game/draw_rate`, `game/white_win_rate`, `game/black_win_rate`, `game/avg_length`.

**Métriques PPO** : `loss/policy`, `loss/value`, `loss/entropy`, `ppo/clipfrac`, `ppo/approx_kl`, `ppo/entropy_coef`.

**Métriques de rollout** : `rollout/mean_reward`, `rollout/white_mean_reward`, `rollout/black_mean_reward`, `rollout/phase_*_frac`.

**Métriques de parade** : `parry/self_capture_rate`, `parry/good_move_rate`, `parry/skip_rate`, `parry/enemy_capture_rate`.

**Métriques d'entraînement** : `train/fps`, `train/lr`, `train/max_steps` (curriculum).

### 17.2 Signaux d'alerte

- **`ppo/approx_kl > 0.05`** : divergence KL trop élevée → la politique change trop vite, risque d'instabilité.
- **`parry/self_capture_rate > 15%`** : l'agent mange encore ses propres pièces trop souvent.
- **`ppo/clipfrac > 0.25`** : trop de mises à jour clippées → le learning rate ou le clip_eps peut être trop agressif.

### 17.3 Évaluation

`evaluate_model.py` permet une évaluation complète d'un checkpoint :

- Self-play greedy (temperature=0) sur 500+ parties.
- Analyse détaillée des décisions de parade et de défense.
- Comparaison côte-à-côte de deux checkpoints pour mesurer la progression.

---

## 18. Historique des itérations

### v1 — Implémentation initiale

Bugs critiques :
- `_enforce_check` vérifiait le mauvais camp (l'acteur au lieu de l'adversaire).
- Pas de reset du compteur de demi-coups sur les captures/coups de pion.
- Pas de règle des 50 coups.
- Rewards de défense signés relativement à l'attaquant au lieu de l'agent.

### v2 — Corrections de bugs + diagnostics

- Correction des 5 bugs majeurs (marqués BUGFIX #1–#5 dans le code).
- Ajout du tracking des statistiques de parade.
- Ajout des rewards par couleur pour détecter les déséquilibres.

### v3 — Stabilisation + parry fix

Problèmes observés via TensorBoard :
- **Taux d'auto-capture en parade élevé** → augmentation de `PARRY_SELF_CAPTURE` de -0.08 à -0.20 × valeur, augmentation de `PARRY_SKIP` et `PARRY_MOVE_GOOD`.
- **Clipfrac ~0.25** → réduction de `clip_eps` de 0.2 à 0.15, réduction de `ppo_epochs` de 4 à 3.
- **Value loss stagnante ~0.46** → augmentation de `value_head_hidden` de 256 à 384.
- **Convergence prématurée** → augmentation de `entropy_coef` initial de 0.003 à 0.005, minimum de 0.0005 à 0.0015, durée de decay doublée de 100M à 200M.
- **Agent trop agressif** → réduction de `STEP_PENALTY` de -0.003 à -0.002.
- **Ajout du curriculum learning** avec cap à 300.
- **Ajout du reward de match nul basé sur le matériel** pour les timeouts.

---

*Ce document constitue la documentation technique complète du pipeline d'entraînement RL de Chess Obscur. Chaque décision de conception — de l'architecture réseau aux valeurs de reward — est le résultat d'itérations guidées par les métriques observées pendant l'entraînement.*