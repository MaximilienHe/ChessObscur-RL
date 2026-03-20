# Chess Obscur RL - Vue d'ensemble technique courte

Cette note est la version courte du projet pour une review technique.
Elle décrit l'état actuel du code (`v8` -> `v10`), pas les anciens réglages historiques.

## 1. Formulation du problème

On entraîne une politique neuronale unique pour jouer à **Chess Obscur**, une variante stochastique des échecs avec :

- des réactions défensives type QTE sur les captures (`block`, `parry`, `accept loss`)
- une mécanique de **parade** où le défenseur contrôle temporairement la pièce attaquante
- une condition de défaite par **3 échecs subis**

Ce n'est donc pas du chess RL classique de type AlphaZero :

- les transitions sont **stochastiques**
- le jeu a des **tours multi-phases**
- un reward shaping tactique est plus important qu'un apprentissage purement sparse win/loss

Le projet utilise donc du **self-play PPO vectorisé sur GPU**, plutôt qu'une approche centrée sur MCTS.

## 2. Design clef de l'environnement

L'environnement vit presque entièrement sur GPU et fait tourner beaucoup de parties en parallèle.

- `num_envs = 2048`
- l'état du jeu est stocké sous forme de tenseurs batchés
- un rollout = `2048 envs x 256 steps = 524 288 transitions`

Machine à états principale :

```text
PHASE_MOVE
   |
   | coup normal
   v
PHASE_MOVE

PHASE_MOVE
   |
   | tentative de capture
   v
PHASE_DEFENSE -- block reussi --> PHASE_MOVE
      |           parry reussi --> PHASE_PARRY -> PHASE_MOVE
      |           accept/echec --> PHASE_MOVE
      v
PHASE_FINISHED (si condition terminale)
```

Phases actuelles :

- `PHASE_MOVE = 0`
- `PHASE_DEFENSE = 1`
- `PHASE_PARRY = 2`
- `PHASE_FINISHED = 3`

Conditions terminales gérées dans l'environnement :

- capture du roi / situations sans coup légal de type mat ou pat
- **défaite aux 3 échecs**
- **règle des 50 coups**
- **triple répétition** via hachage de Zobrist
- timeout via `full_move_count >= max_steps * 2`

Correctif important en `v10` :

- `full_move_count` n'augmente plus pendant les sous-phases `defense/parry`
- il n'augmente que sur les vrais demi-coups
- cela a supprimé un biais de training sur les timeouts, la pénalité de nul et la longueur moyenne des parties

## 3. Espaces d'observation et d'action

### Observation

Le réseau reçoit un tenseur canonisé :

- shape : **`(19, 8, 8)`**
- les plans encodent :
  - les pièces du joueur actif
  - les pièces adverses
  - l'en passant
  - les droits de roque
  - le camp au trait
  - l'indicateur d'échec
  - la phase courante
  - le compteur de 3 échecs
  - le compteur de demi-coups

Choix de design important :

- l'observation est toujours exprimée du point de vue du **joueur actif**
- cela permet d'apprendre une seule politique partagée pour les deux couleurs

### Espace d'actions

Taille actuelle : **`4099` actions**

```text
0..4095   = coups de plateau (from * 64 + to)
4096      = ATTEMPT_BLOCK
4097      = ATTEMPT_PARRY
4098      = ACCEPT_LOSS
```

Toutes les actions illégales sont masquées avant le softmax via un grand logit négatif.

## 4. Design du reward

Rewards terminaux :

- victoire : `+1.5`
- défaite : `-1.5`
- nul : pénalité progressive de `-0.3` à `-1.0`

Pour les nuls sur timeout, le reward est aussi modulé par l'avantage matériel :

- avantage matériel -> nul moins négatif
- désavantage matériel -> nul plus négatif

Le shaping dense inclut notamment :

- reward de capture
- pénalité de perte de pièce
- reward pour donner échec
- reward pour défense/parade réussie
- pénalité de défense ratée
- forte pénalité pour auto-capture en parade
- petite pénalité par step

Objectif global du shaping :

- garder la victoire comme objectif principal
- mais fournir un signal dense pour des mécaniques rares comme `defense/parry`
- réduire la convergence vers un équilibre trop centré sur le nul

## 5. Réseau de neurones

Le modèle actuel est un ResNet partagé avec deux têtes séparées : politique et valeur.

```text
Entree : obs (19, 8, 8)
   |
   v
Conv 3x3, 19 -> 192
BN + ReLU
   |
   v
15 x Blocs Residuels
(192 canaux constants)
   |
   +-----------------------------+
   |                             |
   v                             v
Tete politique                   Tete valeur
Conv 1x1 : 192 -> 64             Conv 1x1 : 192 -> 4
BN + ReLU                        BN + ReLU
Flatten                          Flatten
Linear : 64*8*8 -> 4099          Linear : 4*8*8 -> 1024
                                 ReLU
                                 Linear : 1024 -> 1
```

Notes :

- la tête politique produit des logits masqués sur `4099` actions
- la tête de valeur n'a **pas de `tanh`**
- en `v10`, le bottleneck de valeur est passé de `1` à `4` canaux pour préserver plus d'information spatiale

Pourquoi cette architecture :

- un trunk résiduel profond pour le traitement spatial
- des têtes `1x1` peu coûteuses pour séparer tardivement politique et valeur
- un encodage d'action plat, simple à masquer

## 6. Boucle d'entraînement PPO

Réglages principaux :

- algorithme : **PPO**
- `gamma = 0.99`
- `gae_lambda = 0.95`
- `clip_eps = 0.12`
- `ppo_epochs = 3`
- `num_minibatches = 8`
- AMP activé sur CUDA

Schedule de learning rate :

- warmup sur `1M` de steps
- puis cosine annealing avec warm restarts tous les `100M` steps
- plancher à `3e-5`
- en `v10`, le decay des restarts est passé de `0.5` à `0.7`

Boucle d'entraînement :

```text
collecte du rollout sur GPU
-> calcul GAE / returns
-> update PPO sur le batch
-> mise a jour du learning rate
-> snapshot league si necessaire
-> repetition jusqu'a 500M steps
```

## 7. League training et curriculum

### League training

Environ `30%` des environnements peuvent jouer contre d'anciens checkpoints au lieu de jouer uniquement contre la politique courante.

Objectif :

- casser les équilibres de self-play trop centrés sur le nul
- forcer l'exploitation de politiques historiques plus faibles
- élargir la distribution des adversaires rencontrés

Comportement actuel du pool :

- snapshot tous les `10M` steps
- `10` checkpoints conservés
- en `v10`, la rétention n'est plus du FIFO pur, l'historique gardé est plus étalé dans le temps

### Curriculum learning

Schedule actuel :

- départ à `150` max steps
- `+10` tous les `5M` timesteps
- cap à `220`

Objectif :

- apprendre d'abord les tactiques sur des parties plus courtes
- puis exposer le modèle à des fins de partie plus longues

En `v10`, le cap est passé de `180` à `220`, car l'ancien curriculum plafonnait trop tôt.

## 8. Principales améliorations v8-v10

### v8

- réseau beaucoup plus grand (`15x192`, têtes plus larges)
- cosine LR avec restarts au lieu d'un LR qui meurt vers zéro
- ajout du league training
- reward shaping anti-draw plus fort

### v9

- action space nettoyé de `4163` à `4099`
- environnement beaucoup plus vectorisé
- triple répétition via Zobrist
- rewards terminaux rendus plus cohérents

### v10

- correction du biais sur `full_move_count`
- value head élargi (`1 -> 4` canaux)
- cap du curriculum relevé à `220`
- decay des restarts LR relâché à `0.7`
- pool league rendu plus diversifié

## 9. Ce qui est techniquement intéressant

Pour une review rapide par quelqu'un de très à l'aise en IA, les idées principales sont :

1. C'est un **environnement RL entièrement batché sur GPU** pour un jeu d'échecs stochastique à transitions multi-phases.
2. Le design repose sur des **observations canonisées** et un **action masking discret** pour garder PPO simple malgré les règles complexes.
3. Le projet a nécessité un reward shaping non trivial parce que le jeu contient des mécaniques rares et un équilibre naturel très orienté vers le nul.
4. Les versions `v8` -> `v10` ont surtout amélioré la **qualité du signal d'apprentissage**, pas seulement le débit.
5. Les fixes les plus importants récemment ont été :
   - correction du biais de comptage des coups
   - augmentation de capacité de la tête de valeur
   - extension du curriculum
   - maintien d'un learning rate réellement actif plus longtemps

## 10. Résumé en une phrase

Chess Obscur RL est un système de self-play PPO vectorisé sur GPU pour une variante d'échecs stochastique, avec un ResNet à 15 blocs, une politique discrète masquée sur `4099` actions, un shaping tactique dense, du league training, et plusieurs correctifs récents visant surtout à améliorer la force finale du modèle.
