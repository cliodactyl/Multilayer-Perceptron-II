# Multilayer Perceptron

Perceptron multicouche écrit from scratch (numpy pour l'algèbre linéaire,
matplotlib pour les courbes), qui classe des tumeurs du sein en **M**alignes
ou **B**énignes à partir du dataset Wisconsin.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
# 1. découper le dataset
python split.py data/data.csv --validation-ratio 0.2 --seed 42

# 2. entraîner
python train.py --train data/data_training.csv --valid data/data_validation.csv \
                --layer 24 24 --epochs 84 --batch_size 8 --learning_rate 0.0314 \
                --loss categoricalCrossentropy

# 3. prédire et évaluer
python predict.py --model saved_model.npy --dataset data/data_validation.csv
```

Tout est modulable, soit par arguments, soit par fichier :

```bash
python train.py --config configs/network.json
python train.py --layer 24 24 24 --activation relu --optimizer adam \
                --learning_rate 0.001 --early-stopping 15
```

Bonus :

```bash
python tools/gradient_check.py                      # prouve que la backprop est juste
python tools/compare_models.py h1.json h2.json      # plusieurs courbes sur un graphe
```

## Organisation

Un fichier = une responsabilité. C'est la traduction du réflexe
« une fonction = une utilité » à l'échelle du module.

| Fichier | Rôle |
|---|---|
| `mlp/activations.py` | sigmoid, tanh, relu, softmax + leurs dérivées |
| `mlp/initializers.py` | heUniform, glorotUniform, … |
| `mlp/layers.py` | `DenseLayer` : forward / backward d'une couche |
| `mlp/network.py` | `Network` : empile les couches, propage dans les deux sens |
| `mlp/losses.py` | categoricalCrossentropy, binaryCrossentropy |
| `mlp/optimizers.py` | SGD, momentum, Nesterov, RMSprop, Adam |
| `mlp/callbacks.py` | early stopping |
| `mlp/model.py` | `create_network` / `fit` / `predict` : la boucle d'entraînement |
| `mlp/dataset.py` | lecture CSV, split stratifié, normalisation |
| `mlp/metrics.py` | accuracy, precision, recall, f1, matrice de confusion |
| `mlp/persistence.py` | sauvegarde / chargement du modèle |
| `mlp/plotting.py` | courbes d'apprentissage |
| `split.py`, `train.py`, `predict.py` | les trois programmes demandés |

Le dépendance va toujours dans le même sens : `model` → `network` → `layers`
→ `activations`. Aucun module ne dépend de ce qui est au-dessus de lui, donc
on peut lire le projet de bas en haut sans jamais avoir à revenir en arrière.

## Les trois notions de la soutenance

### Feedforward

La donnée entre à gauche et ressort à droite. Pour chaque couche :

```
z = a_prev @ W + b        somme pondérée + biais
a = activation(z)         seuil "souple"
```

La sortie de la dernière couche passe par un **softmax**, qui transforme
deux scores quelconques en deux probabilités positives dont la somme fait 1.
`[0.03, 0.97]` se lit donc : 97 % de chances que la tumeur soit maligne.

### Backpropagation

Une fois la perte calculée, on veut savoir **de combien chaque poids est
responsable de l'erreur**, c'est-à-dire ∂E/∂W pour chaque couche.

Le calculer indépendamment pour chaque poids serait absurdement coûteux :
un réseau, c'est une fonction composée, et la dérivée d'une composition se
factorise (règle de la chaîne). La backpropagation exploite ça en partant de
la sortie et en remontant, chaque couche réutilisant le travail de la
suivante :

```
dz      = activation.backward(da)     # traverse l'activation
dW      = a_prev.T @ dz               # ce qu'on va corriger
db      = somme(dz)
da_prev = dz @ W.T                    # ce qu'on transmet à la couche d'avant
```

C'est exactement la même structure que le feedforward, miroir. Le coût
total d'une passe arrière est du même ordre qu'une passe avant.

Le cas softmax + categorical crossentropy se simplifie remarquablement :
en enchaînant les deux dérivées, on obtient `dz = (prédiction − vérité)`.
Le code ne code pas ce raccourci en dur (chaque brique reste indépendante),
mais numériquement il tombe dessus — d'où une erreur de l'ordre de 1e-8
dans `tools/gradient_check.py`.

### Gradient descent

Le gradient indique la direction de plus forte **montée** de l'erreur. On
fait donc un pas dans la direction opposée :

```
W ← W − learning_rate × dW
```

- pas trop petit → l'apprentissage traîne
- pas trop grand → on saute par-dessus le minimum, la loss oscille ou explose

Ici c'est de la descente **mini-batch** : les paramètres sont mis à jour
tous les 8 exemples plutôt qu'après avoir vu tout le dataset. C'est plus
rapide, et le bruit introduit par les petits batches aide même à ne pas
rester coincé dans un mauvais minimum local. Les données sont remélangées à
chaque epoch pour que le réseau n'apprenne pas l'ordre des exemples.

## Décisions à savoir justifier

**La normalisation.** Les colonnes brutes vont de ~0.05 à ~2500. Sans
centrage-réduction, la somme pondérée sature les sigmoïdes dès la première
couche, leur dérivée tombe à ~0 et plus aucun gradient ne remonte. Moyenne
et écart-type sont calculés **uniquement sur l'entraînement**, puis
sauvegardés dans le modèle : sinon on laisse fuiter de l'information de la
validation vers l'entraînement, et `predict.py` ne saurait pas reproduire la
même transformation.

**Le split stratifié.** On conserve la proportion M/B dans les deux
parties, sinon les scores de validation dépendent de la chance du tirage.

**L'initialisation.** À zéro, tous les neurones d'une couche calculent la
même chose et reçoivent le même gradient : ils restent identiques pour
toujours, et la couche ne vaut pas mieux qu'un seul neurone. Il faut donc
des poids aléatoires — mais calibrés (heUniform) pour que les activations
ne saturent pas d'entrée.

**precision / recall.** L'accuracy seule est trompeuse en médecine. Rater un
cas malin (faux négatif) est bien plus grave qu'une fausse alerte : c'est le
**rappel** sur la classe M qu'on regarde en premier.

## Résultats typiques

Avec `--layer 24 24 --epochs 84 --batch_size 8 --learning_rate 0.0314` :
val_loss ≈ 0.09, accuracy de validation ≈ 0.96.
Avec `--optimizer adam --layer 24 24 24 --early-stopping 15` : même niveau
atteint en une vingtaine d'epochs au lieu de 84.

## Note sur `data/data.csv`

Ce fichier est fourni pour pouvoir tester tout de suite. Utilise celui des
ressources du sujet pour la soutenance ; le format attendu est identique
(colonne 0 = identifiant, colonne 1 = M ou B, colonnes 2 à 31 = les
30 features).
