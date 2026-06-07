# Reconnaissance de caractères chinois manuscrits

Classification de caractères chinois manuscrits (CASIA HWDB) à l'aide de méthodes de machine learning classiques avec réduction de dimensionnalité PCA.

---

## Dataset

**Source :** [CASIA HWDB — Handwritten Chinese Characters (Kaggle)](https://www.kaggle.com/datasets/pascalbliem/handwritten-chinese-character-hanzi-datasets)

Le dataset contient des images PNG de caractères chinois manuscrits, organisées en sous-dossiers par caractère dans `CASIA-HWDB_Train/`. Chaque sous-dossier contient jusqu'à ~600 images d'un même caractère.

---

## Structure du projet

```
.
├── CASIA-HWDB_Train/          # Dataset d'entraînement (images PNG par caractère)
├── pixelstrainXclasse.csv     # CSV généré : matrices aplaties des images
├── notebook.ipynb             # Notebook principal
└── README.md
```

---

## Pré-requis

```bash
pip install scikit-learn scikit-image pandas numpy matplotlib pillow opencv-python
```

---

## Pipeline

### 1. Chargement des données

Les noms de dossiers (caractères chinois) sont encodés en entiers via `sklearn.preprocessing.LabelEncoder`, car les noms de fichiers utilisent un encodage non-standard susceptible de provoquer des conflits.

### 2. Analyse des dimensions d'images

La fonction `largeurhauteurdesimagesparcaractere` calcule, pour chaque caractère :
- L'**écart-type** des dimensions (hauteur et largeur)
- Les dimensions **maximales** observées

> Si l'écart-type est faible, un simple **contourage blanc** (padding) suffit pour normaliser toutes les images à la même taille.

### 3. Normalisation — Contourage blanc

La fonction `countourageBlanc` centre chaque image dans une matrice blanche de taille **210 × 179 px** (dimensions maximales observées dans le dataset).

### 4. Génération du CSV

Chaque image normalisée est aplatie en un vecteur 1D de **37 590 pixels** (`210 × 179`). L'ensemble du dataset est sérialisé dans un fichier CSV :

| Colonne `class` | `index0` | `index1` | … | `index37589` |
|:-:|:-:|:-:|:-:|:-:|
| label entier | valeur pixel | valeur pixel | … | valeur pixel |

> **Attention :** La génération est rapide pour 1 à 3 classes, mais ralentit fortement au-delà. Le fichier CSV résultant peut être très volumineux.

### 5. Réduction de dimensionnalité — PCA

Une **PCA à 50 composantes** est appliquée sur les vecteurs de pixels. Ce choix limite le surapprentissage tout en conservant l'essentiel de l'information.

### 6. Entraînement et évaluation des modèles

Trois algorithmes sont comparés, chacun avec une recherche d'hyperparamètres via `GridSearchCV` (validation croisée à 5 plis) :

| Algorithme | Hyperparamètre exploré |
|---|---|
| **SVC** | Type de noyau : `linear`, `poly`, `rbf`, `sigmoid` |
| **K plus proches voisins (KNN)** | Nombre de voisins `k` : 2, 7, 9, 11, 21, 55, 67, 100, 150, 300 |
| **Random Forest** | Nombre d'arbres `n_estimators` : 2, 7, 9, 11, 21, 55, 67, 100, 150 |

Les performances sont visualisées sous forme de courbes (précision en % selon la valeur de l'hyperparamètre).

---

## Résultats

Les résultats varient selon le nombre de classes utilisées. Pour chaque modèle, le notebook affiche :
- La **précision globale** (`accuracy_score`)
- Le **rapport de classification** détaillé (`classification_report`)
- Un **graphique** des performances en fonction des hyperparamètres

---

## Notes

- Le notebook est conçu pour un usage sur **Windows** (chemins avec `\\`). Adapter les séparateurs si vous utilisez Linux/macOS.
- Remplacer `pixelstrainXclasse.csv` par le nom du fichier correspondant au nombre de classes souhaité (ex. `pixelstrain7classe.csv` pour 7 classes).
- La variable `classes=classes[0:N]` permet de contrôler le nombre de classes utilisées pour la génération du CSV.
