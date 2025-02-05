from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

# Chargement des données
with open('true_formatted_cell_dataset.pkl', 'rb') as f:
    labeled_data = pickle.load(f)

# Préparation des données
D_a, y_a, class_names = labeled_data['X'], labeled_data['y'], labeled_data['class_names']

# Paramètres
n_folds = 3
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Équilibrage des classes
balancing_strategy = 'undersampling'
if balancing_strategy == 'oversampling':
    sampler = SMOTE(random_state=42)
elif balancing_strategy == 'undersampling':
    sampler = RandomUnderSampler(random_state=42)

D_a_balanced, y_a_balanced = sampler.fit_resample(D_a, y_a)
print(f"Distribution des classes avant équilibrage : {np.bincount(y_a)}")
print(f"Distribution des classes après équilibrage : {np.bincount(y_a_balanced)}")

# Initialisation du modèle
model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

validation_scores = []
fold_cms = []

for fold, (train_index, val_index) in enumerate(skf.split(D_a_balanced, y_a_balanced), 1):
    print(f"\n--- Fold {fold} ---")

    # Séparation des données d'entraînement et de validation
    D_a_train, D_a_val = D_a_balanced[train_index], D_a_balanced[val_index]
    y_a_train, y_a_val = y_a_balanced[train_index], y_a_balanced[val_index]

    new_model = clone(model)

    # Entraînement du modèle
    new_model.fit(D_a_train, y_a_train)

    # Prédiction et évaluation
    y_val_pred = new_model.predict(D_a_val)
    score = accuracy_score(y_a_val, y_val_pred)
    validation_scores.append(score)
    print(f'Score de validation : {score}')

    # Affichage du classification_report pour chaque pli
    report = classification_report(y_a_val, y_val_pred, target_names=class_names)
    print(f"\nClassification Report pour le pli {fold} :")
    print(report)

    # Matrice de confusion
    cm = confusion_matrix(y_a_val, y_val_pred)
    fold_cms.append(cm)

# Résultats finaux
print(f"Scores de validation : {validation_scores}")
print(f"Score moyen : {np.mean(validation_scores)}")

# Visualisation des matrices de confusion
plt.figure(figsize=(20, 15))

for fold, cm in enumerate(fold_cms, 1):
    plt.subplot(2, 3, fold)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Matrice de confusion - Pli {fold}')
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies valeurs')

# Matrice de confusion agrégée (comptages bruts)
plt.subplot(2, 3, 4)
total_cm = np.sum(fold_cms, axis=0)
sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matrice de confusion agrégée (Comptages bruts)')
plt.xlabel('Prédictions')
plt.ylabel('Vraies valeurs')

# Matrice de confusion agrégée (pourcentages)
plt.subplot(2, 3, 5)
total_cm_percent = total_cm.astype('float') / total_cm.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(total_cm_percent, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matrice de confusion agrégée (Pourcentages)')
plt.xlabel('Prédictions')
plt.ylabel('Vraies valeurs')

plt.tight_layout()
plt.savefig('cv_conf_matrices_supervised.png', dpi=300)
plt.close()
