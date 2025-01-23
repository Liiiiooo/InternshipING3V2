from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle


def get_dynamic_knn(X_train):
    n_neighbors = int(np.sqrt(len(X_train)))
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier(n_neighbors=max(1, n_neighbors)))
    ])
    return model


# Chargement des données
with open('true_formatted_cell_dataset.pkl', 'rb') as f:
    labeled_data = pickle.load(f)
with open('reduced_unlabeled_cell_dataset.pkl', 'rb') as f:
    unlabeled_data = pickle.load(f)

# Préparation des données
D_a, y_a, class_names = labeled_data['X'], labeled_data['y'], labeled_data['class_names']
D_u = unlabeled_data['X_unlabeled']

# Paramètres
max_iterations = 5
confidence_threshold = 0.8
n_folds = 3
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

validation_scores = []
fold_cms = []

for fold, (train_index, val_index) in enumerate(skf.split(D_a, y_a), 1):
    print(f"\n--- Fold {fold} ---")

    # Séparation des données de validation
    D_a_train, D_a_val = D_a[train_index], D_a[val_index]
    y_a_train, y_a_val = y_a[train_index], y_a[val_index]

    # Copie des données non étiquetées
    D_u_current = D_u.copy()

    # Initialisation du modèle pour ce pli
    self_training_model = get_dynamic_knn(D_a_train)
    self_training_model.fit(D_a_train, y_a_train)

    print(f"Données étiquetées initiales : {len(D_a_train)}")
    print(f"Données non étiquetées initiales : {len(D_u_current)}")

    # Processus d'apprentissage automatique
    for iteration in range(max_iterations):
        print(f"\n--- Itération {iteration + 1} ---")

        # Prédictions et confidences
        pseudo_labels = self_training_model.predict(D_u_current)
        confidences = self_training_model.predict_proba(D_u_current).max(axis=1)

        # Filtrage des données à haute confiance
        high_confidence_mask = confidences > confidence_threshold
        D_u_high_conf = D_u_current[high_confidence_mask]
        pseudo_labels_high_conf = pseudo_labels[high_confidence_mask]

        # Mise à jour des données d'entraînement
        D_a_train = np.concatenate([D_a_train, D_u_high_conf])
        y_a_train = np.concatenate([y_a_train, pseudo_labels_high_conf])

        # Mise à jour du modèle avec k dynamique et réapprentissage
        self_training_model = get_dynamic_knn(D_a_train)
        self_training_model.fit(D_a_train, y_a_train)

        # Mise à jour des données non étiquetées
        D_u_current = D_u_current[~high_confidence_mask]

        # Journalisation
        print(f"Données étiquetées après itération : {len(D_a_train)}")
        print(f"Données non étiquetées restantes : {len(D_u_current)}")
        print(f"Données ajoutées à haute confiance : {len(D_u_high_conf)}")
        print(f"K utilisé : {self_training_model.named_steps['classifier'].n_neighbors}")

        # Arrêt si aucune nouvelle donnée n'est ajoutée
        if len(D_u_high_conf) == 0:
            print("Aucune nouvelle donnée n'a été ajoutée. Arrêt du self-training.")
            break

    # Évaluation du modèle
    y_val_pred = self_training_model.predict(D_a_val)
    score = accuracy_score(y_a_val, y_val_pred)
    validation_scores.append(score)
    print(f'Score de validation : {score}')

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
plt.savefig('cv_conf_matrices_knn_sqrtN.png', dpi=300)
plt.close()
