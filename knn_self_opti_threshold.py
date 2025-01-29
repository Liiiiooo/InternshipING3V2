from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.manifold import TSNE
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

# Chargement des données
with open('true_formatted_cell_dataset.pkl', 'rb') as f:
    labeled_data = pickle.load(f)
with open('reduced_unlabeled_cell_dataset.pkl', 'rb') as f:
    unlabeled_data = pickle.load(f)

# Préparation des données
D_a, y_a, class_names = labeled_data['X'], labeled_data['y'], labeled_data['class_names']
D_u = unlabeled_data['X_unlabeled']

# Paramètres
total_folds = 3  # Validation croisée externe
internal_folds = 3  # Validation croisée interne
max_iterations = 5
confidence_thresholds = [0.6, 0.7, 0.8, 0.9]  # Seuils à tester
skf_external = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=42)
skf_internal = StratifiedKFold(n_splits=internal_folds, shuffle=True, random_state=42)
k = 5

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
            ('classifier', CalibratedClassifierCV(
                KNeighborsClassifier(n_neighbors=k),
                method='isotonic',
                cv=3
            ))
        ])

validation_scores = []
fold_cms = []

for fold, (train_index, val_index) in enumerate(skf_external.split(D_a_balanced, y_a_balanced), 1):
    print(f"\n--- Fold {fold} ---")

    # Séparation des données de validation
    D_a_train, D_a_val = D_a_balanced[train_index], D_a_balanced[val_index]
    y_a_train, y_a_val = y_a_balanced[train_index], y_a_balanced[val_index]

    # Validation croisée interne pour optimiser le seuil
    best_threshold = 0.8
    best_score = 0.0

    for threshold in confidence_thresholds:
        internal_scores = []

        for sub_train_idx, sub_val_idx in skf_internal.split(D_a_train, y_a_train):
            D_sub_train, D_sub_val = D_a_train[sub_train_idx], D_a_train[sub_val_idx]
            y_sub_train, y_sub_val = y_a_train[sub_train_idx], y_a_train[sub_val_idx]

            self_training_model = clone(model)
            self_training_model.fit(D_sub_train, y_sub_train)

            pseudo_labels = self_training_model.predict(D_sub_val)
            confidences = self_training_model.predict_proba(D_sub_val).max(axis=1)

            selected = confidences > threshold
            if selected.sum() == 0:
                continue

            acc = accuracy_score(y_sub_val[selected], pseudo_labels[selected])
            internal_scores.append(acc)
            print(acc)

        mean_score = np.mean(internal_scores) if internal_scores else 0
        if mean_score > best_score:
            best_score = mean_score
            best_threshold = threshold

    print(f"Seuil optimal pour ce pli : {best_threshold}")
    # Copie des données non étiquetées
    D_u_current = D_u.copy()

    # Initialisation du modèle pour ce pli
    self_training_model = clone(model)

    # Suivi des données
    fold_data_info = {
        'labeled_data_count': [len(D_a_train)],
        'unlabeled_data_count': [len(D_u_current)]
    }

    print(f"Données étiquetées initiales : {len(D_a_train)}")
    print(f"Données non étiquetées initiales : {len(D_u_current)}")

    # Processus d'apprentissage automatique
    for iteration in range(max_iterations):
        print(f"\n--- Itération {iteration + 1} ---")

        # Entraînement du modèle
        self_training_model.fit(D_a_train, y_a_train)

        # Prédictions et confidences
        pseudo_labels = self_training_model.predict(D_u_current)
        confidences = self_training_model.predict_proba(D_u_current).max(axis=1)

        # Filtrage des données à haute confiance
        high_confidence_mask = confidences > best_threshold
        D_u_high_conf = D_u_current[high_confidence_mask]
        pseudo_labels_high_conf = pseudo_labels[high_confidence_mask]

        # if len(D_u_high_conf) > batch_size:
        #    D_u_high_conf = D_u_high_conf[:batch_size]
        #    pseudo_labels_high_conf = pseudo_labels_high_conf[:batch_size]

        # Mise à jour des données d'entraînement
        D_a_train = np.concatenate([D_a_train, D_u_high_conf])
        y_a_train = np.concatenate([y_a_train, pseudo_labels_high_conf])

        # Mise à jour des données non étiquetées
        D_u_current = D_u_current[~high_confidence_mask]

        # Journalisation
        print(f"Données étiquetées après itération : {len(D_a_train)}")
        print(f"Données non étiquetées restantes : {len(D_u_current)}")
        print(f"Données ajoutées à haute confiance : {len(D_u_high_conf)}")

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

# Trouver l'indice du pli avec le meilleur score de validation
best_fold_idx = np.argmax(validation_scores)
best_validation_score = validation_scores[best_fold_idx]
print(f"\nMeilleur score de validation : {best_validation_score} (Pli {best_fold_idx + 1})")

# Récupérer les données et les prédictions pour le meilleur pli
best_fold_train_index, best_fold_val_index = list(skf_external.split(D_a_balanced, y_a_balanced))[best_fold_idx]
D_a_train_best, D_a_val_best = D_a_balanced[best_fold_train_index], D_a_balanced[best_fold_val_index]
y_a_train_best, y_a_val_best = y_a_balanced[best_fold_train_index], y_a_balanced[best_fold_val_index]

best_self_training_model = clone(model)

# Processus d'apprentissage automatique
for iteration in range(max_iterations):
    print(f"\n--- Itération {iteration + 1} ---")

    # Entraînement du modèle
    best_self_training_model.fit(D_a_train_best, y_a_train_best)

    # Prédictions et confidences
    pseudo_labels_best = best_self_training_model.predict(D_u)
    confidences_best = best_self_training_model.predict_proba(D_u).max(axis=1)

    # Filtrage des données à haute confiance
    high_confidence_mask_best = confidences_best > best_threshold
    D_u_high_conf_best = D_u[high_confidence_mask_best]
    pseudo_labels_high_conf_best = pseudo_labels_best[high_confidence_mask_best]

    # if len(D_u_high_conf_best) > batch_size:
    #    D_u_high_conf_best = D_u_high_conf_best[:batch_size]
    #    pseudo_labels_high_conf_best = pseudo_labels_high_conf_best[:batch_size]

    # Mise à jour des données d'entraînement
    D_a_train_best = np.concatenate([D_a_train_best, D_u_high_conf_best])
    y_a_train_best = np.concatenate([y_a_train_best, pseudo_labels_high_conf_best])

    # Mise à jour des données non étiquetées
    D_u = D_u[~high_confidence_mask_best]

    # Journalisation
    print(f"Données étiquetées après itération : {len(D_a_train_best)}")
    print(f"Données non étiquetées restantes : {len(D_u)}")
    print(f"Données ajoutées à haute confiance : {len(D_u_high_conf_best)}")

    # Arrêt si aucune nouvelle donnée n'est ajoutée
    if len(D_u_high_conf_best) == 0:
        print("Aucune nouvelle donnée n'a été ajoutée. Arrêt du self-training.")
        break

# Prédictions du modèle sur les données de validation
y_val_pred_best = best_self_training_model.predict(D_a_val_best)

# Affichage du classification_report pour le meilleur pli
report = classification_report(y_a_val_best, y_val_pred_best, target_names=class_names)
print("\nClassification Report pour le meilleur pli :")
print(report)

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
plt.savefig('cv_conf_matrices_knn.png', dpi=300)
plt.close()

# Ajout de la visualisation TSNE globale
#D_combined = np.concatenate([D_a_balanced, D_u])
#labels_combined = np.concatenate([y_a_balanced, np.full(len(D_u), -1)])  # -1 pour différencier les pseudo-labels

for idx, class_names in enumerate(class_names):
    print(f"Index {idx} : Classe : {class_names}")

selected_classes = [0, 1, 2]
mask = np.isin(y_a_train_best, selected_classes)

D_filtered = D_a_train_best[mask]
labels_filtered = y_a_train_best[mask]

# Réduction de dimension avec TSNE
tsne = TSNE(n_components=2, random_state=42)
D_2D = tsne.fit_transform(D_a_train_best)

# Création du scatter plot
plt.figure(figsize=(12, 8))
sns.scatterplot(x=D_2D[:, 0], y=D_2D[:, 1], hue=y_a_train_best, palette='Set1', legend='full')
plt.title("t-SNE des données avec pseudo-labels KNN with optimal threshold")
plt.savefig("tsne_pseudo_labels_global.png", dpi=300)
plt.close()

# Réduction de dimension avec TSNE
tsne = TSNE(n_components=2, random_state=42)
D_2D = tsne.fit_transform(D_filtered)

# Création du scatter plot
plt.figure(figsize=(12, 8))
sns.scatterplot(x=D_2D[:, 0], y=D_2D[:, 1], hue=labels_filtered, palette='Set1', legend='full')
plt.title("t-SNE des données avec pseudo-labels KNN with optimal threshold (class 0 to 2")
plt.savefig("tsne_pseudo_labels_filtered.png", dpi=300)
plt.close()
