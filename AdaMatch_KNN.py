from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Chargement des données
with open('true_formatted_cell_dataset.pkl', 'rb') as f:
    labeled_data = pickle.load(f)
with open('reduced_unlabeled_cell_dataset.pkl', 'rb') as f:
    unlabeled_data = pickle.load(f)

D_a, y_a, class_names = labeled_data['X'], labeled_data['y'], labeled_data['class_names']
D_u = unlabeled_data['X_unlabeled']

# Paramètres
max_iterations = 5
n_folds = 3
k = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Équilibrage des classes
balancing_strategy = 'undersampling'
if balancing_strategy == 'oversampling':
    sampler = SMOTE(random_state=42)
elif balancing_strategy == 'undersampling':
    sampler = RandomUnderSampler(random_state=42)

D_a_balanced, y_a_balanced = sampler.fit_resample(D_a, y_a)


class AdaMatchSelfTraining:
    def __init__(self, base_model, confidence_threshold=0.8, max_iterations=10, alpha=0.5):
        self.model = clone(base_model)
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.alpha = alpha  # Facteur de correction de distribution

    def fit(self, D_a, y_a, D_u):
        D_a_train, y_a_train = D_a.copy(), y_a.copy()
        D_u_current = D_u.copy()

        for iteration in range(self.max_iterations):
            # Entraînement du modèle sur les données étiquetées
            self.model.fit(D_a_train, y_a_train)

            # Prédiction sur les données non étiquetées
            pseudo_labels = self.model.predict(D_u_current)
            confidences = self.model.predict_proba(D_u_current).max(axis=1)

            # Distribution alignment : ajustement des seuils de confiance selon la distribution des classes
            class_distribution = np.bincount(y_a_train, minlength=len(np.unique(y_a)))
            total_samples = np.sum(class_distribution)
            class_proportions = class_distribution / total_samples
            dynamic_thresholds = self.confidence_threshold * (
                        1 + self.alpha * (class_proportions - np.mean(class_proportions)))

            # Sélection des pseudo-labels avec seuils adaptatifs
            high_confidence_mask = confidences > dynamic_thresholds[pseudo_labels]
            D_u_high_conf = D_u_current[high_confidence_mask]
            pseudo_labels_high_conf = pseudo_labels[high_confidence_mask]

            # Mise à jour des données d'entraînement
            D_a_train = np.concatenate([D_a_train, D_u_high_conf])
            y_a_train = np.concatenate([y_a_train, pseudo_labels_high_conf])

            # Suppression des données pseudo-étiquetées
            D_u_current = D_u_current[~high_confidence_mask]

            # Journalisation
            print(f"Données étiquetées après itération : {len(D_a_train)}")
            print(f"Données non étiquetées restantes : {len(D_u_current)}")
            print(f"Données ajoutées à haute confiance : {len(D_u_high_conf)}")

            # Arrêt si aucune nouvelle donnée n'est ajoutée
            if len(D_u_high_conf) == 0:
                print("Aucune nouvelle donnée n'a été ajoutée. Arrêt du self-training.")
                break

        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

validation_scores = []
fold_cms = []

# Initialisation du modèle avec AdaMatch
base_model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', CalibratedClassifierCV(
        KNeighborsClassifier(n_neighbors=k),
        method='isotonic',
        cv=3
    ))
])

for fold, (train_index, val_index) in enumerate(skf.split(D_a_balanced, y_a_balanced), 1):
    print(f"\n--- Fold {fold} ---")

    # Séparation des données de validation
    D_a_train, D_a_val = D_a_balanced[train_index], D_a_balanced[val_index]
    y_a_train, y_a_val = y_a_balanced[train_index], y_a_balanced[val_index]

    # Copie des données non étiquetées
    D_u_current = D_u.copy()

    self_training_model = AdaMatchSelfTraining(base_model)

    # Entraînement avec AdaMatch
    self_training_model.fit(D_a_train, y_a_train, D_u_current)

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
best_fold_train_index, best_fold_val_index = list(skf.split(D_a_balanced, y_a_balanced))[best_fold_idx]
D_a_train_best, D_a_val_best = D_a_balanced[best_fold_train_index], D_a_balanced[best_fold_val_index]
y_a_train_best, y_a_val_best = y_a_balanced[best_fold_train_index], y_a_balanced[best_fold_val_index]


# Entraînement du modèle
best_self_training_model = AdaMatchSelfTraining(base_model)

# Entraînement avec AdaMatch
best_self_training_model.fit(D_a_train_best, y_a_train_best, D_u)

# Évaluation du modèle
y_val_pred_best = best_self_training_model.predict(D_a_val_best)
score = accuracy_score(y_a_val_best, y_val_pred_best)
validation_scores.append(score)
print(f'Score de validation : {score}')

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
plt.savefig('cv_conf_matrices_knn_AdaMatch.png', dpi=300)
plt.close()

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
plt.title("t-SNE des données avec pseudo-labels KNN with AdaMatch")
plt.savefig("tsne_pseudo_labels_global_AdaMatch.png", dpi=300)
plt.close()

# Réduction de dimension avec TSNE
tsne = TSNE(n_components=2, random_state=42)
D_2D = tsne.fit_transform(D_filtered)

# Création du scatter plot
plt.figure(figsize=(12, 8))
sns.scatterplot(x=D_2D[:, 0], y=D_2D[:, 1], hue=labels_filtered, palette='Set1', legend='full')
plt.title("t-SNE des données avec pseudo-labels KNN with AdaMatch (class 0 to 2")
plt.savefig("tsne_pseudo_labels_filtered_AdaMatch.png", dpi=300)
plt.close()
