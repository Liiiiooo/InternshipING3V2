from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
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
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Équilibrage des classes
balancing_strategy = 'undersampling'
if balancing_strategy == 'oversampling':
    sampler = SMOTE(random_state=42)
elif balancing_strategy == 'undersampling':
    sampler = RandomUnderSampler(random_state=42)

D_a_balanced, y_a_balanced = sampler.fit_resample(D_a, y_a)

# Initialisation du modèle
base_model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Classe AdaMatchSelfTraining avec Logistic Regression
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

            # Arrêt si aucune nouvelle donnée n'a été ajoutée
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

    # Appliquer le masque pour différencier pseudo-labels et non étiquetés
    pseudo_labels = self_training_model.predict(D_u_current)
    confidences = self_training_model.predict_proba(D_u_current).max(axis=1)

    high_conf_mask = confidences > self_training_model.confidence_threshold
    D_u_high_conf = D_u_current[high_conf_mask]
    pseudo_labels_high_conf = pseudo_labels[high_conf_mask]

    # Mise à jour des ensembles de données
    D_a_train = np.concatenate([D_a_train, D_u_high_conf])
    y_a_train = np.concatenate([y_a_train, pseudo_labels_high_conf])

    # Évaluation du modèle
    y_val_pred = self_training_model.predict(D_a_val)
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

    # Préparation des données pour TSNE
    D_all = np.vstack((D_a_train, D_u_current[~high_conf_mask]))
    y_all = np.concatenate((y_a_train, np.full(D_u_current[~high_conf_mask].shape[0], 4)))  # 4 pour "unlabeled"

    # Transformation TSNE
    tsne = TSNE(n_components=2, random_state=42)
    D_2D = tsne.fit_transform(D_all)

    # Scatter plot TSNE
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=D_2D[:, 0], y=D_2D[:, 1], hue=y_all, palette='tab10', legend='full')
    plt.title(f"t-SNE des données (pli {fold}) avec pseudo-labels Logistic Regression + AdaMatch")
    plt.savefig(f"tsne_pseudo_labels_fold_{fold}_AdaMatchLogReg.png", dpi=300)
    plt.close()

    mask = y_all != 4
    D_filtered = D_all[mask]
    labels_filtered = y_all[mask]

    tsne = TSNE(n_components=2, random_state=42)
    D_2D_filtered = tsne.fit_transform(D_filtered)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=D_2D_filtered[:, 0], y=D_2D_filtered[:, 1], hue=labels_filtered, palette='tab10', legend='full')
    plt.title(f"t-SNE des données (pli {fold}) sans la classe 4")
    plt.savefig(f"tsne_pseudo_labels_fold_{fold}_filtered_AdaMatchLogReg.png", dpi=300)
    plt.close()

    # Visualisation t-SNE des données annotées initiales
    tsne = TSNE(n_components=2, random_state=42)
    D_a_2D = tsne.fit_transform(D_a)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=D_a_2D[:, 0], y=D_a_2D[:, 1], hue=y_a, palette='tab10', legend='full')
    plt.title("t-SNE of labeled data")
    plt.savefig(f"tsne_labeled_data_AdaMatchlogreg{fold}.png", dpi=300)
    plt.close()

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
plt.savefig('cv_conf_matrices_logreg_AdaMatch.png', dpi=300)
plt.close()
