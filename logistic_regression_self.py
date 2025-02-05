from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
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
max_iterations = 5
confidence_threshold = 0.8
batch_size = 200
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

    # Séparation des données de validation
    D_a_train, D_a_val = D_a_balanced[train_index], D_a_balanced[val_index]
    y_a_train, y_a_val = y_a_balanced[train_index], y_a_balanced[val_index]

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
        high_confidence_mask = confidences > confidence_threshold
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

    # Affichage du classification_report pour chaque pli
    report = classification_report(y_a_val, y_val_pred, target_names=class_names)
    print(f"\nClassification Report pour le pli {fold} :")
    print(report)

    # Matrice de confusion
    cm = confusion_matrix(y_a_val, y_val_pred)
    fold_cms.append(cm)

    # Préparation des données pour TSNE
    D_all = np.vstack((D_a_train, D_u_current))  # Toutes les données (étiquetées + non étiquetées)
    y_all = np.concatenate((y_a_train, np.full(D_u_current.shape[0], 4)))  # Étiquettes (4 = non étiquetées)

    # Transformation TSNE
    tsne = TSNE(n_components=2, random_state=42)
    D_2D = tsne.fit_transform(D_all)

    # Scatter plot TSNE
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=D_2D[:, 0], y=D_2D[:, 1], hue=y_all, palette='tab10', legend='full')
    plt.title(f"t-SNE des données (pli {fold}) avec pseudo-labels logreg")
    plt.savefig(f"tsne_pseudo_labels_fold_{fold}_logreg.png", dpi=300)
    plt.close()

    mask = y_all != 4
    D_filtered = D_all[mask]
    labels_filtered = y_all[mask]

    tsne = TSNE(n_components=2, random_state=42)
    D_2D_filtered = tsne.fit_transform(D_filtered)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=D_2D_filtered[:, 0], y=D_2D_filtered[:, 1], hue=labels_filtered, palette='tab10',
                    legend='full')
    plt.title(f"t-SNE des données (pli {fold}) sans la classe 0")
    plt.savefig(f"tsne_pseudo_labels_fold_{fold}_filtered_logreg.png", dpi=300)
    plt.close()

    # Visualisation t-SNE des données annotées initiales
    tsne = TSNE(n_components=2, random_state=42)
    D_a_2D = tsne.fit_transform(D_a)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=D_a_2D[:, 0], y=D_a_2D[:, 1], hue=y_a, palette='tab10', legend='full')
    plt.title("t-SNE of labeled data")
    plt.savefig(f"tsne_labeled_datalogreg{fold}.png", dpi=300)
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
plt.savefig('cv_conf_matrices_logreg.png', dpi=300)
plt.close()



