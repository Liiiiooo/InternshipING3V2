import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.base import clone


def load_and_scale_data():
    """Charger et normaliser les données étiquetées et non étiquetées, avec un découpage en ensemble de test."""
    with open('true_formatted_cell_dataset.pkl', 'rb') as f:
        labeled_data = pickle.load(f)
    with open('reduced_unlabeled_cell_dataset.pkl', 'rb') as f:
        unlabeled_data = pickle.load(f)

    X_labeled = labeled_data['X']
    y_labeled = labeled_data['y']
    X_unlabeled = unlabeled_data['X_unlabeled']

    # Découper les données étiquetées en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X_labeled, y_labeled, test_size=0.2, random_state=42, stratify=y_labeled
    )

    # Normaliser les données pour améliorer la performance du modèle
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_unlabeled = scaler.transform(X_unlabeled)

    return X_train, X_test, y_train, y_test, X_unlabeled


def iterative_self_training(X_labeled, y_labeled, X_unlabeled, n_neighbors, confidence_threshold=0.9, max_iterations=3, min_samples_per_class=10):
    """Effectuer un entraînement itératif auto-supervisé avec vérification de l'équilibre des classes et retourner le modèle entraîné."""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    X_train_augmented = X_labeled.copy()
    y_train_augmented = y_labeled.copy()
    X_remaining = X_unlabeled.copy()

    for iteration in range(max_iterations):
        if len(X_remaining) == 0:
            break

        # Entraîner le modèle sur les données étiquetées actuelles
        knn.fit(X_train_augmented, y_train_augmented)

        # Générer des pseudo-étiquettes
        probs = knn.predict_proba(X_remaining)
        max_probs = np.max(probs, axis=1)
        pseudo_labels = knn.predict(X_remaining)

        # Filtrer les prédictions confiantes
        confident_idx = max_probs >= confidence_threshold

        if not any(confident_idx):
            break

        # Vérifier la distribution des classes
        new_samples = {}
        for class_label in np.unique(y_labeled):
            class_idx = (pseudo_labels == class_label) & confident_idx
            if sum(class_idx) >= min_samples_per_class:
                new_samples[class_label] = class_idx

        if not new_samples:
            break

        # Ajouter les nouveaux échantillons aux données d'entraînement
        mask = np.zeros(len(X_remaining), dtype=bool)
        for class_idx in new_samples.values():
            mask |= class_idx

        X_train_augmented = np.vstack((X_train_augmented, X_remaining[mask]))
        y_train_augmented = np.concatenate((y_train_augmented, pseudo_labels[mask]))
        X_remaining = X_remaining[~mask]

    return knn


def cross_validate_self_training(X_labeled, y_labeled, X_unlabeled, n_splits=5, n_neighbors=5):
    """Effectuer une validation croisée sur les données étiquetées de base après un self-training sur les données étiquetées et non étiquetées."""
    # Faire le self-training avant la validation croisée
    model = iterative_self_training(
        X_labeled, y_labeled, X_unlabeled, n_neighbors=n_neighbors
    )

    # Validation croisée avec les données étiquetées de base
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    models = []  # Stocker les modèles pour chaque pli

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_labeled, y_labeled), 1):
        # Diviser les données en sous-ensembles d'entraînement et de validation (sur les données étiquetées de base)
        X_train, X_val = X_labeled[train_idx], X_labeled[val_idx]
        y_train, y_val = y_labeled[train_idx], y_labeled[val_idx]

        # Cloner le modèle de self-training (pour éviter de réentraîner le modèle pendant la validation croisée)
        model_clone = clone(model)

        # Entraîner le modèle cloné sur les données d'entraînement du pli
        model_clone.fit(X_train, y_train)

        # Prédictions et évaluation sur l'ensemble de validation
        y_pred = model_clone.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        conf_matrix = confusion_matrix(y_val, y_pred)

        # Stocker les résultats pour ce pli
        results.append({
            'fold': fold,
            'y_true': y_val,
            'y_pred': y_pred,
            'conf_matrix': conf_matrix,
            'accuracy': accuracy
        })
        models.append(model_clone)

        print(f"Pli {fold} - Accuracy : {accuracy:.4f}")

    return results, models


def visualize_results(results, n_classes, n_neighbors):
    """Visualiser les matrices de confusion et les précisions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Tracer les matrices individuelles pour chaque pli
    global_matrix = np.zeros((n_classes, n_classes))
    for i, result in enumerate(results):
        if i < 5:  # Tracer les 5 premiers plis seulement
            sns.heatmap(result['conf_matrix'], annot=True, fmt='d',
                        cmap='Blues', ax=axes[i], cbar=False)
            axes[i].set_title(f"Pli {i + 1} (Accuracy : {result['accuracy']:.3f})")
        global_matrix += result['conf_matrix']

    # Tracer la matrice globale
    sns.heatmap(global_matrix, annot=True, fmt='.1f', cmap='Blues',
                ax=axes[-1], cbar=True)
    axes[-1].set_title(f"Global (Accuracy moyenne : {np.mean([r['accuracy'] for r in results]):.3f})")

    plt.tight_layout()
    plt.savefig(f'cv_conf_matrices_knn_k={n_neighbors}_self.png')
    plt.close()


def evaluate_final_performance(y_test, y_pred, n_neighbors):
    """Évaluer la performance finale avec plusieurs métriques."""
    print("\nPerformance finale sur le jeu de test :")
    print("-" * 50)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred))
    print("\nMatrice de confusion :")
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de confusion - Jeu de test')
    plt.savefig(f'test_confusion_matrix_knn_k={n_neighbors}_self.png')
    plt.close()
    print(f"\nMatrice de confusion enregistrée dans test_confusion_matrix_knn_k={n_neighbors}_self.png")


# Exécution principale
n_neighbors = 5
X_train, X_test, y_train, y_test, X_unlabeled = load_and_scale_data()

# Validation croisée sur les données d'entraînement
results, models = cross_validate_self_training(X_train, y_train, X_unlabeled)

final_model = models[-1]  # Choisir le dernier modèle cloné

# Prédictions sur le jeu de test
y_pred = final_model.predict(X_test)

# Évaluation finale sur le jeu de test
evaluate_final_performance(y_test, y_pred, n_neighbors)
