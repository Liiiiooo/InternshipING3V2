import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Attention, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model


def build_mlp_attention_model(input_shape, num_classes):
    input_layer = Input(shape=(input_shape,))
    reshaped_input = Reshape((input_shape, 1))(input_layer)
    attention = Attention()([reshaped_input, reshaped_input, reshaped_input])
    reshaped_input = Flatten()(attention)
    reshaped_input = Dense(64, activation='relu')(reshaped_input)
    reshaped_input = Dense(64, activation='relu')(reshaped_input)
    output_layer = Dense(num_classes, activation='softmax')(reshaped_input)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def get_high_confidence_predictions(model, X_unlabeled, confidence_threshold=0.8):
    probabilities = model.predict(X_unlabeled)
    max_probs = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    confident_indices = np.where(max_probs >= confidence_threshold)[0]
    return confident_indices, predictions[confident_indices]


def self_training_with_external_unlabeled(X_labeled, y_labeled, X_unlabeled, n_iterations=5, confidence_threshold=0.8):
    # Séparation des données étiquetées en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2, random_state=42,
                                                        stratify=y_labeled)

    # Initialisation et ajustement du scaler sur les données d'entraînement
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_unlabeled_scaled = scaler.transform(X_unlabeled)

    # Création du modèle
    model = build_mlp_attention_model(X_train_scaled.shape[1], len(np.unique(y_labeled)))

    # Variables pour suivre les données non étiquetées restantes
    remaining_unlabeled = X_unlabeled_scaled.copy()
    X_current_labeled = X_train_scaled.copy()
    y_current_labeled = y_train.copy()

    # Boucle de self-training
    for iteration in range(n_iterations):
        print(f"\nItération {iteration + 1}/{n_iterations}")
        print("-" * 50)

        # Entraînement sur les données étiquetées actuelles
        model.fit(X_current_labeled, y_current_labeled, epochs=10, batch_size=32, verbose=0)

        # Prédiction sur les données non étiquetées restantes
        confident_indices, confident_predictions = get_high_confidence_predictions(
            model, remaining_unlabeled, confidence_threshold
        )

        if len(confident_indices) == 0:
            print("Aucune prédiction avec une confiance suffisante trouvée.")
            break

        # Ajout des nouvelles données étiquetées
        X_new_labeled = remaining_unlabeled[confident_indices]
        y_new_labeled = confident_predictions

        X_current_labeled = np.vstack((X_current_labeled, X_new_labeled))
        y_current_labeled = np.concatenate((y_current_labeled, y_new_labeled))

        # Mise à jour des données non étiquetées restantes
        remaining_unlabeled = np.delete(remaining_unlabeled, confident_indices, axis=0)

        print(f"Ajout de {len(confident_indices)} nouveaux exemples étiquetés")
        print(f"Nombre total d'exemples étiquetés: {len(y_current_labeled)}")
        print(f"Nombre d'exemples non étiquetés restants: {len(remaining_unlabeled)}")

    return model, scaler, X_current_labeled, y_current_labeled, X_test_scaled, y_test


def evaluate_with_cross_validation(X, y, input_shape, num_classes, model, class_names, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cms, accuracies = [], []
    cm_aggregated = np.zeros((num_classes, num_classes), dtype=int)
    cm_aggregated_percent = np.zeros((num_classes, num_classes), dtype=float)

    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    axes = axes.ravel()

    for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Création d'une nouvelle instance du modèle
        model_fold = tf.keras.models.clone_model(model)
        model_fold.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Entraînement du modèle sur le pli d'entraînement
        model_fold.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, verbose=0)

        # Prédictions sur l'ensemble de validation
        y_pred_val = np.argmax(model_fold.predict(X_val_fold), axis=1)
        acc = accuracy_score(y_val_fold, y_pred_val)
        accuracies.append(acc)

        # Matrice de confusion
        cm = confusion_matrix(y_val_fold, y_pred_val)
        cms.append(cm)
        cm_aggregated += cm

        # Affichage de la matrice de confusion brute
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[idx])
        axes[idx].set_title(f'Pli {idx + 1} (Acc: {acc:.3f})')

    # Matrice agrégée (brute)
    sns.heatmap(cm_aggregated, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[len(cms)])
    axes[len(cms)].set_title(f'Matrice de confusion agrégée (Acc: {np.mean(accuracies):.3f})')

    # Matrice agrégée en pourcentage
    # Calcul de la matrice de confusion en pourcentage
    cm_aggregated_percent = np.zeros((num_classes, num_classes), dtype=float)
    row_sums = cm_aggregated.sum(axis=1, keepdims=True)
    # Éviter la division par zéro
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_aggregated_percent = np.where(row_sums > 0,
                                         (cm_aggregated.astype('float') / row_sums) * 100,
                                         0)

    # Pour gérer les cas où row_sums est 0 (remplacer NaN par 0)
    cm_aggregated_percent = np.nan_to_num(cm_aggregated_percent)
    sns.heatmap(cm_aggregated_percent, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[len(cms) + 1])
    axes[len(cms) + 1].set_title(f'Matrice de confusion agrégée (en %) (Acc: {np.mean(accuracies):.3f})')

    plt.tight_layout()
    plt.savefig('cv_confusion_matrices_mlp_attention_self2.png')
    plt.close(fig)

    return accuracies, cm_aggregated, cm_aggregated_percent


# Chargement des données
with open('true_formatted_cell_dataset.pkl', 'rb') as file:
    labeled_data = pickle.load(file)
X_labeled, y_labeled = labeled_data['X'], labeled_data['y']
class_names = labeled_data['class_names']

with open('reduced_unlabeled_cell_dataset.pkl', 'rb') as file:
    unlabeled_data = pickle.load(file)
X_unlabeled = unlabeled_data['X_unlabeled']

# Exécution de l'auto-formation
final_model, scaler, X_final, y_final, X_test_scaled, y_test = self_training_with_external_unlabeled(
    X_labeled, y_labeled, X_unlabeled, n_iterations=5, confidence_threshold=0.8
)

# Validation croisée sur les données étiquetées de base
accuracies, cm_aggregated, cm_aggregated_percentage = evaluate_with_cross_validation(
    X_labeled, y_labeled, input_shape=X_labeled.shape[1], num_classes=len(class_names), model=final_model, class_names=class_names)

# Évaluation finale sur l'ensemble de test
print("\nPerformances finales sur l'ensemble de test :")
y_pred_test = np.argmax(final_model.predict(X_test_scaled), axis=1)
print(classification_report(y_test, y_pred_test, target_names=class_names))
