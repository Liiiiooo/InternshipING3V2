import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Attention, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model


def create_unlabeled_data(X, percentage=0.5):

    n_samples = len(X)
    n_unlabeled = int(n_samples * percentage)
    unlabeled_indices = np.random.choice(n_samples, n_unlabeled, replace=False)
    return unlabeled_indices


def get_high_confidence_predictions(model, X_unlabeled, confidence_threshold=0.8):
    # Obtenir les probabilités de prédiction
    probabilities = model.predict(X_unlabeled)
    max_probs = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)

    # Sélectionner les indices où la confiance est supérieure au seuil
    confident_indices = np.where(max_probs >= confidence_threshold)[0]

    return confident_indices, predictions[confident_indices]


def build_mlp_attention_model(input_shape, num_classes):

    input_layer = Input(shape=(input_shape,))  # Entrée avec le nombre de features
    reshaped_input = Reshape((input_shape, 1))(input_layer)
    attention = Attention()([reshaped_input, reshaped_input, reshaped_input])
    reshaped_input = Flatten()(attention)
    reshaped_input = Dense(64, activation='relu')(reshaped_input)  # Couche cachée
    reshaped_input = Dense(64, activation='relu')(reshaped_input)  # Couche cachée
    output_layer = Dense(len(class_names), activation='softmax')(
        reshaped_input)  # Couche de sortie pour classification multi-classe
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def plot_cv_confusion_matrices_pretrained(model, X, y, cv, class_names):
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))  # Layout étendu pour éviter les conflits
    axes = axes.ravel()

    cms = []
    accuracies = []
    cm_aggregated = np.zeros((len(class_names), len(class_names)), dtype=int)

    for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # Pas de réinitialisation ou réentraînement du modèle ici
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]

        # Prédictions directement avec le modèle final
        y_pred_fold = np.argmax(model.predict(X_val_fold), axis=1)

        # Calcul de l'accuracy
        acc = accuracy_score(y_val_fold, y_pred_fold)
        accuracies.append(acc)
        print(f"Accuracy pli {idx + 1}: {acc:.3f}")

        # Calcul de la matrice de confusion
        cm = confusion_matrix(y_val_fold, y_pred_fold)
        cms.append(cm)
        cm_aggregated += cm

        # Affichage de la matrice pour le pli actuel
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[idx])
        axes[idx].set_title(f'Pli {idx + 1} (Acc: {acc:.3f})')
        axes[idx].set_ylabel('Vraie classe')
        axes[idx].set_xlabel('Classe prédite')

    # Matrice de confusion agrégée brute
    sns.heatmap(cm_aggregated, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[len(cms)])
    axes[len(cms)].set_title(f'Matrice de confusion agrégée (Acc: {np.mean(accuracies):.3f})')
    axes[len(cms)].set_ylabel('Vraie classe')
    axes[len(cms)].set_xlabel('Classe prédite')

    # Matrice de confusion en pourcentages
    cm_percentage = cm_aggregated.astype('float') / cm_aggregated.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[len(cms) + 1])
    axes[len(cms) + 1].set_title('Matrice de confusion agrégée en pourcentages')
    axes[len(cms) + 1].set_ylabel('Vraie classe')
    axes[len(cms) + 1].set_xlabel('Classe prédite')

    plt.tight_layout()
    plt.savefig('cv_confusion_matrices_mlp_attention_self.png')
    plt.close(fig)  # Fermez la figure pour éviter les superpositions

    return cms, cm_aggregated, cm_percentage, accuracies


def self_training(X, y, unlabeled_percentage=0.3, n_iterations=5, confidence_threshold=0.8):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Création du jeu de données non étiqueté
    unlabeled_indices = create_unlabeled_data(X_train, unlabeled_percentage)
    X_labeled = np.delete(X_train, unlabeled_indices, axis=0)
    y_labeled = np.delete(y_train, unlabeled_indices, axis=0)
    X_unlabeled = X_train[unlabeled_indices]

    # Création du modèle
    model = build_mlp_attention_model(X_labeled.shape[1], len(np.unique(y)))

    # Boucle de self-training
    for iteration in range(n_iterations):
        print(f"\nItération {iteration + 1}/{n_iterations}")
        print("-" * 50)

        # Entraînement sur les données étiquetées
        model.fit(X_labeled, y_labeled, epochs=10, batch_size=32, verbose=0)

        # Prédiction sur les données non étiquetées
        confident_indices, confident_predictions = get_high_confidence_predictions(
            model, X_unlabeled, confidence_threshold
        )

        if len(confident_indices) == 0:
            print("Aucune prédiction avec une confiance suffisante trouvée.")
            continue

        # Ajout des nouvelles données étiquetées
        X_new_labeled = X_unlabeled[confident_indices]
        y_new_labeled = confident_predictions

        X_labeled = np.vstack((X_labeled, X_new_labeled))
        y_labeled = np.concatenate((y_labeled, y_new_labeled))

        # Mise à jour des données non étiquetées
        X_unlabeled = np.delete(X_unlabeled, confident_indices, axis=0)

        print(f"Ajout de {len(confident_indices)} nouveaux exemples étiquetés")
        print(f"Nombre total d'exemples étiquetés: {len(y_labeled)}")

        # Évaluation sur l'ensemble de test
        y_pred_test = model.predict(X_test)
        accuracy = accuracy_score(y_test, np.argmax(y_pred_test, axis=1))
        print(f"Accuracy sur l'ensemble de test: {accuracy:.3f}")

    # Validation croisée finale
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cms, cm_aggregated, cm_percentage, accuracies = plot_cv_confusion_matrices_pretrained(
        model, X_labeled, y_labeled, cv, class_names
    )

    # Évaluation finale sur l'ensemble de test
    print("\nPerformances finales sur l'ensemble de test:")
    y_pred_test = model.predict(X_test)
    print("\nRapport de classification détaillé:")
    print(classification_report(y_test, np.argmax(y_pred_test, axis=1), target_names=class_names))

    return model, X_labeled, y_labeled


# Chargement des données
with open('true_formatted_cell_dataset.pkl', 'rb') as file:
    data = pickle.load(file)
X, y = data['X'], data['y']
class_names = data['class_names']

# Exécution du self-training
final_model, X_final, y_final = self_training(
    X,
    y,
    unlabeled_percentage=0.3,
    n_iterations=5,
    confidence_threshold=0.8
)

# Sauvegarde du modèle final
with open('model_mlp_attention_self_trained.pkl', 'wb') as f:
    pickle.dump({
        'model': final_model,
        'X_final': X_final,
        'y_final': y_final
    }, f)
