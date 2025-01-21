import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Attention, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

print(tf.__version__)

# Chargement des données
with open('true_formatted_cell_dataset.pkl', 'rb') as file:
    data = pickle.load(file)
X, y = data['X'], data['y']
class_names = data['class_names']

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Définition du modèle avec une couche d'attention avant le MLP
input_layer = Input(shape=(X_train.shape[1],))  # Entrée avec le nombre de features

# Reshape des entrées pour utiliser l'attention
reshaped_input = Reshape((X_train.shape[1], 1))(input_layer)

# Couche d'attention
attention = Attention()([reshaped_input, reshaped_input, reshaped_input])

reshaped_input = Flatten()(attention)

# Réseau MLP classique après attention
reshaped_input = Dense(64, activation='relu')(reshaped_input)  # Couche cachée
reshaped_input = Dense(64, activation='relu')(reshaped_input)  # Couche cachée

output_layer = Dense(len(class_names), activation='softmax')(
    reshaped_input)  # Couche de sortie pour classification multi-classe

# Modèle final
model = Model(inputs=input_layer, outputs=output_layer)

# Compilation du modèle
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Résumé du modèle
model.summary()

# Validation croisée
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def plot_cv_confusion_matrices(model, X, y, cv, class_names):
    # Adapter la figure à 8 axes au cas où
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))  # Layout étendu pour éviter les conflits
    axes = axes.ravel()

    cms = []
    accuracies = []
    cm_aggregated = np.zeros((len(class_names), len(class_names)), dtype=int)

    for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # Création d'une nouvelle instance du modèle
        model_fold = tf.keras.models.clone_model(model)
        model_fold.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        # Split des données
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]

        # Standardisation spécifique à chaque pli
        scaler_fold = StandardScaler()
        X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold)
        X_val_fold_scaled = scaler_fold.transform(X_val_fold)

        # Entraînement du modèle
        model_fold.fit(X_train_fold_scaled, y_train_fold, epochs=10, batch_size=32, verbose=0)

        # Prédictions
        y_pred_fold = np.argmax(model_fold.predict(X_val_fold_scaled), axis=1)

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
    plt.savefig('cv_confusion_matrices_mlp_attention.png')
    plt.close(fig)  # Fermez la figure pour éviter les superpositions

    return cms, cm_aggregated, cm_percentage, accuracies


cms, cm_aggregated, cm_percentage, accuracies = plot_cv_confusion_matrices(model, X_train, y_train, cv, class_names)

# Scaler final ajusté sur tout l'ensemble d'entraînement
scaler_final = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X_train)  # Ajuste et transforme l'ensemble d'entraînement
X_test_scaled = scaler_final.transform(X_test)  # Transforme l'ensemble de test avec le même scaler

# Entraînement final sur tout l'ensemble d'entraînement
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32)

# Évaluation finale sur l'ensemble de test
y_pred_test = np.argmax(model.predict(X_test_scaled), axis=1)
print("\nPerformances sur l'ensemble de test:")
print("-" * 50)
print("\nRapport de classification détaillé:")
print(classification_report(y_test, y_pred_test, target_names=class_names))

# Sauvegarde du modèle
model.save('model_mlp_attention.keras')