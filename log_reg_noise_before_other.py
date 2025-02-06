from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Chargement des données
with open('true_formatted_cell_dataset.pkl', 'rb') as f:
    labeled_data = pickle.load(f)

D_a, y_a, class_names = labeled_data['X'], labeled_data['y'], labeled_data['class_names']

# Fonction pour ajouter du bruit à la classe 'other'
def add_noise(data, labels, target_class, noise_level=0.1):
    noisy_data = data.copy()
    target_mask = labels == target_class
    noise = noise_level * np.random.randn(*noisy_data[target_mask].shape)
    noisy_data[target_mask] += noise
    return noisy_data

# Déterminer l'indice de la classe 'other'
class_indices = {cls: i for i, cls in enumerate(class_names)}
other_index = class_indices['other']

# Ajout de bruit uniquement à la classe 'other' avant entraînement
D_a_augmented = add_noise(D_a, y_a, other_index, noise_level=0.1)

# Paramètres
n_folds = 3
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

validation_scores = []
fold_cms = []
models_with_scores = []

for fold, (train_index, val_index) in enumerate(skf.split(D_a_augmented, y_a), 1):
    D_a_train, D_a_val = D_a[train_index], D_a[val_index]
    y_a_train, y_a_val = y_a[train_index], y_a[val_index]

    model.fit(D_a_train, y_a_train)
    y_val_pred = model.predict(D_a_val)
    score = accuracy_score(y_a_val, y_val_pred)
    validation_scores.append(score)
    fold_cms.append(confusion_matrix(y_a_val, y_val_pred))
    models_with_scores.append((model, score))

# Affichage de l'accuracy moyenne
aggregate_accuracy = np.mean(validation_scores)
print(f"\nAccuracy moyenne sur tous les plis : {aggregate_accuracy:.4f}")

best_model, best_score = max(models_with_scores, key=lambda x: x[1])
print(f"\nMeilleur modèle obtenu avec un score de validation de : {best_score}")

# Matrices de confusion
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
plt.savefig('cv_conf_matrices_lr_noise_before_other.png', dpi=300)
plt.close()

# Extraction des caractéristiques et visualisation
scaler = StandardScaler()
D_a_scaled = scaler.fit_transform(D_a_augmented)

# PCA
pca = PCA(n_components=2)
D_pca = pca.fit_transform(D_a_scaled)

# TSNE
tsne = TSNE(n_components=2, random_state=42)
D_tsne = tsne.fit_transform(D_a_scaled)

# UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
D_umap = umap_model.fit_transform(D_a_scaled)

# Visualisation
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, D_vis, title in zip(axes, [D_pca, D_tsne, D_umap], ['PCA', 't-SNE', 'UMAP']):
    sns.scatterplot(x=D_vis[:, 0], y=D_vis[:, 1], hue=y_a, palette='tab10', legend='full', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')

# Ajouter une légende commune
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title='Classes', bbox_to_anchor=(1.05, 0.5), loc='center left')

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Laisser de l'espace pour la légende à droite
plt.savefig("Visualisations_noise_lr_other_before.png", dpi=300)
plt.close()
