import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Configuration de la page Streamlit
st.set_page_config(
    page_title="TP1 - LDA sur MNIST", layout="wide", initial_sidebar_state="expanded"
)

st.title("🧪 TP n°1 : LDA (Analyse Discriminante Linéaire) sur MNIST")
st.write(
    "Application interactive de classification binaire : **Chiffre 0** (Classe Positive) vs **Non-0** (Classe Négative)."
)


# 1. Chargement et préparation des données avec mise en cache
@st.cache_data
def load_and_prep_data():
    # Chargement de MNIST (784 pixels)
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = mnist.data / 255.0, mnist.target.astype(int)

    # Binarisation : 1 si le chiffre est 0, 0 sinon (non-0)
    y_binary = np.where(y == 0, 1, 0)

    # Entraînement du modèle LDA (réduction à 1 dimension)
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(X, y_binary)

    return X, y, y_binary, lda, X_lda


with st.spinner("Chargement et entraînement de la LDA sur MNIST..."):
    X, y, y_binary, lda, X_lda = load_and_prep_data()

# 2. Interface Streamlit à 2 Colonnes
col_left, col_right = st.columns([1, 2])

# --- PANNEAU DE CONTRÔLE (A gauche) ---
with col_left:
    st.header("🎛️ Panneau de Contrôle")

    mode = st.radio(
        "Sélectionnez le mode de visualisation :",
        ["Test 1 : Projection LDA (1D)", "Test 2 : Prédiction Interactive"],
    )

    img_idx = st.slider(
        "Index de l'image de test :",
        min_value=0,
        max_value=len(X) - 1,
        value=0,
        step=1,
    )

# --- VISUALISATION (A droite) ---
with col_right:
    st.header("📊 Visualisation")

    # Calcul de la frontière de décision
    # Pour LDA 1D dans scikit-learn, le seuil se situe à la valeur de décision = 0
    decision_threshold = 0.0
    scores = lda.decision_function(X)

    if mode == "Test 1 : Projection LDA (1D)":
        st.subheader("Visualisation de la séparation des classes (1D)")

        fig, ax = plt.subplots(figsize=(8, 4))

        # Subsampling pour accélérer l'affichage de l'histogramme
        sample_indices = np.random.choice(len(scores), size=5000, replace=False)

        scores_sample = scores[sample_indices]
        y_sample = y_binary[sample_indices]

        # Tracé des histogrammes
        ax.hist(
            scores_sample[y_sample == 1],
            bins=40,
            alpha=0.6,
            color="green",
            label="Classe 0 (Zero)",
        )
        ax.hist(
            scores_sample[y_sample == 0],
            bins=40,
            alpha=0.6,
            color="blue",
            label="Classe Non-0",
        )

        # Ligne de frontière de décision
        ax.axvline(
            x=decision_threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Frontière de décision",
        )

        ax.set_xlabel("Score de projection LDA 1D")
        ax.set_ylabel("Nombre d'échantillons")
        ax.set_title("Séparation optimale des classes par la LDA")
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

    elif mode == "Test 2 : Prédiction Interactive":
        st.subheader("Test individuel d'une image")

        # Récupération de l'image sélectionnée
        img_data = X[img_idx].reshape(28, 28)
        true_label = y[img_idx]
        is_zero_true = y_binary[img_idx]

        # Prédiction
        pred_binary = lda.predict([X[img_idx]])[0]

        # Affichage de l'image
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(img_data, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

        # Rendu du verdict
        st.write(f"**Vrai chiffre dans le dataset :** `{true_label}`")
        st.write(
            f"**Classe réelle :** `{'Classe 0' if is_zero_true == 1 else 'Classe Non-0'}`"
        )

        if pred_binary == is_zero_true:
            st.success("✅ **SUCCÈS :** La LDA a correctement classé cette image !")
        else:
            st.error("❌ **ÉCHEC :** Erreur de classification de la LDA.")