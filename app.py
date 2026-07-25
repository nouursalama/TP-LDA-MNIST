import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Indispensable pour les serveurs headless comme Streamlit
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

st.title("TP LDA - MNIST (0 vs 1)")

# Chargement optimisé avec sous-échantillonnage pour économiser la RAM
@st.cache_data
def load_data():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    # Filtrer uniquement pour conserver les 0 et les 1
    mask = (y == '0') | (y == '1')
    X_filtered, y_filtered = X[mask], y[mask]
    # Prendre 5 000 images max pour éviter le crash mémoire (OOM)
    return X_filtered[:5000] / 255.0, y_filtered[:5000]

with st.spinner("Chargement des données MNIST..."):
    X, y = load_data()

st.success("Données chargées avec succès !")

# Entraînement de la LDA
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X, y)

# Visualisation Streamlit
st.subheader("Projection 1D de la LDA")
fig, ax = plt.subplots(figsize=(8, 4))
for label in np.unique(y):
    ax.hist(X_lda[y == label], bins=30, alpha=0.6, label=f"Chiffre {label}")
ax.legend()
ax.set_title("Séparation des classes 0 et 1 via LDA")

st.pyplot(fig)
        )

        if pred_binary == is_zero_true:
            st.success("✅ **SUCCÈS :** La LDA a correctement classé cette image !")
        else:
            st.error("❌ **ÉCHEC :** Erreur de classification de la LDA.")
