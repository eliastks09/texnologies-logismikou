import streamlit as st
import pandas as pd
import numpy as np

st.title('Data Analysis App')

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

# Προσθήκη 2D Visualization Tab
def visualize_2d(data):
    st.header("2D Visualizations")

    # PCA
    st.subheader("PCA")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data.iloc[:, :-1])
    pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
    pca_df['label'] = data.iloc[:, -1].values
    fig, ax = plt.subplots()
    sns.scatterplot(x='PCA1', y='PCA2', hue='label', data=pca_df, ax=ax)
    st.pyplot(fig)

    # t-SNE
    st.subheader("t-SNE")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_result = tsne.fit_transform(data.iloc[:, :-1])
    tsne_df = pd.DataFrame(data=tsne_result, columns=['TSNE1', 'TSNE2'])
    tsne_df['label'] = data.iloc[:, -1].values
    fig, ax = plt.subplots()
    sns.scatterplot(x='TSNE1', y='TSNE2', hue='label', data=tsne_df, ax=ax)
    st.pyplot(fig)

    # Exploratory Data Analysis
    st.subheader("Exploratory Data Analysis")
    st.write("Παρουσίαση 2-3 διαγραμμάτων EDA:")
    
    # Παράδειγμα EDA 1: Διασπορά χαρακτηριστικών
    st.write("Διάγραμμα 1: Διασπορά Χαρακτηριστικών")
    fig, ax = plt.subplots()
    sns.pairplot(data.iloc[:, :-1])
    st.pyplot(fig)
    
    # Παράδειγμα EDA 2: Ιστόγραμμα για κάθε χαρακτηριστικό
    st.write("Διάγραμμα 2: Ιστόγραμμα Χαρακτηριστικών")
    fig, ax = plt.subplots()
    data.iloc[:, :-1].hist(ax=ax)
    st.pyplot(fig)

    # Παράδειγμα EDA 3: Θερμότητα χαρακτηριστικών (Heatmap)
    st.write("Διάγραμμα 3: Θερμότητα Χαρακτηριστικών")
    fig, ax = plt.subplots()
    sns.heatmap(data.iloc[:, :-1].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Ενημέρωση Streamlit app
st.title("Data Analysis App")
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)

    st.write("Data Preview:")
    st.write(data.head())

    visualize_2d(data)
