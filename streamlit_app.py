

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, silhouette_score
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

st.title("Data Analysis and Machine Learning Application")

# Επιλογή αρχείου προς φόρτωση
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    # Έλεγχος τύπου αρχείου και φόρτωση δεδομένων
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.write("Data Preview:")
    st.write(data.head())

    if data.shape[1] < 2:
        st.error("The data must have at least two columns: features and a label column.")
    else:
        features = data.columns[:-1]
        label = data.columns[-1]

        st.write("Features:", features)
        st.write("Label:", label)
        st.write("Data Summary:")
        st.write(data.describe())

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["2D Visualization", "EDA", "Classification", "Clustering", "Info", "Results Comparison"])

        with tab1:
            st.subheader("2D Visualization")
            X = data[features]
            y = data[label]
            X_scaled = StandardScaler().fit_transform(X)

            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(X_scaled)
            data['pca-one'] = pca_result[:, 0]
            data['pca-two'] = pca_result[:, 1]

            tsne = TSNE(n_components=2, random_state=42)
            tsne_result = tsne.fit_transform(X_scaled)
            data['tsne-one'] = tsne_result[:, 0]
            data['tsne-two'] = tsne_result[:, 1]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            sns.scatterplot(x='pca-one', y='pca-two', hue=label, data=data, ax=ax1, palette="viridis")
            ax1.set_title('PCA')
            sns.scatterplot(x='tsne-one', y='tsne-two', hue=label, data=data, ax=ax2, palette="viridis")
            ax2.set_title('t-SNE')
            st.pyplot(fig)

        with tab2:
            st.subheader("Exploratory Data Analysis (EDA)")
            st.write("Correlation Matrix:")
            corr = data.corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

            st.write("Pairplot:")
            fig = sns.pairplot(data, hue=label, palette="viridis")
            st.pyplot(fig)

            st.write("Boxplot:")
            fig, ax = plt.subplots()
            sns.boxplot(data=data, ax=ax)
            st.pyplot(fig)

        with tab3:
            st.subheader("Classification")
            algo = st.selectbox("Select Classification Algorithm", ["K-Nearest Neighbors"])
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            if algo == "K-Nearest Neighbors":
                k = st.slider("Number of Neighbors (k)", 1, 20, 5)
                model = KNeighborsClassifier(n_neighbors=k)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.write("Classification Report:")
                st.text(classification_report(y_test, y_pred))

        with tab4:
            st.subheader("Clustering")
            algo = st.selectbox("Select Clustering Algorithm", ["K-Means"])
            if algo == "K-Means":
                k = st.slider("Number of Clusters (k)", 1, 10, 3)
                model = KMeans(n_clusters=k, random_state=42)
                model.fit(X)
                labels = model.labels_
                silhouette_avg = silhouette_score(X, labels)
                st.write(f"Silhouette Score: {silhouette_avg}")
                data['cluster'] = labels
                fig, ax = plt.subplots()
                sns.scatterplot(x='pca-one', y='pca-two', hue='cluster', data=data, palette="viridis", ax=ax)
                ax.set_title('K-Means Clustering')
                st.pyplot(fig)

        with tab5:
            st.subheader("Info")
            st.write("""
                This application was developed by our team to perform data analysis and machine learning tasks.
                - **Loading and Previewing Data**: Load CSV or Excel files.
                - **2D Visualization**: Perform PCA and t-SNE visualizations.
                - **Exploratory Data Analysis**: Show correlation matrix, pairplot, and boxplot.
                - **Classification**: Implement K-Nearest Neighbors classification.
                - **Clustering**: Implement K-Means clustering.
                """)

        with tab6:
            st.subheader("Results Comparison")
            st.write("Compare the performance of different algorithms based on metrics and visualizations.")
            # Detailed implementation for comparing results will be added here

    st.download_button(
        label="Download data as CSV",
        data=data.to_csv(index=False),
        file_name='loaded_data.csv',
        mime='text/csv',
    )
