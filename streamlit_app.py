 # streamlit_app.py

pip install streamlit pandas openpyxl

import streamlit as st
import pandas as pd

st.title("Data Loader Application")

# Επιλογή αρχείου προς φόρτωση
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    # Έλεγχος τύπου αρχείου και φόρτωση δεδομένων
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    # Εμφάνιση προεπισκόπησης δεδομένων
    st.write("Data Preview:")
    st.write(data.head())

    # Εμφάνιση βασικών πληροφοριών για τα δεδομένα
    st.write("Summary Statistics:")
    st.write(data.describe())

    # Εμφάνιση στήλες και σχήμα των δεδομένων
    st.write("Columns:")
    st.write(data.columns)
    st.write("Shape:")
    st.write(data.shape)

    # Παροχή δυνατότητας λήψης των φορτωμένων δεδομένων ως CSV
    st.download_button(
        label="Download data as CSV",
        data=data.to_csv(index=False),
        file_name='loaded_data.csv',
        mime='text/csv',
    )
import streamlit as st
import pandas as pd

st.title("Data Loader Application")

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

    # Έλεγχος προδιαγραφών πίνακα
    if data.shape[1] < 2:
        st.error("The data must have at least two columns: features and a label column.")
    else:
        # Διαχωρισμός χαρακτηριστικών και ετικέτας
        features = data.columns[:-1]
        label = data.columns[-1]

        st.write("Features:", features)
        st.write("Label:", label)

        st.write("Data Summary:")
        st.write(data.describe())

        # Εμφάνιση των στηλών και σχήμα των δεδομένων
        st.write("Columns:")
        st.write(data.columns)
        st.write("Shape:")
        st.write(data.shape)

        # Παροχή δυνατότητας λήψης των φορτωμένων δεδομένων ως CSV
        st.download_button(
            label="Download data as CSV",
            data=data.to_csv(index=False),
            file_name='loaded_data.csv',
            mime='text/csv',
        )

pip install streamlit pandas matplotlib seaborn scikit-learn

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

st.title("Data Analysis and Visualization Application")

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

    # Έλεγχος προδιαγραφών πίνακα
    if data.shape[1] < 2:
        st.error("The data must have at least two columns: features and a label column.")
    else:
        # Διαχωρισμός χαρακτηριστικών και ετικέτας
        features = data.columns[:-1]
        label = data.columns[-1]

        st.write("Features:", features)
        st.write("Label:", label)

        st.write("Data Summary:")
        st.write(data.describe())

        # Εμφάνιση των στηλών και σχήμα των δεδομένων
        st.write("Columns:")
        st.write(data.columns)
        st.write("Shape:")
        st.write(data.shape)

        # Tab για 2D Visualizations
        tab1, tab2 = st.tabs(["2D Visualization", "EDA"])

        with tab1:
            st.subheader("2D Visualization")

            # Standardize the features
            X = data[features]
            y = data[label]
            X_scaled = StandardScaler().fit_transform(X)

            # PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(X_scaled)
            data['pca-one'] = pca_result[:, 0]
            data['pca-two'] = pca_result[:, 1]

            # t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            tsne_result = tsne.fit_transform(X_scaled)
            data['tsne-one'] = tsne_result[:, 0]
            data['tsne-two'] = tsne_result[:, 1]

            # Plotting
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            sns.scatterplot(x='pca-one', y='pca-two', hue=label, data=data, ax=ax1, palette="viridis")
            ax1.set_title('PCA')

            sns.scatterplot(x='tsne-one', y='tsne-two', hue=label, data=data, ax=ax2, palette="viridis")
            ax2.set_title('t-SNE')

            st.pyplot(fig)

        with tab2:
            st.subheader("Exploratory Data Analysis (EDA)")

            # Correlation Matrix
            st.write("Correlation Matrix:")
            corr = data.corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

            # Pairplot
            st.write("Pairplot:")
            fig = sns.pairplot(data, hue=label, palette="viridis")
            st.pyplot(fig)

            # Boxplot
            st.write("Boxplot:")
            fig, ax = plt.subplots()
            sns.boxplot(data=data, ax=ax)
            st.pyplot(fig)

        # Παροχή δυνατότητας λήψης των φορτωμένων δεδομένων ως CSV
        st.download_button(
            label="Download data as CSV",
            data=data.to_csv(index=False),
            file_name='loaded_data.csv',
            mime='text/csv',
        )

