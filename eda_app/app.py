# app.py

import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
class_names = iris.target_names

# Set Matplotlib backend
matplotlib.use('Agg')

# Streamlit App
st.title("Iris EDA App")

# Display dataset
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.write(iris_df)

# Sidebar with user options
st.sidebar.header("Select Features and Visualizations")
selected_features = st.sidebar.multiselect("Select features for Pairplot:", iris_df.columns[:-1])

# Check if features are selected
if not selected_features:
    st.warning("Please select at least one feature.")
else:
 
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    corr_matrix = iris_df[selected_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    st.pyplot(fig)

    # Boxplot
    st.subheader("Boxplot")
    for feature in selected_features:
        st.write(f"Boxplot for {feature}")
        fig, ax = plt.subplots()
        sns.boxplot(x='target', y=feature, data=iris_df)
        st.pyplot(fig)

    # Scatter Plot
    if len(selected_features) == 2:
        st.subheader("Scatter Plot")
        fig, ax = plt.subplots()
        sns.scatterplot(x=selected_features[0], y=selected_features[1], hue='target', data=iris_df)
        st.pyplot(fig)
    else:
        st.warning("Please select exactly two features for the Scatter Plot.")
