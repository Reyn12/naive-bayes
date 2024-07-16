import pandas as pd
import numpy as np
import streamlit as st

# Load dataset
file_path = "data.csv"
df = pd.read_csv(file_path)

# Display first few rows of the dataframe for understanding the data
st.write("Data Preview:", df.head())

# Mapping categorical columns to their original values for display
original_columns = [
    "parental level of education",
    "test preparation course",
    "math score",
    "reading score",
    "writing score",
    "Class_lunch",
]

# Pisahkan fitur dan label
X = df.drop(columns=["Class_lunch"])
y = df["Class_lunch"]


# Implementasi Algoritma Naive Bayes tanpa konversi ke numerik
class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.priors = {}
        self.likelihoods = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = X_c.shape[0] / X.shape[0]
            self.likelihoods[c] = {}
            for col in X.columns:
                self.likelihoods[c][col] = (
                    X_c[col].value_counts(normalize=True).to_dict()
                )

    def predict(self, X):
        y_pred = [self._predict(x) for _, x in X.iterrows()]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.priors[c])
            likelihood = sum(
                [np.log(self.likelihoods[c][col].get(x[col], 1e-6)) for col in x.index]
            )
            posterior = prior + likelihood
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]


# Contoh penggunaan
nb = NaiveBayes()
nb.fit(X, y)

# Buat Aplikasi Streamlit
st.title("Naive Bayes Classifier")

# Menggunakan nilai asli dari tabel untuk dropdown
parental_education_options = df["parental level of education"].unique()
test_preparation_options = df["test preparation course"].unique()
score_options = df["math score"].unique()

parental_education = st.selectbox(
    "Parental Level of Education", parental_education_options
)
test_preparation = st.selectbox("Test Preparation Course", test_preparation_options)
math_score = st.selectbox("Math Score", score_options)
reading_score = st.selectbox("Reading Score", score_options)
writing_score = st.selectbox("Writing Score", score_options)

# Buat DataFrame dari input pengguna
input_data = pd.DataFrame(
    {
        "parental level of education": [parental_education],
        "test preparation course": [test_preparation],
        "math score": [math_score],
        "reading score": [reading_score],
        "writing score": [writing_score],
    }
)

prediction = nb.predict(input_data)

st.write("Prediction: ", prediction[0])
