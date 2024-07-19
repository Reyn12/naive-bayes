import pandas as pd
import numpy as np
import streamlit as st

# Load dataset
file_path = "data.csv"
df = pd.read_csv(file_path)

# Display first few rows of the dataframe for understanding the data
st.write("Pratinjau Data:", df.head())

# Mapping kolom kategorikal ke nilai aslinya untuk ditampilkan
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

# Hitung probabilitas prior
total_count = len(df)
class_counts = df["Class_lunch"].value_counts()
prior_probabilities = class_counts / total_count

# Buat DataFrame untuk menampilkan probabilitas prior
prior_df = pd.DataFrame(
    {
        "Kelas": class_counts.index,
        "Jumlah": class_counts.values,
        "Probabilitas Prior": prior_probabilities.values,
    }
)


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
st.title("Klasifikasi Naive Bayes")

# Input pengguna dalam dua baris
col1, col2 = st.columns(2)

with col1:
    pendidikan_orang_tua = st.selectbox(
        "Pendidikan Orang Tua", df["parental level of education"].unique()
    )

with col2:
    kursus_persiapan = st.selectbox(
        "Kursus Persiapan Tes", df["test preparation course"].unique()
    )

col3, col4, col5 = st.columns(3)

with col3:
    nilai_matematika = st.selectbox("Nilai Matematika", df["math score"].unique())

with col4:
    nilai_membaca = st.selectbox("Nilai Membaca", df["reading score"].unique())

with col5:
    nilai_menulis = st.selectbox("Nilai Menulis", df["writing score"].unique())

# Buat DataFrame dari input pengguna
input_data = pd.DataFrame(
    {
        "parental level of education": [pendidikan_orang_tua],
        "test preparation course": [kursus_persiapan],
        "math score": [nilai_matematika],
        "reading score": [nilai_membaca],
        "writing score": [nilai_menulis],
    }
)

if st.button("Mulai Prediksi"):
    prediction = nb.predict(input_data)

    st.subheader("HASIL PREDIKSI : ")
    st.markdown(
        f"<h1 style='text-align: center; color: blue;'>{prediction[0]}</h1>",
        unsafe_allow_html=True,
    )
    # Menampilkan Nilai Prior
    st.subheader("NILAI PRIOR")
    st.write(prior_df)

    # Membuat tabel kontingensi
    st.subheader("TABEL KONTINGENSI")

    contingency_table_test_prep = pd.crosstab(
        df["Class_lunch"], df["test preparation course"]
    )

    contingency_table_math_score = pd.crosstab(df["Class_lunch"], df["math score"])

    contingency_table_reading_score = pd.crosstab(
        df["Class_lunch"], df["reading score"]
    )

    contingency_table_writing_score = pd.crosstab(
        df["Class_lunch"], df["writing score"]
    )

    col6, col7 = st.columns(2)

    with col6:
        st.write("Test Preparation Course:")
        st.write(contingency_table_test_prep)

    with col7:
        st.write("Math Score:")
        st.write(contingency_table_math_score)

    col8, col9 = st.columns(2)

    with col8:
        st.write("Reading Score:")
        st.write(contingency_table_reading_score)

    with col9:
        st.write("Writing Score:")
        st.write(contingency_table_writing_score)

    # Menampilkan Tabel Hasil Pelatihan
    st.subheader("HASIL PELATIHAN")

    training_results = {
        "parental level of education": pd.crosstab(
            df["Class_lunch"], df["parental level of education"]
        ).apply(lambda r: r / r.sum(), axis=1),
        "test preparation course": pd.crosstab(
            df["Class_lunch"], df["test preparation course"]
        ).apply(lambda r: r / r.sum(), axis=1),
        "math score": pd.crosstab(df["Class_lunch"], df["math score"]).apply(
            lambda r: r / r.sum(), axis=1
        ),
        "reading score": pd.crosstab(df["Class_lunch"], df["reading score"]).apply(
            lambda r: r / r.sum(), axis=1
        ),
        "writing score": pd.crosstab(df["Class_lunch"], df["writing score"]).apply(
            lambda r: r / r.sum(), axis=1
        ),
    }

    col10, col11 = st.columns(2)

    with col10:
        st.write("Parental level of education:")
        st.write(training_results["parental level of education"])

        st.write("Test preparation course:")
        st.write(training_results["test preparation course"])

    with col11:
        st.write("Math score:")
        st.write(training_results["math score"])

        st.write("Reading score:")
        st.write(training_results["reading score"])

        st.write("Writing score:")
        st.write(training_results["writing score"])
