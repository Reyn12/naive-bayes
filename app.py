import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report


# Fungsi untuk mengubah nilai huruf ke nilai numerik
def grade_to_numeric(grade):
    grade_dict = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}
    return grade_dict.get(grade, -1)


# Fungsi untuk mengubah nilai numerik ke nilai huruf
def numeric_to_grade(numeric):
    numeric_dict = {4: "A", 3: "B", 2: "C", 1: "D", 0: "E"}
    return numeric_dict.get(numeric, "Unknown")


# Fungsi untuk menghitung nilai prior
def calculate_prior(y_train):
    prior = y_train.value_counts(normalize=True)
    return prior


# Fungsi untuk menghitung tabel kontingensi untuk semua fitur
def calculate_all_contingency_tables(X_train, y_train, original_data, label_encoders):
    contingency_tables = {}
    for column in X_train.columns:
        if column in label_encoders:
            decoded_index = label_encoders[column].inverse_transform(X_train[column])
            decoded_columns = label_encoders["Class_lunch"].inverse_transform(y_train)
            table = pd.crosstab(
                index=pd.Series(decoded_index, name=original_data[column].name),
                columns=pd.Series(decoded_columns, name="Class_lunch"),
                margins=True,
            )
        else:
            if column in ["math score", "reading score", "writing score"]:
                decoded_index = X_train[column].apply(numeric_to_grade)
                table = pd.crosstab(
                    index=pd.Series(decoded_index, name=original_data[column].name),
                    columns=pd.Series(
                        label_encoders["Class_lunch"].inverse_transform(y_train),
                        name="Class_lunch",
                    ),
                    margins=True,
                )
            else:
                table = pd.crosstab(
                    index=X_train[column], columns=y_train, margins=True
                )
        contingency_tables[original_data[column].name] = table
    return contingency_tables


# Fungsi untuk menghitung nilai posterior
def calculate_posterior(model, X_new):
    probabilities = model.predict_proba(X_new)
    return probabilities


# Fungsi untuk mengubah laporan klasifikasi menjadi DataFrame
def classification_report_to_df(report):
    report_data = []
    lines = report.split("\n")
    for line in lines[2:]:
        row = {}
        row_data = line.split()
        if len(row_data) == 0:  # Skip empty lines
            continue
        if len(row_data) == 5:
            row["class"] = row_data[0]
            row["precision"] = float(row_data[1])
            row["recall"] = float(row_data[2])
            row["f1-score"] = float(row_data[3])
            row["support"] = int(row_data[4])
        elif len(row_data) == 6:
            row["class"] = " ".join(row_data[:2])
            row["precision"] = float(row_data[2])
            row["recall"] = float(row_data[3])
            row["f1-score"] = float(row_data[4])
            row["support"] = int(row_data[5])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    return dataframe


# Load dataset
dataset_path = "dataset5.csv"
data = pd.read_csv(dataset_path)

# Simpan data asli sebelum encoding
original_data = data.copy()

# Preprocess dataset
data["math score"] = data["math score"].apply(grade_to_numeric)
data["reading score"] = data["reading score"].apply(grade_to_numeric)
data["writing score"] = data["writing score"].apply(grade_to_numeric)

label_encoders = {}
for column in ["parental level of education", "test preparation course", "Class_lunch"]:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Sidebar
with st.sidebar:
    st.title("Menu Program")
    page = st.selectbox("Pilih Halaman", ["Simulasi Program", "Our Team", "Dataset"])

if page == "Simulasi Program":
    st.title("Naive Bayes Classifier - Exams")

    # Inject custom CSS
    st.markdown(
        """
        <style>
        .stButton button {
            width: 100%;
            background-color: #FFD700; /* Cream background */
            color: black; /* Black text */
            padding: 10px 24px; /* Some padding */
            border: none; /* Remove borders */
            cursor: pointer; /* Add a pointer cursor on hover */
        }
        .stButton button:hover {
            background-color: #FFC300; /* Darker cream background on hover */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    train_size = st.slider("Masukkan jumlah data Training (%)", 10, 90, 70)
    test_size = 100 - train_size

    parental_levels = label_encoders["parental level of education"].inverse_transform(
        data["parental level of education"].unique()
    )
    test_preparations = label_encoders["test preparation course"].inverse_transform(
        data["test preparation course"].unique()
    )

    # Arrange inputs in the desired layout
    col1, col2 = st.columns(2)
    with col1:
        parental_level = st.selectbox(
            "Pilih parental level of education", parental_levels
        )
    with col2:
        test_preparation = st.selectbox(
            "Pilih test preparation course", test_preparations
        )

    col3, col4, col5 = st.columns(3)
    with col3:
        math_score = st.selectbox("Pilih math score", ["A", "B", "C", "D", "E"])
    with col4:
        reading_score = st.selectbox("Pilih reading score", ["A", "B", "C", "D", "E"])
    with col5:
        writing_score = st.selectbox("Pilih writing score", ["A", "B", "C", "D", "E"])

    # Create a button with the custom CSS class
    if st.button("Prediksi", key="prediksi"):
        # Split data into training and testing
        X = data.drop("Class_lunch", axis=1)
        y = data["Class_lunch"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, random_state=42
        )

        # Train Naive Bayes model
        model = GaussianNB()
        model.fit(X_train, y_train)

        # Encode the input data
        new_data = {
            "parental level of education": label_encoders[
                "parental level of education"
            ].transform([parental_level])[0],
            "test preparation course": label_encoders[
                "test preparation course"
            ].transform([test_preparation])[0],
            "math score": grade_to_numeric(math_score),
            "reading score": grade_to_numeric(reading_score),
            "writing score": grade_to_numeric(writing_score),
        }
        new_data_df = pd.DataFrame([new_data])

        # Predict and calculate posterior
        prediction = model.predict(new_data_df)[0]
        prediction_label = label_encoders["Class_lunch"].inverse_transform(
            [prediction]
        )[0]
        posterior = calculate_posterior(model, new_data_df)

        # Calculate prior
        prior = calculate_prior(y_train)
        prior.index = label_encoders["Class_lunch"].inverse_transform(prior.index)

        # Calculate all contingency tables
        contingency_tables = calculate_all_contingency_tables(
            X_train, y_train, original_data, label_encoders
        )

        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, target_names=label_encoders["Class_lunch"].classes_
        )
        report_df = classification_report_to_df(report)

        # Display results
        col1, col2 = st.columns(2)

        with col1:
            st.write("### Hasil Prediksi")
            st.markdown(
                f"<h3><b>{prediction_label}</b></h3>",
                unsafe_allow_html=True,
            )

        with col2:
            st.write("### Akurasi Model")
            st.markdown(
                f"<h3><b>{int(accuracy * 100)}%</b></h3>", unsafe_allow_html=True
            )

        col1, col2 = st.columns(2)
        with col1:
            st.write("### Nilai Prior")
            st.write(prior)

        with col2:
            st.write("### Nilai Posterior")
            st.write(
                pd.DataFrame(
                    posterior, columns=label_encoders["Class_lunch"].classes_
                ).to_html(index=False),
                unsafe_allow_html=True,
            )

        st.write("### Tabel Kontingensi")
        cols = st.columns(2)
        for i, (feature, table) in enumerate(contingency_tables.items()):
            with cols[i % 2]:
                st.write(f"#### Tabel Kontingensi untuk {feature}")
                st.write(table)

        st.write("### Laporan Klasifikasi")
        st.table(report_df)

elif page == "Our Team":
    st.title("Our Team")
    st.markdown(
        """
    **Kelompok: NAIVE BAYES**

    **10122002 - Muhammad Renaldi Maulana**

    **10122020 - M Rizky Firdaus**
    """
    )

elif page == "Dataset":
    st.title("Tampilan Data")
    st.write("### Dataset Lengkap (1000 data)")
    st.dataframe(original_data)
