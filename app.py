import streamlit as st
import pandas as pd
import numpy as np


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
            decoded_index = X_train[column].apply(lambda x: label_encoders[column][x])
            decoded_columns = y_train.apply(lambda x: label_encoders["Class_lunch"][x])
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
                        y_train.apply(lambda x: label_encoders["Class_lunch"][x]),
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
def calculate_posterior_naive_bayes(X_new, class_stats, priors):
    posteriors = {}
    for cls in priors.index:
        prior = np.log(priors[cls])
        conditional = np.sum(
            [
                np.log(
                    gaussian_prob(
                        X_new[feature],
                        class_stats[cls]["mean"][feature],
                        class_stats[cls]["std"][feature],
                    )
                )
                for feature in X_new.index
            ]
        )
        posteriors[cls] = prior + conditional
    return posteriors


# Fungsi untuk mengubah laporan klasifikasi menjadi DataFrame
def classification_report_to_df(y_true, y_pred, target_names):
    report_data = []
    for label in np.unique(y_true):
        precision = np.sum((y_pred == label) & (y_true == label)) / np.sum(
            y_pred == label
        )
        recall = np.sum((y_pred == label) & (y_true == label)) / np.sum(y_true == label)
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        support = np.sum(y_true == label)
        report_data.append(
            {
                "class": target_names[label],
                "precision": precision,
                "recall": recall,
                "f1-score": f1_score,
                "support": support,
            }
        )
    dataframe = pd.DataFrame.from_dict(report_data)
    return dataframe


# Fungsi untuk menghitung mean dan std dev per kelas untuk fitur numerik
def calculate_class_stats(X_train, y_train, classes):
    class_stats = {}
    for cls in classes:
        cls_data = X_train[y_train == cls]
        class_stats[cls] = {"mean": cls_data.mean(), "std": cls_data.std(ddof=0)}
    return class_stats


# Fungsi untuk menghitung probabilitas Gaussian
def gaussian_prob(x, mean, std):
    exponent = np.exp(-((x - mean) ** 2 / (2 * std**2)))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent


# Fungsi manual untuk encoding label
def manual_label_encoding(column):
    unique_values = column.unique()
    encoding_dict = {val: idx for idx, val in enumerate(unique_values)}
    decoding_dict = {idx: val for val, idx in encoding_dict.items()}
    encoded_column = column.map(encoding_dict)
    return encoded_column, encoding_dict, decoding_dict


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
    data[column], encoding_dict, decoding_dict = manual_label_encoding(data[column])
    label_encoders[column] = decoding_dict

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

    parental_levels = list(label_encoders["parental level of education"].values())
    test_preparations = list(label_encoders["test preparation course"].values())

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

        # Manual split
        split_idx = int(len(data) * (train_size / 100))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]

        # Train Naive Bayes model manually
        classes = np.unique(y_train)
        priors = calculate_prior(y_train)
        class_stats = calculate_class_stats(X_train, y_train, classes)

        # Encode the input data
        new_data = {
            "parental level of education": list(
                label_encoders["parental level of education"].keys()
            )[
                list(label_encoders["parental level of education"].values()).index(
                    parental_level
                )
            ],
            "test preparation course": list(
                label_encoders["test preparation course"].keys()
            )[
                list(label_encoders["test preparation course"].values()).index(
                    test_preparation
                )
            ],
            "math score": grade_to_numeric(math_score),
            "reading score": grade_to_numeric(reading_score),
            "writing score": grade_to_numeric(writing_score),
        }
        new_data_df = pd.Series(new_data)

        # Predict and calculate posterior
        posteriors = calculate_posterior_naive_bayes(new_data_df, class_stats, priors)
        prediction = max(posteriors, key=posteriors.get)
        prediction_label = label_encoders["Class_lunch"][prediction]

        # Calculate all contingency tables
        contingency_tables = calculate_all_contingency_tables(
            X_train, y_train, original_data, label_encoders
        )

        # Evaluate model manually
        y_pred = []
        for _, row in X_test.iterrows():
            row_series = pd.Series(row)
            posteriors = calculate_posterior_naive_bayes(
                row_series, class_stats, priors
            )
            y_pred.append(max(posteriors, key=posteriors.get))

        accuracy = np.mean(y_pred == y_test)
        report_df = classification_report_to_df(
            y_test, y_pred, list(label_encoders["Class_lunch"].values())
        )

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
            st.write(priors)

        with col2:
            st.write("### Nilai Posterior")
            st.write(
                pd.DataFrame(
                    [posteriors], columns=list(label_encoders["Class_lunch"].values())
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
    st.title("Our Teams")
    st.markdown(
        """
    **Kelompok : NAIVE BAYES**

    **10122002 - Muhammad Renaldi Maulana**

    **10122024 - Dzaky Farras Fauzan**

    **10122007 - Mochammad Rizky Firdaus**

    **10122028 - Muhamad Hilmi Firdaus**
    """
    )

elif page == "Dataset":
    st.title("Tampilan Data")
    st.write("### Dataset Lengkap Exams (1000 data)")
    st.dataframe(original_data)
