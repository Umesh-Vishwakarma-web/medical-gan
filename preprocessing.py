import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# ============================================================
# 1. Load dataset
# ============================================================

print("Loading dataset...")

df = pd.read_csv("medical_data.csv")

print("Original dataset shape:", df.shape)


# ============================================================
# 2. Remove personal/identifier information
# ============================================================

columns_to_remove = [
    "patient_id",
    "first_name",
    "last_name",
    "phone",
    "email"
]

df = df.drop(columns=columns_to_remove, errors="ignore")

print("After removing personal information:", df.shape)


# ============================================================
# 3. Convert date columns
# ============================================================

df["admission_date"] = pd.to_datetime(
    df["admission_date"],
    errors="coerce"
)

df["discharge_date"] = pd.to_datetime(
    df["discharge_date"],
    errors="coerce"
)


# ============================================================
# 4. Create Length of Stay
# ============================================================

df["length_of_stay"] = (
    df["discharge_date"] - df["admission_date"]
).dt.days


# ============================================================
# 5. Detect invalid negative hospital stays
# ============================================================

invalid_stays = (df["length_of_stay"] < 0).sum()

print("Invalid negative hospital stays:", invalid_stays)


# Replace invalid negative values with NaN
df.loc[df["length_of_stay"] < 0, "length_of_stay"] = np.nan


# ============================================================
# 6. Remove original date columns
# ============================================================

df = df.drop(
    columns=["admission_date", "discharge_date"],
    errors="ignore"
)


# ============================================================
# 7. Identify numerical and categorical columns
# ============================================================

num_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

cat_cols = df.select_dtypes(
    include=["object", "string"]
).columns.tolist()

print("\nNumerical columns:")
print(num_cols)

print("\nCategorical columns:")
print(cat_cols)


# ============================================================
# 8. Check missing values before cleaning
# ============================================================

print("\nMissing values before cleaning:")

missing_before = df.isnull().sum()

print(
    missing_before[missing_before > 0]
)


# ============================================================
# 9. Fill numerical missing values with median
# ============================================================

for column in num_cols:

    if df[column].isnull().any():

        median_value = df[column].median()

        df[column] = df[column].fillna(median_value)


# ============================================================
# 10. Fill categorical missing values with mode
# ============================================================

for column in cat_cols:

    if df[column].isnull().any():

        mode_values = df[column].mode()

        if not mode_values.empty:

            df[column] = df[column].fillna(
                mode_values[0]
            )


# ============================================================
# 11. Check missing values after cleaning
# ============================================================

print("\nMissing values after cleaning:")

missing_after = df.isnull().sum()

print(
    missing_after[missing_after > 0]
)

print(
    "Total missing values:",
    df.isnull().sum().sum()
)


# ============================================================
# 12. Verify negative length of stay
# ============================================================

negative_stays_after = (
    df["length_of_stay"] < 0
).sum()

print(
    "\nNegative length of stay after cleaning:",
    negative_stays_after
)


# ============================================================
# 13. One-Hot Encode categorical columns
# ============================================================

if len(cat_cols) > 0:

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )

    cat_encoded = encoder.fit_transform(
        df[cat_cols]
    )

else:

    cat_encoded = np.empty(
        (len(df), 0)
    )


# ============================================================
# 14. Scale numerical columns
# ============================================================

if len(num_cols) > 0:

    scaler = MinMaxScaler(
        feature_range=(-1, 1)
    )

    num_scaled = scaler.fit_transform(
        df[num_cols]
    )

else:

    num_scaled = np.empty(
        (len(df), 0)
    )


# ============================================================
# 15. Combine numerical and categorical data
# ============================================================

data_processed = np.hstack(
    (
        num_scaled,
        cat_encoded
    )
)


# ============================================================
# 16. Create feature names
# ============================================================

feature_names = []

feature_names.extend(num_cols)

if len(cat_cols) > 0:

    encoded_feature_names = (
        encoder.get_feature_names_out(cat_cols)
    )

    feature_names.extend(
        encoded_feature_names
    )


# ============================================================
# 17. Save cleaned dataset
# ============================================================

df.to_csv(
    "cleaned_medical_data.csv",
    index=False
)


# ============================================================
# 18. Save processed numerical dataset
# ============================================================

np.save(
    "processed_medical_data.npy",
    data_processed
)


# ============================================================
# 19. Save feature names
# ============================================================

pd.DataFrame(
    {
        "feature": feature_names
    }
).to_csv(
    "processed_feature_names.csv",
    index=False
)


# ============================================================
# 20. Final information
# ============================================================

print("\n======================================")
print("PREPROCESSING COMPLETED")
print("======================================")

print(
    "Cleaned dataset shape:",
    df.shape
)

print(
    "Processed data shape:",
    data_processed.shape
)

print(
    "Total features:",
    len(feature_names)
)

print(
    "Total missing values:",
    df.isnull().sum().sum()
)

print(
    "Negative length of stay:",
    (df["length_of_stay"] < 0).sum()
)

print("\nFiles created:")

print("1. cleaned_medical_data.csv")
print("2. processed_medical_data.npy")
print("3. processed_feature_names.csv")

print("\nPreprocessing finished successfully!")