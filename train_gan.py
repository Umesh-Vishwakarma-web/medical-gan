import pandas as pd
from ctgan import CTGAN

# ==========================================
# 1. Load cleaned medical dataset
# ==========================================

df = pd.read_csv("cleaned_medical_data.csv")

print("=" * 50)
print("LOADING CLEANED DATA")
print("=" * 50)

print("\nDataset shape:")
print(df.shape)

print("\nDataset columns:")
print(df.columns.tolist())


# ==========================================
# 2. Define categorical columns
# ==========================================

categorical_columns = [
    "gender",
    "department",
    "diagnosis",
    "treatment",
    "status"
]


# ==========================================
# 3. Display basic information
# ==========================================

print("\nCategorical columns:")
print(categorical_columns)

print("\nFirst 5 records:")
print(df.head())


# ==========================================
# 4. Train CTGAN
# ==========================================

print("\n" + "=" * 50)
print("TRAINING CTGAN")
print("=" * 50)

model = CTGAN(
    epochs=300,
    batch_size=500,
    verbose=True
)

model.fit(
    df,
    discrete_columns=categorical_columns
)


# ==========================================
# 5. Generate synthetic records
# ==========================================

print("\n" + "=" * 50)
print("GENERATING SYNTHETIC DATA")
print("=" * 50)

synthetic_data = model.sample(
    len(df)
)


# ==========================================
# 6. Display synthetic data
# ==========================================

print("\nSynthetic dataset shape:")
print(synthetic_data.shape)

print("\nFirst 5 synthetic records:")
print(synthetic_data.head())


# ==========================================
# 7. Save synthetic dataset
# ==========================================

synthetic_data.to_csv(
    "synthetic_medical_data.csv",
    index=False
)


# ==========================================
# 8. Finished
# ==========================================

print("\n" + "=" * 50)
print("GAN TRAINING COMPLETED SUCCESSFULLY")
print("=" * 50)

print("\nSynthetic dataset saved as:")
print("synthetic_medical_data.csv")