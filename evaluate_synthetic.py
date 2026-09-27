import pandas as pd
import numpy as np

# ==========================================
# 1. Load real and synthetic data
# ==========================================

real = pd.read_csv("cleaned_medical_data.csv")
synthetic = pd.read_csv("synthetic_medical_data.csv")

print("=" * 60)
print("SYNTHETIC DATA EVALUATION")
print("=" * 60)

print("\nReal dataset shape:")
print(real.shape)

print("\nSynthetic dataset shape:")
print(synthetic.shape)


# ==========================================
# 2. Check missing values
# ==========================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print("\nReal data:")
print(real.isnull().sum())

print("\nSynthetic data:")
print(synthetic.isnull().sum())


# ==========================================
# 3. Check negative hospital stays
# ==========================================

print("\n" + "=" * 60)
print("HOSPITAL STAY VALIDATION")
print("=" * 60)

real_negative = (
    real["length_of_stay"] < 0
).sum()

synthetic_negative = (
    synthetic["length_of_stay"] < 0
).sum()

print("\nNegative stays in real data:")
print(real_negative)

print("\nNegative stays in synthetic data:")
print(synthetic_negative)


# ==========================================
# 4. Check age range
# ==========================================

print("\n" + "=" * 60)
print("AGE VALIDATION")
print("=" * 60)

print("\nReal age range:")
print(
    real["age"].min(),
    "to",
    real["age"].max()
)

print("\nSynthetic age range:")
print(
    synthetic["age"].min(),
    "to",
    synthetic["age"].max()
)

synthetic_invalid_age = (
    (synthetic["age"] < 0) |
    (synthetic["age"] > 120)
).sum()

print("\nInvalid synthetic ages:")
print(synthetic_invalid_age)


# ==========================================
# 5. Compare numerical statistics
# ==========================================

print("\n" + "=" * 60)
print("NUMERICAL COMPARISON")
print("=" * 60)

numeric_columns = [
    "age",
    "bill_amount",
    "length_of_stay"
]

comparison = pd.DataFrame({
    "Real Mean": real[numeric_columns].mean(),
    "Synthetic Mean": synthetic[numeric_columns].mean(),
    "Real Median": real[numeric_columns].median(),
    "Synthetic Median": synthetic[numeric_columns].median()
})

print(comparison)


# ==========================================
# 6. Compare categorical distributions
# ==========================================

print("\n" + "=" * 60)
print("CATEGORICAL DISTRIBUTION")
print("=" * 60)

categorical_columns = [
    "gender",
    "department",
    "diagnosis",
    "treatment",
    "status"
]

for column in categorical_columns:

    print("\n" + "-" * 50)
    print(column)

    real_distribution = (
        real[column]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    synthetic_distribution = (
        synthetic[column]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    comparison = pd.DataFrame({
        "Real (%)": real_distribution,
        "Synthetic (%)": synthetic_distribution
    }).fillna(0)

    print(comparison)


# ==========================================
# 7. Final validation
# ==========================================

print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)

if synthetic_negative == 0:
    print("✓ No negative hospital stays")
else:
    print(
        "✗ Negative hospital stays found:",
        synthetic_negative
    )

if synthetic_invalid_age == 0:
    print("✓ All synthetic ages are valid")
else:
    print(
        "✗ Invalid ages found:",
        synthetic_invalid_age
    )

if synthetic.isnull().sum().sum() == 0:
    print("✓ No missing values")
else:
    print("✗ Missing values found")

print("\nEvaluation completed.")