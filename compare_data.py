import os
import pandas as pd
import matplotlib.pyplot as plt
print("=" * 60)
print("REAL VS SYNTHETIC MEDICAL DATA COMPARISON")
print("=" * 60)

# Load datasets
real = pd.read_csv("cleaned_medical_data.csv")
synthetic = pd.read_csv("synthetic_medical_data_cleaned.csv")

print("\nReal dataset shape:")
print(real.shape)

print("\nSynthetic dataset shape:")
print(synthetic.shape)

# Create results folder
os.makedirs("results", exist_ok=True)

# Data quality check
print("\n" + "=" * 60)
print("DATA QUALITY CHECK")
print("=" * 60)

print("\nReal missing values:", real.isnull().sum().sum())
print("Synthetic missing values:", synthetic.isnull().sum().sum())

print(
    "Real negative hospital stays:",
    (real["length_of_stay"] < 0).sum()
)

print(
    "Synthetic negative hospital stays:",
    (synthetic["length_of_stay"] < 0).sum()
)

# Numerical columns
numeric_columns = [
    "age",
    "bill_amount",
    "length_of_stay"
]

# Numerical statistics
print("\n" + "=" * 60)
print("NUMERICAL STATISTICS")
print("=" * 60)

statistics = pd.DataFrame({
    "Real Mean": real[numeric_columns].mean(),
    "Synthetic Mean": synthetic[numeric_columns].mean(),
    "Real Median": real[numeric_columns].median(),
    "Synthetic Median": synthetic[numeric_columns].median(),
    "Real Std": real[numeric_columns].std(),
    "Synthetic Std": synthetic[numeric_columns].std()
})

print(statistics.round(2))

statistics.to_csv(
    "results/numerical_comparison.csv"
)

# Numerical graphs
print("\nCreating numerical graphs...")

for column in numeric_columns:

    plt.figure(figsize=(8, 5))

    plt.hist(
        real[column],
        bins=30,
        alpha=0.5,
        label="Real"
    )

    plt.hist(
        synthetic[column],
        bins=30,
        alpha=0.5,
        label="Synthetic"
    )

    plt.title("Real vs Synthetic - " + column)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()

    filename = "results/" + column + "_comparison.png"

    plt.savefig(filename, dpi=150)
    plt.close()

    print("Saved:", filename)

# Categorical columns
categorical_columns = [
    "gender",
    "department",
    "diagnosis",
    "treatment",
    "status"
]

# Categorical graphs
print("\nCreating categorical graphs...")

for column in categorical_columns:

    real_counts = (
        real[column]
        .value_counts(normalize=True)
        .mul(100)
    )

    synthetic_counts = (
        synthetic[column]
        .value_counts(normalize=True)
        .mul(100)
    )

    comparison = pd.DataFrame({
        "Real": real_counts,
        "Synthetic": synthetic_counts
    }).fillna(0)

    ax = comparison.plot(
        kind="bar",
        figsize=(10, 5)
    )

    ax.set_title("Real vs Synthetic - " + column)
    ax.set_xlabel(column)
    ax.set_ylabel("Percentage (%)")

    plt.xticks(rotation=45)
    plt.tight_layout()

    filename = "results/" + column + "_comparison.png"

    plt.savefig(filename, dpi=150)
    plt.close()

    print("Saved:", filename)

print("\n" + "=" * 60)
print("COMPARISON COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nResults saved in:")
print("D:\\medical_gan\\results")