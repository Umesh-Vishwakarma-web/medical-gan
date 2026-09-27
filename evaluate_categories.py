import pandas as pd
import os

# Load datasets
real = pd.read_csv("cleaned_medical_data.csv")
synthetic = pd.read_csv("synthetic_medical_data_cleaned.csv")

# Create results folder
os.makedirs("results", exist_ok=True)

categorical_columns = [
    "gender",
    "department",
    "diagnosis",
    "treatment",
    "status"
]

print("=" * 60)
print("CATEGORICAL DISTRIBUTION EVALUATION")
print("=" * 60)

results = []

for column in categorical_columns:

    print("\n" + "-" * 60)
    print("Feature:", column)
    print("-" * 60)

    # Calculate percentage distributions
    real_distribution = (
        real[column]
        .value_counts(normalize=True)
        .mul(100)
    )

    synthetic_distribution = (
        synthetic[column]
        .value_counts(normalize=True)
        .mul(100)
    )

    # Combine real and synthetic values
    comparison = pd.DataFrame({
        "Real (%)": real_distribution,
        "Synthetic (%)": synthetic_distribution
    }).fillna(0)

    # Calculate absolute difference
    comparison["Absolute Difference (%)"] = (
        comparison["Real (%)"] -
        comparison["Synthetic (%)"]
    ).abs()

    print(comparison.round(2))

    # Average distribution difference
    average_difference = (
        comparison["Absolute Difference (%)"].mean()
    )

    print(
        "\nAverage distribution difference:",
        round(average_difference, 2),
        "%"
    )

    results.append({
        "Feature": column,
        "Average Difference (%)": round(
            average_difference, 2
        )
    })


# Save evaluation results
results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/categorical_evaluation.csv",
    index=False
)

print("\n" + "=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)

print("\nSummary:")
print(results_df)

print(
    "\nSaved to: results/categorical_evaluation.csv"
)