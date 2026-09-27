import pandas as pd

# ==========================================
# 1. Load synthetic data
# ==========================================

synthetic = pd.read_csv("synthetic_medical_data.csv")

print("=" * 60)
print("SYNTHETIC DATA CLEANING")
print("=" * 60)

print("\nOriginal synthetic dataset:")
print(synthetic.shape)


# ==========================================
# 2. Check invalid ages
# ==========================================

invalid_age_mask = (
    (synthetic["age"] < 0) |
    (synthetic["age"] > 120)
)

invalid_age_count = invalid_age_mask.sum()

print("\nInvalid ages found:")
print(invalid_age_count)

# Replace invalid ages with median valid age
valid_age_median = synthetic.loc[
    ~invalid_age_mask,
    "age"
].median()

synthetic.loc[
    invalid_age_mask,
    "age"
] = valid_age_median


# ==========================================
# 3. Check negative hospital stays
# ==========================================

negative_stay_mask = (
    synthetic["length_of_stay"] < 0
)

negative_stay_count = negative_stay_mask.sum()

print("\nNegative hospital stays found:")
print(negative_stay_count)

# Replace impossible negative stays with 0
synthetic.loc[
    negative_stay_mask,
    "length_of_stay"
] = 0


# ==========================================
# 4. Round numerical values
# ==========================================

synthetic["age"] = (
    synthetic["age"]
    .round()
    .astype(int)
)

synthetic["length_of_stay"] = (
    synthetic["length_of_stay"]
    .round()
    .astype(int)
)

synthetic["bill_amount"] = (
    synthetic["bill_amount"]
    .round(2)
)


# ==========================================
# 5. Final validation
# ==========================================

print("\n" + "=" * 60)
print("FINAL VALIDATION")
print("=" * 60)

remaining_negative = (
    synthetic["length_of_stay"] < 0
).sum()

remaining_invalid_age = (
    (synthetic["age"] < 0) |
    (synthetic["age"] > 120)
).sum()

remaining_missing = (
    synthetic.isnull().sum().sum()
)

print("\nRemaining negative stays:")
print(remaining_negative)

print("\nRemaining invalid ages:")
print(remaining_invalid_age)

print("\nRemaining missing values:")
print(remaining_missing)


# ==========================================
# 6. Save cleaned synthetic data
# ==========================================

synthetic.to_csv(
    "synthetic_medical_data_cleaned.csv",
    index=False
)


# ==========================================
# 7. Display results
# ==========================================

print("\nFirst 5 cleaned synthetic records:")
print(synthetic.head())

print("\nCleaned synthetic dataset shape:")
print(synthetic.shape)

print("\n" + "=" * 60)
print("SYNTHETIC DATA CLEANING COMPLETED")
print("=" * 60)

print("\nSaved as:")
print("synthetic_medical_data_cleaned.csv")