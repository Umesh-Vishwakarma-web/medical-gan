Building Synthetic Medical Records Using GANs


This project is about generating synthetic medical records using CTGAN (Conditional Tabular GAN).
I built this project to understand how generative AI can be used with tabular healthcare data. The project takes a medical dataset, cleans and prepares it, trains a CTGAN model, generates synthetic records, and then compares the generated data with the original data.

This is a learning and portfolio project, not a production healthcare system.

Project Overview
The main goal of this project is to explore how synthetic medical data can be generated while maintaining similar statistical patterns to the original dataset.

The project includes:
Data cleaning
Removal of direct personal identifiers
Feature engineering
Missing-value handling
Categorical data processing
Numerical data scaling
CTGAN model training
Synthetic data generation
Data validation
Real vs synthetic data comparison
Statistical evaluation
Data visualization
Project Workflow
Medical Dataset
&#x20;     &#x20;     
Data Cleaning
&#x20;     |
&#x20;     v
Remove Personal Identifiers
&#x20;     |
&#x20;     v
Feature Engineering
&#x20;     |
&#x20;     v
Encoding + Scaling
&#x20;     |
&#x20;     v
CTGAN Training
&#x20;     |
&#x20;     v
Synthetic Medical Records
&#x20;     |
&#x20;     v
Validation and Cleaning
&#x20;     |
&#x20;     v
Real vs Synthetic Comparison
&#x20;     |
&#x20;     v
Evaluation Results
Dataset

The original dataset contains 17,498 records and 14 columns.
The original dataset included fields such as:
Patient ID
First name
Last name
Age
Gender
Phone
Email
Department
Diagnosis
Treatment
Admission date
Discharge date
Status
Bill amount

For model training, direct identifiers were removed, including:
Patient ID
First name
Last name
Phone number
Email address

The original dataset is not included in this GitHub repository.
Important: The dataset used for this project should only be used when you have permission to use it. 
Do not upload real patient information or other sensitive personal data to GitHub.
Data Preprocessing
Before training CTGAN, the dataset is cleaned and prepared.
The preprocessing steps include:
Loading the medical dataset
Removing direct personal identifiers
Converting admission and discharge dates
Creating a length\_of\_stay feature
Detecting invalid negative hospital stays
Handling missing numerical values
Handling missing categorical values
Encoding categorical variables
Scaling numerical variables
Saving the processed data

The original dataset contained 4,382 invalid negative hospital stay values.
&#x20;These values were identified and handled during preprocessing.
After preprocessing:
Records: 17,498
Cleaned columns: 8
Model features after encoding: 29
Missing values after cleaning: 0
Negative hospital stays after cleaning: 0

CTGAN Model

I used CTGAN (Conditional Tabular GAN) to generate synthetic medical records.
CTGAN is designed for tabular datasets containing both numerical and categorical variables.

The model was trained using:
Model: CTGAN
Epochs: 300
Batch Size: 500
After training, the model generated:
17,498 synthetic medical records

The generated records were then validated and cleaned before performing the final comparison.
Synthetic Data Validation
The generated data was checked for common data-quality problems.
The validation process checks:
Missing values
Invalid ages
Negative hospital stays
Numerical formatting
Categorical values
Some invalid values were found in the generated dataset and were corrected during the validation step.
The cleaned synthetic dataset is saved as:
synthetic\_medical\_data\_cleaned.csv
Generated datasets are excluded from this repository using .gitignore.
Results
The real and synthetic datasets were compared using numerical statistics and categorical distributions.
Numerical Comparison
Feature	Real Mean	Synthetic Mean
Age	45.43	42.60
Bill Amount	100716.90	110896.95
Length of Stay	280.36	263.15
The comparison also includes median and standard deviation values.
The complete numerical comparison is available in:
results/numerical\_comparison.csv
Categorical Evaluation
The categorical distributions were compared between the real and synthetic datasets.

The average distribution difference was approximately:
8.32%
Feature	Distribution Difference
Gender	7.92%
Department	4.67%
Diagnosis	4.64%
Treatment	8.38%
Status	8.32%
These values provide a basic measure of how closely the synthetic categorical distributions match the original dataset.
The complete evaluation is available in:
results/categorical\_evaluation.csv
Visual Comparisons
The project generates comparison charts for the real and synthetic datasets.

The charts include:
Age
Bill Amount
Length of Stay
Gender
Department
Diagnosis
Treatment
Status

The charts are stored in:

results/
Project Structure
medical-gan/
│
├── preprocessing.py
├── train\_gan.py
├── evaluate\_synthetic.py
├── validate\_and\_clean\_synthetic.py
├── compare\_data.py
├── evaluate\_categories.py
│
├── results/
│   ├── age\_comparison.png
│   ├── bill\_amount\_comparison.png
│   ├── length\_of\_stay\_comparison.png
│   ├── gender\_comparison.png
│   ├── department\_comparison.png
│   ├── diagnosis\_comparison.png
│   ├── treatment\_comparison.png
│   ├── status\_comparison.png
│   ├── numerical\_comparison.csv
│   └── categorical\_evaluation.csv
│
├── README.md
├── requirements.txt
└── .gitignore
How to Run
1\. Clone the Repository
git clone https://github.com/Umesh-Vishwakarma-web/medical-gan.git
Move into the project folder:
cd medical-gan
2\. Install Required Libraries
pip install -r requirements.txt
3\. Add the Dataset
Place your authorized dataset inside the project folder with the filename:
medical\_data.csv

The file should be located like this:
medical-gan/
│
├── medical\_data.csv
├── preprocessing.py
├── train\_gan.py
└── ...
The dataset is intentionally not included in this repository.

4\. Run Preprocessing
python preprocessing.py

This creates:
cleaned\_medical\_data.csv
processed\_medical\_data.npy
processed\_feature\_names.csv
5\. Train CTGAN
python train\_gan.py
This generates:
synthetic\_medical\_data.csv
6\. Evaluate the Generated Data
python evaluate\_synthetic.py
7\. Validate and Clean Synthetic Data
python validate\_and\_clean\_synthetic.py
This creates:
synthetic\_medical\_data\_cleaned.csv
8\. Compare Real and Synthetic Data
python compare\_data.py
This generates the comparison charts and:
results/numerical\_comparison.csv
9\. Evaluate Categorical Distributions
python evaluate\_categories.py

This creates:
results/categorical\_evaluation.csv
Complete Command Sequence
If everything is already installed and the dataset is present, the complete workflow is:
python preprocessing.py
python train\_gan.py
python evaluate\_synthetic.py
python validate\_and\_clean\_synthetic.py
python compare\_data.py
python evaluate\_categories.py
Technologies Used
Python
Pandas
NumPy
Scikit-learn
CTGAN
Matplotlib
Git
GitHub
What I Learned

Through this project, I learned about:
Data cleaning and preprocessing
Handling numerical and categorical data
Feature engineering
Missing-value handling
GANs
CTGAN for tabular data
Synthetic data generation
Data validation
Statistical comparison
Data visualization
Git and GitHub
Building an end-to-end machine-learning project
Limitations
This project is mainly for learning and portfolio purposes.
The generated data should not automatically be considered private, anonymous, or suitable for real healthcare applications.
A real healthcare system would require additional:

Privacy testing
Security controls
Bias and fairness evaluation
Memorization testing
Statistical validation
Domain-expert review
Regulatory and legal compliance

The quality of synthetic data also depends on the quality, size, and characteristics of the original training dataset.
Future Improvements
Possible improvements for this project include:
Testing other synthetic data generation models
Adding stronger statistical evaluation methods
Measuring correlations between features
Testing privacy and memorization risks
Comparing downstream machine-learning performance
Adding fairness analysis
Using larger datasets
Experimenting with privacy-preserving techniques
Adding automated evaluation reports

Author

Umesh Vishwakarma
GitHub:
https://github.com/Umesh-Vishwakarma-web
Disclaimer
This project is created for educational and portfolio purposes only.
It is not intended for clinical use, medical diagnosis, patient treatment, or making healthcare decisions.
