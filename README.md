Building Synthetic Medical Records Using GANs



This project is about generating synthetic medical records using CTGAN (Conditional Tabular GAN).



I built this project to understand how generative AI can be used with tabular healthcare data. The project takes a medical dataset, cleans and prepares it, trains a CTGAN model, generates synthetic records, and then compares the generated data with the original data.



This is a learning and portfolio project, not a production healthcare system.



What I worked on



The project follows these main steps:



Clean the medical dataset

Remove direct personal information

Create the length\_of\_stay feature

Handle missing and invalid values

Encode categorical columns

Scale numerical columns

Train a CTGAN model

Generate synthetic medical records

Validate the generated data

Compare real and synthetic data

Project workflow

Medical Dataset

&#x20;     |

&#x20;     v

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



Some of the original columns included:



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



I removed direct identifiers such as names, phone numbers, email addresses, and patient IDs before using the data for model training.



The original dataset is not included in this GitHub repository.



Data preprocessing



Before training the model, I cleaned the data and prepared it for CTGAN.



Some of the preprocessing steps were:



Removed direct identifiers

Converted admission and discharge dates

Created length\_of\_stay

Checked for negative hospital stays

Filled missing numerical values using the median

Filled missing categorical values using the mode

Converted categorical variables using one-hot encoding

Scaled numerical features



After preprocessing, the dataset contained 29 model features.



CTGAN



I used CTGAN because the dataset contains both numerical and categorical information.



The model was trained with:



Epochs: 300

Batch size: 500



After training, the model generated 17,498 synthetic records.



The synthetic records were then checked for invalid ages, negative hospital stays, missing values, and other formatting problems.



Results



I compared the original and synthetic datasets using numerical and categorical distributions.



Some of the numerical results were:



Feature	Real Mean	Synthetic Mean

Age	45.43	42.60

Bill Amount	100716.90	110896.95

Length of Stay	280.36	263.15



For the categorical variables, the average distribution difference was approximately 8.32%.



Feature	Difference

Gender	7.92%

Department	4.67%

Diagnosis	4.64%

Treatment	8.38%

Status	8.32%



These results give a basic indication of how similar the synthetic data is to the original dataset.



Project files

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

How to run



Clone the repository:



git clone https://github.com/Umesh-Vishwakarma-web/medical-gan.git

cd medical-gan



Install the required libraries:



pip install -r requirements.txt



Place your authorized dataset in the project folder as:



medical\_data.csv



Then run the scripts in this order:



python preprocessing.py

python train\_gan.py

python evaluate\_synthetic.py

python validate\_and\_clean\_synthetic.py

python compare\_data.py

python evaluate\_categories.py



The comparison charts and evaluation files will be saved in the results folder.



Technologies

Python

Pandas

NumPy

Scikit-learn

CTGAN

Matplotlib

Git

GitHub

What I learned



While working on this project, I learned about:



Data cleaning and preprocessing

Handling categorical and numerical data

Feature engineering

GANs and CTGAN

Synthetic tabular data generation

Data validation

Comparing datasets

Data visualization

Using Git and GitHub

Limitations



This project is mainly for learning and demonstration.



The generated data should not automatically be considered private or safe for real healthcare use. A real healthcare application would require proper privacy testing, security controls, bias evaluation, domain-expert review, and regulatory compliance.



The quality of synthetic data also depends on the quality and size of the training dataset.



Future improvements



Some things I would like to improve in the future:



Test other synthetic data generation models

Add stronger statistical evaluation

Test privacy and memorization risks

Compare downstream machine-learning performance

Add fairness analysis

Try larger datasets

Experiment with privacy-preserving techniques

Author



Umesh Vishwakarma



GitHub: https://github.com/Umesh-Vishwakarma-web



This project is for educational and portfolio purposes. It is not intended for clinical use or for making medical decisions.

