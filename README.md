# Asthma Attack Prediction using Machine Learning and XAI

![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)

This repository contains the code and resources for the research project **"Asthma Attack Prediction Model: A Data-Driven Approach Using SMOTE and Explainable AI"**. This work was presented at the **16th International IEEE Conference on Computing, Communication and Networking Technologies (ICCCNT) 2025**.

The project focuses on developing a highly accurate machine learning model to predict asthma attacks by analyzing a combination of patient health data, demographic information, and environmental triggers. A key innovation is the integration of Explainable AI (XAI) techniques to ensure the model's predictions are transparent and interpretable for clinical use.

---

## 📋 Table of Contents
* [About The Project](#about-the-project)
* [Key Features](#-key-features)
* [Methodology Workflow](#-methodology-workflow)
* [Getting Started](#-getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#-usage)
* [Results & Evaluation](#-results--evaluation)
* [Explainable AI (XAI) Insights](#-explainable-ai-xai-insights)
* [Future Scope](#-future-scope)
* [License](#-license)
* [Acknowledgments](#-acknowledgments)
* [Publication](#-publication)

---

## 📖 About The Project

[cite_start]Asthma is a chronic respiratory disease that affects millions of people worldwide, characterized by unpredictable and potentially life-threatening attacks[cite: 24]. [cite_start]Traditional methods for predicting attacks often fall short due to the complex interplay of genetic, environmental, and lifestyle factors[cite: 37].

[cite_start]This project leverages machine learning to create a robust predictive model that can provide early warnings for asthma exacerbations[cite: 41]. By analyzing a diverse dataset, our model identifies critical patterns and risk factors. [cite_start]To move beyond "black-box" predictions, we employ **SHAP (SHapley Additive exPlanations)** and **LIME (Local Interpretable Model-agnostic Explanations)** to make the model's reasoning transparent, building trust and facilitating its adoption in clinical decision-making[cite: 29, 53, 55].

### Key Features ✨
* [cite_start]**High-Accuracy Prediction**: The final Random Forest model achieves an accuracy of **97.58%** on the test set[cite: 28, 242].
* [cite_start]**Data-Driven Approach**: Utilizes a comprehensive dataset including patient history, demographics, and environmental data like Air Quality Index (AQI) and temperature[cite: 25, 43].
* [cite_start]**Imbalanced Data Handling**: Implements the **Synthetic Minority Over-sampling Technique (SMOTE)** to effectively address class imbalance between attack and non-attack instances, preventing model bias[cite: 26, 48, 159].
* [cite_start]**Intelligent Feature Selection**: A Random Forest Regressor is used to identify and select the top 10 most influential features, optimizing model performance and reducing noise[cite: 160, 169, 171].
* [cite_start]**Dual-XAI Framework**: Integrates both SHAP for global feature importance and LIME for case-specific prediction explanations, offering a complete view of the model's behavior[cite: 29, 53, 235].

---

## 🔄 Methodology Workflow

The project follows a structured methodology from data acquisition to model interpretation:

1.  [cite_start]**Data Acquisition**: The study uses the IEEE asthma dataset, which contains 584 instances and 32 features covering patient and environmental data[cite: 146, 147].
2.  [cite_start]**Data Pre-processing**: This crucial phase includes handling missing values, removing duplicates, feature scaling (normalization), and outlier detection[cite: 156, 158]. [cite_start]SMOTE is applied to balance the dataset[cite: 159].
3.  [cite_start]**Exploratory Data Analysis (EDA)**: Visual analysis is performed to understand data distributions and feature relationships[cite: 162].
4.  [cite_start]**Feature Selection**: The top 10 most predictive features are selected using Random Forest importance scores[cite: 171]. [cite_start]Key selected features include `medication`, `chest_tig`, and `pefr`[cite: 171].
5.  [cite_start]**Model Training & Tuning**: Twelve different classification algorithms—including Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting—are trained and evaluated[cite: 204, 263]. [cite_start]`GridSearchCV` is used for hyperparameter tuning to optimize performance[cite: 51, 210].
6.  [cite_start]**Model Evaluation**: Models are rigorously assessed using metrics such as Accuracy, Precision, Recall, and F1-Score[cite: 52, 229].
7.  [cite_start]**Model Explainability**: SHAP and LIME are applied to the best-performing model to interpret its predictions globally and locally[cite: 54, 55, 238, 239].

---

## 🚀 Getting Started

Follow these instructions to set up the project environment and run the code on your local machine.

### Prerequisites

This project requires **Python 3.x** and common data science packages. Based on the project notebook, you will need:
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* shap
* lime

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/asthma-prediction.git](https://github.com/your-username/asthma-prediction.git)
    cd asthma-prediction
    ```
2.  **Install the required packages:**
    You can install all dependencies using pip.
    ```sh
    pip install pandas numpy scikit-learn matplotlib seaborn shap lime jupyter
    ```

---

## 💻 Usage

The entire workflow is contained within the `FDS_Project_FinalCode.ipynb` Jupyter Notebook.

1.  Launch Jupyter Notebook:
    ```sh
    jupyter notebook
    ```
2.  Open `FDS_Project_FinalCode.ipynb` in your browser.
3.  Run the cells sequentially from top to bottom to execute the entire pipeline, from data loading and pre-processing to model training, evaluation, and visualization of XAI results.

---

## 📊 Results & Evaluation

The models were evaluated on multiple performance metrics. [cite_start]The **Random Forest Classifier** emerged as the top-performing model, demonstrating a superior balance of accuracy and reliability[cite: 242, 340].

[cite_start]Here is a summary of the performance of all tested classifiers[cite: 263]:

| Classifier               | Accuracy | Precision | Recall   | F1-Score |
| ------------------------ | :------: | :-------: | :------: | :------: |
| **Random Forest** | **0.9758** | **0.9731** | **0.9790** | **0.9759** |
| Gradient Boosting        |  0.9747  |  0.9690   |  0.9811  |  0.9749  |
| Decision Tree            |  0.9663  |  0.9606   |  0.9727  |  0.9666  |
| AdaBoost                 |  0.9631  |  0.9585   |  0.9685  |  0.9634  |
| Logistic Regression      |  0.9293  |  0.9185   |  0.9433  |  0.9305  |
| MLP Classifier | 0.9494 | 0.9362 | 0.9664 | 0.9506 |
| Linear SVC | 0.9557 | 0.9416 | 0.9727 | 0.9568 |
| Ridge Classifier | 0.9293 | 0.8942 | 0.9748 | 0.9327 |
| K-Nearest Neighbors | 0.8850 | 0.8370 | 0.9579 | 0.8932 |
| Gaussian NB | 0.8724 | 0.8435 | 0.9180 | 0.8787 |
| Support Vector Machine | 0.6994 | 0.6596 | 0.8340 | 0.7358 |
| Perceptron | 0.6466 | 0.5346 | 0.6954 | 0.5906 |


---

## 🧠 Explainable AI (XAI) Insights

* [cite_start]**SHAP Summary Plot**: The SHAP analysis revealed that `medication`, `chest_tig` (chest tightness), and `pefr` (peak expiratory flow rate) are the most significant global predictors[cite: 334]. [cite_start]High values for `medication` and low values for `pefr` strongly influence the model's output[cite: 336].
* [cite_start]**LIME Plot**: LIME provides explanations for individual predictions[cite: 337]. [cite_start]For example, for a specific "No Asthma Attack" prediction, LIME highlighted that high medication use and low chest tightness were the primary contributing factors[cite: 338].

---

## 🔭 Future Scope

Future research can extend this work in several exciting directions:
* [cite_start]**Real-time Data Integration**: Incorporate live data from wearable devices and environmental sensors to enhance predictive accuracy[cite: 350].
* [cite_start]**Longitudinal Analysis**: Integrate long-term patient data, including genetic information and medication compliance, for more personalized predictions[cite: 350].
* [cite_start]**Advanced Data Fusion**: Combine structured and unstructured data (e.g., clinical notes) to create a more holistic patient profile and improve asthma care[cite: 352].

---

## 📜 License
This project is licensed under the **GNU General Public License v3.0**. See the `LICENSE` file for more details.
