# IMDB Movie Data ETL Pipeline

## Overview
This project is a robust data processing pipeline built with Python and Pandas. 
It's designed to ingest raw IMDB movie data, perform extensive cleaning, detect statistical outliers, and engineer features suitable for machine learning models.
The pipeline focuses on data quality assurance, ensuring that the final dataset is free of duplicates, corruption, and extreme outliers that could skew predictive modeling.

## Features

### 1. Data Cleaning
* **Deduplication:** Handles both exact row duplicates and partial duplicates based on unique identifiers (e.g., IMDB links).
* **Corruption Filtering:** automatically detects and removes rows with excessive missing data points based on a configurable threshold.
* **Smart Imputation:** logic to handle missing values differently based on data type:
    * **Predefined:** Specific defaults for known columns.
    * **Numeric:** Median imputation to resist skew.
    * **Categorical:** Mode (most frequent) imputation.

### 2. Outlier Detection
Implements two statistical methods for detecting and handling anomalies:
* **IQR Method:** Removes data points falling outside the Interquartile Range (1.5 * IQR).
* **Z-Score Method:** Removes data points with a standard deviation > 3 from the mean.

### 3. Feature Engineering
* **Binning:** Discretizes continuous variables (like Duration and IMDB Score) into categorical bins for classification tasks.
* **One-Hot Encoding:** Converts categorical variables into binary vectors.

## Technologies Used
* **Python 3.x**
* **Pandas:** Core data manipulation.
* **NumPy:** Numerical operations.
* **Matplotlib/Seaborn:** Used during the EDA phase (not included in the script) for visualizing distributions.

## Usage

Ensure you have the required libraries installed:

```bash
pip install pandas numpy
