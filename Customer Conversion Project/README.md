# Customer Conversion Prediction
Project Overview

This project analyzes a customer dataset to predict whether a client will subscribe to a product. The dataset is imbalanced, requiring specialized techniques to handle skewed class distributions. Three machine learning models — Logistic Regression, Random Forest, and XGBoost — are evaluated to determine the best-performing model. The project highlights key steps including data cleaning, exploratory data analysis (EDA), handling outliers, feature encoding, oversampling, scaling, model building, and evaluation.

1. Libraries Used

The following libraries are used:

    Data manipulation: pandas, numpy
    Data visualization: matplotlib, seaborn
    Machine learning: sklearn, imblearn, xgboost

2. Dataset Loading and Cleaning

    Dataset loading: The dataset is loaded from a .csv file.
    Initial insights: Checked for missing values, data types, and duplicate rows.
    Outliers: Replaced with boundary values using the Interquartile Range (IQR) method.

3. Exploratory Data Analysis (EDA)

Performed EDA to understand feature distributions:

    Categorical variables: Visualized using bar charts to observe distributions and target variable correlations.
    Numerical variables: Analyzed using histograms and box plots to detect skewness and outliers.
    Bivariate analysis: Explored relationships between independent features and the target variable.

4. Feature Engineering

    Handling missing values: Replaced unknown values with the mode of categorical features.
    Encoding: Applied one-hot encoding and label encoding for categorical features.
    Feature correlation: Identified and dropped highly correlated features if necessary.

5. Data Preprocessing

    Imbalanced data: Handled using SMOTETomek to oversample the minority class.
    Scaling: Standardized numerical features using StandardScaler.

6. Model Training and Evaluation

Three models were trained and evaluated:

    Logistic Regression: Baseline model with decent performance.
    Random Forest: Performed well with high AUROC and accuracy.
    XGBoost: Achieved the best performance with an AUROC score of 0.986.

7. Feature Importance

    Analyzed feature importance using Random Forest to identify key predictors.
    The most important feature identified was duration.

8. Conclusion

XGBoost outperformed other models and is recommended for deployment in production to predict customer subscriptions. Further optimization is advised for better results.
How to Use the Code

    Install required libraries:

    pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn xgboost

    Run the script step-by-step in a Python environment (e.g., Jupyter Notebook, Google Colab).
    Provide your dataset by replacing the dataset(1).csv file with your dataset.

Project Insights

    Oversampling with SMOTETomek effectively handled the imbalanced dataset.
    XGBoost emerged as the best-performing model with the highest accuracy and AUROC score.
    Feature engineering and preprocessing significantly influenced model performance.

Future Improvements

    Explore additional hyperparameter tuning for all models.
    Consider other resampling techniques to handle imbalances.
    Use advanced algorithms or ensembles for potentially better performance.
