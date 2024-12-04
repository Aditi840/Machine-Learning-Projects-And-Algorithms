# Wikipedia-Article-Classification

This project uses machine learning models to classify Wikipedia articles based on their content. The dataset contains articles with text features such as "Summary", "Full Text", "Links", and "Categories". The goal is to predict whether an article belongs to a specific class ('FA' or 'Non-FA').
Requirements

pip install Wikipedia-API
pip install imbalanced-learn
pip install scikit-learn
pip install xgboost

Data Processing

    Loading and Cleaning Data:
        Read and clean the dataset by handling missing values and duplicates.
        Convert certain columns (like 'Sections') into numerical values.
        Text columns are transformed into their lengths (Summary_Length, Full_Text_length).

    Feature Engineering:
        Count the number of links and categories in each article.

    Encoding:
        The 'Class' column is transformed into a binary label (Class_Binary).

Exploratory Data Analysis (EDA)

    Visualized class distribution with a bar chart showing class proportions.

Model Training

    Split the data into training and testing sets.
    Apply Standard Scaling for feature normalization.
    Use SMOTE for balancing the dataset.

Models Used

    Logistic Regression
    Random Forest Classifier
    SVM (Support Vector Machine)

Each model is trained on the original and resampled datasets, and their performance is evaluated using accuracy and AUC scores.
Results

    Best Model: Random Forest with the highest accuracy (99.38%).
    Feature Importance: Evaluates which features contribute most to model predictions.

Observations

    Using SMOTE decreased accuracy slightly, suggesting that the models perform better on the original dataset.
