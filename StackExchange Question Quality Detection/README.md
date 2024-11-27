Predicting StackExchange Question Quality Using Machine Learning

Overview

This project aims to classify questions on an online Q&A platform into different quality categories: good quality, low quality, and very low quality. By leveraging data preprocessing, feature engineering, and machine learning algorithms, the analysis provides insights into the key factors affecting question quality and builds predictive models for automated classification.

Workflow

1. Dataset Creation

    Extracted data from an XML file using the ElementTree library.
    Transformed the extracted data into a structured CSV format (dataset.csv).

2. Sampling and Labeling

    Performed Simple Random Sampling to create a manageable dataset (sampled_dataset.csv).
    Labeled questions into quality categories based on score and answer count:
        Good Quality: Score > 5 and at least one answer.
        Low Quality: Score between 0 and 5 with any number of answers.
        Very Low Quality: Negative scores.

3. Data Cleaning

    Dropped irrelevant columns and cleaned text data (e.g., removed HTML tags).
    Engineered features such as:
        Title Text Length
        Body Text Length
        Body Sentence Count
    Removed columns like Tags, Title, and Body to reduce complexity.
    Addressed missing values and dropped columns with excessive null entries (e.g., FavoriteCount).

4. Exploratory Data Analysis (EDA)

    Visualized class distributions and performed univariate analysis of features.
    Examined correlations to understand feature relationships.
    Observed data imbalance in question_quality and addressed it with SMOTE.

5. Modeling

    Feature Scaling: Used StandardScaler for normalization.
    Models Implemented:
        Logistic Regression
        Random Forest Classifier
        Multinomial Naive Bayes (with SMOTE)
    Evaluated models using:
        Accuracy
        Precision
        Recall
        F1-score

6. Cross-Validation

    Used StratifiedKFold for robust model evaluation.

7. Feature Importance

    Analyzed feature importance for the Random Forest model to understand key contributors to question quality.

Results
Model Performance
Model	Train Accuracy	Test Accuracy	Test Precision	Test Recall	Test F1-Score
Logistic Regression	83.45%	83.53%	0.77	0.84	0.77
Random Forest	100%	83.12%	0.76	0.83	0.78
Multinomial Naive Bayes	-	70.87%	0.73	0.70	0.72
Key Observations

    Logistic Regression demonstrated balanced performance, simplicity, and interpretability.
    Random Forest achieved strong results but showed potential overfitting.
    Multinomial Naive Bayes struggled with accuracy compared to other models.

Conclusion

    Logistic Regression emerged as a strong candidate for this task due to its high accuracy, interpretability, and generalization on unseen data.
    Feature engineering and SMOTE helped address data imbalance, improving classification accuracy.
    Random Forest provided deeper insights with feature importance analysis, showcasing the influence of different features on predictions.

How to Use

    Clone the repository:

git clone <repository_url>
cd <repository_directory>

Install dependencies:

pip install -r requirements.txt

Run the Python script:

    python main.py

    View results and visualizations generated during the process.

Technologies Used

    Python Libraries:
        Data Handling: pandas, numpy
        Data Visualization: matplotlib, seaborn
        Machine Learning: scikit-learn, xgboost, imblearn
        Natural Language Processing: nltk
    Tools: Jupyter Notebook

Future Scope

    Explore deep learning approaches like BERT for textual feature extraction.
    Apply hyperparameter tuning to further optimize model performance.
    Automate the XML-to-CSV pipeline for scalability.
    
