# HeartDisease_Prediction

This project focuses on building and evaluating machine learning models to predict the likelihood of heart disease in patients based on clinical and diagnostic features. The goal is to support early detection and intervention by using data-driven techniques.

## Objective

To apply various classification algorithms and compare their performance in predicting heart disease using a publicly available dataset. The project explores data preprocessing, feature selection, model training, and evaluation using performance metrics.

## Dataset

- **Source:** [Kaggle – Heart Attack Prediction Dataset](https://www.kaggle.com/datasets/ahmedmohamedibrahim1/heart-attack-prediction-dataset)
- **Attributes:** Includes features like age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, ST depression, and more.

## Machine Learning Models Used

- Logistic Regression  
- Decision Tree Classifier  
- K-Nearest Neighbors (KNN)  
- Random Forest (optional extension)  
- SVM (optional extension)

## Workflow

1. **Data Preprocessing**
   - Handling missing/null values
   - Converting categorical variables
   - Feature scaling

2. **Exploratory Data Analysis (EDA)**
   - Visualizing class balance
   - Correlation heatmaps
   - Distribution plots for key variables

3. **Model Training & Evaluation**
   - Splitting data into training and test sets
   - Training different ML classifiers
   - Evaluating performance using:
     - Accuracy
     - Precision, Recall, F1-Score
     - Confusion Matrix
     - ROC-AUC Score

## Tools & Technologies

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

## Results

Among the models tested, Logistic Regression and Decision Trees showed strong and interpretable performance. Evaluation metrics like F1-score and ROC-AUC were used to assess model effectiveness in classifying positive and negative cases of heart disease.

## Key Takeaways

- Classification models can effectively support early prediction of heart disease.
- Preprocessing and feature engineering play a critical role in healthcare ML models.
- This project demonstrates the end-to-end ML pipeline from raw data to model evaluation.

## Future Improvements

- Add hyperparameter tuning with GridSearchCV
- Test ensemble models like Random Forest and Gradient Boosting
- Deploy model via a web interface (e.g., Streamlit or Flask)

## Author

**Sai Sahi**  
MSc Computer Science – Teesside University  
Email: sahisai141@gmail.com  
GitHub: https://github.com/SahiMLE

---

*This project is part of my AI/ML learning journey. Feedback and collaboration are welcome!*
