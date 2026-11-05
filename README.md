# Credit Card Default Prediction Using Machine Learning

## 1. Project Overview
This project focuses on predicting whether a credit card customer is likely to default on their payment in the following month using machine learning classification algorithms.

## 2. Objective
To develop and evaluate machine learning models that identify customers at risk of credit card default using historical customer and payment information.

## 3. Dataset
- **Dataset:** UCI Default of Credit Card Clients
- **Source:** UCI Machine Learning Repository
- **Records:** 30,000
- **Input Features:** 23
- **Target Variable:** `default.payment.next.month`

## 4. Machine Learning Models Used
The following classification algorithms were implemented and evaluated:

1. Logistic Regression – baseline classification model.
2. Decision Tree – captures nonlinear decision rules.
3. Random Forest – combines multiple decision trees.
4. K-Nearest Neighbors (KNN) – classifies based on neighboring observations.
5. Gradient Boosting – sequentially improves prediction performance.

Gradient Boosting was selected as the final model after model evaluation and hyperparameter tuning.

## 5. Model Evaluation Results

Final Gradient Boosting model results using a decision threshold of 0.35:

| Metric | Score |
|---|---:|
| Accuracy | 81.35% |
| Precision | 58.71% |
| Recall | 46.92% |
| F1-Score | 52.16% |
| ROC-AUC | 78.04% |

A recall-focused alternative threshold of 0.25 was also evaluated, achieving 58.62% recall and 53.47% F1-score, with 77.90% accuracy.

## 6. Model Optimization
- Hyperparameter tuning using GridSearchCV.
- Threshold selection using out-of-fold predictions.
- Group-aware validation to reduce duplicate-record leakage.
- Final model saved using Joblib.
- Selected model threshold: 0.35.

## 7. Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Google Colab

## 8. Project Files
- `Credit_Card_Default_Analysis.ipynb` – Complete analysis and model implementation.
- `UCI_Credit_Card.csv` – Dataset.
- `credit_card_default_model.pkl` – Saved optimized model and decision threshold.
- `README.md` – Project documentation.

## 9. Conclusion
The project demonstrates the application of machine learning classification techniques for predicting credit card default. Gradient Boosting achieved 81.35% accuracy and a ROC-AUC score of 78.04% at the selected threshold of 0.35. A lower threshold of 0.25 provides higher recall when identifying potential defaulters is prioritized.

## 10. Future Scope
- Explore additional feature engineering techniques.
- Evaluate advanced ensemble and boosting algorithms.
- Improve recall while maintaining acceptable precision.
- Develop a web-based interface for real-time default prediction.

---

**Project developed as part of the Entri Elevate Data Science Mini Project.**
