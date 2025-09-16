
# Telco Customer Churn Prediction

This project aims to predict customer churn for a telecommunications company using machine learning techniques. By analyzing customer data, we build a model to identify customers who are likely to churn, allowing the company to take proactive measures to retain them.

## Dataset

The dataset used in this project is the Telco Customer Churn dataset, which contains information about customers, including their services, account information, and whether they churned or not.

## Project Steps

1.  **Data Loading and Initial Exploration:** Load the dataset and perform initial checks on its shape, missing values, and data types.
2.  **Data Cleaning and Preprocessing:** Handle missing values, convert data types, and perform encoding for categorical features. Unnecessary columns were also removed.
3.  **Exploratory Data Analysis (EDA):** Visualize the distribution of various features and explore relationships between features and the target variable (Churn).
4.  **Feature Engineering (Planned):** (This section can be updated after you complete the feature engineering steps)
5.  **Model Building:** Train a K-Nearest Neighbors (KNN) classification model to predict churn.
6.  **Model Evaluation:** Evaluate the performance of the KNN model using metrics like accuracy, precision, recall, and F1-score. A confusion matrix is also generated to visualize the model's predictions. The optimal number of neighbors for the KNN model was determined through experimentation.
7.  **Model Validation:** Cross-validation is used to get a more robust estimate of the model's performance.
8.  **Decision Boundary Visualization:** Visualize the decision boundary of the KNN model using two key features to understand how the model separates the classes.

## Technologies Used

*   Python
*   pandas (for data manipulation and analysis)
*   numpy (for numerical operations)
*   matplotlib and seaborn (for data visualization)
*   scikit-learn (for machine learning model building and evaluation)
*   joblib (for saving and loading the trained model)
*   google.colab.sheets (for interactive data exploration in Google Sheets)

## How to Run

1.  Clone the repository.
2.  Ensure you have the necessary libraries installed (`pip install pandas numpy matplotlib seaborn scikit-learn joblib google.colab`).
3.  Run the Python notebook.
4.  (Optional) If you implement the Streamlit app, follow the instructions to run the app locally.

## Future Work

*   Explore more advanced feature engineering techniques.
*   Experiment with other classification algorithms (e.g., Logistic Regression, Support Vector Machines, Random Forests, Gradient Boosting).
*   Perform hyperparameter tuning for the selected models.
*   Address class imbalance if necessary.
*   Deploy the model as a web application (e.g., using Streamlit or Flask).
