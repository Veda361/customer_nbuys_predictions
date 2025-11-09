🧠 Customer Purchase Prediction using Logistic Regression & Decision Tree

This project aims to predict whether a customer will purchase a product based on demographic and review-based data using Machine Learning Classification Models — Logistic Regression and Decision Tree Classifier.

📋 Project Overview

This project performs a complete end-to-end ML pipeline, including:

Data loading and cleaning

Encoding categorical data

Feature scaling

Training and evaluating two ML models

Comparing model performance

Visualizing Decision Tree structure

The dataset (Customer_Review (1).csv) contains customer information such as:

Age

Gender

Review (e.g., Good/Poor)

Education

Purchased (target variable)

⚙️ Tech Stack
Category	Libraries/Tools
Language	Python 🐍
Data Handling	pandas, numpy
Visualization	matplotlib
Machine Learning	scikit-learn
Models Used	Logistic Regression, Decision Tree Classifier
🧩 Dataset Information

File: Customer_Review (1).csv
Each record represents a customer and their purchasing behavior.

Column	Description	Example
Age	Customer’s age	25
Gender	Gender of the customer	Male/Female
Review	Product review type	Good/Poor
Education	Education level	Graduate/Postgraduate
Purchased	Target (Yes/No)	Yes
🧠 Workflow

Import Required Libraries

pandas, numpy, matplotlib, sklearn

Load Dataset

data = pd.read_csv('Customer_Review (1).csv')


Preprocessing

Handle missing values

Encode categorical columns using One-Hot Encoding

Scale features using StandardScaler

Split Data

train_test_split(X_scaled, y, test_size=0.2, random_state=42)


Train Models

Model 1: Logistic Regression

Model 2: Decision Tree Classifier

Evaluate Models

Accuracy Score

Confusion Matrix

Classification Report

Visualize

Plot the Decision Tree using plot_tree()

🧮 Model Evaluation
Model	Accuracy	Characteristics
Logistic Regression	Predicts linear relationships well	Good for interpretable and simple patterns
Decision Tree Classifier	Handles non-linear and categorical data effectively	Can be visualized and explained easily

Example output:

✅ Logistic Regression Accuracy: 0.84
🌳 Decision Tree Accuracy: 0.90

📊 Decision Tree Visualization

The project includes a visualization of how the Decision Tree makes splitting decisions.

from sklearn import tree
plt.figure(figsize=(12,8))
tree.plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=['No','Yes'])
plt.show()

🧾 Results

The Decision Tree Classifier performed slightly better than Logistic Regression.

It captured non-linear relationships and categorical dependencies effectively.

Visualization of decision paths helps in interpretability.

🚀 How to Run the Project

Clone the repository:

git clone https://github.com/<your-username>/Customer-Purchase-Prediction.git
cd Customer-Purchase-Prediction


Install dependencies:

pip install numpy pandas matplotlib scikit-learn


Run the Python file:

python main.py


(Optional) Open in Jupyter Notebook:

jupyter notebook

📦 Requirements

Create a requirements.txt file with:

numpy
pandas
matplotlib
scikit-learn

🧠 Future Improvements

Add Random Forest and XGBoost models for better generalization.

Use GridSearchCV for hyperparameter tuning.

Deploy the model using Streamlit or Flask.

Visualize feature importance for deeper insight.

👨‍💻 Author

Dev Sahu
🚀 B.Tech Student | Developer | Machine Learning Enthusiast
🔗 LinkedIn
 | GitHub
