import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import shap
import os

# 1. Generate Synthetic Data
np.random.seed(42)
n_samples = 1000

experience = np.random.uniform(0, 20, n_samples)
education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
roles = ['Software Engineer', 'Data Scientist', 'Product Manager', 'Designer', 'DevOps']
locations = ['New York', 'San Francisco', 'Remote', 'London', 'Berlin']

education = np.random.choice(education_levels, n_samples)
role = np.random.choice(roles, n_samples)
location = np.random.choice(locations, n_samples)

# Base salary and multipliers
base_salary = 50000

def calculate_salary(exp, edu, role, loc):
    salary = base_salary
    salary += exp * 4000
    
    if edu == 'Bachelor': salary += 15000
    elif edu == 'Master': salary += 30000
    elif edu == 'PhD': salary += 45000
    
    if role == 'Software Engineer': salary += 20000
    elif role == 'Data Scientist': salary += 25000
    elif role == 'Product Manager': salary += 30000
    elif role == 'DevOps': salary += 18000
    
    if loc == 'San Francisco': salary += 40000
    elif loc == 'New York': salary += 30000
    elif loc == 'London': salary -= 5000
    elif loc == 'Berlin': salary -= 10000
    
    # Add some noise
    salary += np.random.normal(0, 8000)
    return salary

salaries = [calculate_salary(exp, edu, r, loc) for exp, edu, r, loc in zip(experience, education, role, location)]

df = pd.DataFrame({
    'YearsExperience': experience,
    'Education': education,
    'Role': role,
    'Location': location,
    'Salary': salaries
})

# 2. Preprocessing & Training
X = df.drop('Salary', axis=1)
y = df['Salary']

categorical_features = ['Education', 'Role', 'Location']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough')

# We apply the preprocessor first to get the feature names for SHAP
X_processed = preprocessor.fit_transform(X)

# Get feature names after one-hot encoding
cat_encoder = preprocessor.named_transformers_['cat']
cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
feature_names = list(cat_feature_names) + ['YearsExperience']

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Model R^2 score: {model.score(X_test, y_test)}")

# 3. Save Model and Preprocessor
model_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, 'model.joblib'))
joblib.dump(preprocessor, os.path.join(model_dir, 'preprocessor.joblib'))

# 4. Initialize SHAP Explainer
explainer = shap.TreeExplainer(model)
joblib.dump(explainer, os.path.join(model_dir, 'explainer.joblib'))
joblib.dump(feature_names, os.path.join(model_dir, 'feature_names.joblib'))

print("Training completed and artifacts saved.")
