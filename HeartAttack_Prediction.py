# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:53:58 2023

@author: almor
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import shap

def load_data(data_path):
    """
    Load the dataset from the given file path.
    """
    return pd.read_csv(data_path)

def preprocess_data(data):
    """
    Preprocess the dataset by removing outliers and converting class labels to numeric.
    """
    # Remove outliers in 'impluse' column
    data = data[data['impluse'] <= 1000]
    
    # Convert 'class' column to numeric
    data['class'] = data['class'].replace({'positive': 1, 'negative': 0}).astype(int)
    
    return data

def perform_eda(data):
    """
    Perform exploratory data analysis (EDA) and visualize key features.
    """
    # Plot Age
    plt.figure(figsize=(8, 6))
    age_data = data['age']
    plt.hist(age_data, bins=30, alpha=0.5, color='b', label='Data')
    plt.title('Histogram')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    
    # Check for outliers with box plot
    numerical_columns = data.select_dtypes(include=['float64', 'int64']) 
    plt.figure(figsize=(12, 6))  
    sns.boxplot(data=numerical_columns)
    plt.title("Box Plot of Numerical Columns (Check for Outliers)")
    plt.xticks(rotation=45)  
    
    # Impulse has outliers > 1000 - remove them
    data = data[data['impluse'] <= 1000]
    
    # Heatmap for correlations
    corr_matrix = data.corr()
    plt.figure(figsize=(12, 15))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    
    # Plot collage for age
    fig, axis = plt.subplots(1, 3, figsize=(12, 5))
    plt.suptitle('Age', fontsize=30, color='blue')
    sns.scatterplot(data=data, x='age', y='class', ax=axis[0], hue='class')
    sns.boxplot(data=data, x='class', y='age', ax=axis[1])
    sns.histplot(data=data, x='age', ax=axis[2])
    
    # Plot collage for gender
    fig, axis = plt.subplots(1, 3, figsize=(12, 5))
    plt.suptitle('Gender', fontsize=30, color='blue')
    sns.scatterplot(data=data, x='gender', y='class', ax=axis[0], hue='class')
    sns.boxplot(data=data, x='class', y='gender', ax=axis[1])
    sns.histplot(data=data, x='gender', ax=axis[2])
    
    # Plot collage for impulse
    fig, axis = plt.subplots(1, 3, figsize=(12, 5))
    plt.suptitle('Impulse', fontsize=30, color='blue')
    sns.scatterplot(data=data, x='impluse', y='class', ax=axis[0], hue='class')
    sns.boxplot(data=data, x='class', y='impluse', ax=axis[1])
    sns.histplot(data=data, x='impluse', ax=axis[2])
    
    # Plot collage for pressurehight
    fig, axis = plt.subplots(1, 3, figsize=(12, 5))
    plt.suptitle('PressureHight', fontsize=30, color='blue')
    sns.scatterplot(data=data, x='pressurehight', y='class', ax=axis[0], hue='class')
    sns.boxplot(data=data, x='class', y='pressurehight', ax=axis[1])
    sns.histplot(data=data, x='pressurehight', ax=axis[2])
    
    # Plot collage for pressurelow
    fig, axis = plt.subplots(1, 3, figsize=(12, 5))
    plt.suptitle('PressureLow', fontsize=30, color='blue')
    sns.scatterplot(data=data, x='pressurelow', y='class', ax=axis[0], hue='class')
    sns.boxplot(data=data, x='class', y='pressurelow', ax=axis[1])
    sns.histplot(data=data, x='pressurelow', ax=axis[2])
    
    # Plot collage for glucose
    fig, axis = plt.subplots(1, 3, figsize=(12, 5))
    plt.suptitle('Glucose', fontsize=30, color='blue')
    sns.scatterplot(data=data, x='glucose', y='class', ax=axis[0], hue='class')
    sns.boxplot(data=data, x='class', y='glucose', ax=axis[1])
    sns.histplot(data=data, x='glucose', ax=axis[2])
    
    # Plot collage for kcm
    fig, axis = plt.subplots(1, 3, figsize=(12, 5))
    plt.suptitle('KCM', fontsize=30, color='blue')
    sns.scatterplot(data=data, x='kcm', y='class', ax=axis[0], hue='class')
    sns.boxplot(data=data, x='class', y='kcm', ax=axis[1])
    sns.histplot(data=data, x='kcm', ax=axis[2])
    
    # Plot collage for troponin
    fig, axis = plt.subplots(1, 3, figsize=(12, 5))
    plt.suptitle('Troponin', fontsize=30, color='blue')
    sns.scatterplot(data=data, x='troponin', y='class', ax=axis[0], hue='class')
    sns.boxplot(data=data, x='class', y='troponin', ax=axis[1])
    sns.histplot(data=data, x='troponin', ax=axis[2])

    plt.show()

def train_and_evaluate_model(model, param_grid, X_train, X_test, y_train, y_test, cv=3):
    """
    Train and evaluate the given model using GridSearchCV and various metrics.
    """
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=3, scoring='accuracy', n_jobs=-1)
    
    # Perform hyperparameter tuning on training data
    grid_search.fit(X_train, y_train)
    
    # Get best parameters
    best_params = grid_search.best_params_
    
    # Train model with best hyperparameters on training data
    best_model = model.__class__(**best_params)
    best_model.fit(X_train, y_train)
    
    # Make predictions on the testing data
    y_pred = best_model.predict(X_test)
    
    # Evaluate the model with various metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f'Best Hyperparameters: {best_params}')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    
    return best_model, accuracy

if __name__ == "__main__":
    data_path = r"C:\Users\almor\Downloads\Heart Attack.csv"
    data = load_data(data_path)
    
    data = preprocess_data(data)
    perform_eda(data)  
    
    X = data.drop(columns=['class'])
    y = data['class']
    
    # Handling Class Imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)
    
    models = [
        {
            'model': RandomForestClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [10, 50, 100, 150, 250, 500],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        },
        {
            'model': GradientBoostingClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.1, 0.2],
                'min_samples_split': [2, 5, 10]
            }
        },
        {
            'model': KNeighborsClassifier(),
            'param_grid': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            }
        }
    ]
    
    results = []
    
    best_accuracy = 0
    best_model = None

    for model_info in models:
        model = model_info['model']
        param_grid = model_info['param_grid']
        current_model, accuracy = train_and_evaluate_model(model, param_grid, X_train, X_test, y_train, y_test)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = current_model
    
    # Find feature importance with SHAP for the best model
    # Create Explainer
    explainer = shap.TreeExplainer(best_model)

    # Calculate SHAP values with feature names
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=X.columns)

    print(f"Best Model: {best_model.__class__.__name__}, Accuracy: {best_accuracy:.2f}")

    # Best performing model
    final_model = best_model
