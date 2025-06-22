# depression_predictor.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib # For saving and loading the trained model

# --- 1. Data Generation (Synthetic Dataset) ---
# In a real-world scenario, you would load data from a CSV, database, etc.
# For this demonstration, we'll create a synthetic dataset.
# Features: SleepHours, ExerciseHours, SocialSupportScore, StressLevel
# Target: DepressionRisk (0 = Low Risk, 1 = High Risk)
def generate_synthetic_data(num_samples=500):
    """
    Generates a synthetic dataset for depression risk prediction.
    Features include sleep hours, exercise hours, social support, and stress level.
    The 'DepressionRisk' target is assigned based on these features with some randomness.
    """
    np.random.seed(42) # For reproducibility of the generated data

    data = {
        'SleepHours': np.random.uniform(4, 10, num_samples),        # Daily sleep hours (e.g., 4 to 10)
        'ExerciseHours': np.random.uniform(0, 7, num_samples),      # Weekly exercise hours (e.g., 0 to 7)
        'SocialSupportScore': np.random.randint(1, 11, num_samples), # Score 1-10, higher is better support
        'StressLevel': np.random.randint(1, 11, num_samples),       # Score 1-10, higher is worse stress
    }
    df = pd.DataFrame(data)

    # Define a simple rule to assign 'DepressionRisk' (0 or 1)
    # Individuals with low sleep, low exercise, low social support, or high stress
    # are more likely to be in the 'High Risk' group.
    df['DepressionRisk'] = 0 # Default to low risk

    # Assign high risk (1) based on a combination of factors
    # About 80% chance of being high risk if criteria met, 20% chance of still being low risk (for noise)
    df.loc[
        (df['SleepHours'] < 6) |               # Less than 6 hours of sleep
        (df['ExerciseHours'] < 2) |            # Less than 2 hours of exercise per week
        (df['SocialSupportScore'] < 5) |       # Low social support
        (df['StressLevel'] > 6),               # High stress level
        'DepressionRisk'
    ] = np.random.choice([0, 1], size=len(df[(df['SleepHours'] < 6) | (df['ExerciseHours'] < 2) | (df['SocialSupportScore'] < 5) | (df['StressLevel'] > 6)]), p=[0.2, 0.8])

    # For those with generally healthy habits, mostly low risk, but with some chance of high risk (for realism)
    # About 90% chance of being low risk if criteria met, 10% chance of still being high risk (for noise)
    df.loc[
        (df['SleepHours'] >= 6) &
        (df['ExerciseHours'] >= 2) &
        (df['SocialSupportScore'] >= 5) &
        (df['StressLevel'] <= 6),
        'DepressionRisk'
    ] = np.random.choice([0, 1], size=len(df[(df['SleepHours'] >= 6) & (df['ExerciseHours'] >= 2) & (df['SocialSupportScore'] >= 5) & (df['StressLevel'] <= 6)]), p=[0.9, 0.1])

    # Ensure the target column is integer type
    df['DepressionRisk'] = df['DepressionRisk'].astype(int)
    return df

# --- 2. Main Machine Learning Pipeline ---
def main():
    """
    Executes the depression risk prediction pipeline:
    1. Generates synthetic data.
    2. Splits data into training and testing sets.
    3. Trains a Logistic Regression model.
    4. Evaluates the model's performance.
    5. Saves the trained model.
    6. Demonstrates how to load the model and make new predictions.
    """
    print("--- Depression Risk Predictor Project ---")
    print("Generating synthetic data for demonstration...")
    df = generate_synthetic_data(num_samples=500) # Generate 500 data samples
    print("Data generated successfully. First 5 rows:")
    print(df.head())
    print(f"\nDistribution of 'DepressionRisk':\n{df['DepressionRisk'].value_counts()}")

    # Separate features (X) and target (y) variable
    features = ['SleepHours', 'ExerciseHours', 'SocialSupportScore', 'StressLevel']
    X = df[features]
    y = df['DepressionRisk']

    # Split the dataset into training and testing sets
    # test_size=0.2 means 20% of data will be used for testing, 80% for training
    # random_state ensures reproducibility of the split
    # stratify=y ensures that the proportion of target classes is maintained in both train and test sets
    print("\nSplitting data into training and testing sets (80/20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")

    # Initialize and train the Logistic Regression model
    # Logistic Regression is a simple yet powerful algorithm for binary classification
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(random_state=42, solver='liblinear') # 'liblinear' solver is good for small datasets
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Make predictions on the test set to evaluate the model's performance
    print("\nEvaluating model performance on the test set...")
    y_pred = model.predict(X_test)

    # Evaluate the model using accuracy and a classification report
    # Accuracy: Proportion of correctly predicted instances
    # Classification Report: Provides precision, recall, f1-score for each class
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk'])

    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)

    # --- 3. Save the Trained Model ---
    # Saving the model allows you to use it later without retraining
    model_filename = 'depression_risk_model.pkl'
    joblib.dump(model, model_filename)
    print(f"\nTrained model saved as '{model_filename}'")
    print("You can deploy this model for new predictions.")

    # --- 4. Demonstrate Model Loading and New Predictions ---
    print("\n--- Example Prediction using the Saved Model ---")
    print(f"Loading the model '{model_filename}' for new predictions...")
    loaded_model = joblib.load(model_filename)
    print("Model loaded successfully.")

    # Example new data points for prediction
    # Create new data in the same format (DataFrame with feature columns)
    print("\nPredicting risk for example new individuals:")

    # Individual 1: Healthy lifestyle profile (expected Low Risk)
    new_individual_1 = pd.DataFrame([[7, 4, 9, 3]], columns=features) # Sleep:7, Exercise:4, Social:9, Stress:3
    prediction_1 = loaded_model.predict(new_individual_1)[0]
    prediction_proba_1 = loaded_model.predict_proba(new_individual_1)[0] # Probabilities for [Low Risk, High Risk]

    print(f"\nIndividual 1 (Sleep:7, Exercise:4, Social:9, Stress:3):")
    print(f"  Predicted Risk: {'HIGH' if prediction_1 == 1 else 'LOW'} ({prediction_1})")
    print(f"  Probability of Low Risk: {prediction_proba_1[0]:.4f}")
    print(f"  Probability of High Risk: {prediction_proba_1[1]:.4f}")

    # Individual 2: Less healthy lifestyle profile (expected High Risk)
    new_individual_2 = pd.DataFrame([[5, 1, 3, 8]], columns=features) # Sleep:5, Exercise:1, Social:3, Stress:8
    prediction_2 = loaded_model.predict(new_individual_2)[0]
    prediction_proba_2 = loaded_model.predict_proba(new_individual_2)[0]

    print(f"\nIndividual 2 (Sleep:5, Exercise:1, Social:3, Stress:8):")
    print(f"  Predicted Risk: {'HIGH' if prediction_2 == 1 else 'LOW'} ({prediction_2})")
    print(f"  Probability of Low Risk: {prediction_proba_2[0]:.4f}")
    print(f"  Probability of High Risk: {prediction_proba_2[1]:.4f}")

    print("\n--- Project End ---")

if __name__ == "__main__":
    main()

