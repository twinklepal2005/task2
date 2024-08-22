import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Create and Save the CSV File

# Sample data
data = {
    'timestamp': [
        '2024-08-01 00:00:00', '2024-08-01 01:00:00', '2024-08-01 02:00:00',
        '2024-08-01 03:00:00', '2024-08-01 04:00:00', '2024-08-01 05:00:00'
    ],
    'sensor_reading': [100, 150, None, 130, 170, 200],
    'target_column': [0, 0, 1, 0, 1, 1],
    'last_maintenance_date': [
        '2024-07-01 00:00:00', '2024-07-01 00:00:00', '2024-07-01 00:00:00',
        '2024-07-01 00:00:00', '2024-07-01 00:00:00', '2024-07-01 00:00:00'
    ],
    'last_failure_date': [
        '2024-07-20 00:00:00', '2024-07-20 00:00:00', '2024-07-20 00:00:00',
        '2024-07-20 00:00:00', '2024-07-20 00:00:00', '2024-07-20 00:00:00'
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert datetime columns to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['last_maintenance_date'] = pd.to_datetime(df['last_maintenance_date'])
df['last_failure_date'] = pd.to_datetime(df['last_failure_date'])

# Save to CSV
df.to_csv('data.csv', index=False)

print("CSV file 'data.csv' created successfully.")

# Step 2: Load and Preprocess the Data

# Load the data from the CSV file
df = pd.read_csv('data.csv')

# 1. Handle Missing Values
df['sensor_reading'] = df['sensor_reading'].fillna(df['sensor_reading'].mean())

# 2. Handle Outliers
Q1 = df['sensor_reading'].quantile(0.25)
Q3 = df['sensor_reading'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df = df[(df['sensor_reading'] >= lower_bound) & (df['sensor_reading'] <= upper_bound)]

# 3. Normalization/Scaling
# Exclude non-numeric columns
numeric_features = df.select_dtypes(include=[np.number])

# Separate features and target
X = numeric_features.drop(columns=['target_column'])
y = numeric_features['target_column']

# Standardize the numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame for easier handling
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 4. Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Check the shapes of the resulting datasets
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
