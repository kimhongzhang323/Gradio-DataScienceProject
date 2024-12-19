import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the data
data = pd.read_csv('data/cleaned_financial_data.csv')

# Define features and target variable
features = [
    'EBITDA (millions)', 'Revenue (millions)', 'Gross Profit (millions)', 
    'Op Income (millions)', 'EPS', 'Shares Outstanding', 'Year Close Price', 
    'Total Assets (millions)', 'Cash on Hand (millions)', 'Long Term Debt (millions)', 
    'Total Liabilities (millions)', 'Gross Margin', 'PE ratio', 'Employees'
]
X = data[features]
y = data['Net Income (millions)']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling - Standardize the features to have mean = 0 and variance = 1
scaler = StandardScaler()

# Fit the scaler on the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Only transform the test data (don't fit on test data)
X_test_scaled = scaler.transform(X_test)

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model with Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculate R-squared (R²) to see how well the model explains the variance in the target
r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')

# Perform cross-validation to get a better estimate of model performance
cv_scores = cross_val_score(model, scaler.fit_transform(X), y, cv=5, scoring='neg_mean_squared_error')
mse_cv = -cv_scores.mean()  # neg_mean_squared_error returns negative values
print(f'Mean Squared Error after Cross-Validation: {mse_cv}')

# Save the trained model and scaler
joblib.dump(model, 'linear_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler for future use

# Optional: View the coefficients of the trained model
coefficients = pd.DataFrame(model.coef_, features, columns=['Coefficient'])
print("Model Coefficients:")
print(coefficients)

# Optional: Show the first few rows of predictions vs actual values
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(predictions_df.head())
