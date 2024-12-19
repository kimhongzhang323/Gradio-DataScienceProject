import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting function for residuals
def plot_residuals(model):
    data = pd.read_csv('data/cleaned_financial_data.csv')
    features = [
        'EBITDA (millions)', 'Revenue (millions)', 'Gross Profit (millions)', 
        'Op Income (millions)', 'EPS', 'Shares Outstanding', 'Year Close Price', 
        'Total Assets (millions)', 'Cash on Hand (millions)', 'Long Term Debt (millions)', 
        'Total Liabilities (millions)', 'Gross Margin', 'PE ratio', 'Employees'
    ]
    X = data[features]
    y = data['Net Income (millions)']
    y_pred = model.predict(X)
    residuals = y - y_pred

    # Debugging: Print shapes and first few rows
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("y_pred shape:", y_pred.shape)
    print("First few rows of X:\n", X.head())
    print("First few rows of y:\n", y.head())
    print("First few rows of y_pred:\n", y_pred[:5])
    print("First few rows of residuals:\n", residuals[:5])

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, color='b', edgecolors='black')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('residuals_plot.png')
    plt.close()
    return 'residuals_plot.png'

# Plotting function for feature importance
def plot_feature_importance(model):
    feature_importance = model.coef_
    features = [
        'EBITDA (millions)', 'Revenue (millions)', 'Gross Profit (millions)', 
        'Op Income (millions)', 'EPS', 'Shares Outstanding', 'Year Close Price', 
        'Total Assets (millions)', 'Cash on Hand (millions)', 'Long Term Debt (millions)', 
        'Total Liabilities (millions)', 'Gross Margin', 'PE ratio', 'Employees'
    ]

    # Debugging: Print feature importance
    print("Feature importance:", feature_importance)

    plt.figure(figsize=(10, 6))
    plt.barh(features, feature_importance, color='purple')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    return 'feature_importance.png'

# Plotting function for year-over-year revenue growth
def plot_revenue_growth():
    data = pd.read_csv('data/cleaned_financial_data.csv')
    
    # Debugging: Print first few rows of data
    print("First few rows of data:\n", data.head())

    plt.figure(figsize=(10, 6))
    plt.plot(data['year'], data['Revenue Growth (%)'], marker='s', color='orange', linestyle='-', label='Revenue Growth (%)')
    plt.xlabel('Year')
    plt.ylabel('Revenue Growth (%)')
    plt.title('Year-over-Year Revenue Growth')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('revenue_growth.png')
    plt.close()
    return 'revenue_growth.png'

# Scatter plot of revenue vs net income
def plot_scatter_revenue_net_income():
    data = pd.read_csv('data/cleaned_financial_data.csv')
    
    # Debugging: Print first few rows of data
    print("First few rows of data:\n", data.head())

    plt.figure(figsize=(10, 6))
    plt.scatter(data['Revenue (millions)'], data['Net Income (millions)'], color='r')
    plt.xlabel('Revenue (Millions)')
    plt.ylabel('Net Income (Millions)')
    plt.title('Revenue vs. Net Income')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('revenue_vs_net_income.png')
    plt.close()
    return 'revenue_vs_net_income.png'

# Plotting function for correlation heatmap
def plot_correlation_heatmap():
    data = pd.read_csv('data/cleaned_financial_data.csv')
    correlation_data = data[['EBITDA (millions)', 'Revenue (millions)', 'Gross Profit (millions)', 
                             'Op Income (millions)', 'Net Income (millions)', 'EPS', 'Shares Outstanding', 
                             'Year Close Price', 'Total Assets (millions)', 'Cash on Hand (millions)', 
                             'Long Term Debt (millions)', 'Total Liabilities (millions)', 'Gross Margin', 
                             'PE ratio', 'Employees']]
    corr_matrix = correlation_data.corr()

    # Debugging: Print correlation matrix
    print("Correlation matrix:\n", corr_matrix)

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Financial Data')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()
    return 'correlation_heatmap.png'