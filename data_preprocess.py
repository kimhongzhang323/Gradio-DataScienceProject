import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error
import seaborn as sns

# Load the dataset (replace 'financial_data.csv' with your actual file path)
file_path = 'data/data.csv'
df = pd.read_csv(file_path)

# 1. Remove Special Characters (Dollar Signs and Commas) in Numeric Columns
columns_to_clean = [
    "EBITDA (millions)", "Revenue (millions)", "Gross Profit (millions)",
    "Op Income (millions)", "Net Income (millions)", "Total Assets (millions)",
    "Cash on Hand (millions)", "Long Term Debt (millions)", "Total Liabilities (millions)",
    "Year Close Price", "Shares Outstanding", "Employees", "EPS"
]

# Remove '$' and ',' from the numeric columns and convert them to float
for col in columns_to_clean:
    df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)

# 2. Handle Percentage Columns (e.g., Gross Margin and PE ratio)
df["Gross Margin"] = df["Gross Margin"].replace('%', '', regex=True).astype(float)
df["PE ratio"] = df["PE ratio"].replace('N/A', None).astype(float)

# 3. Inspect for Missing or Inconsistent Data
missing_data_summary = df.isnull().sum()
print("Missing Data Summary:\n", missing_data_summary)

# Handle missing data
df["PE ratio"].fillna(df["PE ratio"].median(), inplace=True)

# 4. Add Derived Columns
df["Debt to Asset Ratio"] = df["Long Term Debt (millions)"] / df["Total Assets (millions)"]
df["Revenue Growth (%)"] = df["Revenue (millions)"].pct_change() * 100

# 5. Convert 'Employees' and 'Shares Outstanding' from String to Integer (e.g., '164,000' to 164000)
df['Employees'] = df['Employees'].replace('[,]', '', regex=True).astype(int)
df['Shares Outstanding'] = df['Shares Outstanding'].replace('[,]', '', regex=True).astype(int)

# 6. Save the Cleaned Data
cleaned_file_path = 'cleaned_financial_data.csv'
df.to_csv(cleaned_file_path, index=False)

print(f"Cleaned data saved to {cleaned_file_path}")

# Optional: Show the cleaned dataframe or analysis
print("\nCleaned Dataframe:")
print(df.head())

# Load the trained model and scaler
model = joblib.load('linear_regression_model.pkl')

# Define the prediction function
def predict_net_income(ebitda, revenue, gross_profit, op_income, eps, shares_outstanding,
                       year_close_price, total_assets, cash_on_hand, long_term_debt, 
                       total_liabilities, gross_margin, pe_ratio, employees):
    input_data = pd.DataFrame({
        'EBITDA (millions)': [ebitda],
        'Revenue (millions)': [revenue],
        'Gross Profit (millions)': [gross_profit],
        'Op Income (millions)': [op_income],
        'EPS': [eps],
        'Shares Outstanding': [shares_outstanding],
        'Year Close Price': [year_close_price],
        'Total Assets (millions)': [total_assets],
        'Cash on Hand (millions)': [cash_on_hand],
        'Long Term Debt (millions)': [long_term_debt],
        'Total Liabilities (millions)': [total_liabilities],
        'Gross Margin': [gross_margin],
        'PE ratio': [pe_ratio],
        'Employees': [employees]
    })
    prediction = model.predict(input_data)
    return prediction[0]

# Define the plotting functions
def plot_residuals():
    # Load the data
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

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, color='b', edgecolors='black')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('residuals_plot.png')
    return 'residuals_plot.png'

def plot_feature_importance():
    # Assuming the model has a coef_ attribute (like LinearRegression)
    feature_importance = model.coef_
    features = [
        'EBITDA (millions)', 'Revenue (millions)', 'Gross Profit (millions)', 
        'Op Income (millions)', 'EPS', 'Shares Outstanding', 'Year Close Price', 
        'Total Assets (millions)', 'Cash on Hand (millions)', 'Long Term Debt (millions)', 
        'Total Liabilities (millions)', 'Gross Margin', 'PE ratio', 'Employees'
    ]

    plt.figure(figsize=(10, 6))
    plt.barh(features, feature_importance, color='purple')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    return 'feature_importance.png'

def plot_revenue_growth():
    # Load data and plot the year-over-year revenue growth
    data = pd.read_csv('data/cleaned_financial_data.csv')
    plt.figure(figsize=(10, 6))
    plt.plot(data['year'], data['Revenue Growth (%)'], marker='s', color='orange', linestyle='-', label='Revenue Growth (%)')
    plt.xlabel('Year')
    plt.ylabel('Revenue Growth (%)')
    plt.title('Year-over-Year Revenue Growth')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('revenue_growth.png')
    return 'revenue_growth.png'

def plot_scatter_revenue_net_income():
    # Scatter plot of Revenue vs. Net Income
    data = pd.read_csv('data/cleaned_financial_data.csv')
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Revenue (millions)'], data['Net Income (millions)'], color='r')
    plt.xlabel('Revenue (Millions)')
    plt.ylabel('Net Income (Millions)')
    plt.title('Revenue vs. Net Income')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('revenue_vs_net_income.png')
    return 'revenue_vs_net_income.png'

def plot_correlation_heatmap():
    # Correlation heatmap
    data = pd.read_csv('data/cleaned_financial_data.csv')
    correlation_data = data[['EBITDA (millions)', 'Revenue (millions)', 'Gross Profit (millions)', 
                             'Op Income (millions)', 'Net Income (millions)', 'EPS', 'Shares Outstanding', 
                             'Year Close Price', 'Total Assets (millions)', 'Cash on Hand (millions)', 
                             'Long Term Debt (millions)', 'Total Liabilities (millions)', 'Gross Margin', 
                             'PE ratio', 'Employees']]
    corr_matrix = correlation_data.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Financial Data')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    return 'correlation_heatmap.png'

# Create the Gradio interface
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Prediction"):
            with gr.Row():
                with gr.Column():
                    ebitda = gr.Number(label="EBITDA (millions)", precision=2)
                    revenue = gr.Number(label="Revenue (millions)", precision=2)
                    gross_profit = gr.Number(label="Gross Profit (millions)", precision=2)
                    op_income = gr.Number(label="Op Income (millions)", precision=2)
                    eps = gr.Number(label="EPS", precision=2)
                    shares_outstanding = gr.Number(label="Shares Outstanding", precision=0)
                    year_close_price = gr.Number(label="Year Close Price", precision=2)
                    total_assets = gr.Number(label="Total Assets (millions)", precision=2)
                    cash_on_hand = gr.Number(label="Cash on Hand (millions)", precision=2)
                    long_term_debt = gr.Number(label="Long Term Debt (millions)", precision=2)
                    total_liabilities = gr.Number(label="Total Liabilities (millions)", precision=2)
                    gross_margin = gr.Number(label="Gross Margin", precision=2)
                    pe_ratio = gr.Number(label="PE ratio", precision=2)
                    employees = gr.Number(label="Employees", precision=0)

                with gr.Column():
                    output = gr.Number(label="Predicted Net Income (millions)")
                    predict_btn = gr.Button("Predict")

            # Prediction button functionality
            predict_btn.click(predict_net_income, inputs=[ebitda, revenue, gross_profit, op_income, eps, shares_outstanding,
                                                          year_close_price, total_assets, cash_on_hand, long_term_debt, 
                                                          total_liabilities, gross_margin, pe_ratio, employees], outputs=output)

        with gr.TabItem("Graphs"):
            gr.Markdown("## Model Evaluation Plots")
            gr.Markdown("### Residuals Plot")
            residuals_img = gr.Image(label="Residuals Plot")
            gr.Markdown("### Feature Importance")
            feature_importance_img = gr.Image(label="Feature Importance")
            gr.Markdown("### Year-over-Year Revenue Growth")
            revenue_growth_img = gr.Image(label="Revenue Growth")
            gr.Markdown("### Revenue vs. Net Income Scatter Plot")
            revenue_net_income_img = gr.Image(label="Revenue vs Net Income")
            gr.Markdown("### Correlation Heatmap")
            correlation_heatmap_img = gr.Image(label="Correlation Heatmap")

            # Button to load and display the graphs
            graph_btn = gr.Button("Show Graphs")
            graph_btn.click(fn=plot_residuals, outputs=residuals_img)
            graph_btn.click(fn=plot_feature_importance, outputs=feature_importance_img)
            graph_btn.click(fn=plot_revenue_growth, outputs=revenue_growth_img)
            graph_btn.click(fn=plot_scatter_revenue_net_income, outputs=revenue_net_income_img)
            graph_btn.click(fn=plot_correlation_heatmap, outputs=correlation_heatmap_img)

# Launch the Gradio app
demo.launch(share=True)
