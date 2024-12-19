import streamlit as st
import joblib
import pandas as pd
from graph_generation import plot_residuals, plot_feature_importance, plot_revenue_growth, plot_scatter_revenue_net_income, plot_correlation_heatmap

# Load the trained model
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

# Streamlit app
st.title("Financial Data Prediction and Analysis")

# Sidebar for input
st.sidebar.header("Input Parameters")
ebitda = st.sidebar.number_input("EBITDA (millions)", value=0.0)
revenue = st.sidebar.number_input("Revenue (millions)", value=0.0)
gross_profit = st.sidebar.number_input("Gross Profit (millions)", value=0.0)
op_income = st.sidebar.number_input("Op Income (millions)", value=0.0)
eps = st.sidebar.number_input("EPS", value=0.0)
shares_outstanding = st.sidebar.number_input("Shares Outstanding", value=0)
year_close_price = st.sidebar.number_input("Year Close Price", value=0.0)
total_assets = st.sidebar.number_input("Total Assets (millions)", value=0.0)
cash_on_hand = st.sidebar.number_input("Cash on Hand (millions)", value=0.0)
long_term_debt = st.sidebar.number_input("Long Term Debt (millions)", value=0.0)
total_liabilities = st.sidebar.number_input("Total Liabilities (millions)", value=0.0)
gross_margin = st.sidebar.number_input("Gross Margin", value=0.0)
pe_ratio = st.sidebar.number_input("PE ratio", value=0.0)
employees = st.sidebar.number_input("Employees", value=0)

# Prediction
if st.sidebar.button("Predict"):
    prediction = predict_net_income(ebitda, revenue, gross_profit, op_income, eps, shares_outstanding,
                                    year_close_price, total_assets, cash_on_hand, long_term_debt, 
                                    total_liabilities, gross_margin, pe_ratio, employees)
    st.write(f"Predicted Net Income (millions): {prediction}")

# Graphs
st.header("Model Evaluation Plots")

if st.button("Show Residuals Plot"):
    residuals_plot_path = plot_residuals(model)
    st.image(residuals_plot_path)

if st.button("Show Feature Importance"):
    feature_importance_plot_path = plot_feature_importance(model)
    st.image(feature_importance_plot_path)

if st.button("Show Year-over-Year Revenue Growth"):
    revenue_growth_plot_path = plot_revenue_growth()
    st.image(revenue_growth_plot_path)

if st.button("Show Revenue vs. Net Income Scatter Plot"):
    revenue_net_income_plot_path = plot_scatter_revenue_net_income()
    st.image(revenue_net_income_plot_path)

if st.button("Show Correlation Heatmap"):
    correlation_heatmap_plot_path = plot_correlation_heatmap()
    st.image(correlation_heatmap_plot_path)