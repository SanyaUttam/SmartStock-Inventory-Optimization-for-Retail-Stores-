import streamlit as st
import pandas as pd
import numpy as np

# ========================== STREAMLIT CONFIG ==========================
st.set_page_config(
    page_title="SmartStock - Inventory Optimization Dashboard",
    layout="wide",
    page_icon="📊"
)

st.title("📦 SmartStock Inventory Optimization Dashboard")
st.markdown("""
Welcome to the **SmartStock** analytics dashboard (Milestone 3).  
This application visualizes inventory performance, demand forecasts, and reorder optimization insights.
""")

# ========================== CACHED DATA LOAD ==========================
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"❌ Error loading file {file_path}: {e}")
        return pd.DataFrame()

# ========================== PREPARE INVENTORY DATA ==========================
@st.cache_data
def prepare_inventory_data(df_raw, item_cost_map, df_forecast, days_in_year, holding_rate_pct):
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    df_raw['Units Sold'] = df_raw['Units Sold'].clip(lower=0)

    # --- Detect or create 'Category Name' column ---
    possible_names = ['Category Name', 'Category', 'Category_Name', 'Product Category']
    category_col = None
    for col in possible_names:
        if col in df_raw.columns:
            category_col = col
            break

    if category_col is None:
        st.warning("⚠️ No category column found in dataset. Defaulting to 'Unknown Category'.")
        df_raw['Category Name'] = 'Unknown'
    elif category_col != 'Category Name':
        df_raw.rename(columns={category_col: 'Category Name'}, inplace=True)

    # --- Aggregate daily sales ---
    df_daily_sales = df_raw.groupby(['Date', 'Category Name', 'Product ID'])['Units Sold'].sum().reset_index()

    # --- Historical ADD & Std. Dev ---
    df_stats = df_daily_sales.groupby(['Category Name', 'Product ID'])['Units Sold'].agg(
        ADD_hist='mean', STD_DEV_D='std'
    ).fillna(0).reset_index()

    # --- Merge with forecast data if available ---
    if not df_forecast.empty and 'Predicted Units Sold' in df_forecast.columns:
        if 'Category Name' not in df_forecast.columns:
            # try mapping category from original df_raw
            cat_map = df_raw[['Product ID', 'Category Name']].drop_duplicates()
            df_forecast = df_forecast.merge(cat_map, on='Product ID', how='left')
            df_forecast['Category Name'].fillna('Unknown', inplace=True)

        df_forecasted_add = df_forecast.groupby(['Category Name', 'Product ID'])['Predicted Units Sold'].mean().reset_index()
        df_forecasted_add.rename(columns={'Predicted Units Sold': 'Forecasted_ADD'}, inplace=True)

        df_stats = df_stats.merge(df_forecasted_add, on=['Category Name', 'Product ID'], how='left')
        df_stats['ADD'] = df_stats['Forecasted_ADD'].fillna(df_stats['ADD_hist'])
    else:
        df_stats['ADD'] = df_stats['ADD_hist']

    # --- Compute cost and inventory parameters ---
    df_stats['Unit_Cost'] = df_stats['Product ID'].map(item_cost_map).fillna(25.00)
    df_stats['Annual_Demand'] = df_stats['ADD'] * days_in_year
    df_stats['Holding_Cost_H'] = df_stats['Unit_Cost'] * (holding_rate_pct / 100)

    return df_stats

# ========================== PAGE SECTIONS ==========================
st.sidebar.header("⚙️ Configuration")

cleaned_file = st.sidebar.file_uploader("Upload Cleaned Sales Data (CSV)", type=["csv"])
forecast_file = st.sidebar.file_uploader("Upload Forecasted Data (CSV)", type=["csv"])

days_in_year = st.sidebar.number_input("Days in Year", value=365)
holding_rate = st.sidebar.slider("Annual Holding Cost Rate (%)", 5.0, 30.0, 15.0)
item_cost_default = st.sidebar.number_input("Default Item Cost (₹)", value=25.0)

if cleaned_file:
    df_raw = load_data(cleaned_file)
    st.success("✅ Cleaned data loaded successfully.")
else:
    st.warning("⚠️ Please upload your cleaned sales dataset to continue.")
    st.stop()

if forecast_file:
    df_forecast = load_data(forecast_file)
    st.success("✅ Forecast data loaded successfully.")
else:
    df_forecast = pd.DataFrame()

# Dummy cost map (in real case, fetched from DB or CSV)
item_costs = {pid: np.random.uniform(20, 50) for pid in df_raw['Product ID'].unique()}

# ========================== DATA PREPARATION ==========================
st.subheader("📊 Data Processing")
df_stats = prepare_inventory_data(df_raw, item_costs, df_forecast, days_in_year, holding_rate)

st.dataframe(df_stats.head(), use_container_width=True)

# ========================== VISUALIZATION ==========================
st.subheader("📈 Category-wise Average Daily Demand (ADD)")
avg_demand = df_stats.groupby('Category Name')['ADD'].mean().reset_index()

st.bar_chart(data=avg_demand, x='Category Name', y='ADD', use_container_width=True)

st.subheader("📉 Inventory Metrics Summary")
st.write("""
- **ADD:** Average Daily Demand (from forecast or historical)
- **STD_DEV_D:** Daily demand variability  
- **Holding_Cost_H:** Annual holding cost per unit  
- **Annual_Demand:** Estimated yearly sales volume
""")

# ========================== DOWNLOAD SECTION ==========================
st.subheader("⬇️ Export Results")
csv_data = df_stats.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Processed Inventory Data as CSV",
    data=csv_data,
    file_name="inventory_categorywise_metrics.csv",
    mime="text/csv"
)

st.success("✅ Inventory analysis and data preparation completed successfully.")
st.markdown("---")
st.markdown("**Milestone 3 Completed — Inventory Optimization (Category-wise)** 🚀")
