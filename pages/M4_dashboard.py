import streamlit as st
import pandas as pd
import numpy as np
import os
import io

st.set_page_config(
    page_title="SmartStock Operational Dashboard",
    page_icon="📊",
    layout="wide"
)

DATA_PATH = "data"
CLEANED_SALES_PATH = os.path.join(DATA_PATH, "cleaned_sales_data.csv")
FORECAST_PATH = os.path.join(DATA_PATH, "forecast_per_product_prophet.csv") 
INVENTORY_PATH = os.path.join(DATA_PATH, "inventory_optimization_parameters.csv")

@st.cache_data
def load_data(path, required_cols=None, rename_cols=None):
    """Loads CSV data with file path handling and column checks."""
    if not os.path.exists(path):
        st.error(f"⚠️ Data file not found: {path}. Please check your 'data/' folder.")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Error loading {os.path.basename(path)}: {e}")
        return pd.DataFrame()

    if rename_cols:
        actual_rename = {k: v for k, v in rename_cols.items() if k in df.columns}
        df.rename(columns=actual_rename, inplace=True)
    
    if required_cols and not set(required_cols).issubset(df.columns):
        missing = set(required_cols) - set(df.columns)
        st.warning(f"⚠️ Missing required columns in {os.path.basename(path)}: {missing}. "
                   f"The file contains: {list(df.columns)}. Data may not function correctly.")
        
    return df

def prepare_inventory_data(df):
    """
    Ensures the Inventory DataFrame has the necessary columns (ROP, EOQ, Current_Stock)
    for the Stock Alerts tab to function.
    """
    df_copy = df.copy()
    if 'ROP' not in df_copy.columns and 'Reorder_Point' in df_copy.columns:
        df_copy.rename(columns={'Reorder_Point': 'ROP'}, inplace=True)
    if 'Current_Stock' not in df_copy.columns and 'ROP' in df_copy.columns:
        st.info("ℹ️ Synthesizing 'Current_Stock' for Stock Alerts demo, as it was missing from the file.")
        np.random.seed(42)
        random_factor = np.random.choice([0.9, 1.2], size=len(df_copy), p=[0.4, 0.6])
        df_copy['Current_Stock'] = (df_copy['ROP'] * random_factor).apply(np.ceil).astype(int)

    return df_copy

INV_COLS = {"Product_ID", "ROP", "EOQ", "Current_Stock"}
FORECAST_COLS = {"Product_ID", "ds", "yhat"}

INV_RENAME = {
    "Product ID": "Product_ID", 
    "Reorder_Point": "ROP",      
    "EOQ": "EOQ", 
    "Stock": "Current_Stock", 
    "Total_Units_Sold": "Current_Stock", 
    "Current Stock": "Current_Stock"
}
FORECAST_RENAME = {
    "Product ID": "Product_ID",    
    "Date": "ds",                  
    "Predicted Units Sold": "yhat", 
    "product_id": "Product_ID", 
    "Product": "Product_ID", 
    "date": "ds", 
    "ds": "ds", 
    "forecast": "yhat", 
    "yhat": "yhat", 
    "prediction": "yhat",
    "yhat_upper": "yhat_upper",
    "yhat_lower": "yhat_lower"
}

st.session_state.inventory_df = load_data(INVENTORY_PATH, required_cols={"Product_ID", "ROP", "EOQ"}, rename_cols=INV_RENAME)
st.session_state.inventory_df = prepare_inventory_data(st.session_state.inventory_df)

st.session_state.forecast_df = load_data(FORECAST_PATH, required_cols=FORECAST_COLS, rename_cols=FORECAST_RENAME)
st.session_state.cleaned_sales_df = load_data(CLEANED_SALES_PATH)
st.title("📊 SmartStock Operational Dashboard")
st.markdown("""
This is the central control point for inventory management, combining **predictive analytics** with **actionable business logic**.
""")

st.markdown("---")
st.sidebar.title("📂 Navigation")
selected_tab = st.sidebar.radio(
    "Select a Module:",
    ["📈 Demand Forecast", "⚠️ Stock Alerts", "📂 Data Management"]
)
if selected_tab == "📈 Demand Forecast":
    st.header("📈 Category-wise Demand Forecast")
    st.markdown("""
    Visualize forecasted sales trends aggregated **by product category**.  
    If the dataset doesn’t contain category information, the dashboard automatically reverts to product-level visualization.
    """)

    forecast_df = st.session_state.forecast_df

    if not forecast_df.empty and set(["ds", "yhat"]).issubset(forecast_df.columns):
        try:
            forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
            forecast_df['yhat'] = pd.to_numeric(forecast_df['yhat'], errors='coerce').fillna(0)
        except Exception as e:
            st.error(f"Error parsing date/numeric columns in forecast data: {e}")
            st.stop()

        # Try to find category column
        possible_category_cols = ['Category', 'Product_Category', 'Item_Category', 'category', 'Product_Type']
        category_col = None
        for col in possible_category_cols:
            if col in forecast_df.columns:
                category_col = col
                break

        if category_col:
            # Aggregate forecast by category
            st.success(f"✅ Detected Category Column: `{category_col}`")
            cat_list = sorted(forecast_df[category_col].dropna().unique())
            selected_category = st.selectbox("Select Category:", cat_list)

            cat_data = (
                forecast_df[forecast_df[category_col] == selected_category]
                .groupby("ds")["yhat"]
                .sum()
                .reset_index()
                .sort_values("ds")
            )

            st.subheader(f"Forecasted Demand Trend — {selected_category}")
            st.line_chart(cat_data.set_index("ds")["yhat"])

            with st.expander("Show Aggregated Forecast Data"):
                st.dataframe(cat_data.tail(10), use_container_width=True)

        else:
            # Fallback to product-level visualization
            st.warning("⚠️ No category column found. Showing forecast by Product ID instead.")
            if "Product_ID" in forecast_df.columns:
                products = sorted(forecast_df["Product_ID"].unique())
                selected_product = st.selectbox("Select Product to Visualize:", products)

                product_data = forecast_df[forecast_df["Product_ID"] == selected_product].sort_values('ds')
                st.subheader(f"Forecasted Demand for Product: {selected_product}")
                st.line_chart(product_data.set_index('ds')['yhat'])

                with st.expander("Show Raw Forecast Data"):
                    st.dataframe(product_data.tail(10), use_container_width=True)
            else:
                st.error("❌ Neither Category nor Product_ID column found for visualization.")
    else:
        st.error("Cannot display Demand Forecast. Required columns (`ds`, `yhat`) are missing or the forecast file is empty.")


elif selected_tab == "⚠️ Stock Alerts":
    st.header("⚠️ Stock Alerts & Reorder Recommendations")
    st.markdown("Identify critical low-stock items and generate the daily reorder list.")

    inventory_df = st.session_state.inventory_df

    if not inventory_df.empty and INV_COLS.issubset(inventory_df.columns):
        low_stock_df = inventory_df[inventory_df["Current_Stock"] <= inventory_df["ROP"]].copy()
        low_stock_df['Recommended_Order'] = low_stock_df['EOQ'] + low_stock_df['ROP'] - low_stock_df['Current_Stock']
        low_stock_df['Recommended_Order'] = low_stock_df['Recommended_Order'].apply(lambda x: max(0, int(x)))
        reorder_report = low_stock_df[[
            "Product_ID", 
            "Current_Stock", 
            "ROP", 
            "EOQ", 
            "Recommended_Order"
        ]].sort_values(by="Current_Stock")
        total_products = inventory_df.shape[0]
        alert_count = reorder_report.shape[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Products Monitored", total_products)
        col2.metric("Products in Alert Status", alert_count, delta=f"{alert_count} items", delta_color="inverse")
        col3.metric("Total Recommended Order Units", reorder_report['Recommended_Order'].sum())

        st.markdown("---")

        if not reorder_report.empty:
            st.subheader(f"🚨 {alert_count} Products Require Immediate Reorder")
            st.dataframe(
                reorder_report.style.background_gradient(cmap='Reds', subset=['Current_Stock']),
                use_container_width=True
            )
            csv_data = reorder_report.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Reorder Report CSV",
                data=csv_data,
                file_name=f"smartstock_reorder_report_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.success("✅ Excellent! All products are currently above their Reorder Point.")
    else:
        st.error("Cannot perform Stock Alerts. Inventory data preparation failed or essential columns are missing.")


elif selected_tab == "📂 Data Management":
    st.header("📂 Data Upload & Management")
    st.markdown("Upload a new sales dataset to refresh analysis or review the current working data.")

    uploaded = st.file_uploader("📤 Upload New Sales Data (CSV)", type=["csv"])
    if uploaded is not None:
        try:
            user_df = pd.read_csv(uploaded)
            st.success(f"✅ File '{uploaded.name}' uploaded successfully. Showing first 5 rows.")
            st.dataframe(user_df.head(), use_container_width=True)
            st.info("Uploaded data can now be used for downstream analysis/forecasting.")
            
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            
    else:
        st.info("No file uploaded. Displaying the default `cleaned_sales_data.csv`.")
        if not st.session_state.cleaned_sales_df.empty:
            st.dataframe(st.session_state.cleaned_sales_df.head(10), use_container_width=True)
            with st.expander("View Data Details"):
                st.write(f"Total Rows: {st.session_state.cleaned_sales_df.shape[0]}")
                st.write(f"Total Columns: {st.session_state.cleaned_sales_df.shape[1]}")
                st.write("Column Types:")
                st.write(st.session_state.cleaned_sales_df.dtypes.astype(str))
        else:
            st.warning("Default sales data is missing or empty.")

st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:gray; font-size:12px;'>
        © 2025 SmartStock | Milestone 4 Implementation
    </div>
    """,
    unsafe_allow_html=True
)
