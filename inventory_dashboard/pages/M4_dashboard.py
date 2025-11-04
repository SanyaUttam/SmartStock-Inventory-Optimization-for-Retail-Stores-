import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from datetime import datetime

PRIMARY_COLOR = "#1976D2"
SECONDARY_COLOR = "#4FC3F7"
TEXT_COLOR = "#212121"
MUTED_COLOR = "#757575"
BLACK_HEADING = "#000000"

st.set_page_config(
    page_title="Operational Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown(f"""
<style>
    /* 1. Global text color consistency */
    html, body, [class*="css"] {{
        color: {TEXT_COLOR};
        font-family: 'Poppins', sans-serif;
    }}
    
    /* 2. Main Title (Big-Font) - Black */
    .big-font {{
        font-size: 48px !important;
        font-weight: 700;
        color: {BLACK_HEADING};
    }}
    
    /* 3. Subheader below Title */
    .subheader {{
        font-size: 22px !important;
        color: {MUTED_COLOR};
    }}
    
    /* 4. Section Headers - Black */
    .section-header {{
        font-size: 28px !important;
        color: {BLACK_HEADING};
        margin-top: 20px;
    }}
    
    /* 5. Logo */
    .emoji-logo {{
        font-size: 80px;
        text-align: center;
        padding-top: 5px;
        line-height: 1;
    }}
    
    /* 6. Ensure default Streamlit headers (st.header, st.subheader) are Black */
    h1, h2, h3 {{
        color: {BLACK_HEADING};
    }}
    
    /* 7. Retain the professional look for the sidebar and other elements */
    [data-testid="stSidebar"] {{
      background: #F5F5F5;
      color: {TEXT_COLOR};
      border-right: 1px solid #E0E0E0;
    }}
    
    /* 8. Metric Cards (KPIs) */
    .metric-card {{
      border-radius: 8px;
      padding: 12px;
      text-align: center;
      color: {TEXT_COLOR};
      box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
      background: #FFFFFF;
      border: 1px solid #E0E0E0; 
    }}
    .metric-label {{ font-size: 12px; color: #757575; margin-bottom:6px; }}
    .metric-value {{ font-size:22px; font-weight:700; }}
</style>
""",
    unsafe_allow_html=True,
)


st.title("📊Operational Dashboard")

DATA_PATH = "data"
FORECAST_PATH = os.path.join(DATA_PATH, "forecast_per_product_prophet.csv")
INVENTORY_PATH = os.path.join(DATA_PATH, "inventory_optimization_parameters.csv")
LOG_PATH = os.path.join(DATA_PATH, "inventory_validation_log backup.csv")

required_files = [FORECAST_PATH, INVENTORY_PATH]
missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    st.error(f"⚠ Missing files: {', '.join([os.path.basename(f) for f in missing_files])}. Please place them in the `{DATA_PATH}` folder.")
    st.stop()


def render_metric(label, value, color=PRIMARY_COLOR):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color};">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

@st.cache_data
def detect_column(df, possible_names):
    if df is None: return None
    for name in possible_names:
        if name in df.columns:
            return name
    return None

@st.cache_data
def load_data(path, date_cols=None):
    try:
        df = pd.read_csv(path)
        if date_cols:
             for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce') 
        return df
    except Exception as e:
        if path != LOG_PATH:
             st.error(f"Error loading {os.path.basename(path)}: {e}")
        return pd.DataFrame()

def process_uploaded_log(uploaded_file):
    """Processes the uploaded CSV and attempts to convert the Date column."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            for dcol in ["Date", "date", "Decision_Date"]:
                if dcol in df.columns:
                    df["Date"] = pd.to_datetime(df[dcol], errors='coerce')
                    return df
            return df 
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
            return None
    return None


forecast_df = load_data(FORECAST_PATH, date_cols=["Date"])
inv_df = load_data(INVENTORY_PATH)
log_df_initial = load_data(LOG_PATH, date_cols=["Date"])

if 'uploaded_log_df' not in st.session_state:
    st.session_state.uploaded_log_df = None

log_df = st.session_state.uploaded_log_df if st.session_state.uploaded_log_df is not None else log_df_initial

P_COL_OPTIONS = ["Product ID", "Product_ID", "Product", "product", "SKU"]
F_COL = detect_column(forecast_df, P_COL_OPTIONS)
I_COL = detect_column(inv_df, P_COL_OPTIONS)
FORECAST_VALUE_COL = detect_column(forecast_df, ["Predicted Units Sold", "forecast_best"]) or "Predicted Units Sold" 

if F_COL is None or I_COL is None:
    st.error("⚠ Could not find a common Product ID column in the data files. Check for 'Product ID' or 'SKU'.")
    st.stop()

forecast_df.rename(columns={F_COL: "Product_ID", FORECAST_VALUE_COL: "Forecast_Value"}, inplace=True)
inv_df.rename(columns={I_COL: "Product_ID"}, inplace=True)

if "Category" in forecast_df.columns:
    category_map = forecast_df[["Product_ID", "Category"]].drop_duplicates()
    inv_df = inv_df.merge(category_map, on="Product_ID", how="left")
else:
    inv_df["Category"] = "Unknown"

inv_df["DisplayName"] = inv_df.apply(
    lambda r: f"{r['Product_ID']} — {r['Category']}" if pd.notna(r["Category"]) else str(r["Product_ID"]),
    axis=1
)

ROP_COL = detect_column(inv_df, ["Reorder_Point", "Reorder Point", "ROP"]) or "Reorder_Point"
EOQ_COL = detect_column(inv_df, ["EOQ", "Economic Order Quantity"]) or "EOQ"

if not log_df.empty:
    latest_stock = log_df.sort_values('Date', ascending=False).drop_duplicates(subset=['Product_ID'])
    STOCK_COL = detect_column(latest_stock, ["Stock_Level_at_Decision", "Current_Stock"]) or "Stock_Level_at_Decision"
    latest_stock = latest_stock.rename(columns={STOCK_COL: 'Current_Stock'})[['Product_ID', 'Current_Stock']]
else:
    latest_stock = pd.DataFrame(columns=['Product_ID', 'Current_Stock'])


inv_df = inv_df.merge(latest_stock, on='Product_ID', how='left')
inv_df['Current_Stock'] = inv_df['Current_Stock'].fillna(0) 
inv_df["Action"] = np.where(inv_df["Current_Stock"] <= inv_df[ROP_COL], "Reorder ⚠", "OK ✅")
inv_df["Order_Quantity"] = np.where(
    inv_df["Action"] == "Reorder ⚠",
    np.maximum(0, inv_df[EOQ_COL] + inv_df[ROP_COL] - inv_df["Current_Stock"]),
    0
).round(0).astype(int)


st.sidebar.header("🗺️ Main Navigation")
selected_tab = st.sidebar.radio(
    "Go to Section:",
    ("📈 Forecast Visualization", "🚨 Stock Alerts", "📝 Validation Log")
)

st.sidebar.markdown("---")
st.sidebar.header("⚙ Product Selection")

cat_list = sorted(inv_df["Category"].dropna().unique())
selected_cat = st.sidebar.selectbox("Filter by Category", ["All"] + cat_list)

if selected_cat != "All":
    product_choices = sorted(inv_df[inv_df["Category"] == selected_cat]["DisplayName"].unique())
else:
    product_choices = sorted(inv_df["DisplayName"].unique())

selected_display = st.sidebar.selectbox("Select Product", product_choices)
selected_product = selected_display.split(" — ")[0]
prod_forecast_df = forecast_df[forecast_df["Product_ID"] == selected_product].sort_values("Date")
prod_inventory_row = inv_df[inv_df["Product_ID"] == selected_product].iloc[0]


if selected_tab == "📈 Forecast Visualization":
    st.subheader(f"Demand Forecast for {selected_display}")
    
    if not prod_forecast_df.empty and "Forecast_Value" in prod_forecast_df.columns:
        avg_forecast = prod_forecast_df["Forecast_Value"].mean()
        peak = prod_forecast_df["Forecast_Value"].max()
        
        col1, col2, col3 = st.columns(3)
        with col1: render_metric("Average Forecast", f"{avg_forecast:,.0f}")
        with col2: render_metric("Peak Demand", f"{peak:,.0f}")
        with col3: render_metric("Forecast Range", f"{(peak - prod_forecast_df['Forecast_Value'].min()):,.0f}")

        fig1 = px.line(prod_forecast_df, x="Date", y="Forecast_Value", markers=True, 
                       title="Predicted Units Sold Over Time", color_discrete_sequence=[PRIMARY_COLOR])
        fig1.update_layout(template="plotly_white", xaxis_tickangle=-45) 
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Forecast data missing for the selected product.")



elif selected_tab == "🚨 Stock Alerts":
    st.subheader("🚨 Real-Time Stock Alerts")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a: render_metric("Current Stock", f"{prod_inventory_row['Current_Stock']:,.0f}")
    with col_b: render_metric("Reorder Point (ROP)", f"{prod_inventory_row[ROP_COL]:,.0f}")
    with col_c: render_metric("Economic Order Qty (EOQ)", f"{prod_inventory_row[EOQ_COL]:,.0f}")
    with col_d: 
        action_color = "#D32F2F" if prod_inventory_row["Action"].startswith("Reorder") else "#388E3C"
        render_metric("Action Required", prod_inventory_row["Action"], color=action_color)

    st.markdown("---")

    st.subheader("Global Inventory Status")
    alert_df = inv_df[["Product_ID", "Category", "Current_Stock", ROP_COL, EOQ_COL, "Action", "Order_Quantity"]].sort_values("Action", ascending=False)
    st.dataframe(alert_df, use_container_width=True, hide_index=True)
    
    low_count = alert_df[alert_df["Action"].str.contains("Reorder")].shape[0]
    if low_count > 0: st.warning(f"⚠️ **{low_count}** products are currently **BELOW** their Reorder Point. Recommended reorder quantities are listed above.")
    else: st.success("✅ All products are currently above their Reorder Point (ROP).")



elif selected_tab == "📝 Validation Log":
    st.subheader("📝 Inventory Validation Log & Data Injection")
    
    with st.expander("⬆️ **Upload New Inventory Log (CSV)**", expanded=False):
        uploaded_file = st.file_uploader(
            "Upload a new CSV file containing inventory/stock levels (Must have 'Product ID' and a date/stock column).", 
            type="csv"
        )
        if uploaded_file is not None:
            new_log_df = process_uploaded_log(uploaded_file)
            if new_log_df is not None:
                st.session_state.uploaded_log_df = new_log_df
                st.success("✅ Log file uploaded successfully! Data has been refreshed. Switch tabs to see updated alerts.")
                st.dataframe(new_log_df.head(), use_container_width=True)
                
    st.markdown("---")
    
    if not log_df.empty:
        st.subheader("Log History")
        
        prod_log_df = log_df[log_df["Product_ID"] == selected_product].sort_values("Date", ascending=False)
  
        col_log_a, col_log_b = st.columns(2)
        with col_log_a: 
            latest_date = prod_log_df['Date'].max().strftime('%Y-%m-%d') if not prod_log_df.empty and pd.notna(prod_log_df['Date'].max()) else 'N/A'
            render_metric("Last Log Entry Date", latest_date, SECONDARY_COLOR)
        with col_log_b: 
            total_entries = prod_log_df.shape[0]
            render_metric("Total Log Entries", f"{total_entries:,}", SECONDARY_COLOR)

        st.dataframe(prod_log_df, use_container_width=True, hide_index=True)

        csv_bytes = prod_log_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"⬇ Download Log for {selected_product}",
            data=csv_bytes,
            file_name=f"inventory_log_{selected_product}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Inventory validation log file not found or is empty. Please use the upload feature above to load a log file.")