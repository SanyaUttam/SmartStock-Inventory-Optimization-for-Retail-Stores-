import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import warnings
import os 
SALES_FILE_PATH = "cleaned_sales_data.csv"
FORECAST_FILE_PATH = "forecast_per_product_prophet.csv"

# Suppress Prophet/pandas future warnings
warnings.filterwarnings("ignore")

# --- 1. APP CONFIGURATION & UI ---
st.set_page_config(
    page_title="SmartStock Inventory Optimization Dashboard",
    page_icon="📦",
    layout="wide"
)

st.title("📦 SmartStock Inventory Optimization Dashboard")
st.markdown("""
This application analyzes historical sales data and forecasted demand to calculate optimal inventory levels.
- **ABC Analysis** classifies products based on their value.
- **EOQ, Safety Stock, and Reorder Point** are calculated for each product.
- A **Simulation** validates the ordering rules for high-value items against historical sales data.
""")

# --- Sidebar for User Inputs (Configuration Parameters) ---
with st.sidebar:
    st.header("⚙️ Configuration Parameters")

    # --- STATIC DATA CHECK ---
    st.subheader("1. Data Status (Read from File System)")
    if os.path.exists(SALES_FILE_PATH):
        st.success(f"✅ Sales Data Found: {SALES_FILE_PATH}")
    else:
        st.error(f"❌ SALES FILE NOT FOUND: {SALES_FILE_PATH}")

    if os.path.exists(FORECAST_FILE_PATH):
        st.info(f"☑️ Forecast Data Found: {FORECAST_FILE_PATH}")
    else:
        st.warning(f"⚠️ FORECAST FILE NOT FOUND: {FORECAST_FILE_PATH}")
    
    st.markdown("---")
    
    # 2. Inventory Control Parameters (Dynamic Inputs for Store Owner)
    st.subheader("2. Inventory & Financial Settings")
    service_level = st.slider(
        "Service Level (Target)", 0.80, 0.99, 0.95, 0.01, help="Desired probability of not stocking out."
    )
    lead_time_days = st.number_input(
        "Lead Time (Days)", min_value=1, value=7, help="Time from placing an order to receiving it."
    )
    holding_rate = st.slider(
        "Annual Holding Rate (%)", 1, 50, 15, help="Annual cost to hold one unit as a percentage of its cost."
    )
    ordering_cost = st.number_input(
        "Fixed Ordering Cost ($)", min_value=1.0, value=15.00, step=1.0
    )

    # 3. Simulation Parameters
    st.subheader("3. Simulation Settings")
    simulation_days = st.slider(
        "Simulation Period (Days)", 7, 90, 30, help="How many days to run the validation simulation for."
    )
    
    st.markdown("---")


# --- 2. CACHED DATA PROCESSING FUNCTIONS (Optimized for Streamlit) ---

@st.cache_data
def load_data_from_disk(sales_path, forecast_path):
    """Loads sales and forecast dataframes directly from disk paths."""
    
    # Load Sales Data (Required)
    try:
        df_raw = pd.read_csv(sales_path)
    except FileNotFoundError:
        st.error(f"FATAL: Sales data file not found at '{sales_path}'. Cannot proceed.")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"Error reading sales data: {e}")
        return pd.DataFrame(), pd.DataFrame()

    # Load Forecast Data (Optional)
    df_forecast = pd.DataFrame()
    try:
        df_forecast = pd.read_csv(forecast_path)
    except FileNotFoundError:
        st.warning(f"Forecast data file not found at '{forecast_path}'. Using historical average demand instead.")
    except Exception as e:
        st.warning(f"Error reading forecast data: {e}. Using historical average demand instead.")
        
    return df_raw, df_forecast

@st.cache_data
def prepare_inventory_data(df_raw, item_cost_map, df_forecast, days_in_year, holding_rate_pct):
    """Aggregates data, integrates forecasts, and prepares it for calculation."""
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    df_raw['Units Sold'] = df_raw['Units Sold'].clip(lower=0)

    # Calculate historical stats
    df_daily_sales = df_raw.groupby(['Date', 'Product ID'])['Units Sold'].sum().reset_index()
    df_stats = df_daily_sales.groupby('Product ID')['Units Sold'].agg(
        ADD_hist='mean',
        STD_DEV_D='std'
    ).fillna(0).reset_index()
    df_stats['STD_DEV_D'] = df_stats['STD_DEV_D'].fillna(0) # Handle single-sale products

    # Integrate forecast
    if not df_forecast.empty and 'Predicted Units Sold' in df_forecast.columns:
        df_forecasted_add = df_forecast.groupby('Product ID')['Predicted Units Sold'].mean().reset_index()
        df_forecasted_add.rename(columns={'Predicted Units Sold': 'Forecasted_ADD'}, inplace=True)
        df_stats = df_stats.merge(df_forecasted_add, on='Product ID', how='left')
        # Use forecast if available, otherwise fallback to historical
        df_stats['ADD'] = df_stats['Forecasted_ADD'].fillna(df_stats['ADD_hist'])
    else:
        df_stats['ADD'] = df_stats['ADD_hist']

    # Finalize inputs
    df_stats['Unit_Cost'] = df_stats['Product ID'].map(item_cost_map).fillna(25.00)
    df_stats['Annual_Demand'] = df_stats['ADD'] * days_in_year
    df_stats['Holding_Cost_H'] = df_stats['Unit_Cost'] * (holding_rate_pct / 100)
    return df_stats

@st.cache_data
def run_abc_analysis(df_inventory_stats, a_cutoff=0.80, b_cutoff=0.95):
    """Performs ABC classification."""
    df_abc = df_inventory_stats.copy()
    df_abc['Annual_Usage_Value'] = df_abc['Annual_Demand'] * df_abc['Unit_Cost']
    df_abc = df_abc.sort_values(by='Annual_Usage_Value', ascending=False)
    df_abc['Cumulative_Value'] = df_abc['Annual_Usage_Value'].cumsum()
    total_value = df_abc['Annual_Usage_Value'].sum()
    df_abc['Cumulative_Percentage'] = np.where(total_value > 0, df_abc['Cumulative_Value'] / total_value, 0)

    df_abc['ABC_Category'] = 'C'
    df_abc.loc[df_abc['Cumulative_Percentage'] <= b_cutoff, 'ABC_Category'] = 'B'
    df_abc.loc[df_abc['Cumulative_Percentage'] <= a_cutoff, 'ABC_Category'] = 'A'
    return df_abc.sort_index()

@st.cache_data
def calculate_inventory_levels(df_classified_data, z_score, lead_time, ordering_cost_val, lead_time_std_dev=1.0, min_safety_stock=5):
    """Calculates EOQ, Safety Stock, and Reorder Point."""
    df_calc = df_classified_data.copy()

    # EOQ
    df_calc['EOQ'] = np.sqrt((2 * df_calc['Annual_Demand'] * ordering_cost_val) / df_calc['Holding_Cost_H'].replace(0, np.nan))
    df_calc['EOQ'].fillna(0, inplace=True)
    
    # Safety Stock
    variance_demand_lt = lead_time * df_calc['STD_DEV_D']**2
    variance_lt_demand = (df_calc['ADD']**2) * lead_time_std_dev**2
    std_dev_combined = np.sqrt(variance_demand_lt + variance_lt_demand)
    df_calc['Safety_Stock'] = z_score * std_dev_combined
    
    # Robustness: Apply minimum safety stock for A-items
    df_calc['Safety_Stock'] = np.where(
        (df_calc['Safety_Stock'] < min_safety_stock) & (df_calc['ABC_Category'] == 'A'),
        min_safety_stock,
        df_calc['Safety_Stock']
    )
    
    # Reorder Point (ROP)
    df_calc['Demand_During_Lead_Time'] = df_calc['ADD'] * lead_time
    df_calc['Reorder_Point'] = df_calc['Demand_During_Lead_Time'] + df_calc['Safety_Stock']
    
    # Final cleanup
    for col in ['EOQ', 'Safety_Stock', 'Reorder_Point']:
        df_calc[col] = df_calc[col].clip(lower=0).round(0).astype(int)
    return df_calc

# MODIFIED FUNCTION to return the full daily stock log for visualization
@st.cache_data
def validate_decision_rules(df_raw, df_params, sample_days, lead_time):
    """Simulates inventory decisions for top 'A' items and returns the full daily log."""
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    df_daily_sales = df_raw.groupby(['Date', 'Product ID'])['Units Sold'].sum().reset_index()
    # Select top 3 'A' items
    a_items = df_params[(df_params['ABC_Category'] == 'A') & (df_params['EOQ'] > 0)]['Product ID'].head(3).tolist()

    if not a_items: return pd.DataFrame(), pd.DataFrame()

    df_sample_sales = df_daily_sales[df_daily_sales['Product ID'].isin(a_items)]
    end_date = df_sample_sales['Date'].max()
    start_date = end_date - pd.Timedelta(days=sample_days - 1)
    df_sample_sales = df_sample_sales[df_sample_sales['Date'] >= start_date]

    inventory_log = []
    daily_stock_log = [] # To store daily stock levels for plotting
    
    # Initial stock set higher than ROP
    initial_stock = {row['Product ID']: row['Reorder_Point'] + row['EOQ'] for _, row in df_params.iterrows() if row['Product ID'] in a_items}
    stock_levels = initial_stock.copy()
    
    for date in sorted(df_sample_sales['Date'].unique()):
        orders_received_today = [log for log in inventory_log if log.get('Decision') == 'ORDERED' and log.get('Arrival_Date') == date]
        order_placed_today = {item_id: False for item_id in a_items}

        for item_id in a_items:
            if item_id not in stock_levels: continue
            
            params = df_params[df_params['Product ID'] == item_id].iloc[0]
            rop, eoq, ss = params['Reorder_Point'], params['EOQ'], params['Safety_Stock']
            
            # 1. Receive shipments
            received_qty = sum(o['Order_Qty'] for o in orders_received_today if o['Product_ID'] == item_id)
            if received_qty > 0:
                stock_levels[item_id] += received_qty
                inventory_log.append({'Date': date, 'Product_ID': item_id, 'Decision': 'SHIPMENT RECEIVED', 'Stock_Level_at_Decision': stock_levels[item_id] - received_qty, 'Order_Qty': received_qty, 'ROP': rop, 'EOQ': eoq})
            
            # 2. Fulfill demand
            demand_today = df_sample_sales[(df_sample_sales['Date'] == date) & (df_sample_sales['Product ID'] == item_id)]['Units Sold'].sum()
            stock_after_demand = stock_levels[item_id] - demand_today
            stock_levels[item_id] = max(0, stock_after_demand) # New EOD Stock

            # 3. Check ROP and order
            if stock_after_demand <= rop and not order_placed_today[item_id]:
                order_placed_today[item_id] = True
                arrival_date = date + pd.Timedelta(days=lead_time)
                inventory_log.append({'Date': date, 'Product_ID': item_id, 'Decision': 'ORDERED', 'Stock_Level_at_Decision': stock_after_demand, 'Order_Qty': eoq, 'Arrival_Date': arrival_date, 'ROP': rop, 'EOQ': eoq})
            
            # 4. Log daily stock for plotting
            daily_stock_log.append({
                'Date': date, 
                'Product_ID': item_id, 
                'Inventory_Level': stock_levels[item_id],
                'ROP': rop,
                'Safety_Stock': ss
            })

    df_log = pd.DataFrame(inventory_log)
    df_daily_log = pd.DataFrame(daily_stock_log)
    key_events = ['ORDERED', 'SHIPMENT RECEIVED']
    df_events = df_log[df_log['Decision'].isin(key_events)].sort_values(['Date', 'Product_ID'])
    
    return df_events, df_daily_log

def create_inventory_plot(df_daily_log, product_id):
    """Creates a time-series plot of Inventory Level, ROP, and Safety Stock."""
    df_plot = df_daily_log[df_daily_log['Product_ID'] == product_id]
    if df_plot.empty:
        return None

    # Use melt to prepare data for easy line plotting with Plotly
    df_melt = df_plot.melt(
        id_vars=['Date', 'Product_ID'],
        value_vars=['Inventory_Level', 'ROP', 'Safety_Stock'],
        var_name='Metric',
        value_name='Quantity'
    )
    
    # Define colors
    color_map = {
        'Inventory_Level': 'blue',
        'ROP': 'orange',
        'Safety_Stock': 'red'
    }

    fig = px.line(
        df_melt,
        x='Date',
        y='Quantity',
        color='Metric',
        title=f"Inventory Levels vs. Reorder Points for {product_id}",
        color_discrete_map=color_map,
        height=400
    )
    fig.update_layout(legend_title_text='')
    return fig


# --- 3. MAIN APP LOGIC ---

# The logic is now unconditional and starts immediately by attempting to load files from disk.
df_raw, df_forecast = load_data_from_disk(SALES_FILE_PATH, FORECAST_FILE_PATH)

if df_raw.empty:
    st.info("⚠️ Dashboard waiting: Please place the 'cleaned_sales_data.csv' file in the directory and reload the app.")
else:
    # Static/Fallback Inputs (based on original milestone code)
    item_costs = {'P001': 250.00, 'P002': 180.00, 'P003': 100.00, 'P004': 75.00, 'P005': 60.00, 'P006': 50.00, 'P007': 30.00, 'P008': 20.00, 'P009': 15.00, 'P010': 10.00}
    for pid in df_raw['Product ID'].unique():
        item_costs.setdefault(pid, 25.00)
    
    # --- Calculation Workflow ---
    with st.spinner("Calculating inventory parameters..."):
        df_stats = prepare_inventory_data(df_raw, item_costs, df_forecast, 365, holding_rate)
        df_classified = run_abc_analysis(df_stats)
        z_score = norm.ppf(service_level)
        df_final = calculate_inventory_levels(df_classified, z_score, lead_time_days, ordering_cost)
        
        # Run simulation, getting both events and daily log
        df_validation_log, df_daily_log = validate_decision_rules(df_raw, df_final, simulation_days, lead_time_days)

    st.success("✅ Calculation Complete! Optimization Parameters Generated.")

    # --- Display Results ---
    
    # 1. ABC Analysis Summary
    col1, col2 = st.columns([1, 2])
    with col1:
        st.header("📊 ABC Analysis")
        abc_counts = df_final['ABC_Category'].value_counts()
        fig_abc = px.pie(
            names=abc_counts.index, 
            values=abc_counts.values, 
            title='Product Distribution by Annual Usage Value',
            color_discrete_sequence=px.colors.sequential.Agsunset,
            height=300
        )
        st.plotly_chart(fig_abc, use_container_width=True)
        
    # 2. Inventory Levels & ROP Plot (The required visualization)
    with col2:
        st.header("📈 Inventory Level Simulation")
        if not df_daily_log.empty:
            # Get the 'A' items that were simulated
            a_items_simulated = df_daily_log['Product_ID'].unique()
            product_to_plot = st.selectbox(
                "Select Product to Visualize", 
                options=a_items_simulated,
                help="Shows a simulation of stock level, ROP, and Safety Stock for the selected product."
            )
            
            # Render the plot
            fig_inv = create_inventory_plot(df_daily_log, product_to_plot)
            st.plotly_chart(fig_inv, use_container_width=True)
        else:
            st.warning("No 'A' items with valid EOQ were found or no simulation was possible.")

    # 3. Final Parameters Table and Download
    st.header("📋 Final Inventory Optimization Parameters")
    st.markdown("Recommended **ROP (Reorder Point)**, **EOQ (Order Quantity)**, and **Safety Stock** for all products.")

    display_cols = ['Product ID', 'ABC_Category', 'Unit_Cost', 'Annual_Demand', 'ADD', 'Reorder_Point', 'Safety_Stock', 'EOQ']
    
    tab_a, tab_b, tab_c = st.tabs(["Category A (High Value)", "Category B (Medium Value)", "Category C (Low Value)"])

    with tab_a:
        df_a = df_final[df_final['ABC_Category'] == 'A'].sort_values('Annual_Usage_Value', ascending=False)
        st.dataframe(df_a[display_cols].head(10), use_container_width=True)
    
    with tab_b:
        df_b = df_final[df_final['ABC_Category'] == 'B'].sort_values('Annual_Usage_Value', ascending=False)
        st.dataframe(df_b[display_cols].head(10), use_container_width=True)
    
    with tab_c:
        df_c = df_final[df_final['ABC_Category'] == 'C'].sort_values('Annual_Usage_Value', ascending=False)
        st.dataframe(df_c[display_cols].head(10), use_container_width=True)
        
    st.download_button(
        label="📥 Download Full Optimization Data (CSV)",
        data=df_final.to_csv(index=False).encode('utf-8'),
        file_name='inventory_optimization_parameters_final.csv',
        mime='text/csv',
    )
    
    # 4. Decision Validation Log
    st.header("🔬 Decision Rule Validation (Simulation Events)")
    if not df_validation_log.empty:
        log_display_cols = ['Date', 'Product_ID', 'Decision', 'Stock_Level_at_Decision', 'Order_Qty', 'ROP', 'EOQ']
        st.dataframe(df_validation_log[log_display_cols], use_container_width=True)
        st.download_button(
            label="📥 Download Full Simulation Log (CSV)",
            data=df_validation_log.to_csv(index=False).encode('utf-8'),
            file_name='inventory_validation_log_events.csv',
            mime='text/csv',
        )
    else:
        st.info("No reorder events were triggered for top 'A' items in the simulation period with current parameters.")