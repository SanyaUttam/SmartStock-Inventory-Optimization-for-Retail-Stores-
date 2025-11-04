import pandas as pd
import numpy as np
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore") 

INVENTORY_PARAMS = {
    'HOLDING_RATE': 0.15,
    'ORDERING_COST': 15.00,
    'LEAD_TIME_DAYS': 7,
    'LEAD_TIME_STD_DEV': 1.0,
    'SERVICE_LEVEL': 0.95,
    'DAYS_IN_YEAR': 365,
    'MIN_SAFETY_STOCK': 5,
    'Z_SCORE': norm.ppf(0.95)
}

# --- 2. Data Aggregation and Preparation (Integrated with Forecast) ---

def prepare_inventory_data(df_raw, item_cost_map, df_forecast):
    """
    Aggregates data by item, calculates key statistical metrics, 
    and uses Forecasted ADD for better ROP/EOQ.
    """
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    df_raw['Units Sold'] = df_raw['Units Sold'].clip(lower=0) 
    
    # Calculate historical stats (used for STD_DEV_D and as ADD fallback)
    df_daily_sales = df_raw.groupby(['Date', 'Product ID'])['Units Sold'].sum().reset_index()
    df_stats = df_daily_sales.groupby('Product ID')['Units Sold'].agg(
        ADD='mean',
        STD_DEV_D='std',
        Total_Units_Sold='sum'
    ).fillna(0).reset_index()
    df_stats['STD_DEV_D'] = df_stats['STD_DEV_D'].fillna(0)
    
    # --- INTEGRATE FORECASTED DEMAND (Milestone 2 output) ---
    if not df_forecast.empty:
        # Calculate Mean Forecasted Demand (the preferred ADD)
        df_forecasted_add = df_forecast.groupby('Product ID')['Predicted Units Sold'].mean().reset_index()
        df_forecasted_add.rename(columns={'Predicted Units Sold': 'Forecasted_ADD'}, inplace=True)
        
        df_stats = df_stats.merge(df_forecasted_add, on='Product ID', how='left')

        # Replace Historical ADD with Forecasted ADD where available
        df_stats['ADD'] = np.where(
            df_stats['Forecasted_ADD'].notna(), 
            df_stats['Forecasted_ADD'], 
            df_stats['ADD']
        )
        df_stats.drop(columns=['Forecasted_ADD'], inplace=True, errors='ignore')
    
    # Finalize inputs based on the chosen ADD
    df_stats['Unit_Cost'] = df_stats['Product ID'].map(item_cost_map).fillna(25.00) 
    df_stats['Annual_Demand'] = df_stats['ADD'] * INVENTORY_PARAMS['DAYS_IN_YEAR']
    df_stats['Holding_Cost_H'] = df_stats['Unit_Cost'] * INVENTORY_PARAMS['HOLDING_RATE']

    return df_stats

# --- 3. Inventory Classification (ABC Analysis) ---

def run_abc_analysis(df_inventory_stats, a_cutoff=0.80, b_cutoff=0.95):
    """Performs ABC classification based on Annual Usage Value."""
    df_abc = df_inventory_stats.copy()
    
    df_abc['Annual_Usage_Value'] = df_abc['Annual_Demand'] * df_abc['Unit_Cost']
    
    df_abc = df_abc.sort_values(by='Annual_Usage_Value', ascending=False)
    df_abc['Cumulative_Value'] = df_abc['Annual_Usage_Value'].cumsum()
    
    total_value = df_abc['Annual_Usage_Value'].sum()
    df_abc['Cumulative_Percentage'] = np.where(total_value > 0, df_abc['Cumulative_Value'] / total_value, 0)

    df_abc['ABC_Category'] = 'C'
    df_abc.loc[df_abc['Cumulative_Percentage'] <= b_cutoff, 'ABC_Category'] = 'B'
    df_abc.loc[df_abc['Cumulative_Percentage'] <= a_cutoff, 'ABC_Category'] = 'A'
    
    return df_abc.sort_index().drop(columns=['Cumulative_Value'])

# --- 4. Inventory Calculation Logic ---

def calculate_inventory_levels(df_classified_data):
    """Calculates EOQ, Safety Stock, and Reorder Point."""
    df_calc = df_classified_data.copy()
    P = INVENTORY_PARAMS
    
    # 1. Economic Order Quantity (EOQ)
    df_calc['EOQ'] = np.where(
        (df_calc['Annual_Demand'] > 0) & (df_calc['Holding_Cost_H'] > 0),
        np.sqrt( (2 * df_calc['Annual_Demand'] * P['ORDERING_COST']) / df_calc['Holding_Cost_H'] ),
        0
    )
    
    # 2. Safety Stock (SS)
    variance_demand_during_lead_time = P['LEAD_TIME_DAYS'] * df_calc['STD_DEV_D']**2
    variance_lead_time_demand = (df_calc['ADD']**2) * P['LEAD_TIME_STD_DEV']**2
    
    std_dev_combined = np.sqrt(variance_demand_during_lead_time + variance_lead_time_demand)
    df_calc['Safety_Stock'] = P['Z_SCORE'] * std_dev_combined
    
    # Robustness: Apply minimum safety stock for A-items
    df_calc['Safety_Stock'] = np.where(
        (df_calc['Safety_Stock'] < P['MIN_SAFETY_STOCK']) & (df_calc['ABC_Category'] == 'A'),
        P['MIN_SAFETY_STOCK'],
        df_calc['Safety_Stock']
    )
    
    # 3. Reorder Point (ROP)
    df_calc['Demand_During_Lead_Time'] = df_calc['ADD'] * P['LEAD_TIME_DAYS']
    df_calc['Reorder_Point'] = df_calc['Demand_During_Lead_Time'] + df_calc['Safety_Stock']
    
    # Final cleanup: Round to integers for practical inventory counts
    for col in ['EOQ', 'Safety_Stock', 'Reorder_Point']:
        df_calc[col] = df_calc[col].clip(lower=0).round(0).astype(int)
        
    return df_calc

# --- 5. Reporting ---

def generate_reports(df_final):
    """Generates structured reports for A, B, and C category items."""
    
    display_cols = ['Product ID', 'ABC_Category', 'Unit_Cost', 'Annual_Demand', 
                    'ADD', 'STD_DEV_D', 'Reorder_Point', 'Safety_Stock', 'EOQ']
    
    print("\n" + "="*80)
    print("--- FINAL INVENTORY OPTIMIZATION PARAMETERS ---")
    
    df_a = df_final[df_final['ABC_Category'] == 'A'].sort_values('Annual_Usage_Value', ascending=False)
    print("\n>> Category A (High Value / High Control):")
    print(df_a[display_cols].head(5).to_markdown(index=False, floatfmt=".2f"))

    df_b = df_final[df_final['ABC_Category'] == 'B'].sort_values('Annual_Usage_Value', ascending=False)
    print("\n>> Category B (Medium Value / Moderate Control):")
    print(df_b[display_cols].head(5).to_markdown(index=False, floatfmt=".2f"))
    
    df_c = df_final[df_final['ABC_Category'] == 'C'].sort_values('Annual_Usage_Value', ascending=False)
    print("\n>> Category C (Low Value / Simplified Control):")
    print(df_c[display_cols].head(5).to_markdown(index=False, floatfmt=".2f"))
    
    print("="*80)

# --- 6. Decision Rule Validation (Simulation) ---

def validate_decision_rules(df_raw, df_params, sample_days=30):
    """
    Simulates inventory decisions for a sample of 'A' items.
    Returns:
        - df_order_events: Log of ORDERED and SHIPMENT RECEIVED events (for M3 output).
        - df_daily_log: Daily stock status including Stockout_Qty (for M4 KPI output).
    """
    P = globals().get('INVENTORY_PARAMS', {})
    if not P:
        return pd.DataFrame(), pd.DataFrame()

    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    df_daily_sales = df_raw.groupby(['Date', 'Product ID'])['Units Sold'].sum().reset_index()

    a_items = df_params[(df_params['ABC_Category'] == 'A') & (df_params['EOQ'] > 0)]['Product ID'].head(3).tolist()
    
    # Fallback check
    if not a_items and not df_params.empty:
        a_items = ['P0016', 'P0005', 'P0004']
        a_items = [item for item in a_items if item in df_params['Product ID'].tolist()]
        
    if not a_items:
        return pd.DataFrame(), pd.DataFrame()

    df_sample_sales = df_daily_sales[df_daily_sales['Product ID'].isin(a_items)].copy()
    
    end_date = df_sample_sales['Date'].max()
    start_date = end_date - pd.Timedelta(days=sample_days - 1)
    df_sample_sales = df_sample_sales[df_sample_sales['Date'] >= start_date]

    # Initialize Stock and Log
    order_event_log = []
    daily_stock_log = []
    
    initial_stock = {
        row['Product ID']: row['Reorder_Point'] + row['Safety_Stock'] + row['EOQ'] * 2
        for index, row in df_params.iterrows() if row['Product ID'] in a_items
    }
    stock_levels = initial_stock.copy()
    
    # 2. Daily Simulation Loop
    for date in sorted(df_sample_sales['Date'].unique()):
        
        # Check for Incoming Shipments for the day
        orders_received_today = [
            log for log in order_event_log 
            if log.get('Decision') == 'ORDERED' and log.get('Arrival_Date') == date
        ]
        
        order_placed_today = {item_id: False for item_id in a_items} 

        for item_id in a_items:
            if item_id not in stock_levels: continue
            
            params = df_params[df_params['Product ID'] == item_id].iloc[0]
            rop = params['Reorder_Point']
            eoq = params['EOQ']
            safety_stock = params['Safety_Stock']
            
            stock_level_at_start_of_day = stock_levels[item_id]
            
            # --- STOCK UPDATE 1: RECEIVE SHIPMENTS ---
            received_qty = sum(o['Order_Qty'] for o in orders_received_today if o['Product_ID'] == item_id)
            if received_qty > 0:
                stock_levels[item_id] += received_qty
                # Log the receipt event
                order_event_log.append({
                    'Date': date, 'Product_ID': item_id, 'Decision': 'SHIPMENT RECEIVED',
                    'Stock_Level_at_Decision': stock_level_at_start_of_day, 'Order_Qty': received_qty,
                    'ROP': rop, 'EOQ': eoq
                })
                
            # Get today's demand
            demand_today = df_sample_sales[
                (df_sample_sales['Date'] == date) & 
                (df_sample_sales['Product ID'] == item_id)
            ]['Units Sold'].sum()
            
            stock_level_before_demand = stock_levels[item_id] 
            
            # --- STOCK UPDATE 2: FULFILL DEMAND ---
            sales_fulfilled = min(demand_today, stock_level_before_demand)
            stockout_qty = demand_today - sales_fulfilled
            
            stock_levels[item_id] -= sales_fulfilled
            end_stock = stock_levels[item_id]
            
            # --- ROP CHECK AND ORDER PLACEMENT ---
            if end_stock <= rop and not order_placed_today[item_id]:
                order_qty = eoq
                arrival_date = date + pd.Timedelta(days=P['LEAD_TIME_DAYS'])
                
                order_placed_today[item_id] = True
                
                # Log the order placed 
                order_event_log.append({
                    'Date': date, 'Product_ID': item_id, 'Decision': 'ORDERED',
                    'Stock_Level_at_Decision': end_stock, 
                    'Order_Qty': order_qty, 'Arrival_Date': arrival_date,
                    'ROP': rop, 'EOQ': eoq
                })
                
            # --- DAILY STOCK LOG (M4 KPI Output) ---
            daily_stock_log.append({
                'Date': date, 
                'Product_ID': item_id, 
                'End_Stock': end_stock,
                'Stockout_Qty': stockout_qty,
                'ROP': rop, 
                'EOQ': eoq,
                'Safety_Stock': safety_stock
            })

    # Final Log generation
    df_order_events = pd.DataFrame(order_event_log)
    df_daily_log = pd.DataFrame(daily_stock_log)
    
    final_cols = ['Date', 'Product_ID', 'Decision', 'Stock_Level_at_Decision', 'Order_Qty', 'ROP', 'EOQ']
    df_order_events = df_order_events[df_order_events['Decision'].isin(['ORDERED', 'SHIPMENT RECEIVED'])].sort_values(['Date', 'Product_ID'])
    
    return df_order_events[final_cols], df_daily_log

# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    
    FILE_PATH = "cleaned_sales_data.csv"
    FORECAST_FILE = "forecast_per_product_prophet.csv" 
    
    # --- 1. Load Data ---
    try:
        df_raw = pd.read_csv(FILE_PATH)
        print(f"✅ Successfully loaded historical data from: {FILE_PATH}. Records: {len(df_raw)}")
    except FileNotFoundError:
        print(f"\nFATAL ERROR: The required file '{FILE_PATH}' was not found.")
        exit()
        
    try:
        df_forecast = pd.read_csv(FORECAST_FILE)
        print(f"✅ Loaded {len(df_forecast)} forecasted records for ROP/EOQ calculation.")
    except FileNotFoundError:
        print(f"⚠️ Warning: Forecast file '{FORECAST_FILE}' not found. Using historical averages only.")
        df_forecast = pd.DataFrame(columns=['Product ID', 'Predicted Units Sold']) 
        
    # --- 2. Define Unit Costs (CRITICAL INPUT) ---
    item_costs = {
        'P001': 250.00, 'P002': 180.00, 'P003': 100.00, 'P004': 75.00, 'P005': 60.00,'P006': 50.00, 'P007': 30.00, 'P008': 20.00, 'P009': 15.00, 'P010': 10.00 
    }
    
    product_list = df_raw['Product ID'].unique()
    for pid in product_list:
        if pid not in item_costs:
            item_costs[pid] = 25.00 # Default cost for unlisted products
            
    print(f"Total Unique Products found: {len(product_list)}")
    print("--- Milestone 3: Inventory Optimization Logic ---")

    # --- A. Data Preparation (Uses Forecasted ADD) ---
    df_stats = prepare_inventory_data(df_raw, item_costs, df_forecast)
    print("✅ Initial statistics calculated (ADD uses Forecasted Demand where available).")

    # --- B. ABC Classification ---
    df_classified = run_abc_analysis(df_stats)
    print("✅ ABC Classification completed.")

    # --- C. Inventory Calculations ---
    df_final = calculate_inventory_levels(df_classified)
    print("✅ ROP, EOQ, and Safety Stock calculated.")

    # --- D. Final Output and Reporting ---
    generate_reports(df_final)

    # --- E. Decision Rule Validation (Simulation) ---
    print("\n--- Decision Rule Validation (Simulation) ---")
    df_params = df_final.copy()
    
    # Run simulation for the last 30 days on the top 3 'A' items
    df_validation_log, df_daily_log = validate_decision_rules(df_raw, df_params, sample_days=30)
    
    if not df_validation_log.empty:
        simulated_products = len(df_validation_log['Product_ID'].unique())
        print(f"✅ Simulation run for the last 30 days on {simulated_products} A-Items.")
        print("Sample Order Events Log (Top 5 Orders Placed):")
        
        # Display sample events
        df_orders_placed = df_validation_log[
            (df_validation_log['Decision'] == 'ORDERED') &
            (df_validation_log['Order_Qty'].notna()) 
        ].head(5).copy()
        
        display_cols_log = ['Date', 'Product_ID', 'Decision', 'Stock_Level_at_Decision', 'ROP', 'Order_Qty', 'EOQ']
        print(df_orders_placed[display_cols_log].to_markdown(index=False, floatfmt=".0f"))
        
        # Save the order event log for potential use
        df_validation_log.to_csv('inventory_validation_log.csv', index=False)
        print("📄 Full validation log saved to: inventory_validation_log.csv")
    else:
        print("⚠️ No 'A' items with valid EOQ found or no reorder events triggered in the simulation period. Check data.")

    # --- F. Final File Generation for Milestone 4 Dashboard ---
    
    # 1. Save the ROP/EOQ table with the required name
    df_final.to_csv('inventory_optimization_parameters_final.csv', index=False)
    print(f"\n✅ ROP/EOQ Parameters saved to: inventory_optimization_parameters_final.csv (Required for M4 Recommendations)")
    
    # 2. Save the Daily Stock Log (CRITICAL for M4 KPIs)
    if not df_daily_log.empty:
        df_daily_log.to_csv('simulated_stock_log_M4.csv', index=False)
        print(f"✅ Daily Stock Log saved to: simulated_stock_log_M4.csv (Required for M4 KPIs/Visualization)")
    else:
        print(f"⚠️ Cannot save simulated_stock_log_M4.csv as the simulation log is empty.")