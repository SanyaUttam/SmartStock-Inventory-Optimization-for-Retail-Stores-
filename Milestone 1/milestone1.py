import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
PROFESSIONAL_PALETTE = 'viridis' 
LINE_COLORS = {'raw': '#2C3E50', 'lag': '#27AE60', 'roll': '#C0392B'}

def create_synthetic_data(file_path):
    """Creates a synthetic retail sales dataset."""
    print("Creating synthetic dataset...")
    
    # Parameters
    n_days = 365
    n_stores = 5
    n_products = 3
    start_date = '2023-01-01'
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    from itertools import product
    store_ids = [f'S{i+1}' for i in range(n_stores)]
    product_ids = [f'P{i+1}' for i in range(n_products)]
    base_data = list(product(store_ids, product_ids, dates))
    df = pd.DataFrame(base_data, columns=['Store ID', 'Product ID', 'Date'])
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Units Sold'] = (
        10 +                                        
        np.random.randint(0, 5, size=len(df)) +     
        df['DayOfWeek'].apply(lambda x: 5 if x >= 5 else 0) + 
        (df.index / (n_stores * n_products * n_days) * 50)  
    ).astype(int)
    
    price_map = {f'P{i+1}': 5 + i * 2 for i in range(n_products)}
    df['Price'] = df['Product ID'].map(price_map) + np.random.uniform(-0.5, 0.5, size=len(df))
    
    df.to_csv(file_path, index=False)
    print(f"Synthetic data saved to: {file_path}")
    return df
def initial_cleaning(file_path):
    """Loads the data, converts 'Date' to datetime, and object columns to category."""
    print("\n--- Starting Milestone 1: Data Cleaning and Feature Engineering ---")

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}. Creating synthetic data instead.")
        df = create_synthetic_data(file_path)
    
    df['Date'] = pd.to_datetime(df['Date'])
    object_cols = df.select_dtypes(include='object').columns
    for col in object_cols:
        df[col] = df[col].astype('category')

    print("Initial cleaning complete: Date and Categorical columns optimized.")
    return df
def feature_engineering(df):
    """Creates time-based, time-series, and price-related features."""
    
    df = df.sort_values(by=['Store ID', 'Product ID', 'Date']).reset_index(drop=True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfMonth'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    print("Calendar features (Year, Month, DayOfWeek, etc.) created.")

    TARGET = 'Units Sold'
    GROUP_KEYS = ['Store ID', 'Product ID']
    LAG_DAYS = [1, 7]
    ROLLING_WINDOWS = [7, 30]
    
    for lag in LAG_DAYS:
        df[f'{TARGET}_Lag_{lag}'] = df.groupby(GROUP_KEYS)[TARGET].shift(lag)

    for window in ROLLING_WINDOWS:
        df[f'{TARGET}_RollingMean_{window}'] = (
            df.groupby(GROUP_KEYS)[TARGET]
            .rolling(window=window)
            .mean()
            .reset_index(level=GROUP_KEYS, drop=True)
            .shift(1)
        )

    df['Price_Lag_1'] = df.groupby(GROUP_KEYS)['Price'].shift(1)
    df['Price_Lag_7'] = df.groupby(GROUP_KEYS)['Price'].shift(7)
    df['Price_Change_7Day'] = (df['Price'] - df['Price_Lag_7']) / df['Price_Lag_7'].replace(0, np.nan)
    df.drop(columns=['Price_Lag_7'], inplace=True)

    print("Time-series and Price-related features created.")
    return df
def handle_missing_values(df):
    """Handles NaNs primarily introduced by Lag and Rolling Mean features."""
    print("\n--- Starting Missing Value Handling ---")
    
    if df['Units Sold'].isnull().sum() > 0:
        print(f"Dropping {df['Units Sold'].isnull().sum()} rows with missing 'Units Sold'.")
        df.dropna(subset=['Units Sold'], inplace=True)

    ts_cols = df.columns[df.columns.str.contains('Lag|RollingMean|Price_Change')]
    
    for col in ts_cols:
        df[col] = df[col].fillna(0)
    
    remaining_nans = df.isnull().sum().loc[lambda x: x > 0]
    if not remaining_nans.empty:
        print("Remaining NaNs in critical columns after imputation (dropping these rows):")
        print(remaining_nans)
        df.dropna(inplace=True)
    else:
        print("No remaining NaNs in critical columns.")
    
    print(f"Final dataset shape after handling NaNs: {df.shape}")
    return df

def visualize_features(df):
    """Generates comparison graphs one at a time for professional review."""
    print("\n--- Starting Visualization and Verification (6 Graphs, One at a Time) ---")

    sample_group = df.groupby(['Store ID', 'Product ID']).first().index[0]
    sample_df = df[(df['Store ID'] == sample_group[0]) & (df['Product ID'] == sample_group[1])].copy()
    
    plt.figure(figsize=(14, 6))
    plt.plot(sample_df['Date'], sample_df['Units Sold'], label='Units Sold (Raw)', color=LINE_COLORS['raw'], linewidth=2)
    plt.plot(sample_df['Date'], sample_df['Units Sold_Lag_7'], label='Units Sold Lag 7', color=LINE_COLORS['lag'], linestyle='--')
    plt.plot(sample_df['Date'], sample_df['Units Sold_RollingMean_30'], label='Units Sold Rolling Mean 30', color=LINE_COLORS['roll'], linestyle='-.')
    plt.title(f'1. Time Series Comparison (Sample: {sample_group[0]}, {sample_group[1]})')
    plt.xlabel('Date')
    plt.ylabel('Units Sold')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='DayOfWeek', y='Units Sold', data=df, palette='viridis')
    plt.title('2. Units Sold by Day of Week (0=Mon, 6=Sun) - Intra-Week Seasonality')
    plt.xlabel('Day of Week')
    plt.ylabel('Units Sold')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Month', y='Units Sold', data=df, palette='plasma')
    plt.title('3. Units Sold by Month - Inter-Year Seasonality')
    plt.xlabel('Month')
    plt.ylabel('Units Sold')
    plt.show()
    
    temp_df = df.dropna(subset=['Units Sold_Lag_7'])
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='Units Sold_Lag_7', y='Units Sold', data=temp_df, alpha=0.1, color=LINE_COLORS['raw'])
    
    max_sales = temp_df[['Units Sold', 'Units Sold_Lag_7']].max().max()
    plt.plot([0, max_sales], [0, max_sales], color=LINE_COLORS['roll'], linestyle='--', label='Perfect Correlation (y=x)')
    
    plt.title('4. Lag 7 Verification (Weekly Seasonality Check)')
    plt.xlabel('Units Sold Lag 7')
    plt.ylabel('Units Sold (Current)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    # Use the raw color for the histogram
    sns.histplot(df['Units Sold'], kde=True, bins=30, color=LINE_COLORS['raw'])
    plt.title('5. Distribution of Units Sold')
    plt.xlabel('Units Sold')
    plt.ylabel('Frequency')
    plt.show()
    
    df['Price_Change_Bins'] = pd.cut(df['Price_Change_7Day'], bins=5, labels=False, include_lowest=True)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Price_Change_Bins', y='Units Sold', data=df, palette='gray')
    plt.title('6. Sales vs. 7-Day Price Change Bins (0=Lowest Change)')
    plt.xlabel('7-Day Price Change Bin')
    plt.ylabel('Units Sold')
    plt.show()
    
    print("Visualization complete. Please close all plot windows.")

if __name__ == "__main__":
    FILE_PATH = "retail_store_inventory.csv"
    OUTPUT_FILE = "cleaned_sales_data.csv"

    df_cleaned = initial_cleaning(FILE_PATH)

    df_engineered = feature_engineering(df_cleaned)

    df_final = handle_missing_values(df_engineered)
    
    visualize_features(df_final.copy())

    df_final.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nMilestone 1 Complete! Final dataset saved as: {OUTPUT_FILE}")
    print(f"Shape of the final dataset: {df_final.shape}")
    print("\nNext step is Milestone 2: Train-Test Split and Model Training.")