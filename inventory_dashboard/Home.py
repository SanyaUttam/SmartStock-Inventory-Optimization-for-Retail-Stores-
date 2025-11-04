import streamlit as st
import os

PRIMARY_COLOR = "#1976D2"
TEXT_COLOR = "#212121"
MUTED_COLOR = "#757575"
BLACK_HEADING = "#000000"

st.set_page_config(
    page_title="SmartStock | Inventory Optimization System",
    page_icon="📦",
    layout="wide"
)

st.markdown(f"""
<style>
    /* 1. Global text color consistency */
    html, body, [class*="css"] {{
        color: {TEXT_COLOR};
        font-family: 'Poppins', sans-serif;
    }}
    
    /* 2. Main Title */
    .big-font {{
        font-size: 48px !important;
        font-weight: 700;
        color: {BLACK_HEADING};
        margin-bottom: 0px;
    }}
    
    /* 3. Subtitle (Subheader below Title) */
    .subtitle {{
        font-size: 22px !important;
        color: {MUTED_COLOR};
        margin-top: 5px;
    }}
    
    /* 4. Section Headers - Black */
    .section-header, h1, h2, h3 {{
        color: {BLACK_HEADING};
    }}
    
    /* 5. Sidebar and other elements */
    [data-testid="stSidebar"] {{
      background: #F5F5F5;
      color: {TEXT_COLOR};
      border-right: 1px solid #E0E0E0;
    }}
</style>
""", unsafe_allow_html=True)

st.markdown(f'<p class="big-font">SmartStock: Inventory Optimization System</p>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitle">A data-driven platform for smarter retail decisions and maximized capital efficiency.</p>', unsafe_allow_html=True)

st.markdown("---")


st.header("💡 Dashboard Capabilities")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📈 Demand Forecast", "Product-Level Trend", "Prophet & ML")
    st.markdown("Displays time-series demand forecasts and future sales estimates for selected products.")
with col2:
    st.metric("⚠ Operational Alerts", "ROP & Stock Check", "Live Status")
    st.markdown("Flags items critically low on stock by comparing current levels to the calculated **Reorder Point (ROP)**.")
with col3:
    st.metric("📝 Validation Log", "Auditing & History", "Decision Record")
    st.markdown("Provides a searchable log of historical inventory movements and reorder decisions for auditing purposes.")
st.markdown("---")


st.header(" Why SmartStock? ")
col_problem, col_solution = st.columns(2)
with col_problem:
    st.subheader("The Challenge")
    st.markdown("""
Traditional inventory management leads to significant profit erosion:
- **Stockouts** → Lost revenue and customer dissatisfaction.  
- **Overstocking** → Increased holding costs and product obsolescence.  
- **Inefficient Ordering** → Suboptimal capital allocation.  
""")
with col_solution:
    st.subheader("The Solution")
    st.markdown("""
SmartStock automates and optimizes inventory policy based on predictive modeling:
- **Accurate Forecasting** → Precise demand prediction.  
- **Safety Stock & ROP Logic** → Maintains high service level reliability.  
- **EOQ Optimization** → Minimizes total ordering and holding costs.  
""")

st.markdown("---")

st.header("🧭 Navigation Guide")
st.markdown("Use the **sidebar** to access the operational pages. The **'Main Navigation'** radio buttons allow you to switch instantly between **Forecasts**, **Stock Alerts**, and the **Validation Log**.")

st.markdown(
    """
    <div style='text-align:right; color:gray; font-size:12px; margin-top: 40px;'>
        &copy; 2025 SmartStock | Developed by Sanya Uttam
    </div>
    """,
    unsafe_allow_html=True
)