import streamlit as st
import os

# ==============================================
# PAGE CONFIGURATION
# ==============================================
st.set_page_config(
    page_title="SmartStock | Inventory Optimization System",
    page_icon="📦",
    layout="wide"
)

# ==============================================
# HEADER SECTION (LOGO + TITLE)
# ==============================================
st.markdown("""
<style>
    .big-font {
        font-size: 48px !important;
        font-weight: 700;
        color: #2e86de;
    }
    .subheader {
        font-size: 22px !important;
        color: #535c68;
    }
    .section-header {
        font-size: 28px !important;
        color: #1e3799;
        margin-top: 20px;
    }
    /* NEW STYLE: Large, Centered Cart Emoji as Logo */
    .emoji-logo {
        font-size: 80px; /* Make the emoji big */
        text-align: center;
        padding-top: 5px;
        line-height: 1;
    }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 6])

# MODIFIED LOGO SECTION: Using a large cart emoji
with col1:
    st.markdown('<div class="emoji-logo">🛒</div>', unsafe_allow_html=True)
with col2:
    # CLEANED TITLE: Removed the cart emoji from the heading text
    st.markdown('<p class="big-font">SmartStock: Inventory Optimization</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">A data-driven system for smarter retail decisions.</p>', unsafe_allow_html=True)

st.markdown("---")


# ==============================================
# DASHBOARD FEATURES OVERVIEW
# ==============================================
st.header("💡 Dashboard Features")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📈 Demand Forecast", "Prophet Model", "Visual Trends")
    st.markdown("Displays time-series demand forecasts using Prophet model outputs.")
with col2:
    st.metric("⚠ Stock Alerts", "ROP & EOQ Logic", "Live Warnings")
    st.markdown("Automatically flags items below Reorder Point (ROP) or safety stock levels.")
with col3:
    st.metric("📂 Data Management", "Smart Upload", "Instant Processing")
    st.markdown("Upload new CSV data or review default processed files for quick updates.")
with col4:
    st.metric("📥 Reports", "CSV Export", "Ready for Action")
    st.markdown("Easily download reorder recommendations in CSV format for purchasing teams.")

st.markdown("---")

# ==============================================
# WHY SMARTSTOCK SECTION
# ==============================================
st.header("🤖 Why SmartStock? Solving the Retail Inventory Problem")
col_problem, col_solution = st.columns(2)
with col_problem:
    st.subheader("The Challenge 📉")
    st.markdown("""
Retailers often rely on manual or outdated methods for inventory management, leading to:
- **Stockouts** → Lost revenue & unsatisfied customers  
- **Overstocking** → Higher holding costs & waste  
- **Unoptimized Orders** → Capital misallocation  
""")
with col_solution:
    st.subheader("The SmartStock Solution ✨")
    st.markdown("""
SmartStock automates and enhances inventory decisions with data-driven intelligence:
- **Demand Forecasting** → Predicts product demand accurately  
- **Safety Stock & Reorder Logic** → Maintains 95% service level reliability  
- **EOQ Optimization** → Minimizes holding and ordering costs  
""")

st.markdown("---")

# ==============================================
# MILESTONE SUMMARY (SHORT VERSION)
# ==============================================
st.header("📊 Milestone Summary")
st.markdown("""
**Milestone 1:** Data Cleaning & Preprocessing  
**Milestone 2:** Demand Forecasting (Prophet/LSTM)  
**Milestone 3:** Inventory Optimization (ROP/EOQ Calculation)  
**Milestone 4:** Dashboard Integration (Streamlit Interface)  
""")

st.markdown("---")

# ==============================================
# NAVIGATION
# ==============================================
st.header("🧭 Navigation")
st.success("To access the operational tools, open the sidebar and select **‘M4 Dashboard’**.")
st.markdown("You can upload data, view forecast charts, check stock alerts, and download reorder reports directly from there.")

# ==============================================
# FOOTER
# ==============================================
st.markdown(
    """
    <div style='text-align:center; color:gray; font-size:12px; margin-top: 40px;'>
        &copy; 2025 <b>SmartStock</b> | Integrating Machine Learning & Inventory Intelligence for Retail Stores.<br>
        Developed by <b>Sanya Uttam</b>
    </div>
    """,
    unsafe_allow_html=True
)
