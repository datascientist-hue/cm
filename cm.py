import streamlit as st
import pandas as pd
from streamlit_echarts import st_echarts
import numpy as np
import calendar

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Contribution Margin Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------
# Custom CSS
# ---------------------------------
st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    .stApp[theme="dark"] .main { background-color: #0E1117; }
    .kpi-card {
        background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 10px;
        padding: 20px 25px; display: flex; flex-direction: column; justify-content: center;
        height: 140px; box-shadow: 0 4px 8px rgba(0,0,0,0.05); transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-5px); }
    .stApp[theme="dark"] .kpi-card { background-color: #1a1c22; border: 1px solid #31333F; }
    .kpi-label { font-size: 14px; color: #555555; margin-bottom: 8px; }
    .stApp[theme="dark"] .kpi-label { color: #a0a0a0; }
    .kpi-value { font-size: 28px; font-weight: 700; color: #1f77b4; }
    .stApp[theme="dark"] .kpi-value { color: #58b0f7; }
    .dsm-container {
        background-color: #ffffff; border-radius: 10px; padding: 25px; margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08); border: 1px solid #e6e6e6;
    }
    .stApp[theme="dark"] .dsm-container { background-color: #1a1c22; border: 1px solid #31333F; }
    .dsm-title {
        font-size: 24px; font-weight: 600; color: #1f77b4;
        margin-bottom: 15px; padding-bottom: 10px;
    }
    .stApp[theme="dark"] .dsm-title { color: #58b0f7; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Helper & Data Functions
# ---------------------------------
def format_indian_currency(n):
    if not isinstance(n, (int, float)): return n
    sign = '-' if n < 0 else ''
    n = abs(n)
    if n >= 1e7: return f"{sign}₹ {n/1e7:,.2f} Cr"
    if n >= 1e5: return f"{sign}₹ {n/1e5:,.2f} L"
    return f"{sign}₹ {n:,.0f}"

@st.cache_data
def load_data(main_path, cat_path):
    try:
        # 1. Load Main Data
        df = pd.read_csv(main_path, encoding='latin1')
        df['Inv Date'] = pd.to_datetime(df['Inv Date'], errors='coerce')
        
        # Numeric Clean up
        qty_candidates = ['Qty in Ltrs/Kgs', 'Qty', 'Billed Qty', 'Quantity', 'Eq Qty']
        qty_col = next((col for col in qty_candidates if col in df.columns), None)
        
        if qty_col:
            df['Qty'] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
        else:
            df['Qty'] = 0 

        numeric_cols = ['Net Value', 'Net Rate', 'COGS']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0
        df.dropna(subset=['Inv Date'], inplace=True)
        
        # Create Month-Year column
        df['Month_Year'] = df['Inv Date'].dt.strftime('%b-%y')
        df['Month_Sort'] = df['Inv Date'].dt.to_period('M')

        # 2. Load Mapping Data (mastermap.csv)
        try:
            cat_df = pd.read_csv(cat_path, encoding='latin1')
            if 'Prod Ctg' in df.columns and 'Prod Ctg' in cat_df.columns and 'Prod_cat_master' in cat_df.columns:
                df = df.merge(cat_df[['Prod Ctg', 'Prod_cat_master']], on='Prod Ctg', how='left')
                df['Prod_cat_master'] = df['Prod_cat_master'].fillna('Unmapped')
            else:
                df['Prod_cat_master'] = df.get('Prod Ctg', 'Unmapped')
        except FileNotFoundError:
            df['Prod_cat_master'] = df.get('Prod Ctg', 'Unmapped')

        # 3. Logic Implementation
        df['DSM'] = df['DSM'].fillna('').astype(str)
        
        # Ensure Item Name exists for SKU analysis
        if 'Item Name' in df.columns:
            df['Item Name'] = df['Item Name'].fillna('Unknown SKU').astype(str)
        else:
            df['Item Name'] = 'Unknown SKU'
        
        if 'State' not in df.columns:
             if 'Region' in df.columns: df['State'] = df['Region']
             else: df['State'] = 'Unknown'

        conditions = [df['DSM'].str.strip() == 'DSM-AP']
        choices = [0.10] 
        df['VC_Percent'] = np.select(conditions, choices, default=0.05)
        
        df['Variable Cost'] = df['Net Rate'] * df['VC_Percent']
        df['COD'] = df['COGS'] + df['Variable Cost']
        df['CM'] = df['Net Value'] - df['COD']
        
        return df
    except FileNotFoundError:
        st.error(f"Error: File not found at {main_path}. Please check the path.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def get_static_fixed_expenses():
    data = {
        "GL Code & Description": [
            "E201010001 - Salary", "E201010002 - Wes", "E201010003 - Employer share of PF",
            "E201010004 - Employer share of ESI", "E205010001 - Telephone Charges",
            "E301020001 - Rent - CFA", "E301020002 - Commission - CFA",
            "E301020003 - Reimbursement Exp -CFA"
        ],
        "Amount": [699409.0, 127697.1, 45878.0, 0.0, 2000.0, 40000.0, 23000.0, 26000.0]
    }
    return pd.DataFrame(data)

# ---------------------------------
# UI Components
# ---------------------------------
def display_kpi_cards(df, fixed_expenses_val, is_percentage_mode):
    if df.empty:
        total_net_value, total_cm, total_cod = 0, 0, 0
    else:
        total_net_value = df['Net Value'].sum()
        total_cm = df['CM'].sum()
        total_cod = df['COD'].sum()
    
    net_profit = total_cm - fixed_expenses_val
    profit_color = "#d32f2f" if net_profit < 0 else "#2e7d32" 
    
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1: st.markdown(f'<div class="kpi-card"><div class="kpi-label">💰 Sales (Net Value)</div><div class="kpi-value">{format_indian_currency(total_net_value)}</div></div>', unsafe_allow_html=True)
    with kpi2: st.markdown(f'<div class="kpi-card"><div class="kpi-label">📦 Total COD</div><div class="kpi-value">{format_indian_currency(total_cod)}</div></div>', unsafe_allow_html=True)
    with kpi3: st.markdown(f'<div class="kpi-card"><div class="kpi-label">📉 Contribution Margin</div><div class="kpi-value">{format_indian_currency(total_cm)}</div></div>', unsafe_allow_html=True)
    with kpi4: 
        lbl = "🏢 Fixed Cost (18%)" if is_percentage_mode else "🏢 Fixed Cost (Actual)"
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">{lbl}</div><div class="kpi-value">{format_indian_currency(fixed_expenses_val)}</div></div>', unsafe_allow_html=True)
    with kpi5: st.markdown(f'<div class="kpi-card"><div class="kpi-label">💵 Net Profit</div><div class="kpi-value" style="color:{profit_color}">{format_indian_currency(net_profit)}</div></div>', unsafe_allow_html=True)

def create_horizontal_bar_chart(df, y_col, x_col, title, theme):
    df_sorted = df.sort_values(by=x_col, ascending=True)
    options = {
        "title": {"text": title, "left": "center", "textStyle": {"color": "#666" if theme=='light' else '#ddd'}},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}}, 
        "grid": {"left": '3%', "right": '10%', "bottom": '3%', "containLabel": True},
        "xAxis": {"type": "value", "axisLabel": {"formatter": '{value}'}}, 
        "yAxis": {"type": "category", "data": df_sorted[y_col].tolist()},
        "series": [{
            "name": "CM %", 
            "type": "bar", 
            "data": [{"value": round(val, 2), "itemStyle": {"color": '#1f77b4'}} for val in df_sorted[x_col].tolist()],
            "label": {"show": True, "position": 'right', "formatter": '{c}%', "color": 'black' if theme == 'light' else 'white'}
        }]
    }
    return options

def create_sales_donut_chart(above_val, below_val, theme):
    options = {
        "tooltip": {
            "trigger": "item",
            "formatter": "{b}: {d}%" 
        },
        "legend": {"orient": "vertical", "left": "left", "textStyle": {"color": "#333" if theme == 'light' else '#ccc'}},
        "series": [{"name": "Sales Share", "type": "pie", "radius": ['50%', '70%'], "avoidLabelOverlap": False,
            "label": {"show": False, "position": "center"},
            "emphasis": {"label": {"show": True, "fontSize": "20", "fontWeight": "bold"}},
            "labelLine": {"show": False},
            "data": [
                {"value": above_val, "name": "Sales > Target"}, 
                {"value": below_val, "name": "Sales < Target"}
            ],
            "color": ['#3BA272', '#EE6666']
        }]
    }
    return options

def create_dual_axis_chart(df, x_col, bar_col, line_col, title, theme):
    if x_col != 'Month_Year':
        df = df.sort_values(by=bar_col, ascending=False).head(20)

    x_data = df[x_col].tolist()
    bar_data = df[bar_col].tolist()
    line_data = [round(x, 2) for x in df[line_col].tolist()]
    
    text_color = "#ccc" if theme == "dark" else "#333"

    options = {
        "title": {"text": title, "left": "center", "textStyle": {"color": text_color}},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
        "legend": {"data": [bar_col, line_col], "bottom": 0, "textStyle": {"color": text_color}},
        "grid": {"left": '3%', "right": '4%', "bottom": '10%', "containLabel": True},
        "xAxis": [{"type": "category", "data": x_data, "axisLabel": {"rotate": 45, "interval": 0}}],
        "yAxis": [
            {"type": "value", "name": bar_col, "position": "left", "axisLabel": {"formatter": '{value}'}},
            {"type": "value", "name": line_col, "position": "right", "axisLabel": {"formatter": '{value} %'}}
        ],
        "series": [
            {"name": bar_col, "type": "bar", "data": bar_data, "itemStyle": {"color": "#5470C6"}},
            {"name": line_col, "type": "line", "yAxisIndex": 1, "data": line_data, "itemStyle": {"color": "#91CC75"}, "smooth": True}
        ]
    }
    return options

def create_multi_bar_chart(df, x_col, col1, col2, title, theme):
    if x_col != 'Month_Year':
        df = df.sort_values(by=col1, ascending=False).head(20)

    x_data = df[x_col].tolist()
    text_color = "#ccc" if theme == "dark" else "#333"

    options = {
        "title": {"text": title, "left": "center", "textStyle": {"color": text_color}},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"data": [col1, col2], "bottom": 0, "textStyle": {"color": text_color}},
        "grid": {"left": '3%', "right": '4%', "bottom": '10%', "containLabel": True},
        "xAxis": [{"type": "category", "data": x_data, "axisLabel": {"rotate": 45}}],
        "yAxis": [{"type": "value"}],
        "series": [
            {"name": col1, "type": "bar", "data": df[col1].tolist(), "itemStyle": {"color": "#fac858"}},
            {"name": col2, "type": "bar", "data": df[col2].tolist(), "itemStyle": {"color": "#ee6666"}}
        ]
    }
    return options

# ---------------------------------
# Main App Logic
# ---------------------------------
FILE_PATH = r"E:\scm\cm.csv"
CAT_FILE_PATH = r"E:\scm\mastermap.csv"

df = load_data(FILE_PATH, CAT_FILE_PATH)

if df is not None:
    # --- Sidebar Filters ---
    st.sidebar.header("🌍 Global Filters")
    chart_theme = st.sidebar.selectbox("🎨 Select Chart Theme", ["light", "dark"])
    
    # View Mode
    view_mode = st.sidebar.radio(
        "Select View",
        ["Current JC", "YTD"],
        index=0, 
        horizontal=True
    )
    
    min_date, max_date = df['Inv Date'].min().date(), df['Inv Date'].max().date()
    
    date_range = st.sidebar.date_input("📅 Invoice Date Range", value=[min_date, max_date], min_value=min_date, max_value=max_date)
    
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
    else:
        st.error("Invalid date.")
        st.stop()
    
    # Apply Date Filter
    df_filtered = df[(df['Inv Date'] >= start_date) & (df['Inv Date'] <= end_date)]

    # JC Logic
    if view_mode == "Current JC":
        if 'JCPeriod' in df.columns and not df['JCPeriod'].dropna().empty:
            latest_jc_period = sorted(df['JCPeriod'].dropna().unique())[-1]
            df_filtered = df_filtered[df_filtered['JCPeriod'] == latest_jc_period]
            st.sidebar.success(f"📌 Showing Data for **{latest_jc_period}**")
        else:
            st.sidebar.warning("JCPeriod column missing. Showing selected date range.")

    def get_options(dataframe, col):
        if col in dataframe.columns:
            return ["All"] + sorted(dataframe[col].dropna().unique().tolist())
        return ["All"]

    # 1. State Filter
    state_opts = get_options(df_filtered, 'State')
    sel_state = st.sidebar.selectbox('🗺️ State', state_opts)
    if sel_state != 'All': df_filtered = df_filtered[df_filtered['State'] == sel_state]

    # 2. Month Filter
    sorted_months = df_filtered.sort_values('Inv Date')['Month_Year'].unique().tolist()
    sel_months = st.sidebar.multiselect('🗓️ Select Month(s)', sorted_months, default=sorted_months)
    if sel_months:
        df_filtered = df_filtered[df_filtered['Month_Year'].isin(sel_months)]

    st.sidebar.markdown("---")
    
    # Hierarchy Filters
    period_opts = get_options(df_filtered, 'JCPeriod')
    is_jc_filter_disabled = (view_mode == "Current JC") 
    sel_period = st.sidebar.selectbox('📆 JC Period', period_opts, disabled=is_jc_filter_disabled)
    if sel_period != 'All': df_filtered = df_filtered[df_filtered['JCPeriod'] == sel_period]

    week_opts = get_options(df_filtered, 'JCWeek')
    sel_week = st.sidebar.selectbox('🗓️ JC Week', week_opts)
    if sel_week != 'All': df_filtered = df_filtered[df_filtered['JCWeek'] == sel_week]

    dsm_opts = get_options(df_filtered, 'DSM')
    sel_dsm = st.sidebar.selectbox('👤 Sales Category (DSM)', dsm_opts)
    if sel_dsm != 'All': df_filtered = df_filtered[df_filtered['DSM'] == sel_dsm]

    if 'Prod_cat_master' in df_filtered.columns:
        prod_opts = get_options(df_filtered, 'Prod_cat_master')
        sel_prod = st.sidebar.selectbox('📦 Product Category', prod_opts)
        if sel_prod != 'All': df_filtered = df_filtered[df_filtered['Prod_cat_master'] == sel_prod]
    
    if 'Item Name' in df_filtered.columns:
        sku_opts = sorted(df_filtered['Item Name'].unique().tolist())
        sel_sku = st.sidebar.multiselect('🔖 Select SKU (Item Name)', sku_opts)
        if sel_sku:
            df_filtered = df_filtered[df_filtered['Item Name'].isin(sel_sku)]

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Cost Settings")
    use_18_percent = st.sidebar.checkbox("✅ Use 18% of Sales as Fixed Cost", value=False)
    
    fixed_expenses_df = get_static_fixed_expenses()
    static_total_fixed = fixed_expenses_df['Amount'].sum()
    current_sales = df_filtered['Net Value'].sum() if not df_filtered.empty else 0

    if use_18_percent:
        final_fixed_cost = current_sales * 0.18
    else:
        final_fixed_cost = static_total_fixed

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "**🚀 Sales CM Dashboard**", 
        "**🏢 Sales by JC**", 
        "**🎯 Target Analysis**",
        "**💰 Profitability**",
        "**📈 Month-wise Trend**",
        "**🔍 Detailed Analysis**" 
    ])

    def render_kpis():
        display_kpi_cards(df_filtered, final_fixed_cost, use_18_percent)

    # --- TAB 1: Main Dashboard ---
    with tab1:
        render_kpis()
        st.markdown("<hr>", unsafe_allow_html=True)
        
        if df_filtered.empty:
            st.warning("No data available for the selected filters.")
        else:
            dsms_to_show = sorted(df_filtered['DSM'].dropna().unique().tolist())
            for dsm in dsms_to_show:
                st.markdown(f'<div class="dsm-container"><p class="dsm-title">📍 {dsm}</p>', unsafe_allow_html=True)
                dsm_data = df_filtered[df_filtered['DSM'] == dsm]
                
                master_agg = dsm_data.groupby('Prod_cat_master').agg(
                    Net_Value=('Net Value', 'sum'), 
                    CM=('CM', 'sum')
                ).reset_index()
                master_agg['CM %'] = np.where(master_agg['Net_Value'] != 0, (master_agg['CM'] / master_agg['Net_Value']) * 100, 0)

                detail_agg = dsm_data.groupby(['Prod Ctg', 'Prod_cat_master']).agg(
                    Net_Value=('Net Value', 'sum'),
                    CM=('CM', 'sum')
                ).reset_index()
                detail_agg['CM %'] = np.where(detail_agg['Net_Value'] != 0, (detail_agg['CM'] / detail_agg['Net_Value']) * 100, 0)
                
                if not master_agg.empty:
                    col1, col2 = st.columns([6, 5])
                    with col1:
                        chart_options = create_horizontal_bar_chart(master_agg, 'Prod_cat_master', 'CM %', 'Contribution Margin % by Category', chart_theme)
                        st_echarts(options=chart_options, theme=chart_theme, height="350px")
                    
                    with col2:
                        st.subheader("📝 Detailed Metrics")
                        tmd_tpu_opts = ["All"] + sorted(detail_agg['Prod_cat_master'].unique().tolist())
                        sel_type = st.radio(f"Filter Table for {dsm}:", tmd_tpu_opts, horizontal=True, key=f"rad_{dsm}")
                        
                        table_view = detail_agg.copy()
                        if sel_type != "All":
                            table_view = table_view[table_view['Prod_cat_master'] == sel_type]
                        
                        display_df = table_view[['Prod Ctg', 'Net_Value', 'CM', 'CM %']].sort_values('CM %', ascending=False)
                        display_df['Net_Value'] = display_df['Net_Value'].apply(format_indian_currency)
                        display_df['CM'] = display_df['CM'].apply(format_indian_currency)
                        display_df['CM %'] = display_df['CM %'].apply(lambda x: f"{x:.2f}%")
                        
                        st.dataframe(display_df, hide_index=True, use_container_width=True)
                        
                st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 2: JC View (MODIFIED AS REQUESTED) ---
    with tab2:
        render_kpis()
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # NOTE: Using original 'df' with date filters here to show ALL periods back-to-back 
        # even if "Current JC" is selected in sidebar (which normally limits to 1 period).
        df_jc_view = df[(df['Inv Date'] >= start_date) & (df['Inv Date'] <= end_date)]
        
        # Re-apply sidebar filters to this view to keep consistency
        if sel_state != 'All': df_jc_view = df_jc_view[df_jc_view['State'] == sel_state]
        if sel_months and len(sel_months) < len(sorted_months):
             df_jc_view = df_jc_view[df_jc_view['Month_Year'].isin(sel_months)]
        # We purposely skip 'sel_period' filter here if Current JC is selected to show the history
        if sel_week != 'All': df_jc_view = df_jc_view[df_jc_view['JCWeek'] == sel_week]
        if sel_dsm != 'All': df_jc_view = df_jc_view[df_jc_view['DSM'] == sel_dsm]
        if 'Prod_cat_master' in df_jc_view.columns and sel_prod != 'All':
             df_jc_view = df_jc_view[df_jc_view['Prod_cat_master'] == sel_prod]
        if 'Item Name' in df_jc_view.columns and sel_sku:
             df_jc_view = df_jc_view[df_jc_view['Item Name'].isin(sel_sku)]

        if df_jc_view.empty:
            st.warning("No data available for the selected filters.")
        else:
            # Get Unique Periods and Sort Descending (JC 11, JC 10, ...)
            if 'JCPeriod' in df_jc_view.columns:
                unique_periods = sorted(df_jc_view['JCPeriod'].dropna().unique().tolist(), reverse=True)
                
                for period in unique_periods:
                    st.markdown(f"### 🗓️ JC Period: {period}")
                    
                    # Filter data for this period
                    period_data = df_jc_view[df_jc_view['JCPeriod'] == period]
                    
                    # Group by Week
                    jc_week_wise = period_data.groupby('JCWeek').agg(
                        Net_Value=('Net Value', 'sum'), 
                        COD=('COD', 'sum'), 
                        CM=('CM', 'sum')
                    ).reset_index()
                    
                    jc_week_wise['CM %'] = np.where(jc_week_wise['Net_Value'] != 0, (jc_week_wise['CM'] / jc_week_wise['Net_Value']) * 100, 0)
                    
                    # Display Table
                    st.dataframe(jc_week_wise.style.format({'Net_Value': '{:,.2f}', 'COD': '{:,.2f}', 'CM': '{:,.2f}', 'CM %': '{:.2f}%'}), use_container_width=True)
                    st.markdown("---")
            else:
                 st.error("JCPeriod column not found.")
    
    # --- TAB 3: Target Analysis ---
    with tab3:
        st.header("🎯 SKU Contribution Margin vs. Target (Sales Value Share)")
        if df_filtered.empty:
            st.warning("No data available for the selected filters.")
        else:
            sku_agg_dsm = df_filtered.groupby(['DSM', 'Item Name']).agg(
                Net_Value=('Net Value', 'sum'), 
                CM=('CM', 'sum')
            ).reset_index()
            
            sku_agg_dsm = sku_agg_dsm[sku_agg_dsm['Net_Value'] > 0]
            sku_agg_dsm['CM %'] = (sku_agg_dsm['CM'] / sku_agg_dsm['Net_Value']) * 100
            
            high_target_dsms = ['DSM-TN', 'DSM-STN', 'DSM-CTN']
            df_high = sku_agg_dsm[sku_agg_dsm['DSM'].isin(high_target_dsms)]
            df_std = sku_agg_dsm[~sku_agg_dsm['DSM'].isin(high_target_dsms)]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔥 High Priority DSMs (Target > 15%)")
                if df_high.empty: 
                    st.write("No Data.")
                else:
                    above = df_high[df_high['CM %'] >= 15]
                    below = df_high[df_high['CM %'] < 15]
                    
                    val_above = above['Net_Value'].sum()
                    val_below = below['Net_Value'].sum()
                    count_above = len(above)
                    count_below = len(below)
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Sales Above Target", format_indian_currency(val_above), f"{count_above} SKUs")
                    c2.metric("Sales Below Target", format_indian_currency(val_below), f"{count_below} SKUs", delta_color="inverse")
                    
                    st_echarts(create_sales_donut_chart(val_above, val_below, chart_theme), height="250px")
                    
                    with st.expander("View SKUs Below Target"):
                        show_df = below[['DSM', 'Item Name', 'Net_Value', 'CM %']].sort_values('CM %')
                        show_df['Sales'] = show_df['Net_Value'].apply(format_indian_currency)
                        st.dataframe(show_df[['DSM', 'Item Name', 'Sales', 'CM %']].style.format({'CM %': '{:.2f}%'}), use_container_width=True)

            with col2:
                st.subheader("🛡️ Standard DSMs (Target > 10%)")
                if df_std.empty: 
                    st.write("No Data.")
                else:
                    above = df_std[df_std['CM %'] >= 10]
                    below = df_std[df_std['CM %'] < 10]
                    
                    val_above = above['Net_Value'].sum()
                    val_below = below['Net_Value'].sum()
                    count_above = len(above)
                    count_below = len(below)
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Sales Above Target", format_indian_currency(val_above), f"{count_above} SKUs")
                    c2.metric("Sales Below Target", format_indian_currency(val_below), f"{count_below} SKUs", delta_color="inverse")
                    
                    st_echarts(create_sales_donut_chart(val_above, val_below, chart_theme), height="250px")
                    
                    with st.expander("View SKUs Below Target"):
                        show_df = below[['DSM', 'Item Name', 'Net_Value', 'CM %']].sort_values('CM %')
                        show_df['Sales'] = show_df['Net_Value'].apply(format_indian_currency)
                        st.dataframe(show_df[['DSM', 'Item Name', 'Sales', 'CM %']].style.format({'CM %': '{:.2f}%'}), use_container_width=True)

    # --- TAB 4: Profitability ---
    with tab4:
        st.header("💰 Profit & Loss Analysis")
        render_kpis()
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🧮 Logic")
            st.markdown(r"""
            1. **Map:** `Prod Ctg` $\rightarrow$ `Prod_cat_master`
            2. **VC:** 10% (AP), 5% (Others)
            3. **Profit:** CM - Fixed Cost
            """)
        with c2:
            st.subheader("📋 Fixed Cost")
            if use_18_percent:
                st.warning(f"Using 18%: {format_indian_currency(final_fixed_cost)}")
            else:
                st.success(f"Using Actuals: {format_indian_currency(final_fixed_cost)}")
                st.dataframe(fixed_expenses_df, hide_index=True, use_container_width=True)

    # --- TAB 5: Trend Analysis ---
    with tab5:
        st.header("📈 Month-wise Product Trend (CM %)")
        
        if df_filtered.empty:
            st.warning("No data available for the selected filters.")
        else:
            base_agg = df_filtered.groupby('Prod Ctg').agg(
                Sum_Qty=('Qty', 'sum'),
                Contribution=('CM', 'sum')
            )
            
            pivot_nv = df_filtered.pivot_table(
                index='Prod Ctg', columns='Month_Sort', values='Net Value', 
                aggfunc='sum', fill_value=0
            )

            pivot_cm = df_filtered.pivot_table(
                index='Prod Ctg', columns='Month_Sort', values='CM', 
                aggfunc='sum', fill_value=0
            )

            with np.errstate(divide='ignore', invalid='ignore'):
                pivot_pct = np.divide(pivot_cm, pivot_nv) * 100
            
            pivot_pct_df = pd.DataFrame(pivot_pct, index=pivot_cm.index, columns=pivot_cm.columns)
            pivot_pct_df.fillna(0, inplace=True)
            
            pivot_pct_df.columns = [c.strftime('%b-%y') for c in pivot_pct_df.columns]
            
            final_trend_df = base_agg.join(pivot_pct_df)
            
            final_trend_df.index.name = "Description"
            final_trend_df.rename(columns={'Sum_Qty': 'Sum of Qty in Kgs/Ltrs'}, inplace=True)
            
            format_dict = {
                'Sum of Qty in Kgs/Ltrs': '{:,.0f}',
                'Contribution': '{:,.2f}'
            }
            for col in pivot_pct_df.columns:
                format_dict[col] = '{:.2f}%'

            st.dataframe(final_trend_df.style.format(format_dict), use_container_width=True)

    # --- TAB 6: DETAILED ANALYSIS ---
    with tab6:
        st.header("🔍 Detailed Analysis")
        
        if df_filtered.empty:
            st.warning("No data available for the selected filters.")
        else:
            # Dimension Selector
            c_sel, c_blank = st.columns([1, 3])
            with c_sel:
                dim_choice = st.radio("📊 Analyze Graphs By:", ["Month", "Product Category", "SKU"], horizontal=True)
            
            # Map choice to column name
            dim_map = {"Month": "Month_Year", "Product Category": "Prod Ctg", "SKU": "Item Name"}
            group_col = dim_map[dim_choice]
            
            # Aggregation for Graphs
            graph_agg = df_filtered.groupby(group_col).agg(
                Sales=('Net Value', 'sum'),
                COGS=('COD', 'sum'),
                CM=('CM', 'sum')
            ).reset_index()
            graph_agg['CM %'] = np.where(graph_agg['Sales'] != 0, (graph_agg['CM'] / graph_agg['Sales']) * 100, 0)
            
            # Graph 1: Contribution INR vs %
            st.subheader(f"Contribution vs % by {dim_choice}")
            st_echarts(options=create_dual_axis_chart(graph_agg, group_col, 'CM', 'CM %', f"Contribution & % by {dim_choice}", chart_theme), height="400px")
            
            st.markdown("---")
            
            # Graph 2: Sales vs COGS
            st.subheader(f"Sales vs COGS by {dim_choice}")
            st_echarts(options=create_multi_bar_chart(graph_agg, group_col, 'Sales', 'COGS', f"Sales vs Cost by {dim_choice}", chart_theme), height="400px")
            
            st.markdown("---")
            
            # Detailed Table
            st.subheader("📋 Detailed Data Table")
            table_agg = df_filtered.groupby('Prod Ctg').agg(
                Ltrs=('Qty', 'sum'),
                Sales_Value=('Net Value', 'sum'),
                COGS=('COD', 'sum'),
                Contribution=('CM', 'sum')
            ).reset_index()
            
            table_agg['Contribution %'] = np.where(table_agg['Sales_Value'] != 0, (table_agg['Contribution'] / table_agg['Sales_Value']) * 100, 0)
            
            # Renaming columns as requested
            final_table = table_agg.rename(columns={
                'Prod Ctg': 'Product Category',
                'Sales_Value': 'Sales Value',
                'Contribution': 'Contribution INR',
                'Contribution %': 'Contribution %'
            })
            
            # Formatting for display
            display_tbl = final_table[['Product Category', 'Ltrs', 'Sales Value', 'COGS', 'Contribution INR', 'Contribution %']].copy()
            
            st.dataframe(
                display_tbl.style.format({
                    'Ltrs': '{:,.0f}',
                    'Sales Value': '₹ {:,.0f}',
                    'COGS': '₹ {:,.0f}',
                    'Contribution INR': '₹ {:,.0f}',
                    'Contribution %': '{:.2f}%'
                }),
                use_container_width=True,
                hide_index=True
            )

else:
    st.info("Please fix the data loading issue to see the dashboard.")