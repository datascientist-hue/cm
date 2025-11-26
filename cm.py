import streamlit as st
import pandas as pd
from streamlit_echarts import st_echarts
import numpy as np
import io
from ftplib import FTP

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
# Custom CSS for the Professional UI
# ---------------------------------
st.markdown("""
<style>
    /* Main app styling */
    .main {
        background-color: #f0f2f6;
    }
    .stApp[theme="dark"] .main {
        background-color: #0E1117;
    }

    /* KPI Card Styling */
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

    /* DSM Container Styling */
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
    """Formats a number into Indian currency style (Lakhs, Crores)."""
    if not isinstance(n, (int, float)): return n
    sign = '-' if n < 0 else ''
    n = abs(n)
    if n >= 1e7: return f"{sign}₹ {n/1e7:,.2f} Cr"
    if n >= 1e5: return f"{sign}₹ {n/1e5:,.2f} L"
    return f"{sign}₹ {n:,.0f}"

@st.cache_data(ttl=300)
def load_data_from_ftp(_ftp_config): # Added underscore to ignore this arg for hashing
    """
    Connects to an FTP server using credentials from st.secrets,
    retrieves files into memory, and loads them into a pandas DataFrame.
    """
    try:
        # Establish FTP connection using details from secrets.toml
        ftp = FTP(_ftp_config["host"])
        ftp.login(user=_ftp_config["user"], passwd=_ftp_config["password"])

        def download_file(ftp_path):
            """Downloads a single file from FTP into an in-memory buffer."""
            flo = io.BytesIO()
            ftp.retrbinary(f'RETR {ftp_path}', flo.write)
            flo.seek(0)
            return pd.read_parquet(flo)

        # Download both the main sales data and the master mapping file
        df = download_file(_ftp_config["main_file_path"])
        master_df = download_file(_ftp_config["master_file_path"])

        # Close the FTP connection
        ftp.quit()

        # --- Data Processing Logic ---
        # MODIFIED LINE: Added the format parameter to correctly parse dd-mm-yyyy dates
        df['Inv Date'] = pd.to_datetime(df['Inv Date'], format='%d-%m-%Y', errors='coerce')

        df = pd.merge(df, master_df, left_on='Prod Ctg', right_on='Prod_cat', how='left')
        df.rename(columns={'Prod_cat_master': 'Prod Ctg Master'}, inplace=True)
        df['Prod Ctg Master'].fillna('Uncategorized', inplace=True)
        numeric_cols = ['Net Value', 'Net Rate', 'COGS']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df['Variable Cost'] = df['Net Value'] * 0.05
        df['COD'] = df['COGS'] + df['Variable Cost']
        df['CM'] = df['Net Value'] - df['COD']
        df.dropna(subset=['Inv Date'], inplace=True)
        return df

    except Exception as e:
        st.error("FTP Error: Failed to load data. Please check your credentials and file paths in secrets.toml.")
        st.error(f"Details: {e}")
        return None

# ---------------------------------
# Reusable UI Component Functions
# ---------------------------------
def display_kpi_cards(df):
    """Displays the four main KPI cards at the top of a page."""
    if df.empty:
        total_net_value, total_cm, total_cogs, cm_percentage = 0, 0, 0, 0
    else:
        total_net_value = df['Net Value'].sum()
        total_cm = df['CM'].sum()
        total_cogs = df['COGS'].sum()
        cm_percentage = (total_cm / total_net_value * 100) if total_net_value != 0 else 0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1: st.markdown(f'<div class="kpi-card"><div class="kpi-label">💰 Total Net Value</div><div class="kpi-value">{format_indian_currency(total_net_value)}</div></div>', unsafe_allow_html=True)
    with kpi2: st.markdown(f'<div class="kpi-card"><div class="kpi-label">📉 Total Contribution Margin</div><div class="kpi-value">{format_indian_currency(total_cm)}</div></div>', unsafe_allow_html=True)
    with kpi3: st.markdown(f'<div class="kpi-card"><div class="kpi-label">📊 Overall CM %</div><div class="kpi-value">{cm_percentage:.2f}%</div></div>', unsafe_allow_html=True)
    with kpi4: st.markdown(f'<div class="kpi-card"><div class="kpi-label">📦 Total Cost of Goods</div><div class="kpi-value">{format_indian_currency(total_cogs)}</div></div>', unsafe_allow_html=True)

def create_horizontal_bar_chart(df, y_col, x_col, title, theme):
    """Creates a configured horizontal bar chart for ECharts."""
    df_sorted = df.sort_values(by=x_col, ascending=True)
    colors = ['#5470C6', '#91CC75', '#EE6666', '#73C0DE', '#3BA272', '#FC8452']
    options = {
        "title": {"text": title, "left": "center", "textStyle": {"color": "#666" if theme=='light' else '#ddd'}},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": '3%', "right": '10%', "bottom": '3%', "containLabel": True},
        "xAxis": {"type": "value", "axisLabel": {"formatter": '{value}%'}},
        "yAxis": {"type": "category", "data": df_sorted[y_col].tolist()},
        "series": [{
            "name": "CM %",
            "type": "bar",
            "data": [{"value": round(val, 2), "itemStyle": {"color": '#EE6666' if val < 0 else colors[i % len(colors)]}} for i, val in enumerate(df_sorted[x_col].tolist())],
            "label": {"show": True, "position": 'right', "formatter": '{c}%', "color": 'black' if theme == 'light' else 'white'},
        }]
    }
    return options

def create_donut_chart(above_count, below_count, theme):
    """Creates a configured donut chart for ECharts."""
    options = {
        "tooltip": {"trigger": "item"},
        "legend": {"orient": "vertical", "left": "left", "textStyle": {"color": "#333" if theme == 'light' else '#ccc'}},
        "series": [{
            "name": "SKU Status", "type": "pie", "radius": ['50%', '70%'], "avoidLabelOverlap": False,
            "label": {"show": False, "position": "center"},
            "emphasis": {"label": {"show": True, "fontSize": "30", "fontWeight": "bold"}},
            "labelLine": {"show": False},
            "data": [
                {"value": above_count, "name": "Above Target"},
                {"value": below_count, "name": "Below Target"}
            ],
            "color": ['#3BA272', '#EE6666']
        }]
    }
    return options

# ---------------------------------
# Main App Logic
# ---------------------------------
df = load_data_from_ftp(st.secrets["ftp"])

if df is not None:
    # --- Sidebar Filters ---
    st.sidebar.header("🌍 Global Filters")
    chart_theme = st.sidebar.selectbox("🎨 Select Chart Theme", ["dark", "light"])

    min_date, max_date = df['Inv Date'].min().date(), df['Inv Date'].max().date()
    date_range = st.sidebar.date_input("📅 Invoice Date Range", value=[min_date, max_date], min_value=min_date, max_value=max_date)

    if date_range and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    else:
        st.error("Please select a valid date range.")
        st.stop()

    df_filtered = df[(df['Inv Date'] >= start_date) & (df['Inv Date'] <= end_date)]

    def get_options(dataframe, col):
        return ["All"] + sorted(dataframe[col].dropna().unique().tolist())

    # Apply filters sequentially
    period_opts = get_options(df_filtered, 'JCPeriod')
    sel_period = st.sidebar.selectbox('📆 JC Period', period_opts)
    if sel_period != 'All':
        df_filtered = df_filtered[df_filtered['JCPeriod'] == sel_period]

    week_opts = get_options(df_filtered, 'JCWeek')
    sel_week = st.sidebar.selectbox('🗓️ JC Week', week_opts)
    if sel_week != 'All':
        df_filtered = df_filtered[df_filtered['JCWeek'] == sel_week]

    dsm_opts = get_options(df_filtered, 'DSM')
    sel_dsm = st.sidebar.selectbox('👤 DSM', dsm_opts)
    if sel_dsm != 'All':
        df_filtered = df_filtered[df_filtered['DSM'] == sel_dsm]

    asm_opts = get_options(df_filtered, 'ASM')
    sel_asm = st.sidebar.selectbox('👥 ASM', asm_opts)
    if sel_asm != 'All':
        df_filtered = df_filtered[df_filtered['ASM'] == sel_asm]

    doc_type_opts = get_options(df_filtered, 'Document Type')
    sel_doc_type = st.sidebar.selectbox('📄 Document Type', doc_type_opts)
    if sel_doc_type != 'All':
        df_filtered = df_filtered[df_filtered['Document Type'] == sel_doc_type]

    prod_ctg_opts = get_options(df_filtered, 'Prod Ctg Master')
    sel_prod_ctg = st.sidebar.selectbox('📦 Product Category', prod_ctg_opts)
    if sel_prod_ctg != 'All':
        df_filtered = df_filtered[df_filtered['Prod Ctg Master'] == sel_prod_ctg]

    so_type_opts = get_options(df_filtered, 'SOType')
    sel_so_type = st.sidebar.selectbox('🏷️ SO Type', so_type_opts)
    if sel_so_type != 'All':
        df_filtered = df_filtered[df_filtered['SOType'] == sel_so_type]

    # --- Tabs & Dashboard Content ---
    tab1, tab2, tab3 = st.tabs(["**🚀 Sales CM Dashboard**", "**🏢 Sales by JC**", "**🎯 Target Analysis**"])

    with tab1:
        display_kpi_cards(df_filtered)
        st.markdown("<hr>", unsafe_allow_html=True)
        if df_filtered.empty:
            st.warning("No data available for the selected filters.")
        else:
            dsms_to_show = sorted(df_filtered['DSM'].dropna().unique().tolist())
            if not dsms_to_show:
                st.info("No DSMs found for the current filter selection.")
            for dsm in dsms_to_show:
                st.markdown(f'<div class="dsm-container"><p class="dsm-title">📍 {dsm}</p>', unsafe_allow_html=True)
                dsm_data = df_filtered[df_filtered['DSM'] == dsm]
                category_agg = dsm_data.groupby('Prod Ctg Master').agg(Net_Value=('Net Value', 'sum'), COD=('COD', 'sum'), CM=('CM', 'sum')).reset_index()
                category_agg['CM %'] = np.where(category_agg['Net_Value'] != 0, (category_agg['CM'] / category_agg['Net_Value']) * 100, 0)
                if not category_agg.empty:
                    col1, col2 = st.columns([6, 5])
                    with col1:
                        chart_options = create_horizontal_bar_chart(category_agg, 'Prod Ctg Master', 'CM %', '📊 Contribution Margin % by Product Category', chart_theme)
                        if chart_options: st_echarts(options=chart_options, theme=chart_theme, height="350px")
                    with col2:
                        st.subheader("📝 Detailed Metrics")
                        display_df = category_agg[['Prod Ctg Master', 'Net_Value', 'COD', 'CM', 'CM %']].copy().sort_values('CM %', ascending=False)
                        display_df['Net_Value'] = display_df['Net_Value'].apply(format_indian_currency)
                        display_df['COD'] = display_df['COD'].apply(format_indian_currency)
                        display_df['CM'] = display_df['CM'].apply(format_indian_currency)
                        display_df['CM %'] = display_df['CM %'].apply(lambda x: f"{x:.2f}%")
                        st.dataframe(display_df, hide_index=True, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        display_kpi_cards(df_filtered)
        st.markdown("<hr>", unsafe_allow_html=True)
        if df_filtered.empty:
            st.warning("No data available for the selected filters in the sidebar.")
        else:
            st.header("📈 Performance by JC Week")
            jc_week_wise = df_filtered.groupby('JCWeek').agg(Net_Value=('Net Value', 'sum'), COD=('COD', 'sum'), CM=('CM', 'sum')).reset_index()
            jc_week_wise['CM %'] = np.where(jc_week_wise['Net_Value'] != 0, (jc_week_wise['CM'] / jc_week_wise['Net_Value']) * 100, 0)
            jc_week_wise = jc_week_wise.sort_values('JCWeek', ascending=True)
            display_jc_df = jc_week_wise.copy()
            display_jc_df['Net_Value'] = display_jc_df['Net_Value'].apply(format_indian_currency)
            display_jc_df['COD'] = display_jc_df['COD'].apply(format_indian_currency)
            display_jc_df['CM'] = display_jc_df['CM'].apply(format_indian_currency)
            display_jc_df['CM %'] = display_jc_df['CM %'].apply(lambda x: f"{x:.2f}%")
            st.dataframe(display_jc_df, use_container_width=True, hide_index=True)

    with tab3:
        st.header("🎯 SKU Contribution Margin vs. Target (by DSM)")
        if df_filtered.empty:
            st.warning("No data available for the selected filters.")
        else:
            sku_agg_dsm = df_filtered.groupby(['DSM', 'Item Name']).agg(Net_Value=('Net Value', 'sum'), CM=('CM', 'sum')).reset_index()
            sku_agg_dsm = sku_agg_dsm[sku_agg_dsm['Net_Value'] > 0]
            sku_agg_dsm['CM %'] = (sku_agg_dsm['CM'] / sku_agg_dsm['Net_Value']) * 100

            high_target_dsms = ['DSM-TN', 'DSM-STN', 'DSM-CTN']
            df_high_target_group = sku_agg_dsm[sku_agg_dsm['DSM'].isin(high_target_dsms)]
            df_standard_target_group = sku_agg_dsm[~sku_agg_dsm['DSM'].isin(high_target_dsms)]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🔥 High Priority DSMs")
                st.info("🎯 Target CM%: 15%")
                if df_high_target_group.empty:
                    st.write("No SKU data for high priority DSMs in this selection.")
                else:
                    target_high = 15
                    above_target = df_high_target_group[df_high_target_group['CM %'] >= target_high]
                    below_target = df_high_target_group[df_high_target_group['CM %'] < target_high]
                    m1, m2 = st.columns(2)
                    m1.metric("✅ SKUs Above Target", f"{len(above_target)}")
                    m2.metric("⚠️ SKUs Below Target", f"{len(below_target)}")
                    
                    # --- ADDED CHECK ---
                    if len(above_target) > 0 or len(below_target) > 0:
                        donut_chart = create_donut_chart(len(above_target), len(below_target), chart_theme)
                        st_echarts(options=donut_chart, theme=chart_theme, height="250px")
                    else:
                        st.info("No SKUs to display in chart for this group.")

                    with st.expander("👀 View SKUs Above 15%"):
                        st.dataframe(above_target[['DSM', 'Item Name', 'CM %']].style.format({'CM %': '{:.2f}%'}), use_container_width=True)
                    with st.expander("👀 View SKUs Below 15%"):
                        st.dataframe(below_target[['DSM', 'Item Name', 'CM %']].style.format({'CM %': '{:.2f}%'}), use_container_width=True)
            with col2:
                st.subheader("🛡️ Standard DSMs")
                st.info("🎯 Target CM%: 10%")
                if df_standard_target_group.empty:
                    st.write("No SKU data for standard DSMs in this selection.")
                else:
                    target_standard = 10
                    above_target = df_standard_target_group[df_standard_target_group['CM %'] >= target_standard]
                    below_target = df_standard_target_group[df_standard_target_group['CM %'] < target_standard]
                    m1, m2 = st.columns(2)
                    m1.metric("✅ SKUs Above Target", f"{len(above_target)}")
                    m2.metric("⚠️ SKUs Below Target", f"{len(below_target)}")

                    # --- ADDED CHECK ---
                    if len(above_target) > 0 or len(below_target) > 0:
                        donut_chart = create_donut_chart(len(above_target), len(below_target), chart_theme)
                        st_echarts(options=donut_chart, theme=chart_theme, height="250px")
                    else:
                        st.info("No SKUs to display in chart for this group.")

                    with st.expander("👀 View SKUs Above 10%"):
                        st.dataframe(above_target[['DSM', 'Item Name', 'CM %']].style.format({'CM %': '{:.2f}%'}), use_container_width=True)
                    with st.expander("👀 View SKUs Below 10%"):
                        st.dataframe(below_target[['DSM', 'Item Name', 'CM %']].style.format({'CM %': '{:.2f}%'}), use_container_width=True)
else:

    st.info("Please fix the data loading issue to see the dashboard.")
