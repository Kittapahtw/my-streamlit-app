# safety_stock_app_v13_improved_analytics.py
import io
import math
import logging
from typing import Tuple

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Config ---
st.set_page_config(layout="wide", page_title="Safety Stock Calculator V13 - Improved Analytics", page_icon="📊")
logging.basicConfig(level=logging.INFO)

# --- Constants ---
SEASONALITY_MAP = {
    'เลิกขาย/EOL (x0.0)': 0.0,
    'ช่วง Low Season (x0.5)': 0.5,
    'ต่ำกว่าปกติ (x0.75)': 0.75,
    'ปกติ (x1.0)': 1.0,
    'สูงกว่าปกติ (x1.25)': 1.25,
    'ช่วง High Season (x1.5)': 1.5,
    'ช่วงพีคสูงสุด (x2.0)': 2.0
}

DEFAULT_SL_MAP = {'A': 99.0, 'B': 95.0, 'C': 90.0}

# --- Helpers ---
@st.cache_data
def sample_sales_csv() -> bytes:
    # เพิ่มข้อมูลตัวอย่างให้หลากหลายขึ้น
    dates = pd.date_range(end=pd.Timestamp.today(), periods=60)  # เพิ่มข้อมูลเป็น 60 วัน
    np.random.seed(42)  # สำหรับความสม่ำเสมอ
    
    data = []
    for i, date in enumerate(dates):
        # SKU001 - Stable product with slight trend
        base_demand_1 = 8 + (i * 0.02)  # เพิ่ม trend เล็กน้อย
        sales_1 = max(0, int(np.random.normal(base_demand_1, 2)))
        data.append([date.strftime("%Y-%m-%d"), "SKU001", sales_1, "Shopee"])
        
        # SKU002 - Variable product with seasonality
        seasonal_factor = 1 + 0.3 * np.sin(i * 0.1)  # seasonal pattern
        base_demand_2 = 5 * seasonal_factor
        sales_2 = max(0, int(np.random.normal(base_demand_2, 2)))
        data.append([date.strftime("%Y-%m-%d"), "SKU002", sales_2, "Lazada"])
        
        # SKU003 - High variation product
        sales_3 = max(0, int(np.random.normal(12, 6)))
        data.append([date.strftime("%Y-%m-%d"), "SKU003", sales_3, "Shopee"])
        
        # SKU004 - Low demand product
        sales_4 = max(0, int(np.random.normal(2, 1)))
        data.append([date.strftime("%Y-%m-%d"), "SKU004", sales_4, "TikTok"])
    
    df = pd.DataFrame(data, columns=["Date", "code", "PCS", "Sales_Channel"])
    return df.to_csv(index=False).encode('utf-8-sig')

@st.cache_data
def sample_master_csv() -> bytes:
    df = pd.DataFrame({
        "code": ["SKU001", "SKU002", "SKU003", "SKU004"],
        "ProductName": ["Stable Product A", "Seasonal Product B", "High Variation Product C", "Low Demand Product D"],
        "Leadtime day": [7, 14, 21, 10],
        "pcs_per_carton": [12, 24, 6, 48],
        "cost_per_pcs": [15.5, 45.0, 125.0, 8.75]
    })
    return df.to_csv(index=False).encode('utf-8-sig')

@st.cache_data
def perform_abc_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with columns: code, total_volume, abc_class, cumulative_percent"""
    try:
        if df is None or df.empty:
            return pd.DataFrame(columns=['code', 'total_volume', 'abc_class', 'cumulative_percent'])
        if 'PCS' not in df.columns or 'code' not in df.columns:
            return pd.DataFrame(columns=['code', 'total_volume', 'abc_class', 'cumulative_percent'])
        
        temp = df.groupby('code', as_index=False)['PCS'].sum().rename(columns={'PCS': 'total_volume'})
        temp = temp.sort_values('total_volume', ascending=False).reset_index(drop=True)
        total = temp['total_volume'].sum()
        
        if total == 0:
            temp['cumulative_percent'] = 0.0
        else:
            temp['cumulative_volume'] = temp['total_volume'].cumsum()
            temp['cumulative_percent'] = (temp['cumulative_volume'] / total) * 100
        
        def cls(r):
            if r <= 80: return 'A'
            elif r <= 95: return 'B'
            else: return 'C'
        
        temp['abc_class'] = temp['cumulative_percent'].apply(cls)
        return temp[['code', 'total_volume', 'abc_class', 'cumulative_percent']]
    except Exception as e:
        st.error(f"Error in ABC analysis: {e}")
        return pd.DataFrame(columns=['code', 'total_volume', 'abc_class', 'cumulative_percent'])

def safe_read_csv(uploaded_file) -> Tuple[pd.DataFrame, str]:
    try:
        df = pd.read_csv(uploaded_file)
        return df, ""
    except Exception as e:
        return pd.DataFrame(), f"อ่านไฟล์ล้มเหลว: {e}"

def compute_z_from_sl(sl_percent: float) -> float:
    """Clip service level to (eps, 0.99999) because norm.ppf(0 or 1) -> inf"""
    try:
        eps = 1e-6
        p = max(eps, min(0.99999, float(sl_percent) / 100.0))
        return float(norm.ppf(p))
    except Exception:
        return 1.645  # Default Z-score for 95%

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8-sig')

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    try:
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='safety_stock')
            workbook = writer.book
            worksheet = writer.sheets['safety_stock']
            # format sample: money format for safety_stock_value if exists
            money_fmt = workbook.add_format({'num_format': '#,##0.00'})
            try:
                col_idx = df.columns.get_loc('safety_stock_value')
                worksheet.set_column(col_idx, col_idx, 15, money_fmt)
            except Exception:
                pass
        return out.getvalue()
    except Exception as e:
        st.error(f"Error creating Excel file: {e}")
        return b""

def create_improved_time_series_chart(sales_df, selected_skus=None):
    """สร้างกราฟ Time Series แบบง่ายๆ คล้ายกราฟหุ้น"""
    try:
        if sales_df.empty:
            return None
        
        # กรองข้อมูลตาม SKU ที่เลือก
        if selected_skus:
            filtered_data = sales_df[sales_df['code'].isin(selected_skus)].copy()
        else:
            filtered_data = sales_df.copy()
        
        if filtered_data.empty:
            return None
        
        # สร้างกราฟแยกตาม SKU
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for i, sku in enumerate(filtered_data['code'].unique()):
            sku_data = filtered_data[filtered_data['code'] == sku].sort_values('Date')
            
            fig.add_trace(go.Scatter(
                x=sku_data['Date'],
                y=sku_data['PCS'],
                mode='lines+markers',
                name=sku,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=4),
                hovertemplate=f'<b>{sku}</b><br>Date: %{{x}}<br>Demand: %{{y}} PCS<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text='📈 Demand Time Series Analysis',
                font=dict(size=20, color='darkblue'),
                x=0.5
            ),
            xaxis=dict(
                title='Date',
                title_font=dict(size=14),
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title='Demand (PCS)',
                title_font=dict(size=14),
                showgrid=True,
                gridcolor='lightgray'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # เพิ่ม range selector buttons
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="7D", step="day", stepmode="backward"),
                        dict(count=14, label="14D", step="day", stepmode="backward"),
                        dict(count=30, label="30D", step="day", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating time series chart: {e}")
        return None

def create_improved_distribution_chart(sales_df, selected_skus=None):
    """สร้างกราฟ Distribution แบบง่ายๆ"""
    try:
        if sales_df.empty:
            return None
        
        # กรองข้อมูลตาม SKU ที่เลือก
        if selected_skus:
            filtered_data = sales_df[sales_df['code'].isin(selected_skus)].copy()
        else:
            filtered_data = sales_df.copy()
        
        if filtered_data.empty:
            return None
        
        # สร้างกราฟ histogram
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for i, sku in enumerate(filtered_data['code'].unique()):
            sku_data = filtered_data[filtered_data['code'] == sku]['PCS']
            
            fig.add_trace(go.Histogram(
                x=sku_data,
                name=sku,
                opacity=0.7,
                nbinsx=20,
                marker_color=colors[i % len(colors)],
                hovertemplate=f'<b>{sku}</b><br>Demand Range: %{{x}}<br>Count: %{{y}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text='📊 Demand Distribution',
                font=dict(size=20, color='darkblue'),
                x=0.5
            ),
            xaxis=dict(
                title='Demand (PCS)',
                title_font=dict(size=14),
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title='Frequency',
                title_font=dict(size=14),
                showgrid=True,
                gridcolor='lightgray'
            ),
            barmode='overlay',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating distribution chart: {e}")
        return None

def create_statistics_summary_table(sales_df, selected_skus=None):
    """สร้างตารางสถิติแบบครบถ้วน"""
    try:
        if sales_df.empty:
            return None
        
        # กรองข้อมูลตาม SKU ที่เลือก
        if selected_skus:
            filtered_data = sales_df[sales_df['code'].isin(selected_skus)].copy()
        else:
            filtered_data = sales_df.copy()
        
        if filtered_data.empty:
            return None
        
        # คำนวณสถิติ
        stats = filtered_data.groupby('code')['PCS'].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(2)
        
        stats.columns = ['Total Days', 'Avg Demand', 'Std Dev', 'Min', 'Max', 'Median']
        
        # คำนวณ CV และ Risk Level
        stats['CV'] = (stats['Std Dev'] / stats['Avg Demand']).replace([np.inf, -np.inf], np.nan).fillna(0).round(3)
        
        def get_risk_level(cv):
            if cv < 0.5:
                return 'Low Risk'
            elif cv < 1.0:
                return 'Medium Risk'
            elif cv < 1.5:
                return 'High Risk'
            else:
                return 'Very High Risk'
        
        stats['Risk Level'] = stats['CV'].apply(get_risk_level)
        
        # เพิ่ม Total Volume
        total_vol = filtered_data.groupby('code')['PCS'].sum()
        stats['Total Volume'] = total_vol
        
        # จัดเรียงคอลัมน์
        stats = stats[['Total Days', 'Total Volume', 'Avg Demand', 'Std Dev', 'CV', 
                      'Min', 'Max', 'Median', 'Risk Level']]
        
        return stats.reset_index()
    except Exception as e:
        st.error(f"Error creating statistics summary: {e}")
        return None

def create_simple_what_if_analysis():
    """What-If Analysis แบบง่ายๆ"""
    st.markdown("### 🎯 Simple What-If Analysis")
    
    if 'final_df_calc' not in st.session_state:
        st.info("💡 ต้องคำนวณ Safety Stock ก่อนใช้ What-If Analysis")
        return
    
    df = st.session_state['final_df_calc'].copy()
    
    # เลือก SKU
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_sku = st.selectbox(
            "เลือก SKU สำหรับวิเคราะห์:",
            options=df['code'].tolist(),
            index=0
        )
    
    with col2:
        analysis_type = st.radio(
            "ประเภทการวิเคราะห์:",
            ["Service Level", "Lead Time", "Demand"]
        )
    
    if selected_sku:
        sku_data = df[df['code'] == selected_sku].iloc[0]
        
        # แสดงข้อมูลปัจจุบัน
        st.markdown(f"**📊 Current Status: {selected_sku}**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current SS (PCS)", int(sku_data['safety_stock_pcs']))
        with col2:
            st.metric("Current SS Value", f"฿{sku_data['safety_stock_value']:,.0f}")
        with col3:
            st.metric("Service Level", f"{sku_data['service_level_percent']:.1f}%")
        with col4:
            st.metric("Lead Time", f"{sku_data['Leadtime day']:.0f} days")
        
        # Scenario Analysis
        st.markdown("**🔮 Scenario Analysis**")
        
        if analysis_type == "Service Level":
            # Service Level Analysis
            new_sl = st.slider(
                "New Service Level (%)",
                min_value=80.0,
                max_value=99.9,
                value=float(sku_data['service_level_percent']),
                step=0.1
            )
            
            # Calculate new safety stock
            new_z = compute_z_from_sl(new_sl)
            new_ss = new_z * sku_data['combined_std_dev']
            new_ss_pcs = int(math.ceil(max(0, new_ss)))
            new_ss_value = new_ss_pcs * sku_data['cost_per_pcs']
            
            # Show changes
            ss_change = new_ss_pcs - sku_data['safety_stock_pcs']
            value_change = new_ss_value - sku_data['safety_stock_value']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("New SS (PCS)", new_ss_pcs, delta=int(ss_change))
            with col2:
                st.metric("New SS Value", f"฿{new_ss_value:,.0f}", delta=f"฿{value_change:,.0f}")
            with col3:
                change_pct = (ss_change / sku_data['safety_stock_pcs'] * 100) if sku_data['safety_stock_pcs'] > 0 else 0
                st.metric("Change (%)", f"{change_pct:+.1f}%")
        
        elif analysis_type == "Lead Time":
            # Lead Time Analysis
            new_lt = st.slider(
                "New Lead Time (days)",
                min_value=1.0,
                max_value=60.0,
                value=float(sku_data['Leadtime day']),
                step=1.0
            )
            
            # Recalculate with new lead time
            time_divisor = 7.0 if 'weekly' in str(st.session_state.get('period', '')).lower() else 1.0
            new_term1 = (new_lt / time_divisor) * (sku_data['adj_std_dev_demand'] ** 2)
            new_term2 = sku_data['term2_leadtime_var']  # Lead time variability term stays same
            new_combined_std = np.sqrt(max(0, new_term1 + new_term2))
            
            new_ss = sku_data['z_score'] * new_combined_std
            new_ss_pcs = int(math.ceil(max(0, new_ss)))
            new_ss_value = new_ss_pcs * sku_data['cost_per_pcs']
            
            # Show changes
            ss_change = new_ss_pcs - sku_data['safety_stock_pcs']
            value_change = new_ss_value - sku_data['safety_stock_value']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("New SS (PCS)", new_ss_pcs, delta=int(ss_change))
            with col2:
                st.metric("New SS Value", f"฿{new_ss_value:,.0f}", delta=f"฿{value_change:,.0f}")
            with col3:
                change_pct = (ss_change / sku_data['safety_stock_pcs'] * 100) if sku_data['safety_stock_pcs'] > 0 else 0
                st.metric("Change (%)", f"{change_pct:+.1f}%")
        
        else:  # Demand Analysis
            # Demand Change Analysis
            demand_change_pct = st.slider(
                "Demand Change (%)",
                min_value=-50.0,
                max_value=100.0,
                value=0.0,
                step=5.0
            )
            
            # Recalculate with new demand
            new_avg_demand = sku_data['avg_demand'] * (1 + demand_change_pct/100)
            new_std_demand = sku_data['std_dev_demand'] * (1 + demand_change_pct/100)
            
            # Apply seasonal factor
            new_adj_avg = new_avg_demand * sku_data['seasonal_factor']
            new_adj_std = new_std_demand * sku_data['seasonal_factor']
            
            time_divisor = 7.0 if 'weekly' in str(st.session_state.get('period', '')).lower() else 1.0
            std_dev_lead_time_days = st.session_state.get('std_dev_lead_time_days', 2.0)
            
            new_term1 = (sku_data['Leadtime day'] / time_divisor) * (new_adj_std ** 2)
            new_term2 = (new_adj_avg ** 2) * ((std_dev_lead_time_days / time_divisor) ** 2)
            new_combined_std = np.sqrt(max(0, new_term1 + new_term2))
            
            new_ss = sku_data['z_score'] * new_combined_std
            new_ss_pcs = int(math.ceil(max(0, new_ss)))
            new_ss_value = new_ss_pcs * sku_data['cost_per_pcs']
            
            # Show changes
            ss_change = new_ss_pcs - sku_data['safety_stock_pcs']
            value_change = new_ss_value - sku_data['safety_stock_value']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("New SS (PCS)", new_ss_pcs, delta=int(ss_change))
            with col2:
                st.metric("New SS Value", f"฿{new_ss_value:,.0f}", delta=f"฿{value_change:,.0f}")
            with col3:
                change_pct = (ss_change / sku_data['safety_stock_pcs'] * 100) if sku_data['safety_stock_pcs'] > 0 else 0
                st.metric("Change (%)", f"{change_pct:+.1f}%")

def create_comprehensive_risk_dashboard(df):
    """Risk Analysis Dashboard ที่ดูได้ทั้งหมด"""
    try:
        # Risk categorization
        df['risk_category'] = pd.cut(df['demand_cv'], 
                                   bins=[0, 0.5, 1.0, 1.5, float('inf')],
                                   labels=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'],
                                   include_lowest=True)
        
        st.markdown("### 📈 Comprehensive Risk Analysis Dashboard")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_items = len(df)
            st.metric("Total Items", total_items)
        with col2:
            high_risk_items = len(df[df['demand_cv'] >= 1.0])
            st.metric("High Risk Items", high_risk_items, delta=f"{high_risk_items/total_items*100:.1f}%")
        with col3:
            total_ss_value = df['safety_stock_value'].sum()
            st.metric("Total SS Investment", f"฿{total_ss_value:,.0f}")
        with col4:
            avg_cv = df['demand_cv'].mean()
            st.metric("Portfolio Avg CV", f"{avg_cv:.3f}")
        
        # Risk distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk category pie chart
            risk_counts = df['risk_category'].value_counts()
            fig_pie = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title='Risk Distribution by Item Count',
                color_discrete_map={
                    'Low Risk': '#2ecc71',
                    'Medium Risk': '#f39c12',
                    'High Risk': '#e74c3c',
                    'Very High Risk': '#8e44ad'
                }
            )
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Safety stock value by risk
            risk_value = df.groupby('risk_category')['safety_stock_value'].sum().reset_index()
            fig_bar = px.bar(
                risk_value,
                x='risk_category',
                y='safety_stock_value',
                title='Safety Stock Investment by Risk Level',
                color='risk_category',
                color_discrete_map={
                    'Low Risk': '#2ecc71',
                    'Medium Risk': '#f39c12',
                    'High Risk': '#e74c3c',
                    'Very High Risk': '#8e44ad'
                }
            )
            fig_bar.update_layout(height=300, xaxis_title="Risk Level", yaxis_title="SS Value (฿)")
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # CV vs Safety Stock scatter plot
        fig_scatter = px.scatter(
            df,
            x='demand_cv',
            y='safety_stock_pcs',
            size='safety_stock_value',
            color='risk_category',
            hover_data=['code', 'ProductName', 'abc_class'],
            title='Coefficient of Variation vs Safety Stock Requirements',
            color_discrete_map={
                'Low Risk': '#2ecc71',
                'Medium Risk': '#f39c12',
                'High Risk': '#e74c3c',
                'Very High Risk': '#8e44ad'
            }
        )
        fig_scatter.update_layout(height=400, xaxis_title="Coefficient of Variation (CV)", yaxis_title="Safety Stock (PCS)")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Detailed risk tables
        st.markdown("### 📋 Detailed Risk Analysis Tables")
        
        # Risk summary table
        risk_summary = df.groupby('risk_category').agg({
            'code': 'count',
            'safety_stock_pcs': ['sum', 'mean'],
            'safety_stock_value': ['sum', 'mean'],
            'demand_cv': 'mean',
            'service_level_percent': 'mean'
        }).round(2)
        
        # Flatten column names
        risk_summary.columns = ['Item Count', 'Total SS PCS', 'Avg SS PCS', 'Total SS Value', 'Avg SS Value', 'Avg CV', 'Avg Service Level']
        risk_summary = risk_summary.reset_index()
        
        st.markdown("**📊 Risk Summary by Category**")
        st.dataframe(risk_summary, use_container_width=True)
        
        # Top risk items
        high_risk_items = df[df['demand_cv'] >= 1.0].sort_values('safety_stock_value', ascending=False)
        
        if not high_risk_items.empty:
            st.markdown("**🚨 High Risk Items (CV ≥ 1.0) - Sorted by SS Value**")
            
            # Select columns for display
            display_cols = ['code', 'ProductName', 'abc_class', 'demand_cv', 'avg_demand', 
                          'std_dev_demand', 'Leadtime day', 'service_level_percent',
                          'safety_stock_pcs', 'safety_stock_value', 'risk_category']
            existing_cols = [col for col in display_cols if col in high_risk_items.columns]
            
            risk_display = high_risk_items[existing_cols].head(20)  # Show top 20
            
            # Format the display
            if 'safety_stock_value' in risk_display.columns:
                risk_display['safety_stock_value'] = risk_display['safety_stock_value'].apply(lambda x: f"฿{x:,.0f}")
            if 'demand_cv' in risk_display.columns:
                risk_display['demand_cv'] = risk_display['demand_cv'].round(3)
            
            st.dataframe(risk_display, use_container_width=True, height=400)
        
        # ABC vs Risk analysis
        if 'abc_class' in df.columns:
            st.markdown("**🔍 ABC Class vs Risk Level Analysis**")
            
            abc_risk_crosstab = pd.crosstab(df['abc_class'], df['risk_category'], margins=True)
            st.dataframe(abc_risk_crosstab, use_container_width=True)
            
            # Visualization of ABC vs Risk
            abc_risk_data = df.groupby(['abc_class', 'risk_category']).size().reset_index(name='count')
            
            fig_heatmap = px.density_heatmap(
                abc_risk_data,
                x='abc_class',
                y='risk_category',
                z='count',
                title='ABC Class vs Risk Level Heatmap',
                color_continuous_scale='Viridis'
            )
            fig_heatmap.update_layout(height=300)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        return df
    except Exception as e:
        st.error(f"Error creating risk dashboard: {e}")
        return df

# --- UI: Header & samples ---
st.title("📊 Safety Stock Calculator V13 — Improved Analytics & Visualization")
st.markdown("""
**เวอร์ชันปรับปรุง Analytics** — Time Series แบบกราฟหุ้น, Distribution ที่ดูง่าย, What-If Analysis ที่ใช้งานง่าย และ Risk Dashboard แบบครบถ้วน
""")

with st.expander("📁 ดาวน์โหลดไฟล์ตัวอย่าง (Improved Template)", expanded=False):
    c1, c2 = st.columns(2)
    c1.download_button("📊 Enhanced Sales CSV", data=sample_sales_csv(), 
                       file_name="improved_sales_template.csv", mime="text/csv")
    c2.download_button("📋 Enhanced Master CSV", data=sample_master_csv(), 
                       file_name="improved_master_template.csv", mime="text/csv")

# --- Sidebar: Global filters / parameters ---
with st.sidebar:
    st.header("🎛️ ตัวกรอง / พารามิเตอร์")
    period = st.radio("คำนวณ Demand จาก:", ('รายวัน (Daily)', 'รายสัปดาห์ (Weekly)'), horizontal=True)
    std_dev_lead_time_days = st.number_input("ความผันผวนของ Lead Time (วัน)", min_value=0.0, value=2.0, step=0.5,
                                           help="σ_LT ในสูตร Safety Stock")
    unit_for_pareto = st.radio("หน่วย Pareto (Dashboard):", ("PCS", "Carton"), index=0)
    show_math_details = st.checkbox("แสดงรายละเอียดคณิตศาสตร์", value=False)
    reset_btn = st.button("🔄 Reset Session State")

if reset_btn:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# --- Main: file upload ---
st.header("1️⃣ อัปโหลดไฟล์ Sales และ Master")
col1, col2 = st.columns(2)
with col1:
    sales_file = st.file_uploader("📊 Sales Data (CSV)", type=['csv'], key='sales',
                                 help="คอลัมน์ที่ต้องมี: Date, code, PCS, Sales_Channel")
with col2:
    master_file = st.file_uploader("📋 Master Data (CSV)", type=['csv'], key='master',
                                  help="ต้องมี: code, ProductName, Leadtime day, pcs_per_carton, cost_per_pcs")

# read files safely
if sales_file:
    sales_df, err = safe_read_csv(sales_file)
    if err:
        st.error(err)
        st.stop()
else:
    sales_df = pd.DataFrame()

if master_file:
    master_df, err = safe_read_csv(master_file)
    if err:
        st.error(err)
        st.stop()
else:
    master_df = pd.DataFrame()

# show quick preview
if not sales_df.empty:
    st.markdown("**📊 Preview Sales Data**")
    st.dataframe(sales_df.head(5), use_container_width=True)

if not master_df.empty:
    st.markdown("**📋 Preview Master Data**")
    st.dataframe(master_df.head(5), use_container_width=True)

# --- Validation and preparation ---
if not sales_df.empty and not master_df.empty:
    # basic column validations
    req_master = {'code', 'ProductName', 'Leadtime day', 'pcs_per_carton', 'cost_per_pcs'}
    if not req_master.issubset(set(master_df.columns)):
        st.error(f"❌ ไฟล์ Master data ต้องมีคอลัมน์: {sorted(list(req_master))}")
        st.stop()

    req_sales = {'Date', 'code', 'PCS', 'Sales_Channel'}
    if not req_sales.issubset(set(sales_df.columns)):
        st.error(f"❌ ไฟล์ Sales ต้องมีคอลัมน์: {sorted(list(req_sales))}")
        st.stop()

    # convert types
    try:
        sales_df['Date'] = pd.to_datetime(sales_df['Date'])
    except Exception as e:
        st.error(f"❌ ไม่สามารถแปลงคอลัมน์ Date ได้: {e}")
        st.stop()

    # ensure numeric pcs
    try:
        sales_df['PCS'] = pd.to_numeric(sales_df['PCS'], errors='coerce').fillna(0)
        master_df['pcs_per_carton'] = pd.to_numeric(master_df['pcs_per_carton'], errors='coerce').fillna(1)
        master_df['pcs_per_carton'] = master_df['pcs_per_carton'].astype(int)
        master_df['Leadtime day'] = pd.to_numeric(master_df['Leadtime day'], errors='coerce').fillna(0).astype(float)
        master_df['cost_per_pcs'] = pd.to_numeric(master_df['cost_per_pcs'], errors='coerce').fillna(0.0).astype(float)
    except Exception as e:
        st.error(f"❌ Error converting data types: {e}")
        st.stop()

    # channel filter
    try:
        channels = sales_df['Sales_Channel'].dropna().unique().tolist()
        if len(channels) == 0:
            st.warning("⚠️ ไม่มี Sales_Channel ในข้อมูล Sales — จะใช้ข้อมูลทั้งหมด")
            selected_channels = channels
        else:
            selected_channels = st.multiselect("🎯 เลือกช่องทางการขาย (Filter)", 
                                             options=channels, default=channels)

        if len(selected_channels) == 0 and len(channels) > 0:
            st.warning("⚠️ กรุณาเลือกช่องทางการขาย อย่างน้อย 1 ช่องทาง")
            st.stop()

        # filter data
        filtered_df = sales_df[sales_df['Sales_Channel'].isin(selected_channels)].copy() if selected_channels else sales_df.copy()
        if filtered_df.empty:
            st.warning("⚠️ ไม่มีข้อมูลหลังกรองด้วย Sales_Channel ที่เลือก")
            st.stop()
    except Exception as e:
        st.error(f"Error filtering sales channels: {e}")
        st.stop()

    # แสดงกราฟวิเคราะห์ Demand ใหม่
    st.markdown("---")
    st.header("📊 Improved Demand Analysis & Visualization")
    
    # SKU selector สำหรับกราฟ
    all_skus = filtered_df['code'].unique().tolist()
    selected_skus_for_chart = st.multiselect(
        "🎯 เลือก SKUs สำหรับแสดงในกราฟ (ไม่เลือก = แสดงทั้งหมด):",
        options=all_skus,
        default=all_skus[:3] if len(all_skus) > 3 else all_skus,
        help="เลือกสูงสุด 5 SKUs เพื่อความชัดเจนของกราฟ"
    )
    
    # จำกัดจำนวน SKUs ที่แสดงในกราฟ
    if len(selected_skus_for_chart) > 5:
        st.warning("⚠️ แนะนำให้เลือก SKU ไม่เกิน 5 รายการ เพื่อความชัดเจนของกราฟ")
        selected_skus_for_chart = selected_skus_for_chart[:5]
    
    # สร้างกราฟ Time Series และ Distribution แบบใหม่
    col1, col2 = st.columns([3, 2])
    
    with col1:
        time_series_fig = create_improved_time_series_chart(filtered_df, selected_skus_for_chart)
        if time_series_fig:
            st.plotly_chart(time_series_fig, use_container_width=True)
    
    with col2:
        dist_fig = create_improved_distribution_chart(filtered_df, selected_skus_for_chart)
        if dist_fig:
            st.plotly_chart(dist_fig, use_container_width=True)
    
    # Statistics Summary Table
    st.markdown("**📈 Statistics Summary**")
    stats_table = create_statistics_summary_table(filtered_df, selected_skus_for_chart)
    if stats_table is not None:
        st.dataframe(stats_table, use_container_width=True)

    # compute demand stats (daily or weekly)
    with st.spinner("🔄 กำลังคำนวณสถิติ demand..."):
        try:
            if period.startswith('รายสัปดาห์'):
                # aggregate weekly per SKU
                weekly = (filtered_df.set_index('Date')
                                  .groupby('code')['PCS']
                                  .resample('W-SUN')
                                  .sum()
                                  .reset_index())
                demand_stats = weekly.groupby('code', as_index=False)['PCS'].agg(['mean', 'std']).reset_index()
            else:
                daily = filtered_df.copy()
                demand_stats = daily.groupby('code', as_index=False)['PCS'].agg(['mean', 'std']).reset_index()
            
            demand_stats.rename(columns={'mean': 'avg_demand', 'std': 'std_dev_demand'}, inplace=True)
            demand_stats['std_dev_demand'] = demand_stats['std_dev_demand'].fillna(0.0)
            
        except Exception as e:
            st.error(f"Error calculating demand statistics: {e}")
            st.stop()

    # abc analysis
    abc_df = perform_abc_analysis(filtered_df)

    # base merge
    try:
        final_df_base = pd.merge(demand_stats, abc_df, on='code', how='left')
        final_df_base = pd.merge(final_df_base, master_df, on='code', how='left')
        final_df_base = final_df_base.fillna({'Leadtime day': 0, 'pcs_per_carton': 1, 'cost_per_pcs': 0, 
                                             'ProductName': "!! ไม่พบใน Master !!", 'abc_class': 'C', 
                                             'total_volume': 0, 'cumulative_percent': 0})
        
        # คำนวณ CV (Coefficient of Variation) - ป้องกัน division by zero
        final_df_base['demand_cv'] = np.where(
            final_df_base['avg_demand'] > 0,
            final_df_base['std_dev_demand'] / final_df_base['avg_demand'],
            0
        )
        
        # Replace any inf values
        final_df_base['demand_cv'] = final_df_base['demand_cv'].replace([np.inf, -np.inf], 0).fillna(0)
        
    except Exception as e:
        st.error(f"Error merging data: {e}")
        st.stop()

    # --- params editor in session state ---
    if 'params_df' not in st.session_state:
        try:
            params_df = final_df_base[['code', 'ProductName', 'abc_class']].copy()
            params_df['service_level_percent'] = params_df['abc_class'].map(DEFAULT_SL_MAP).fillna(DEFAULT_SL_MAP['C'])
            params_df['seasonal_factor_key'] = 'ปกติ (x1.0)'
            st.session_state.params_df = params_df
        except Exception as e:
            st.error(f"Error initializing parameters: {e}")

    st.header("2️⃣ ปรับพารามิเตอร์ราย SKU")
    st.info("💡 แก้ Service Level หรือ Seasonality ต่อ SKU หรือใช้ฟังก์ชัน Bulk ด้านล่าง")

    # bulk editor
    st.markdown("**⚡ Bulk Apply Service Level ตาม Class**")
    c1, c2, c3, c4 = st.columns([2,2,2,3])
    sl_a = c1.number_input("Service Level Class A (%)", min_value=50.0, max_value=99.99, 
                          value=99.0, step=0.1, key="sl_a")
    sl_b = c2.number_input("Service Level Class B (%)", min_value=50.0, max_value=99.99, 
                          value=95.0, step=0.1, key="sl_b")
    sl_c = c3.number_input("Service Level Class C (%)", min_value=50.0, max_value=99.99, 
                          value=90.0, step=0.1, key="sl_c")
    if c4.button("✅ Apply to all in table"):
        try:
            tmp = st.session_state.params_df.copy()
            tmp['service_level_percent'] = tmp['abc_class'].map({'A': sl_a, 'B': sl_b, 'C': sl_c}).fillna(sl_c)
            st.session_state.params_df = tmp
            st.success("✅ Applied bulk service levels")
        except Exception as e:
            st.error(f"Error applying bulk changes: {e}")

    # editable table
    try:
        edited = st.data_editor(
            st.session_state.params_df,
            column_config={
                "code": st.column_config.TextColumn("SKU", disabled=True),
                "ProductName": st.column_config.TextColumn("ProductName", disabled=True),
                "abc_class": st.column_config.TextColumn("ABC", disabled=True),
                "service_level_percent": st.column_config.NumberColumn("Service Level (%)", 
                                                                      min_value=50.0, max_value=99.99, step=0.1),
                "seasonal_factor_key": st.column_config.SelectboxColumn("Seasonality", 
                                                                       options=list(SEASONALITY_MAP.keys()))
            },
            hide_index=True,
            key="params_editor",
            height=350
        )
        # store back
        st.session_state.params_df = edited
    except Exception as e:
        st.error(f"Error with parameter editor: {e}")

    # calculate button
    if st.button("🚀 คำนวณ SAFETY STOCK (Compute)", type="primary"):
        with st.spinner("🔄 กำลังคำนวณ..."):
            try:
                params = st.session_state.params_df[['code', 'service_level_percent', 'seasonal_factor_key']].copy()
                final_df_calc = pd.merge(final_df_base, params, on='code', how='left')
                final_df_calc['seasonal_factor'] = final_df_calc['seasonal_factor_key'].map(SEASONALITY_MAP).fillna(1.0)

                # z score - ป้องกัน NaN values
                def safe_z_score(sl):
                    if pd.isna(sl):
                        return compute_z_from_sl(DEFAULT_SL_MAP.get('C', 90.0))
                    return compute_z_from_sl(sl)
                
                final_df_calc['z_score'] = final_df_calc['service_level_percent'].apply(safe_z_score)

                # adjust demand & std
                final_df_calc['avg_demand_calc'] = final_df_calc['avg_demand'].fillna(0).clip(lower=0)
                final_df_calc['adj_avg_demand'] = final_df_calc['avg_demand_calc'] * final_df_calc['seasonal_factor']
                final_df_calc['adj_std_dev_demand'] = final_df_calc['std_dev_demand'].fillna(0) * final_df_calc['seasonal_factor']

                time_divisor = 7.0 if period.startswith('รายสัปดาห์') else 1.0

                # คำนวณแยกเป็น components เพื่อแสดง
                final_df_calc['term1_demand_var'] = (final_df_calc['Leadtime day'] / time_divisor) * (final_df_calc['adj_std_dev_demand'] ** 2)
                final_df_calc['term2_leadtime_var'] = (final_df_calc['adj_avg_demand'] ** 2) * ((std_dev_lead_time_days / time_divisor) ** 2)
                final_df_calc['combined_variance'] = final_df_calc['term1_demand_var'] + final_df_calc['term2_leadtime_var']
                final_df_calc['combined_std_dev'] = np.sqrt(final_df_calc['combined_variance'].clip(lower=0))
                
                # Safety Stock calculation
                final_df_calc['safety_stock_pcs'] = (final_df_calc['z_score'] * final_df_calc['combined_std_dev']).fillna(0)
                final_df_calc['safety_stock_pcs'] = final_df_calc['safety_stock_pcs'].apply(
                    lambda x: int(math.ceil(max(0, x))) if not pd.isna(x) else 0)

                # cartons & value
                final_df_calc['pcs_per_carton'] = final_df_calc['pcs_per_carton'].apply(
                    lambda x: max(1, int(x)) if not pd.isna(x) and x > 0 else 1)
                
                final_df_calc['safety_stock_carton'] = final_df_calc.apply(
                    lambda row: int(math.ceil(row['safety_stock_pcs'] / row['pcs_per_carton'])) if row['pcs_per_carton'] > 0 else 0,
                    axis=1
                )
                
                final_df_calc['safety_stock_value'] = final_df_calc['safety_stock_pcs'] * final_df_calc['cost_per_pcs']

                st.session_state['final_df_calc'] = final_df_calc
                st.session_state['period'] = period
                st.session_state['std_dev_lead_time_days'] = std_dev_lead_time_days
                st.success("✅ คำนวณเรียบร้อย! ไปดูแดชบอร์ดได้เลย")
                
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดขณะคำนวณ: {e}")
                logging.exception(e)

# --- Results / downloads ---
if 'final_df_calc' in st.session_state:
    st.markdown("---")
    st.header("3️⃣ ผลลัพธ์การคำนวณ (Result Preview)")
    try:
        df_res = st.session_state['final_df_calc'].copy()
        display_cols = ['ProductName', 'code', 'abc_class', 'Leadtime day', 'service_level_percent', 
                       'seasonal_factor_key', 'demand_cv', 'safety_stock_pcs', 'safety_stock_carton', 'safety_stock_value']
        
        # Filter existing columns
        existing_display_cols = [col for col in display_cols if col in df_res.columns]
        display_df = df_res[existing_display_cols].copy()
        
        # format numeric columns for display
        if 'safety_stock_value' in display_df.columns:
            display_df['safety_stock_value'] = display_df['safety_stock_value'].fillna(0).astype(float)
        if 'demand_cv' in display_df.columns:
            display_df['demand_cv'] = display_df['demand_cv'].fillna(0).round(3)
        
        st.dataframe(display_df.reset_index(drop=True), use_container_width=True, height=400)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_ss_value = df_res['safety_stock_value'].sum()
            st.metric("💰 Total SS Value", f"฿{total_ss_value:,.2f}")
        with col2:
            total_ss_pcs = df_res['safety_stock_pcs'].sum()
            st.metric("📦 Total SS PCS", f"{int(total_ss_pcs):,}")
        with col3:
            avg_service_level = df_res['service_level_percent'].mean()
            st.metric("🎯 Avg Service Level", f"{avg_service_level:.1f}%")
        with col4:
            high_cv_items = len(df_res[df_res['demand_cv'] > 1.0])
            st.metric("⚠️ High Variability Items", high_cv_items)

        # Download buttons
        st.markdown("**📁 Download Results**")
        cold1, cold2 = st.columns(2)
        cold1.download_button("📊 Download CSV", data=df_to_csv_bytes(df_res), 
                             file_name="safety_stock_results_v13_improved.csv", mime="text/csv")
        
        excel_data = df_to_excel_bytes(df_res)
        if excel_data:  # Only show Excel download if file was created successfully
            cold2.download_button("📈 Download Excel", data=excel_data, 
                                 file_name="safety_stock_results_v13_improved.xlsx", 
                                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.error(f"Error displaying results: {e}")

# --- Improved What-If Analysis ---
if 'final_df_calc' in st.session_state:
    st.markdown("---")
    create_simple_what_if_analysis()

# --- Enhanced Dashboard ---
st.markdown("---")
st.header("📊 Enhanced Analytics Dashboard")

if 'final_df_calc' not in st.session_state:
    st.info("💡 ยังไม่มีผลลัพธ์ — ให้ไปกดคำนวณที่ส่วนบนก่อน")
else:
    try:
        df = st.session_state['final_df_calc'].copy()
        
        # Dashboard tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 ABC Analysis", "📈 Risk Analysis", "🔬 Deep Dive"])
        
        with tab1:
            st.subheader("📊 Portfolio Overview")
            
            # KPI Row
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            with kpi1:
                st.metric("Total SKUs", len(df))
            with kpi2:
                st.metric("Total SS Value", f"฿{df['safety_stock_value'].sum():,.0f}")
            with kpi3:
                st.metric("Avg Lead Time", f"{df['Leadtime day'].mean():.1f} days")
            with kpi4:
                st.metric("Highest SS Item", f"฿{df['safety_stock_value'].max():,.0f}")
            with kpi5:
                risk_items = len(df[df['demand_cv'] > 1.5])
                st.metric("Very High Risk Items", risk_items)
            
            # Safety Stock Distribution
            col1, col2 = st.columns(2)
            with col1:
                if 'safety_stock_value' in df.columns:
                    fig_dist = px.histogram(df, x='safety_stock_value', nbins=20, 
                                           title='Safety Stock Value Distribution')
                    fig_dist.update_layout(height=300)
                    st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                if all(col in df.columns for col in ['avg_demand', 'safety_stock_pcs', 'abc_class']):
                    fig_scatter = px.scatter(df, x='avg_demand', y='safety_stock_pcs', 
                                           color='abc_class', size='safety_stock_value',
                                           hover_data=['code', 'ProductName'],
                                           title='Demand vs Safety Stock')
                    fig_scatter.update_layout(height=300)
                    st.plotly_chart(fig_scatter, use_container_width=True)
        
        with tab2:
            st.subheader("🎯 ABC Analysis Deep Dive")
            
            # ABC Summary
            abc_summary = df.groupby('abc_class').agg({
                'code': 'count',
                'total_volume': 'sum',
                'safety_stock_pcs': 'sum',
                'safety_stock_value': 'sum',
                'service_level_percent': 'mean'
            }).reset_index()
            abc_summary.columns = ['ABC Class', 'SKU Count', 'Total Volume', 'Total SS PCS', 'Total SS Value', 'Avg Service Level']
            
            st.dataframe(abc_summary, use_container_width=True)
            
            # ABC Charts
            col1, col2 = st.columns(2)
            with col1:
                fig_abc_vol = px.bar(abc_summary, x='ABC Class', y='Total Volume',
                                    title='Volume by ABC Class', color='ABC Class')
                st.plotly_chart(fig_abc_vol, use_container_width=True)
            
            with col2:
                fig_abc_ss = px.bar(abc_summary, x='ABC Class', y='Total SS Value',
                                   title='Safety Stock Value by ABC Class', color='ABC Class')
                st.plotly_chart(fig_abc_ss, use_container_width=True)
        
        with tab3:
            st.subheader("📈 Comprehensive Risk Analysis")
            # Use the new comprehensive risk dashboard
            df = create_comprehensive_risk_dashboard(df)
        
        with tab4:
            st.subheader("🔬 Mathematical Deep Dive")
            
            if show_math_details and 'term1_demand_var' in df.columns:
                # Component analysis
                st.markdown("**Safety Stock Components Analysis**")
                
                df['term1_contribution'] = np.sqrt(df['term1_demand_var'].clip(lower=0))
                df['term2_contribution'] = np.sqrt(df['term2_leadtime_var'].clip(lower=0))
                
                components_cols = ['code', 'ProductName', 'term1_contribution', 'term2_contribution', 'combined_std_dev']
                existing_comp_cols = [col for col in components_cols if col in df.columns]
                components_df = df[existing_comp_cols].copy()
                
                if 'combined_std_dev' in components_df.columns:
                    components_df['demand_var_pct'] = np.where(
                        df['combined_std_dev'] > 0,
                        (df['term1_contribution'] / df['combined_std_dev'] * 100),
                        0
                    )
                    components_df['leadtime_var_pct'] = np.where(
                        df['combined_std_dev'] > 0,
                        (df['term2_contribution'] / df['combined_std_dev'] * 100),
                        0
                    )
                
                st.dataframe(components_df.round(3), use_container_width=True)
            else:
                st.info("💡 เปิด 'แสดงรายละเอียดคณิตศาสตร์' ใน Sidebar เพื่อดู Deep Dive Analysis")
    except Exception as e:
        st.error(f"Error in dashboard: {e}")

# --- Enhanced Methodology ---
with st.expander("🔬 Enhanced Methodology & Mathematical Details", expanded=False):
    st.markdown("""
    ## 📊 Safety Stock Calculation Methodology
    
    ### สูตรหลัก (Main Formula):
    **SS = Z × √[(LT × σd²) + (μd² × σLT²)]**
    
    ### คำอธิบายสัญลักษณ์ (Symbol Definitions):
    - **SS** = Safety Stock (units)
    - **Z** = Z-score จาก Service Level α
    - **LT** = Average Lead Time (days)
    - **σd** = Adjusted Demand Standard Deviation
    - **μd** = Adjusted Average Demand  
    - **σLT** = Lead Time Standard Deviation
    
    ### การปรับค่าตาม Seasonality:
    - μd(adj) = μd × Seasonal_Factor
    - σd(adj) = σd × Seasonal_Factor
    
    ### Z-Score Conversion:
    Service Level (%) → Z-Score ผ่าน Inverse Normal Distribution:
    - 90% → 1.282
    - 95% → 1.645  
    - 99% → 2.326
    - 99.9% → 3.090
    
    ### ABC Classification:
    - **Class A**: Cumulative 0-80% of volume
    - **Class B**: Cumulative 80-95% of volume  
    - **Class C**: Cumulative 95-100% of volume
    
    ### Risk Assessment (Coefficient of Variation):
    **CV = σd / μd**
    - CV < 0.5: Low Risk
    - 0.5 ≤ CV < 1.0: Medium Risk
    - 1.0 ≤ CV < 1.5: High Risk
    - CV ≥ 1.5: Very High Risk
    
    ### Component Breakdown:
    1. **Term 1 (Demand Variability)**: √(LT × σd²)
    2. **Term 2 (Lead Time Variability)**: √(μd² × σLT²)
    3. **Combined**: √(Term1² + Term2²)
    
    ### การแปลงหน่วยเวลา:
    - รายวัน (Daily): time_divisor = 1.0
    - รายสัปดาห์ (Weekly): time_divisor = 7.0
    
    ### การคำนวณ Carton:
    - Safety Stock (Carton) = ⌈Safety Stock (PCS) / PCS per Carton⌉
    
    ### การคำนวณมูลค่า:
    - Safety Stock Value = Safety Stock (PCS) × Cost per PCS
    
    ### การปรับปรุงในเวอร์ชัน V13:
    1. **Time Series Analysis**: กราฟแบบ candlestick/line chart พร้อม range selector
    2. **Distribution Analysis**: Histogram แบบ overlay สำหรับหลาย SKUs
    3. **What-If Analysis**: UI แบบง่ายๆ เลือก SKU และพารามิเตอร์ที่ต้องการทดสอบ
    4. **Risk Dashboard**: แสดงผลครบถ้วน พร้อม heatmap และ detailed tables
    5. **Statistics Summary**: ตารางสถิติครบถ้วนพร้อม Risk Level classification
    
    ---
    *📚 อ้างอิง: Silver, Pyke & Peterson (1998), Inventory Management and Production Planning*
    """)

st.markdown("---")
st.markdown("**🎉 Safety Stock Calculator V13 - Improved Analytics & Visualization**")
st.markdown("*Enhanced with better Time Series, Distribution, What-If Analysis, and Comprehensive Risk Dashboard*")