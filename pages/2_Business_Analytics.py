import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Business Analytics Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Enhanced styling
st.markdown("""
    <style>
    .analytics-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .insight-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .metric-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .control-section {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    .performance-indicator {
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
        margin: 0.25rem 0;
    }
    .performance-high { background: #d4edda; color: #155724; }
    .performance-medium { background: #fff3cd; color: #856404; }
    .performance-low { background: #f8d7da; color: #721c24; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file):
    """Load and cache data from uploaded file"""
    return pd.read_csv(file)

@st.cache_data
def generate_sample_business_data():
    """Generate comprehensive sample business data"""
    np.random.seed(42)
    
    # Create more realistic business data
    states = ['California', 'Texas', 'New York', 'Florida', 'Illinois', 'Pennsylvania', 'Ohio', 'Georgia']
    cities = ['Los Angeles', 'Houston', 'New York', 'Miami', 'Chicago', 'Philadelphia', 'Columbus', 'Atlanta']
    products = ['Laptop', 'Desktop', 'Phone', 'Tablet', 'Monitor', 'Keyboard', 'Mouse', 'Printer']
    sales_methods = ['Online', 'In-Store', 'Phone', 'B2B Direct']
    
    n_records = 1000
    
    data = {
        'State': np.random.choice(states, n_records),
        'City': np.random.choice(cities, n_records),
        'Product': np.random.choice(products, n_records),
        'Sales Method': np.random.choice(sales_methods, n_records),
        'Total Sales': np.random.lognormal(8, 0.5, n_records).round(2),
        'Units Sold': np.random.poisson(25, n_records),
        'Operating Profit': np.random.normal(1500, 800, n_records).round(2),
        'Customer Satisfaction': np.random.uniform(3.0, 5.0, n_records).round(1),
        'Days to Ship': np.random.poisson(3, n_records),
        'Sales Rep': np.random.choice(['John Smith', 'Sarah Johnson', 'Mike Wilson', 'Lisa Brown', 'David Lee'], n_records)
    }
    
    df = pd.DataFrame(data)
    
    # Add calculated fields
    df['Profit Margin'] = (df['Operating Profit'] / df['Total Sales'] * 100).round(2)
    df['Revenue per Unit'] = (df['Total Sales'] / df['Units Sold']).round(2)
    
    # Add date column
    start_date = datetime.now() - timedelta(days=365)
    df['Order Date'] = pd.date_range(start=start_date, periods=n_records, freq='H')
    
    return df

def clean_business_data(df):
    """Clean and prepare business data for analysis"""
    df_clean = df.copy()
    
    # Clean monetary columns
    monetary_columns = ['Total Sales', 'Operating Profit']
    for col in monetary_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.replace(r'[\$,]', '', regex=True)
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Clean units column
    if 'Units Sold' in df_clean.columns:
        df_clean['Units Sold'] = df_clean['Units Sold'].astype(str).str.replace(r'[\$,]', '', regex=True)
        df_clean['Units Sold'] = pd.to_numeric(df_clean['Units Sold'], errors='coerce')
    
    # Drop unnecessary columns
    columns_to_drop = ['Region', 'Retailer ID']
    df_clean = df_clean.drop(columns=[col for col in columns_to_drop if col in df_clean.columns])
    
    # Remove rows with NaN values in critical columns
    df_clean = df_clean.dropna()
    
    return df_clean

def main():
    """Main business analytics application"""
    # Header
    st.markdown("""
    <div class="analytics-header">
        <h1>🔍 Advanced Business Analytics Dashboard</h1>
        <p>Comprehensive insights and performance metrics for data-driven decisions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Setup and load data
    df = setup_data_source()
    
    if df is not None:
        # Executive summary
        display_executive_summary(df)
        
        # Main analytics sections
        create_analytics_interface(df)
    else:
        st.info("👆 Please upload a CSV file to begin advanced analytics")

def setup_data_source():
    """Setup data source with upload option"""
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown('<div class="control-section">', unsafe_allow_html=True)
        st.markdown("#### 📁 Data Source")
        
        uploaded_file = st.file_uploader(
            "Upload Business Data",
            type=['csv'],
            help="Upload your business data in CSV format"
        )
        
        use_sample = st.checkbox("Use sample data", value=True if uploaded_file is None else False)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            st.success("✅ Custom data loaded successfully!")
            return clean_business_data(df)
        elif use_sample:
            df = generate_sample_business_data()
            st.info("📊 Using sample business data for demonstration")
            return df
        else:
            return None

def display_executive_summary(df):
    """Display executive summary with key business metrics"""
    st.markdown("### 📊 Executive Summary")
    
    # Key Performance Indicators
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_revenue = df['Total Sales'].sum()
        st.markdown(f"""
        <div class="metric-highlight">
            <h3>${total_revenue:,.0f}</h3>
            <p>Total Revenue</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_profit = df['Operating Profit'].sum()
        profit_margin = (total_profit / total_revenue) * 100
        st.markdown(f"""
        <div class="metric-highlight">
            <h3>${total_profit:,.0f}</h3>
            <p>Total Profit ({profit_margin:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_units = df['Units Sold'].sum()
        avg_order_value = total_revenue / len(df)
        st.markdown(f"""
        <div class="metric-highlight">
            <h3>{total_units:,}</h3>
            <p>Units Sold</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-highlight">
            <h3>${avg_order_value:,.0f}</h3>
            <p>Avg Order Value</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        unique_customers = len(df) if 'Customer ID' not in df.columns else df['Customer ID'].nunique()
        st.markdown(f"""
        <div class="metric-highlight">
            <h3>{unique_customers:,}</h3>
            <p>Total Orders</p>
        </div>
        """, unsafe_allow_html=True)

def create_analytics_interface(df):
    """Create comprehensive analytics interface"""
    st.markdown("---")
    
    # Analytics tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Performance Analysis", 
        "🎯 Segmentation Analysis", 
        "📊 Comparative Analytics",
        "🔮 Advanced Insights"
    ])
    
    with tab1:
        performance_analysis(df)
    
    with tab2:
        segmentation_analysis(df)
    
    with tab3:
        comparative_analytics(df)
    
    with tab4:
        advanced_insights(df)

def performance_analysis(df):
    """Create performance analysis dashboard"""
    st.markdown("#### 📈 Performance Analysis Dashboard")
    
    # Control panel
    col1, col2, col3 = st.columns(3)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    with col1:
        dimension = st.selectbox("Analyze by:", categorical_cols, key="perf_dimension")
    
    with col2:
        metric = st.selectbox("Performance metric:", numerical_cols, key="perf_metric")
    
    with col3:
        aggregation = st.selectbox("Aggregation:", ['sum', 'mean', 'median', 'count', 'max', 'min'], key="perf_agg")
    
    # Perform analysis
    try:
        if aggregation == 'sum':
            results = df.groupby(dimension)[metric].sum().sort_values(ascending=False)
        elif aggregation == 'mean':
            results = df.groupby(dimension)[metric].mean().sort_values(ascending=False)
        elif aggregation == 'median':
            results = df.groupby(dimension)[metric].median().sort_values(ascending=False)
        elif aggregation == 'count':
            results = df.groupby(dimension)[metric].count().sort_values(ascending=False)
        elif aggregation == 'max':
            results = df.groupby(dimension)[metric].max().sort_values(ascending=False)
        else:  # min
            results = df.groupby(dimension)[metric].min().sort_values(ascending=False)
        
        # Display results
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("##### 📊 Performance Rankings")
            
            # Create performance indicators
            for i, (category, value) in enumerate(results.head(10).items()):
                if i < 3:
                    performance_class = "performance-high"
                elif i < 7:
                    performance_class = "performance-medium"
                else:
                    performance_class = "performance-low"
                
                st.markdown(f"""
                <div class="performance-indicator {performance_class}">
                    #{i+1} {category}: {value:,.2f}
                </div>
                """, unsafe_allow_html=True)
            
            # Key insights
            st.markdown("##### 🎯 Key Insights")
            top_performer = results.index[0]
            top_value = results.iloc[0]
            avg_value = results.mean()
            
            st.markdown(f"""
            <div class="insight-card">
                <strong>Top Performer:</strong> {top_performer}<br>
                <strong>Value:</strong> {top_value:,.2f}<br>
                <strong>Above Average:</strong> {((top_value/avg_value-1)*100):+.1f}%
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Visualization
            fig = px.bar(
                x=results.head(15).index,
                y=results.head(15).values,
                title=f"{aggregation.title()} of {metric} by {dimension}",
                labels={'x': dimension, 'y': metric}
            )
            
            fig.update_layout(
                height=500,
                title_x=0.5,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")

def segmentation_analysis(df):
    """Create customer/product segmentation analysis"""
    st.markdown("#### 🎯 Advanced Segmentation Analysis")
    
    # Segmentation controls
    col1, col2, col3 = st.columns(3)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    with col1:
        primary_segment = st.selectbox("Primary segment:", categorical_cols, key="seg_primary")
    
    with col2:
        secondary_segment = st.selectbox("Secondary segment:", ['None'] + categorical_cols, key="seg_secondary")
    
    with col3:
        value_metric = st.selectbox("Value metric:", numerical_cols, key="seg_metric")
    
    # Create segmentation analysis
    if secondary_segment == 'None':
        # Single dimension analysis
        segment_data = df.groupby(primary_segment).agg({
            value_metric: ['sum', 'mean', 'count'],
            'Units Sold': 'sum' if 'Units Sold' in df.columns else value_metric
        }).round(2)
        
        segment_data.columns = ['Total_Value', 'Avg_Value', 'Count', 'Total_Units']
        segment_data = segment_data.sort_values('Total_Value', ascending=False)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart for segment contribution
            fig_pie = px.pie(
                values=segment_data['Total_Value'],
                names=segment_data.index,
                title=f"{value_metric} Distribution by {primary_segment}"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart for segment performance
            fig_bar = px.bar(
                x=segment_data.index,
                y=segment_data['Avg_Value'],
                title=f"Average {value_metric} by {primary_segment}"
            )
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Segment details table
        st.markdown("##### 📋 Segment Performance Details")
        st.dataframe(segment_data, use_container_width=True)
    
    else:
        # Two-dimensional analysis
        pivot_data = df.pivot_table(
            values=value_metric,
            index=primary_segment,
            columns=secondary_segment,
            aggfunc='sum',
            fill_value=0
        )
        
        # Heatmap
        fig_heatmap = px.imshow(
            pivot_data,
            title=f"{value_metric} Heatmap: {primary_segment} vs {secondary_segment}",
            aspect="auto",
            color_continuous_scale="Blues"
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Cross-tabulation analysis
        st.markdown("##### 📊 Cross-Segment Analysis")
        st.dataframe(pivot_data, use_container_width=True)

def comparative_analytics(df):
    """Create comparative analytics dashboard"""
    st.markdown("#### 📊 Comparative Analytics Dashboard")
    
    # Comparison setup
    col1, col2 = st.columns(2)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    with col1:
        compare_dimension = st.selectbox("Compare by:", categorical_cols, key="comp_dimension")
        compare_metrics = st.multiselect(
            "Select metrics to compare:",
            numerical_cols,
            default=numerical_cols[:3] if len(numerical_cols) >= 3 else numerical_cols,
            key="comp_metrics"
        )
    
    with col2:
        chart_type = st.selectbox(
            "Chart type:",
            ["Bar Chart", "Radar Chart", "Box Plot", "Violin Plot"],
            key="comp_chart"
        )
        
        top_n = st.slider("Show top N categories:", 5, 20, 10, key="comp_top_n")
    
    if compare_metrics:
        # Prepare comparison data
        comparison_data = df.groupby(compare_dimension)[compare_metrics].mean().head(top_n)
        
        if chart_type == "Bar Chart":
            # Multiple bar chart
            fig = px.bar(
                comparison_data,
                title=f"Comparative Analysis: {compare_dimension}",
                height=500
            )
            fig.update_layout(title_x=0.5, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Radar Chart":
            # Radar chart for top categories
            fig = go.Figure()
            
            for category in comparison_data.index[:5]:  # Top 5 for readability
                fig.add_trace(go.Scatterpolar(
                    r=comparison_data.loc[category].values,
                    theta=compare_metrics,
                    fill='toself',
                    name=category
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, comparison_data.values.max()]
                    )),
                showlegend=True,
                title=f"Radar Chart: {compare_dimension} Comparison",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type in ["Box Plot", "Violin Plot"]:
            # Box or violin plot for distribution comparison
            metric_to_plot = st.selectbox("Select metric for distribution:", compare_metrics, key="dist_metric")
            
            if chart_type == "Box Plot":
                fig = px.box(df, x=compare_dimension, y=metric_to_plot,
                           title=f"Distribution of {metric_to_plot} by {compare_dimension}")
            else:
                fig = px.violin(df, x=compare_dimension, y=metric_to_plot,
                              title=f"Distribution of {metric_to_plot} by {compare_dimension}")
            
            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Comparison table
        st.markdown("##### 📋 Detailed Comparison")
        st.dataframe(comparison_data, use_container_width=True)

def advanced_insights(df):
    """Generate advanced business insights"""
    st.markdown("#### 🔮 Advanced Business Insights")
    
    # Insight categories
    insight_type = st.selectbox(
        "Select insight type:",
        ["Profitability Analysis", "Efficiency Metrics", "Growth Indicators", "Risk Assessment"],
        key="insight_type"
    )
    
    if insight_type == "Profitability Analysis":
        profitability_insights(df)
    elif insight_type == "Efficiency Metrics":
        efficiency_insights(df)
    elif insight_type == "Growth Indicators":
        growth_insights(df)
    else:
        risk_assessment(df)

def profitability_insights(df):
    """Generate profitability insights"""
    st.markdown("##### 💰 Profitability Analysis")
    
    if 'Operating Profit' in df.columns and 'Total Sales' in df.columns:
        # Calculate profit margins
        df['Profit_Margin'] = (df['Operating Profit'] / df['Total Sales']) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Most profitable segments
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            for col in categorical_cols[:2]:  # Top 2 categorical columns
                profit_by_segment = df.groupby(col)['Profit_Margin'].mean().sort_values(ascending=False)
                
                st.markdown(f"**Most Profitable {col}:**")
                st.write(f"🥇 {profit_by_segment.index[0]}: {profit_by_segment.iloc[0]:.1f}%")
                st.write(f"🥈 {profit_by_segment.index[1]}: {profit_by_segment.iloc[1]:.1f}%")
                st.write(f"🥉 {profit_by_segment.index[2]}: {profit_by_segment.iloc[2]:.1f}%")
                st.markdown("---")
        
        with col2:
            # Profit margin distribution
            fig = px.histogram(df, x='Profit_Margin', nbins=30,
                             title="Profit Margin Distribution")
            st.plotly_chart(fig, use_container_width=True)

def efficiency_insights(df):
    """Generate efficiency insights"""
    st.markdown("##### ⚡ Efficiency Metrics")
    
    # Calculate efficiency metrics
    if 'Units Sold' in df.columns and 'Total Sales' in df.columns:
        df['Revenue_per_Unit'] = df['Total Sales'] / df['Units Sold']
        
        col1, col2 = st.columns(2)
        
        with col1:
            avg_revenue_per_unit = df['Revenue_per_Unit'].mean()
            high_efficiency = df[df['Revenue_per_Unit'] > avg_revenue_per_unit * 1.2]
            
            st.metric("Average Revenue per Unit", f"${avg_revenue_per_unit:.2f}")
            st.metric("High Efficiency Orders", f"{len(high_efficiency):,}")
            
            if 'Days to Ship' in df.columns:
                avg_shipping_time = df['Days to Ship'].mean()
                st.metric("Average Shipping Time", f"{avg_shipping_time:.1f} days")
        
        with col2:
            # Efficiency by category
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                efficiency_col = st.selectbox("Analyze efficiency by:", categorical_cols, key="eff_col")
                efficiency_data = df.groupby(efficiency_col)['Revenue_per_Unit'].mean().sort_values(ascending=False)
                
                fig = px.bar(x=efficiency_data.index, y=efficiency_data.values,
                           title=f"Revenue per Unit by {efficiency_col}")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

def growth_insights(df):
    """Generate growth insights"""
    st.markdown("##### 📈 Growth Indicators")
    
    if 'Order Date' in df.columns:
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['Month'] = df['Order Date'].dt.to_period('M')
        
        # Monthly growth analysis
        monthly_sales = df.groupby('Month')['Total Sales'].sum()
        monthly_growth = monthly_sales.pct_change() * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            latest_growth = monthly_growth.iloc[-1]
            avg_growth = monthly_growth.mean()
            
            st.metric("Latest Month Growth", f"{latest_growth:.1f}%")
            st.metric("Average Monthly Growth", f"{avg_growth:.1f}%")
            
            # Growth trend indicator
            if latest_growth > avg_growth:
                st.success("📈 Growth accelerating!")
            elif latest_growth > 0:
                st.info("📊 Positive growth maintained")
            else:
                st.warning("📉 Growth slowing")
        
        with col2:
            # Growth trend chart
            fig = px.line(x=monthly_growth.index.astype(str), y=monthly_growth.values,
                         title="Monthly Growth Rate (%)")
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Date column required for growth analysis. Using sample insights:")
        
        # Alternative growth metrics
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            growth_dimension = st.selectbox("Growth dimension:", categorical_cols, key="growth_dim")
            
            # Simulate growth by comparing top vs bottom performers
            performance = df.groupby(growth_dimension)['Total Sales'].sum().sort_values(ascending=False)
            top_performer = performance.iloc[0]
            bottom_performer = performance.iloc[-1]
            growth_potential = ((top_performer - bottom_performer) / bottom_performer) * 100
            
            st.metric("Growth Potential", f"{growth_potential:.0f}%")
            st.info(f"Top performer ({performance.index[0]}) vs bottom performer ({performance.index[-1]})")

def risk_assessment(df):
    """Generate risk assessment insights"""
    st.markdown("##### ⚠️ Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Concentration Risk Analysis**")
        
        # Customer concentration (if applicable)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols[:2]:
            concentration = df.groupby(col)['Total Sales'].sum().sort_values(ascending=False)
            top_3_share = concentration.head(3).sum() / concentration.sum() * 100
            
            risk_level = "High" if top_3_share > 60 else "Medium" if top_3_share > 40 else "Low"
            risk_color = "🔴" if risk_level == "High" else "🟡" if risk_level == "Medium" else "🟢"
            
            st.write(f"{risk_color} **{col} Concentration:** {top_3_share:.1f}% ({risk_level} Risk)")
    
    with col2:
        st.markdown("**Performance Variability**")
        
        # Calculate coefficient of variation for key metrics
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        for col in numerical_cols[:3]:
            if col in df.columns:
                cv = (df[col].std() / df[col].mean()) * 100
                volatility = "High" if cv > 50 else "Medium" if cv > 25 else "Low"
                volatility_color = "🔴" if volatility == "High" else "🟡" if volatility == "Medium" else "🟢"
                
                st.write(f"{volatility_color} **{col} Volatility:** {cv:.1f}% ({volatility})")
    
    # Risk mitigation recommendations
    st.markdown("##### 💡 Risk Mitigation Recommendations")
    
    recommendations = [
        "📊 Diversify customer base to reduce concentration risk",
        "📈 Monitor key performance indicators regularly",
        "🎯 Implement performance standards and targets",
        "🔄 Develop contingency plans for underperforming segments",
        "📋 Regular review of pricing and profitability strategies"
    ]
    
    for rec in recommendations:
        st.write(rec)

if __name__ == "__main__":
    main()