import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Data Visualization Hub",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .chart-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .control-panel {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file):
    """Load and cache data from uploaded file"""
    return pd.read_csv(file)

@st.cache_data
def generate_sample_data():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    data = {
        'State': np.random.choice(['California', 'Texas', 'New York', 'Florida', 'Illinois'], 500),
        'City': np.random.choice(['Los Angeles', 'Houston', 'New York', 'Miami', 'Chicago'], 500),
        'Product': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard'], 500),
        'Sales Method': np.random.choice(['Online', 'In-Store', 'Phone'], 500),
        'Total Sales': np.random.normal(5000, 2000, 500).round(2),
        'Units Sold': np.random.poisson(50, 500),
        'Operating Profit': np.random.normal(1200, 500, 500).round(2),
        'Operating Margin': np.random.uniform(0.1, 0.3, 500).round(3)
    }
    return pd.DataFrame(data)

def clean_data(df):
    """Clean and prepare data for analysis"""
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
    """Main visualization application"""
    st.title("📊 Advanced Data Visualization Hub")
    st.markdown("### Interactive charts and insights for comprehensive data analysis")
    
    # Sidebar for data upload and controls
    setup_sidebar()
    
    # Get data
    df = get_data()
    
    if df is not None:
        # Data overview
        display_data_overview(df)
        
        # Main visualization sections
        create_visualization_tabs(df)
    else:
        st.info("👆 Please upload a CSV file to begin analysis")

def setup_sidebar():
    """Setup sidebar controls"""
    st.sidebar.header("🎛️ Control Panel")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload CSV File",
        type=['csv'],
        help="Upload your business data in CSV format"
    )
    
    if uploaded_file is not None:
        st.session_state['uploaded_file'] = uploaded_file
        st.sidebar.success("✅ File uploaded successfully!")
    else:
        st.sidebar.info("📋 Using sample data for demo")
        if 'uploaded_file' in st.session_state:
            del st.session_state['uploaded_file']

def get_data():
    """Load data from file or generate sample data"""
    if 'uploaded_file' in st.session_state:
        df = load_data(st.session_state['uploaded_file'])
        return clean_data(df)
    else:
        return clean_data(generate_sample_data())

def display_data_overview(df):
    """Display data overview and key metrics"""
    st.markdown("---")
    st.markdown("### 📈 Data Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>{len(df):,}</h3>
            <p>Total Records</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h3>{len(df.columns)}</h3>
            <p>Data Columns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if 'Total Sales' in df.columns:
            total_sales = df['Total Sales'].sum()
            st.markdown(f"""
            <div class="metric-container">
                <h3>${total_sales:,.0f}</h3>
                <p>Total Sales</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if 'Units Sold' in df.columns:
            total_units = df['Units Sold'].sum()
            st.markdown(f"""
            <div class="metric-container">
                <h3>{total_units:,}</h3>
                <p>Units Sold</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Data preview with controls
    st.markdown("### 📋 Data Preview")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        n_rows = st.slider('Rows to display:', 5, min(len(df), 100), 10)
        
        available_columns = df.columns.tolist()
        default_columns = available_columns[:6] if len(available_columns) > 6 else available_columns
        
        selected_columns = st.multiselect(
            'Select columns:',
            available_columns,
            default=default_columns
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if selected_columns:
            st.dataframe(
                df[selected_columns].head(n_rows),
                use_container_width=True
            )

def create_visualization_tabs(df):
    """Create tabbed interface for different visualizations"""
    st.markdown("---")
    st.markdown("### 🎨 Interactive Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Scatter Analysis", 
        "📈 Distribution Analysis", 
        "📉 Time Series", 
        "🔍 Advanced Charts"
    ])
    
    numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    with tab1:
        create_scatter_plots(df, numerical_columns, categorical_columns)
    
    with tab2:
        create_distribution_plots(df, numerical_columns, categorical_columns)
    
    with tab3:
        create_time_series_plots(df, numerical_columns)
    
    with tab4:
        create_advanced_charts(df, numerical_columns, categorical_columns)

def create_scatter_plots(df, numerical_cols, categorical_cols):
    """Create interactive scatter plots"""
    st.markdown("#### 📊 Scatter Plot Analysis")
    
    if len(numerical_cols) < 2:
        st.warning("Need at least 2 numerical columns for scatter plots")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        x_axis = st.selectbox("X-axis:", numerical_cols, key="scatter_x")
    
    with col2:
        y_axis = st.selectbox("Y-axis:", numerical_cols, index=1, key="scatter_y")
    
    with col3:
        color_by = st.selectbox("Color by:", ['None'] + categorical_cols, key="scatter_color")
    
    with col4:
        size_by = st.selectbox("Size by:", ['None'] + numerical_cols, key="scatter_size")
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x=x_axis,
        y=y_axis,
        color=None if color_by == 'None' else color_by,
        size=None if size_by == 'None' else size_by,
        title=f"Scatter Plot: {x_axis} vs {y_axis}",
        height=600,
        hover_data=numerical_cols[:3]  # Show top 3 numerical columns on hover
    )
    
    fig.update_layout(
        title_font_size=16,
        title_x=0.5,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    if st.checkbox("Show correlation analysis", key="scatter_corr"):
        correlation = df[x_axis].corr(df[y_axis])
        st.metric("Correlation Coefficient", f"{correlation:.3f}")
        
        if abs(correlation) > 0.7:
            st.success("Strong correlation detected!")
        elif abs(correlation) > 0.3:
            st.info("Moderate correlation detected")
        else:
            st.warning("Weak correlation")

def create_distribution_plots(df, numerical_cols, categorical_cols):
    """Create distribution analysis plots"""
    st.markdown("#### 📈 Distribution Analysis")
    
    if not numerical_cols:
        st.warning("No numerical columns available for distribution analysis")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_column = st.selectbox("Select variable:", numerical_cols, key="dist_col")
    
    with col2:
        chart_type = st.selectbox("Chart type:", ["Histogram", "Box Plot", "Violin Plot"], key="dist_type")
    
    with col3:
        if categorical_cols:
            group_by = st.selectbox("Group by:", ['None'] + categorical_cols, key="dist_group")
        else:
            group_by = 'None'
    
    # Create distribution plot
    if chart_type == "Histogram":
        if group_by == 'None':
            fig = px.histogram(df, x=selected_column, nbins=30, title=f"Distribution of {selected_column}")
        else:
            fig = px.histogram(df, x=selected_column, color=group_by, nbins=30, 
                             title=f"Distribution of {selected_column} by {group_by}")
    
    elif chart_type == "Box Plot":
        if group_by == 'None':
            fig = px.box(df, y=selected_column, title=f"Box Plot of {selected_column}")
        else:
            fig = px.box(df, x=group_by, y=selected_column, 
                        title=f"Box Plot of {selected_column} by {group_by}")
    
    else:  # Violin Plot
        if group_by == 'None':
            fig = px.violin(df, y=selected_column, title=f"Violin Plot of {selected_column}")
        else:
            fig = px.violin(df, x=group_by, y=selected_column, 
                           title=f"Violin Plot of {selected_column} by {group_by}")
    
    fig.update_layout(height=500, title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    st.markdown("##### 📊 Statistical Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    stats = df[selected_column].describe()
    
    with col1:
        st.metric("Mean", f"{stats['mean']:.2f}")
    with col2:
        st.metric("Median", f"{stats['50%']:.2f}")
    with col3:
        st.metric("Std Dev", f"{stats['std']:.2f}")
    with col4:
        st.metric("Range", f"{stats['max'] - stats['min']:.2f}")

def create_time_series_plots(df, numerical_cols):
    """Create time series analysis plots"""
    st.markdown("#### 📉 Time Series Analysis")
    
    if not numerical_cols:
        st.warning("No numerical columns available for time series analysis")
        return
    
    # Create synthetic time series for demonstration
    df_ts = df.copy()
    df_ts['Date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
    df_ts = df_ts.sort_values('Date')
    
    col1, col2 = st.columns(2)
    
    with col1:
        y_column = st.selectbox("Select metric:", numerical_cols, key="ts_metric")
    
    with col2:
        aggregation = st.selectbox("Aggregation:", ["Daily", "Weekly", "Monthly"], key="ts_agg")
    
    # Aggregate data based on selection
    if aggregation == "Weekly":
        df_ts_agg = df_ts.groupby(pd.Grouper(key='Date', freq='W'))[y_column].mean().reset_index()
    elif aggregation == "Monthly":
        df_ts_agg = df_ts.groupby(pd.Grouper(key='Date', freq='M'))[y_column].mean().reset_index()
    else:
        df_ts_agg = df_ts.groupby('Date')[y_column].mean().reset_index()
    
    # Create time series plot
    fig = px.line(df_ts_agg, x='Date', y=y_column, 
                  title=f"{aggregation} {y_column} Trend",
                  height=500)
    
    fig.update_layout(title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)
    
    # Trend analysis
    if len(df_ts_agg) > 1:
        trend = "Increasing" if df_ts_agg[y_column].iloc[-1] > df_ts_agg[y_column].iloc[0] else "Decreasing"
        change = ((df_ts_agg[y_column].iloc[-1] - df_ts_agg[y_column].iloc[0]) / df_ts_agg[y_column].iloc[0]) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Trend", trend)
        with col2:
            st.metric("Total Change", f"{change:.1f}%")

def create_advanced_charts(df, numerical_cols, categorical_cols):
    """Create advanced visualization charts"""
    st.markdown("#### 🔍 Advanced Analytics Charts")
    
    chart_option = st.selectbox(
        "Select chart type:",
        ["Correlation Heatmap", "3D Scatter Plot", "Parallel Coordinates", "Sunburst Chart"],
        key="advanced_chart"
    )
    
    if chart_option == "Correlation Heatmap":
        if len(numerical_cols) >= 2:
            corr_matrix = df[numerical_cols].corr()
            
            fig = px.imshow(corr_matrix, 
                           text_auto=True, 
                           aspect="auto",
                           title="Correlation Heatmap",
                           height=500)
            
            fig.update_layout(title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 numerical columns for correlation analysis")
    
    elif chart_option == "3D Scatter Plot":
        if len(numerical_cols) >= 3:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                x_3d = st.selectbox("X-axis:", numerical_cols, key="3d_x")
            with col2:
                y_3d = st.selectbox("Y-axis:", numerical_cols, index=1, key="3d_y")
            with col3:
                z_3d = st.selectbox("Z-axis:", numerical_cols, index=2, key="3d_z")
            with col4:
                color_3d = st.selectbox("Color by:", ['None'] + categorical_cols, key="3d_color")
            
            fig = px.scatter_3d(df, x=x_3d, y=y_3d, z=z_3d,
                               color=None if color_3d == 'None' else color_3d,
                               title="3D Scatter Plot",
                               height=600)
            
            fig.update_layout(title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 3 numerical columns for 3D scatter plot")
    
    elif chart_option == "Parallel Coordinates":
        if len(numerical_cols) >= 3:
            selected_cols = st.multiselect(
                "Select columns for parallel coordinates:",
                numerical_cols,
                default=numerical_cols[:4],
                key="parallel_cols"
            )
            
            if len(selected_cols) >= 2:
                color_col = categorical_cols[0] if categorical_cols else None
                
                fig = px.parallel_coordinates(df, 
                                            dimensions=selected_cols,
                                            color=color_col,
                                            title="Parallel Coordinates Plot",
                                            height=500)
                
                fig.update_layout(title_x=0.5)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 3 numerical columns for parallel coordinates")
    
    elif chart_option == "Sunburst Chart":
        if len(categorical_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                path_cols = st.multiselect(
                    "Select hierarchy path:",
                    categorical_cols,
                    default=categorical_cols[:2],
                    key="sunburst_path"
                )
            
            with col2:
                value_col = st.selectbox("Value column:", numerical_cols, key="sunburst_value")
            
            if len(path_cols) >= 1:
                fig = px.sunburst(df, path=path_cols, values=value_col,
                                 title="Sunburst Chart",
                                 height=600)
                
                fig.update_layout(title_x=0.5)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 categorical columns for sunburst chart")

if __name__ == "__main__":
    main()