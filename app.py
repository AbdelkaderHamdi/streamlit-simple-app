import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Advanced Data Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .subtitle {
        font-size: 1.3rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    data = {
        'State': np.random.choice(['California', 'Texas', 'New York', 'Florida', 'Illinois'], 1000),
        'City': np.random.choice(['Los Angeles', 'Houston', 'New York', 'Miami', 'Chicago'], 1000),
        'Product': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard'], 1000),
        'Sales Method': np.random.choice(['Online', 'In-Store', 'Phone'], 1000),
        'Total Sales': np.random.normal(5000, 2000, 1000).round(2),
        'Units Sold': np.random.poisson(50, 1000),
        'Operating Profit': np.random.normal(1200, 500, 1000).round(2)
    }
    return pd.DataFrame(data)

def main():
    """Main application function"""
    # Header section
    st.markdown('<h1 class="main-header">📊 Advanced Data Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Interactive Business Intelligence & Data Visualization Platform</p>', 
                unsafe_allow_html=True)
    
        
    st.markdown("""
    This professional-grade data analytics platform is built with **Streamlit** and **Plotly** to provide 
    comprehensive business intelligence capabilities.

    ### ✨ Key Features

    - **📊 Interactive Visualizations**: Scatter plots, histograms, and bar charts with real-time filtering
    - **🔍 Advanced Analytics**: Group-by operations, aggregations, and statistical insights
    - **📁 File Upload Support**: Seamlessly work with your own CSV data
    - **📱 Responsive Design**: Professional UI that works across all devices
    - **⚡ High Performance**: Optimized with Streamlit caching for fast data processing

    ### 🛠️ Technologies Used

    - **Frontend**: Streamlit
    - **Data Processing**: Pandas, NumPy
    - **Visualizations**: Plotly Express
    - **Styling**: Custom CSS

    ### 📈 Use Cases

    - Sales performance analysis
    - Business intelligence reporting
    - Data exploration and visualization
    - Statistical analysis and insights generation

    ### 👨‍💻 Developer

    This project demonstrates advanced Python skills in data science, data visualization, 
    and business analytics. Perfect for showcasing technical expertise on LinkedIn and GitHub.
    """)


    
  
if __name__ == "__main__":
    main()