import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

file = st.file_uploader("Upload your file...", type=["csv"])

if file is not None:
    df = load_data(file)

    # Basic cleaning
    df.drop(columns=['Region', 'Retailer ID'], inplace=True, errors='ignore')
    df['Total Sales'] = df['Total Sales'].replace(r'[\$,]', '', regex=True).astype(float)
    df['Units Sold'] = df['Units Sold'].replace(r'[\$,]', '', regex=True).astype(float)
    df['Operating Profit'] = df['Operating Profit'].replace(r'[\$,]', '', regex=True).astype(float)

    st.sidebar.title("Section 2: Grouped Insights")

    # 🔹 User selects aggregation method
    agg_method = st.sidebar.radio("Select aggregation", ['sum', 'mean', 'max'])

    # 🔹 Choose feature and group by
    group_by_feature = st.sidebar.selectbox("Group by:", ['State', 'City', 'Product', 'Sales Method'])
    target_metric = st.sidebar.selectbox("Target metric:", ['Total Sales', 'Units Sold', 'Operating Profit'])

    # 🔹 Compute result
    if agg_method == 'sum':
        grouped_df = df.groupby(group_by_feature)[target_metric].sum().sort_values(ascending=False)
    elif agg_method == 'mean':
        grouped_df = df.groupby(group_by_feature)[target_metric].mean().sort_values(ascending=False)
    else:
        grouped_df = df.groupby(group_by_feature)[target_metric].max().sort_values(ascending=False)

    st.write(f"### {agg_method.title()} of {target_metric} by {group_by_feature}")
    st.dataframe(grouped_df)

    # 🔹 Visualization
    st.plotly_chart(px.bar(grouped_df, x=grouped_df.index, y=target_metric, title=f"{agg_method.title()} of {target_metric} by {group_by_feature}"))

