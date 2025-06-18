import streamlit 
import pandas
import plotly.express as px
import numpy 

@streamlit.cache_data
def load_data(file):
    return pandas.read_csv(file)

file = streamlit.file_uploader('Upload your file.. ', type=['csv'])

if file is not None:
    df = load_data(file)
    
    #cleaning the data
    df.drop(columns=['Region','Retailer ID'], inplace=True)
    df['Total Sales']= df['Total Sales'].replace(r'[\$,]','', regex=True).astype(float)
    df['Unit Sold']= df['Units Sold'].replace(r'[\$,]', '',regex=True).astype(float)
    df['Operating Profit']= df['Operating Profit'].replace(r'[\$,]', '',regex=True).astype(float)


    n_rows = streamlit.slider('choose number of row to display..', min_value=0, max_value=len(df), step=1)

    column_options= streamlit.multiselect('select columns to show', df.columns.to_list(), default=df.columns.to_list())

    streamlit.write(df[:n_rows][column_options])

    tab1, tab2 = streamlit.tabs(['Scatter', 'Histogram'])

    with tab1:
        col1, col2 = streamlit.columns(2)
        numerical_columns= df.select_dtypes(include=numpy.number).columns.to_list() 

        #accept only the numerical columns
        with col1:
            x_column= streamlit.selectbox('select for x axis: ', numerical_columns)
        with col2:
            y_column= streamlit.selectbox('select for y axis: ', numerical_columns)

        figure_scatter= px.scatter(df , x=x_column, y=y_column)
        streamlit.plotly_chart(figure_scatter)
    with tab2:
        histogram_feature= streamlit.selectbox('select feature', numerical_columns)
        figure_hist= px.histogram(df , x=histogram_feature)
        streamlit.plotly_chart(figure_hist)


