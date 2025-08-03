# Streamlit Data Analysis and Visualization Project

## Overview

This project showcases a powerful and interactive data analysis and visualization application built using Streamlit. It provides functionalities for uploading CSV files, performing data cleaning, generating various plots (histograms and scatter plots), and conducting advanced groupby operations to extract meaningful business insights. The application is designed to be user-friendly, allowing individuals with varying technical backgrounds to explore their data effectively.

## Features

### 1. `app.py` (Main Application)

- **Interactive Welcome Page**: A welcoming interface for the Streamlit application.
- **Sidebar Navigation**: Easy navigation between different sections of the application.
- **Quick Demo**: A simple interactive element to demonstrate Streamlit's capabilities.
- **Custom Styling**: Enhanced visual appeal through custom CSS.

### 2. `1_Data_Visualization.py` (Data Visualization)

- **CSV File Upload**: Securely upload your CSV datasets.
- **Data Cleaning**: Automated cleaning of common data issues (e.g., removing currency symbols, handling missing values).
- **Interactive Data Preview**: View a customizable preview of your uploaded data.
- **Scatter Plots**: Generate interactive scatter plots to visualize relationships between numerical variables.
- **Histograms**: Create histograms to understand the distribution of individual numerical features.

### 3. `2_Business_Analytics.py` (Groupby Operations & Insights)

- **Advanced Groupby Operations**: Perform complex data aggregations (sum, mean, max, min, count) based on selected categorical and numerical columns.
- **Dynamic Visualization**: Visualize grouped data using interactive bar charts.
- **Key Insights Display**: Summarize important findings from the grouped analysis.
- **Download Results**: Option to download the aggregated results as a CSV file.

## Getting Started

### Prerequisites

To run this project locally, you need to have Python installed. It is recommended to use a virtual environment.

- Python 3.8+

### Installation

1. **Clone the repository (or download the files):**

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required libraries:**

   ```bash
   pip install -r requirements.txt
   ```

   *(Note: You will need to create a `requirements.txt` file. See the next section.)*



## How to Run the Application

After installing the prerequisites and creating the `requirements.txt` file, you can run the Streamlit application:

1. **Navigate to the project directory:**

   ```bash
   cd /path/to/your/project
   ```

2. **Run the Streamlit application:**

   ```bash
   streamlit run app.py
   ```

   This command will open the Streamlit application in your default web browser. You can then navigate through the different sections using the sidebar.

## Usage

- **Upload Data**: On the "Data Visualization" and "Groupby & Insights" pages, use the file uploader to select your CSV file.
- **Explore Visualizations**: Use the sliders, select boxes, and tabs to interact with the data and generate different plots.
- **Analyze Insights**: Configure groupby operations and view the aggregated results and key insights.
- **Download Results**: Download the processed data for further analysis.

## Project Structure

```
streamlit-analytics-dashboard/
├── app.py
├── pages/
│   ├── 1_Data_Visualization.py
│   └── 2_Business_Analytics.py
├── requirements.txt
└── README.md
```


## Contributing

Feel free to fork this repository, submit pull requests, or open issues for any bugs or feature requests.

