import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from langchain_cohere import ChatCohere
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from logging_config import get_logger
logger = get_logger(__name__)

# Initialize LLM
llm = ChatCohere()

from db_functions import load_from_sqlite

# Set page configuration
st.set_page_config(page_title="Data Analyzer", layout="wide")

# Page title
st.title("Data Analyzer")

# Sidebar options
with st.sidebar:
    st.header("Data Source")
    data_source = st.radio(
        "Choose data source:",
        ("Customer Database", "Customer TestBed Database", "Upload CSV")
    )
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("📤 Upload your data file:", type=["csv"])

# Main content
def load_data(source):
    if source == "Customer Database":
        try:
            return load_from_sqlite(table_name="customer", db_name="customer.db")
        except Exception as e:
            st.error(f"Error loading customer database: {e}")
            return None
    elif source == "Customer TestBed Database":
        try:
            return load_from_sqlite(table_name="customer", db_name="customer_testbed.db")
        except Exception as e:
            st.error(f"Error loading customer testbed database: {e}")
            return None
    else:  # Uploaded file
        if 'uploaded_file' in locals() and uploaded_file is not None:
            return pd.read_csv(uploaded_file)
        else:
            st.warning("Please upload a CSV file")
            return None

# Load data based on selection
df = load_data(data_source)

if df is not None and not df.empty:
    st.success(f"Data loaded successfully: {len(df)} records")
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(
        df.head(10),
        column_config={
            "acct_num": st.column_config.NumberColumn(format="%d"),
            "zip4_code": st.column_config.NumberColumn(format="%d"),
            "govt_issued_id": st.column_config.NumberColumn(format="%d"),
            "phone_number": st.column_config.NumberColumn(format="%d"),
            "postal_code": st.column_config.NumberColumn(format="%d"),
            "index": st.column_config.NumberColumn(format="%d"),
        },
        hide_index=True,
    )
    
    # Analysis options
    st.subheader("Analysis Options")
    
    analysis_tabs = st.tabs(["Basic Stats", "Visualizations", "AI Analysis"])
    
    with analysis_tabs[0]:
        st.header("Basic Statistics")
        
        # Data info
        st.subheader("Data Information")
        
        # Display basic info in a cleaner format
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", len(df))
            st.metric("Total Columns", len(df.columns))
            
        with col2:
            st.metric("Missing Values", df.isna().sum().sum())
            st.metric("Duplicate Rows", df.duplicated().sum())
        
        # Data types
        st.subheader("Data Types")
        dtypes_df = pd.DataFrame(df.dtypes, columns=["Data Type"])
        dtypes_df.index.name = "Column Name"
        st.dataframe(dtypes_df.reset_index())
        
        # Numerical summary
        st.subheader("Numerical Summary")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe())
        else:
            st.info("No numerical columns found")
            
        # Categorical summary
        st.subheader("Categorical Summary")
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if categorical_cols:
            cat_col_to_analyze = st.selectbox("Select a categorical column:", categorical_cols)
            st.dataframe(df[cat_col_to_analyze].value_counts().reset_index().rename(
                columns={"index": cat_col_to_analyze, cat_col_to_analyze: "Count"}
            ))
        else:
            st.info("No categorical columns found")
    
    with analysis_tabs[1]:
        st.header("Data Visualizations")
        
        viz_type = st.selectbox(
            "Select visualization type:",
            ["Distribution", "Correlation", "Categorical Counts"]
        )
        
        if viz_type == "Distribution":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                col_to_plot = st.selectbox("Select column to visualize:", numeric_cols)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df[col_to_plot].dropna(), kde=True, ax=ax)
                ax.set_title(f"Distribution of {col_to_plot}")
                st.pyplot(fig)
            else:
                st.info("No numerical columns available for distribution plot")
                
        elif viz_type == "Correlation":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(12, 10))
                corr_matrix = df[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                ax.set_title("Correlation Matrix")
                st.pyplot(fig)
            else:
                st.info("Need at least two numerical columns for correlation analysis")
                
        elif viz_type == "Categorical Counts":
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            if categorical_cols:
                col_to_plot = st.selectbox("Select categorical column:", categorical_cols)
                
                # Get top N values for plotting
                top_n = st.slider("Show top N categories:", min_value=3, max_value=min(20, df[col_to_plot].nunique()), value=10)
                
                top_cats = df[col_to_plot].value_counts().head(top_n).index
                plot_data = df[df[col_to_plot].isin(top_cats)]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.countplot(y=col_to_plot, data=plot_data, order=top_cats, ax=ax)
                ax.set_title(f"Top {top_n} categories in {col_to_plot}")
                st.pyplot(fig)
            else:
                st.info("No categorical columns available for plotting")
    
    with analysis_tabs[2]:
        st.header("AI-Powered Analysis")
        
        analysis_question = st.text_area(
            "Ask a question about your data:",
            placeholder="Example: What are the key patterns in this data? or How are the different categories distributed?"
        )
        
        if st.button("Generate Analysis"):
            if analysis_question:
                with st.spinner("AI is analyzing your data..."):
                    # Prepare data summary to send to LLM
                    data_summary = f"""
                    Data Summary:
                    - Total records: {len(df)}
                    - Columns: {', '.join(df.columns.tolist())}
                    - Numerical columns: {', '.join(df.select_dtypes(include=[np.number]).columns.tolist())}
                    - Categorical columns: {', '.join(df.select_dtypes(exclude=[np.number]).columns.tolist())}
                    - First few records:\n{df.head(3).to_string()}
                    
                    Basic statistics:
                    {df.describe().to_string()}
                    
                    Question: {analysis_question}
                    """
                    
                    # Call LLM for analysis
                    response = llm.invoke([
                        {"role": "system", "content": "You are a data analysis expert. Analyze the provided data summary and answer the user's question. Provide insights, patterns, and observations based on the available information."},
                        {"role": "user", "content": data_summary}
                    ])
                    
                    # Display results
                    st.subheader("AI Analysis")
                    st.markdown(response.content)
                    
                    # Suggest followup questions
                    st.subheader("Suggested Follow-up Questions")
                    followup_prompt = f"Based on the data summary and the original question '{analysis_question}', suggest 3 follow-up questions that would be valuable to ask about this dataset. Format each suggestion as a bullet point."
                    
                    followup_response = llm.invoke([
                        {"role": "system", "content": "You are a data analysis expert. Suggest follow-up questions."},
                        {"role": "user", "content": followup_prompt}
                    ])
                    
                    st.markdown(followup_response.content)
            else:
                st.warning("Please enter a question to analyze")

else:
    st.info("No data loaded. Please select a data source.")

# Footer
st.markdown("---")
st.caption("@Copyright - Tata Consultancy Services | 2025")