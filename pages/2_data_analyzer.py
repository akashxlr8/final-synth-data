import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from langchain_cohere import ChatCohere
from dotenv import load_dotenv

from pages.analyzer.llm_analyzer import DatasetAnalyzer, AnalyticalQuestion
from pages.analyzer.db import AnalysisDatabase
from prompts import CODE_GENERATOR_PROMPT

# Load environment variables
load_dotenv()

# Initialize analyzer and database
analyzer = DatasetAnalyzer()
db = AnalysisDatabase()

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
        
        # AI-Generated questions from llm_analyzer.py
        if "dataset_id" not in st.session_state:
            # Save dataset to database
            if df is not None and not df.empty:
                dataset_name = "Uploaded CSV" if data_source == "Upload CSV" else f"{data_source}"
                try:
                    dataset_id = db.save_dataset(
                        filename=dataset_name,
                        row_count=df.shape[0],
                        column_count=df.shape[1],
                        columns=df.columns.tolist()
                    )
                    st.session_state.dataset_id = dataset_id
                except Exception as e:
                    st.error(f"Error saving dataset to database: {str(e)}")
        
        if st.button("Generate Analytical Questions", key="generate_ai_questions"):
            with st.spinner("AI is analyzing your dataset to generate insightful questions..."):
                try:
                    questions = analyzer.generate_questions(df)
                    st.session_state.ai_questions = questions
                    
                    # Save questions to database if we have a dataset ID
                    if "dataset_id" in st.session_state:
                        for q in questions:
                            question_id = db.save_question(
                                dataset_id=st.session_state.dataset_id,
                                question=q.question,
                                category=q.category,
                                reasoning=q.reasoning
                            )
                            # Store question_id for later use
                            q.db_id = question_id
                except Exception as e:
                    st.error(f"Error generating questions: {str(e)}")
        
        # Display generated questions
        if "ai_questions" in st.session_state:
            st.subheader("AI-Generated Analysis Questions")
            for i, q in enumerate(st.session_state.ai_questions, 1):
                with st.expander(f"{i}. {q.question}"):
                    st.write(f"**Category:** {q.category}")
                    st.write(f"**Reasoning:** {q.reasoning}")
                    
                    # Add "Analyze this question" button
                    if st.button(f"Analyze Question {i}", key=f"analyze_q_{i}"):
                        with st.spinner("Analyzing question..."):
                            # Prepare data summary
                            data_summary = f"""
                            Data Summary:
                            - Total records: {len(df)}
                            - Question: {q.question}
                            - Question Category: {q.category}
                            - Question Reasoning: {q.reasoning}
                            
                            The first few rows of the data:
                            {df.head(5).to_string()}
                            
                            Statistical summary:
                            {df.describe().to_string()}
                            """
                            
                            prompt = CODE_GENERATOR_PROMPT.format(
                                df_preview=df.head().to_string(),
                                question=q.question,
                                category=q.category
                            )
                            
                            # Call LLM for analysis
                            response = llm.invoke([
                                {"role": "system", "content": prompt},
                                {"role": "user", "content": data_summary}
                            ])
                            
                            # Store and display the analysis
                            if "dataset_id" in st.session_state and hasattr(q, "db_id"):
                                db.save_code_analysis(
                                    question_id=q.db_id,
                                    code="",  # No actual code here
                                    explanation=response.content,
                                    result=""
                                )

                            # First display the raw response content
                            st.markdown(response.content)

                            # Try to parse the response content as JSON to extract code and explanation
                            try:
                                import json
                                import re
                                
                                # Try to find and extract JSON content from the response
                                json_match = re.search(r'(\{.*?\})', response.content, re.DOTALL)
                                
                                if json_match:
                                    json_str = json_match.group(1)
                                    response_json = json.loads(json_str)
                                    
                                    if "code" in response_json and "explanation" in response_json:
                                        # Display the code with proper syntax highlighting using st.code()
                                        st.subheader("Generated Code")
                                        st.code(response_json["code"], language="python")
                                        
                                        # Display the explanation
                                        st.subheader("Explanation")
                                        st.write(response_json["explanation"])
                                        
                                        # Add a button to run the code on the current data
                                        if st.button("Run Analysis", key=f"run_analysis_{i}"):
                                            try:
                                                with st.spinner("Running analysis..."):
                                                    # Create a local environment with the dataframe
                                                    local_namespace = {"df": df}
                                                    # Execute the code
                                                    exec(response_json["code"], {}, local_namespace)
                                                    
                                                    # If the code produces a result variable, display it
                                                    if "result" in local_namespace:
                                                        st.subheader("Analysis Result")
                                                        result = local_namespace["result"]
                                                        
                                                        # Check if result is a DataFrame
                                                        if isinstance(result, pd.DataFrame):
                                                            st.dataframe(result)
                                                        # Check if result is a matplotlib figure
                                                        elif "matplotlib.figure" in str(type(result)):
                                                            st.pyplot(result)
                                                        # Check if result is a plotly figure
                                                        elif "plotly.graph_objects" in str(type(result)):
                                                            st.plotly_chart(result)
                                                        # Otherwise, just display as text
                                                        else:
                                                            st.write(result)
                                            except Exception as run_err:
                                                st.error(f"Error running the analysis: {str(run_err)}")
                            except Exception as e:
                                st.warning(f"Could not parse JSON from response: {str(e)}")
        
        # Add custom question input
        st.subheader("Add Your Own Analysis Question")
        custom_question = st.text_input("Enter your own analytical question:")
        custom_category = st.selectbox(
            "Select a category for your question:",
            ['Trend Analysis', 'Correlation', 'Distribution', 'Outliers', 'Pattern Recognition', 'Custom']
        )
        custom_reasoning = st.text_area("Explain your reasoning for this question:")
        
        if st.button("Add Question", key="add_custom_question"):
            if custom_question:
                # Create a custom question
                custom_q = AnalyticalQuestion(
                    question=custom_question,
                    category=custom_category or "Custom",
                    reasoning=custom_reasoning or "No reasoning provided"
                )
                
                # Add to session state
                if "ai_questions" not in st.session_state:
                    st.session_state.ai_questions = [custom_q]
                else:
                    st.session_state.ai_questions.append(custom_q)
                
                # Save to database if we have a dataset ID
                if "dataset_id" in st.session_state:
                    question_id = db.save_question(
                        dataset_id=st.session_state.dataset_id,
                        question=custom_question,
                        category=custom_category or "Custom",
                        reasoning=custom_reasoning or "No reasoning provided",
                        is_custom=True
                    )
                    custom_q.db_id = question_id
                
                st.success("Question added successfully!")
                st.experimental_rerun()  # Rerun to show the updated questions list
            else:
                st.warning("Please enter a question to add.")
                
        # Show analysis history if available
        if "dataset_id" in st.session_state:
            st.subheader("Analysis History")
            try:
                history = db.get_analysis_history()
                if history:
                    for i, entry in enumerate(history):
                        with st.expander(f"{entry['question']} ({entry['category']})"):
                            st.write(f"**Dataset:** {entry['filename']}")
                            st.write(f"**Analysis:**\n{entry['explanation']}")
                            st.write(f"**Time:** {entry['execution_time']}")
                else:
                    st.info("No analysis history found.")
            except Exception as e:
                st.error(f"Error loading analysis history: {str(e)}")

else:
    st.info("No data loaded. Please select a data source.")

# Footer
st.markdown("---")
st.caption("@Copyright - Tata Consultancy Services | 2025")