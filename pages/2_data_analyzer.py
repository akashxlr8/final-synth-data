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

# Initialize uploaded_file as None first
uploaded_file = None

# Sidebar options
with st.sidebar:
    st.header("Data Source")
    data_source = st.radio(
        "Choose data source:",
        ("Customer Database", "Customer TestBed Database", "Upload CSV")
    )
    
    if data_source == "Upload CSV":
        # Now the uploaded_file variable is accessible outside the sidebar
        uploaded_file = st.file_uploader("📤 Upload your data file:", type=["csv"])

# Main content
def load_data(source):
    global uploaded_file  # Access the global variable
    
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
        if uploaded_file is not None:  # Check the global variable
            try:
                return pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                return None
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
                                # The regex needs to be more comprehensive to capture the entire JSON object
                                json_match = re.search(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}', response.content, re.DOTALL)
                                
                                if json_match:
                                    json_str = json_match.group(0)  # Changed from group(1) to group(0) to get the entire match
                                    
                                    # Clean up any extra characters that might interfere with JSON parsing
                                    json_str = json_str.strip()
                                    
                                    # Try parsing the JSON
                                    try:
                                        response_json = json.loads(json_str)
                                        
                                        if "code" in response_json and "explanation" in response_json:
                                            # Don't display raw content anymore, just show the formatted parts
                                            st.markdown("### Generated Code")
                                            st.code(response_json["code"], language="python")
                                            
                                            st.markdown("### Explanation")
                                            st.write(response_json["explanation"])
                                            
                                            # Save the parsed code to the database
                                            if "dataset_id" in st.session_state and hasattr(q, "db_id"):
                                                db.save_code_analysis(
                                                    question_id=q.db_id,
                                                    code=response_json["code"],
                                                    explanation=response_json["explanation"],
                                                    result=""
                                                )
                                            
                                            # Add a button to run the code on the current data
                                            if st.button("Run Analysis", key=f"run_analysis_{i}"):
                                                try:
                                                    with st.spinner("Running analysis..."):
                                                        # Create a local environment with the dataframe and necessary libraries
                                                        import plotly.express as px
                                                        import plotly.graph_objects as go
                                                        
                                                        local_namespace = {
                                                            "df": df, 
                                                            "plt": plt, 
                                                            "np": np, 
                                                            "pd": pd, 
                                                            "sns": sns,
                                                            "px": px,
                                                            "go": go,
                                                            "st": st
                                                        }
                                                        
                                                        # Modify code to remove fig.show() if present
                                                        code_to_execute = response_json["code"]
                                                        code_to_execute = code_to_execute.replace("fig.show()", "")
                                                        
                                                        # Execute the code
                                                        exec(code_to_execute, {}, local_namespace)
                                                        
                                                        # Display status
                                                        st.success("Analysis completed successfully")
                                                        
                                                        # Flag to track if we've displayed a visualization
                                                        displayed_viz = False
                                                        
                                                        # Check for plotly figure in the namespace
                                                        if 'fig' in local_namespace:
                                                            st.subheader("Analysis Visualization")
                                                            st.plotly_chart(local_namespace['fig'], use_container_width=True)
                                                            displayed_viz = True
                                                        
                                                        # Check for other plotly figures with different names
                                                        for var_name, var_value in local_namespace.items():
                                                            if not displayed_viz and str(type(var_value)).find("plotly.graph_objs") > -1:
                                                                st.subheader("Analysis Visualization")
                                                                st.plotly_chart(var_value, use_container_width=True)
                                                                displayed_viz = True
                                                                break
                                                        
                                                        # Also check for matplotlib figures (as a fallback)
                                                        if not displayed_viz:
                                                            for var_name, var_value in local_namespace.items():
                                                                if str(type(var_value)).find("matplotlib.figure.Figure") > -1:
                                                                    st.pyplot(var_value)
                                                                    displayed_viz = True
                                                                    break
                                                        
                                                        # If the code produces a result variable, display it
                                                        if "result" in local_namespace:
                                                            st.subheader("Analysis Result")
                                                            result = local_namespace["result"]
                                                            
                                                            # Handle different result types
                                                            if isinstance(result, pd.DataFrame):
                                                                st.dataframe(result)
                                                            elif not displayed_viz and "plotly" in str(type(result)):
                                                                st.plotly_chart(result, use_container_width=True)
                                                            elif not displayed_viz and "matplotlib" in str(type(result)):
                                                                st.pyplot(result)
                                                            else:
                                                                st.write(result)
                                                        
                                                        # If no visualization or result was displayed, look for printed output
                                                        if not displayed_viz and "result" not in local_namespace:
                                                            import io
                                                            import sys
                                                            
                                                            # Capture print output
                                                            old_stdout = sys.stdout
                                                            new_stdout = io.StringIO()
                                                            sys.stdout = new_stdout
                                                            
                                                            # Re-execute to capture prints
                                                            exec(code_to_execute, {}, local_namespace)
                                                            
                                                            # Get printed output
                                                            output = new_stdout.getvalue()
                                                            sys.stdout = old_stdout
                                                            
                                                            if output.strip():
                                                                st.subheader("Output")
                                                                st.code(output, language="text")
                                                
                                                except Exception as run_err:
                                                    st.error(f"Error running the analysis: {str(run_err)}")
                                                    import traceback
                                                    st.code(traceback.format_exc(), language="python")
                                        else:
                                            st.warning("Generated response isn't in the expected format. Missing 'code' or 'explanation' fields.")
                                    except json.JSONDecodeError as json_err:
                                        st.error(f"Failed to parse JSON from response: {json_err}")
                                        st.code(json_str, language="json")  # Show the problematic JSON string
                                else:
                                    # If no JSON format is detected, display the raw content
                                    st.warning("The response doesn't contain properly formatted JSON.")
                                    st.markdown("### Raw Response")
                                    st.markdown(response.content)
                            except Exception as e:
                                st.error(f"Error processing response: {str(e)}")
        
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