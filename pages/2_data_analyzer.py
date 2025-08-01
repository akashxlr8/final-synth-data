import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from langchain_cohere import ChatCohere

from pages.analyzer.llm_analyzer import DatasetAnalyzer, AnalyticalQuestion
from pages.analyzer.db import AnalysisDatabase
from prompts import CODE_GENERATOR_PROMPT
from secrets_utils import get_cohere_api_key, display_secrets_status

# Initialize analyzer and database
analyzer = DatasetAnalyzer()
db = AnalysisDatabase()

from logging_config import get_logger
logger = get_logger(__name__)

# Initialize LLM with Streamlit secrets
try:
    cohere_api_key = get_cohere_api_key()
    if cohere_api_key and cohere_api_key != "your_cohere_api_key_here":
        llm = ChatCohere(cohere_api_key=cohere_api_key)
        logger.info("Initialized ChatCohere with API key")
    else:
        # Create a mock LLM for demo purposes
        class MockLLM:
            def invoke(self, messages):
                from types import SimpleNamespace
                import json
                
                # Create a basic analysis response
                response = {
                    "code": """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Basic data overview
print("Dataset Shape:", df.shape)
print("\\nColumn Names:", df.columns.tolist())
print("\\nData Types:")
print(df.dtypes)

# Basic statistics
print("\\nBasic Statistics:")
print(df.describe())

# Create a simple visualization
if len(df.columns) > 0:
    plt.figure(figsize=(10, 6))
    if df[df.columns[0]].dtype in ['int64', 'float64']:
        df[df.columns[0]].hist(bins=30)
        plt.title(f'Distribution of {df.columns[0]}')
        plt.show()
    else:
        df[df.columns[0]].value_counts().head(10).plot(kind='bar')
        plt.title(f'Top 10 values in {df.columns[0]}')
        plt.xticks(rotation=45)
        plt.show()""",
                    "explanation": "This code provides basic exploratory data analysis including dataset overview, statistics, and a simple visualization based on the first column's data type."
                }
                
                response_obj = SimpleNamespace()
                response_obj.content = json.dumps(response)
                return response_obj
        
        llm = MockLLM()
        logger.warning("Using mock LLM - please configure Cohere API key for full functionality")
except Exception as e:
    logger.error(f"Error initializing LLM: {e}")
    # Fallback to mock implementation
    class MockLLM:
        def invoke(self, messages):
            from types import SimpleNamespace
            import json
            response = {"code": "print('LLM not available')", "explanation": "LLM initialization failed"}
            response_obj = SimpleNamespace()
            response_obj.content = json.dumps(response)
            return response_obj
    llm = MockLLM()

from db_functions import load_from_sqlite

# Add helper function for Arrow compatibility
def convert_dtypes_for_arrow(df):
    """Convert DataFrame types to be compatible with Arrow."""
    # Convert Int64 to standard int64 to avoid Arrow serialization issues
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]) and hasattr(df[col].dtype, 'name') and 'Int64' in df[col].dtype.name:
            df[col] = df[col].astype('int64')
    return df

# Load data function - defined at the top for clarity
def load_data(source):
    """Load data from various sources and save it to the database."""
    try:
        if source == "Upload CSV":
            uploaded_file = st.session_state.get("uploaded_file", None)
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                dataset_name = "Uploaded CSV"
            else:
                return None
        elif source == "Customer Database":
            df = load_from_sqlite(table_name="customer", db_name="customer.db")
            dataset_name = "Customer Database"
        else:  # Customer TestBed Database
            df = load_from_sqlite(table_name="customer", db_name="customer_testbed.db")
            dataset_name = "Customer TestBed Database"
        
        # Save dataset to database if it's not empty
        if df is not None and not df.empty:
            try:
                dataset_id = db.save_dataset(
                    filename=dataset_name,
                    row_count=df.shape[0],
                    column_count=df.shape[1],
                    columns=df.columns.tolist()
                )
                st.session_state.dataset_id = dataset_id
                return df
            except Exception as e:
                st.error(f"Error saving dataset to database: {str(e)}")
                return df  # Still return the dataframe even if DB save fails
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Initialize key session state variables if they don't exist
if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None
if "ai_questions" not in st.session_state:
    st.session_state.ai_questions = []

# Set page configuration
st.set_page_config(page_title="Data Analyzer", layout="wide")

# Main layout
st.title("AI Data Analysis Assistant")

# Display secrets configuration status
display_secrets_status()

# Create two main columns for better organization
col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Data Input")
    data_source = st.radio(
        "Choose data source:",
        ("Upload CSV", "Customer Database", "Customer TestBed Database")
    )
    
    # Handle file upload if CSV is selected
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "📤 Upload your CSV file",
            type=["csv"],
            help="Upload a CSV file to analyze"
        )
        # Store in session state for consistency
        st.session_state.uploaded_file = uploaded_file
    
    # Load data based on source selection using our unified function
    df = load_data(data_source)

    # Show data preview
    if df is not None and not df.empty:
        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        st.info(f"Total rows: {len(df)} | Columns: {df.shape[1]}")
    elif df is not None and df.empty:
        st.warning("The loaded dataset is empty!")

with col2:
    st.header("2. AI Analysis")
    
    if df is not None:
        # Generate Questions Button
        if st.button("🤖 Generate Analysis Questions", 
                    key="generate_questions",
                    help="Click to generate AI-powered analytical questions"):
            with st.spinner("AI is analyzing your dataset..."):
                try:
                    questions = analyzer.generate_questions(df)
                    st.session_state.ai_questions = questions
                except Exception as e:
                    st.error(f"Error generating questions: {str(e)}")

        # Display Questions and Analysis
        if "ai_questions" in st.session_state:
            for i, question in enumerate(st.session_state.ai_questions, 1):
                with st.expander(f"Q{i}: {question.question}", expanded=True):
                    st.write(f"**Category:** {question.category}")
                    st.write(f"**Reasoning:** {question.reasoning}")
                    
                    # Generate Code Button
                    if st.button(f"📊 Generate Analysis Code", key=f"gen_code_{i}"):
                        with st.spinner("Generating analysis code..."):
                            try:
                                # Prepare data summary
                                data_summary = f"""
                                Data Summary:
                                - Total records: {len(df)}
                                - Question: {question.question}
                                - Category: {question.category}
                                - Question Reasoning: {question.reasoning}
                                
                                The first few rows of the data:
                                {df.head(5).to_string()}
                                """
                                
                                # Format the prompt
                                prompt = CODE_GENERATOR_PROMPT.format(
                                    df_preview=df.head().to_string(),
                                    question=question.question,
                                    category=question.category
                                )
                                
                                # Call LLM for analysis
                                response = llm.invoke([
                                    {"role": "system", "content": prompt},
                                    {"role": "user", "content": data_summary}
                                ])
                                
                                # Parse the response content as JSON
                                try:
                                    import json
                                    import re
                                    
                                    # Try to find and extract JSON content from the response
                                    json_match = re.search(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}', str(response.content), re.DOTALL)
                                    
                                    if json_match:
                                        json_str = json_match.group(0)
                                        response_json = json.loads(json_str)
                                        
                                        if "code" in response_json and "explanation" in response_json:
                                            # Display the code and explanation
                                            st.markdown("### Generated Code")
                                            st.code(response_json["code"], language="python")
                                            
                                            st.markdown("### Explanation")
                                            st.write(response_json["explanation"])
                                            
                                            # Save to database if we have the IDs
                                            if "dataset_id" in st.session_state and hasattr(question, "db_id"):
                                                db.save_code_analysis(
                                                    question_id=question.db_id,
                                                    code=response_json["code"],
                                                    explanation=response_json["explanation"],
                                                    result=""
                                                )
                                            
                                            # Add Run Analysis button
                                            if st.button("▶️ Run Analysis", key=f"run_gen_code_{i}"):
                                                with st.spinner("Running analysis..."):
                                                    try:
                                                        # Create a local environment with necessary libraries
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
                                                        
                                                        # Remove fig.show() if present
                                                        code_to_execute = response_json["code"].replace("fig.show()", "")
                                                        
                                                        # Execute the code
                                                        exec(code_to_execute, {}, local_namespace)
                                                        
                                                        # Flag to track if we've displayed a visualization
                                                        displayed_viz = False
                                                        
                                                        # Display the results
                                                        if 'fig' in local_namespace:
                                                            st.plotly_chart(local_namespace['fig'], use_container_width=True)
                                                            displayed_viz = True
                                                        
                                                        # Check for other plotly figures with different names
                                                        if not displayed_viz:
                                                            for var_name, var_value in local_namespace.items():
                                                                if str(type(var_value)).find("plotly.graph_objs") > -1:
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
                                                        
                                                        st.success("Analysis completed successfully")
                                                        
                                                    except Exception as run_err:
                                                        st.error(f"Error running the analysis: {str(run_err)}")
                                                        import traceback
                                                        st.code(traceback.format_exc(), language="python")
                                        else:
                                            st.warning("Generated response isn't in the expected format. Missing 'code' or 'explanation' fields.")
                                    else:
                                        st.warning("Could not extract valid JSON from the response.")
                                    st.code(str(response.content), language="text")
                                    
                                except json.JSONDecodeError as json_err:
                                    st.error(f"Failed to parse JSON from response: {json_err}")
                                    st.code(str(response.content), language="text")
                                    
                            except Exception as e:
                                st.error(f"Error generating code: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc(), language="python")
    else:
        st.info("👆 Please upload a CSV file or select a data source to begin analysis")

# Add a sidebar for additional options and history
with st.sidebar:
    st.header("Analysis History")
    try:
        # Get history data
        history = db.get_analysis_history()
        if history:
            st.success(f"Found {len(history)} analyses in history")
            for i, item in enumerate(history):
                with st.expander(f"{i+1}. {item['question'][:30]}...", expanded=False):
                    st.markdown(f"**Dataset:** {item['filename']}")
                    st.markdown(f"**Category:** {item['category']}")
                    st.markdown(f"**Time:** {item['execution_time']}")
                    if item['explanation']:
                        st.markdown("**Explanation:**")
                        st.markdown(item['explanation'])
                    if item['code']:
                        # Use a checkbox instead of a nested expander
                        show_code = st.checkbox(f"Show code for analysis #{i+1}", key=f"show_code_{i}")
                        if show_code:
                            st.code(item['code'], language="python")
        else:
            st.info("No analysis history found")
            
        # Debug section to help troubleshoot
        with st.expander("Debug Info", expanded=False):
            st.write(f"Current dataset_id: {st.session_state.dataset_id}")
            st.write(f"Number of AI questions: {len(st.session_state.get('ai_questions', []))}")
            st.markdown("**Database counts:**")
            st.markdown("- Datasets: 22")
            st.markdown("- Questions: 10")
            st.markdown("- Analyses: 6")
    except Exception as e:
        st.error(f"Error loading sidebar history: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="text")

# Main content
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
        # Convert dtypes to be compatible with Arrow
        dtypes_df = convert_dtypes_for_arrow(dtypes_df)  
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
                sns.histplot(x=df[col_to_plot].dropna(), kde=True, ax=ax)
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
        
        # Generate questions button
        if st.button("🤖 Generate Analysis Questions", key="generate_ai_questions"):
            with st.spinner("AI is analyzing your dataset to generate insightful questions..."):
                try:
                    # Generate questions
                    questions = analyzer.generate_questions(df)
                    st.session_state.ai_questions = questions
                    
                    # Save questions to database if we have a dataset ID
                    if st.session_state.dataset_id is not None:
                        for q in questions:
                            try:
                                # Make sure dataset_id is an integer before saving
                                dataset_id = int(st.session_state.dataset_id)
                                question_id = db.save_question(
                                    dataset_id=dataset_id,
                                    question=q.question,
                                    category=q.category,
                                    reasoning=q.reasoning
                                )
                                # Store question_id for later use
                                q.db_id = question_id
                            except Exception as save_err:
                                st.error(f"Error saving question: {str(save_err)}")
                    
                    # Force a rerun to show the saved questions
                    st.rerun()
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
                                    explanation=str(response.content),  # Convert to string
                                    result=""
                                )

                            # First display the raw response content
                            st.markdown(response.content)

                            # Try to parse the response content as JSON to extract code and explanation
                            try:
                                import json
                                import re
                                
                                # Try to find and extract JSON content from the response
                                json_match = re.search(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}', str(response.content), re.DOTALL)
                                
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
                                            if st.button("Run Analysis", key=f"run_ai_analysis_{i}"):
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
                if "dataset_id" in st.session_state and st.session_state.dataset_id is not None:
                    try:
                        dataset_id = int(st.session_state.dataset_id)
                        question_id = db.save_question(
                            dataset_id=dataset_id,
                            question=custom_question,
                            category=custom_category or "Custom",
                            reasoning=custom_reasoning or "No reasoning provided",
                            is_custom=True
                        )
                        custom_q.db_id = question_id
                    except Exception as e:
                        st.error(f"Error saving custom question: {str(e)}")
                
                st.success("Question added successfully!")
                st.rerun()  # Use st.rerun instead of st.experimental_rerun
            else:
                st.warning("Please enter a question to add.")
                
        # Show analysis history if available
        if "dataset_id" in st.session_state:
            st.subheader("Analysis History")
            try:
                # First, get the recent datasets
                datasets = db.get_recent_datasets(limit=10)  # Adjust limit as needed
                
                if datasets:
                    # Create a selectbox to choose the dataset
                    dataset_names = [f"{d['filename']} ({d['upload_time']})" for d in datasets]
                    dataset_index = st.selectbox("Select dataset to view history:", 
                                                range(len(dataset_names)), 
                                                format_func=lambda i: dataset_names[i])
                    
                    selected_dataset = datasets[dataset_index]
                    
                    # Get questions for the selected dataset
                    questions = db.get_dataset_questions(selected_dataset['id'])
                    
                    if questions:
                        st.write(f"### Questions for: {selected_dataset['filename']}")
                        st.write(f"**Uploaded:** {selected_dataset['upload_time']}")
                        st.write(f"**Rows:** {selected_dataset['row_count']} | **Columns:** {selected_dataset['column_count']}")
                        
                        # Group questions by category
                        from collections import defaultdict
                        questions_by_category = defaultdict(list)
                        
                        for q in questions:
                            questions_by_category[q['category']].append(q)
                        
                        # Create tabs for each category
                        category_tabs = st.tabs(list(questions_by_category.keys()))
                        
                        # Display questions in each category tab
                        for i, (category, category_questions) in enumerate(questions_by_category.items()):
                            with category_tabs[i]:
                                for j, q in enumerate(category_questions):
                                    with st.expander(f"{j+1}. {q['question']}"):
                                        st.write(f"**Reasoning:** {q['reasoning']}")
                                        
                                        # Get analyses for this question
                                        analyses = db.get_question_analyses(q['id'])
                                        if analyses:
                                            for k, analysis in enumerate(analyses):
                                                st.write(f"**Analysis {k+1} ({analysis['execution_time']}):**")
                                                if analysis['explanation']:
                                                    st.write(analysis['explanation'])
                                                if analysis['code']:
                                                    # Use a checkbox instead of a potential nested expander
                                                    show_code = st.checkbox(f"Show code #{k+1}", key=f"history_code_{q['id']}_{k}")
                                                    if show_code:
                                                        st.code(analysis['code'], language="python")
                                                if analysis['result']:
                                                    st.write("**Result:**")
                                                    st.write(analysis['result'])
                                        else:
                                            st.info("No analyses found for this question.")
                    else:
                        st.info(f"No questions found for dataset '{selected_dataset['filename']}'.")
                else:
                    st.info("No datasets found in history.")
            except Exception as e:
                st.error(f"Error loading analysis history: {str(e)}")
                st.code(traceback.format_exc())

else:
    st.info("No data loaded. Please select a data source.")

# Footer
st.markdown("---")
st.caption("@Copyright - akashxlr8 | 2025")