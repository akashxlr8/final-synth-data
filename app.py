import streamlit as st
import pandas as pd
import sqlite3
import langchain_openai
import io
import json
from sdv.metadata import Metadata
import cohere
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os, time
import pickle
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer

# Load environment variables
load_dotenv()

from logging_config import get_logger
logger = get_logger(__name__)

# Initialize Azure OpenAI
# llm = AzureChatOpenAI(
#     azure_deployment="bfsi-genai-demo-gpt-4o",
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_version="2024-05-01-preview",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# )

from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage
llm = ChatCohere()

from prompts import (
    SQL_GENERATION_PROMPT,
    PANDAS_QUERY_PROMPT,
    TEST_DATA_GENERATION_SYSTEM_PROMPT,
    TEST_DATA_GENERATION_SQLITE_PROMPT,
    TEST_DATA_GENERATION_PANDAS_PROMPT
)

from db_functions import (
    load_from_sqlite,
    save_to_sqlite,
    create_replica_db,
    upload_to_db,
    execute_sql_on_db,
    execute_pandas_query,
    insert_test_data_incrementally,
    insert_synthetic_data_into_db,
    generate_data_structure,
    extract_sql_query
)

# call llm for writing sql query for test data extraction if exists in sqlite database
def call_llm_for_sql(user_prompt, table_name="customer", db_name="customer_testbed.db"):
    try:
        logger.info("Generating SQL query for test data extraction")
        # Generate table schema dynamically
        conn = sqlite3.connect(db_name)
        schema_query = f"PRAGMA table_info({table_name})"
        table_schema = pd.read_sql(schema_query, conn)
        logger.debug(f"Table schema: {table_schema}")
        conn.close()

        # Build schema description for the LLM
        schema_description = "\n".join(
            [
                f"{row['name']} ({row['type']})"
                for _, row in table_schema.iterrows()
            ]
        )

        # Construct the LLM prompt
        prompt = SQL_GENERATION_PROMPT.format(
            table_name=table_name,
            schema_description=schema_description,
            user_prompt=user_prompt
        )
        
### replace with azure chatgpt 4.0 code
        test_response = llm.invoke("Hello")
        print(test_response)
        response = llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        # Extract the SQL query from the response
        generated_sql = extract_sql_query(response.content)
        logger.info(f"Generated SQL: {generated_sql}")
        return generated_sql

    except Exception as e:
        logger.error(f"Error generating SQL: {e}")
        return None


# call llm for writing pandas query for test data extraction if exists in sqlite database
def call_llm_for_pandas_query(user_prompt, table_name="customer"):
    try:
        # Generate table schema dynamically
        # replica_df = load_from_pickle()
        logger.info("Loading data from sqlite database in the call_llm_for_pandas_query function")
        replica_df = load_from_sqlite()  # instead of load_from_pickle()

        schema_description = "\n".join(
            [
                f"{col} ({dtype})"
                for col, dtype in replica_df.dtypes.items()
            ]
        )

        # Construct the LLM prompt
        prompt = PANDAS_QUERY_PROMPT.format(
            table_name=table_name,
            schema_description=schema_description,
            user_prompt=user_prompt
        )
        
        response = llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        # Extract the Pandas query from the response
        generated_query = clean_pandas_query(response.content)
        logger.info(f"Generated Pandas query: {generated_query}")   
        return generated_query

    except Exception as e:
        logger.error(f"Error generating Pandas query: {e}")
        return None


# call llm for generating test data if data not exists in sqlite database
def call_llm_to_generate_test_data(test_condition, table_name, reference_dataset=None):
    try:
        if data_storage_option == "SQLite":
            # Generate data structure from the database
            # data_structure = generate_data_structure(table_name)

            # Construct the prompt
            system_prompt =f"""You are an advanced data generator. Based on the given conditions and structure, you need to create realistic test data in a tabular format. Follow these instructions strictly:"""
            user_prompt = TEST_DATA_GENERATION_SQLITE_PROMPT.format(
                test_condition=test_condition,
                reference_dataset=reference_dataset
            )
            # Call LLM to generate test data
            
            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            # Get the response text
            generated_text = response.content
            logger.info(f"Generated test data: {generated_text}")
            
            # Convert the CSV response into a pandas DataFrame
            test_data = pd.read_csv(io.StringIO(generated_text)) #type: ignore
            logger.debug(f"Test data: {test_data}")
        else:
            # Construct the LLM prompt for pandas DataFrame
            system_prompt =f"""You are an advanced data generator. Based on the given conditions and structure, you need to create realistic test data in a tabular format. Follow these instructions strictly:"""
            user_prompt = TEST_DATA_GENERATION_PANDAS_PROMPT.format(
                test_condition=test_condition,
                reference_dataset=reference_dataset
            )
            
            
            
            
            # print(user_prompt)
            # Call Cohere to generate test data
            ### replace with azure chatgpt 4.0 code
            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])

            # # Get the response text
            generated_text = response.content
            logger.debug(f"Generated test data: {generated_text}")
            # response = co.chat(
            #     model='command-r-plus-08-2024',
            #     messages = [{"role":"system","content":system_prompt},
            #                 {"role":"user","content":user_prompt}],
            
            # )

            # Get the response text
            # generated_text = response.message.content[0].text
            print(generated_text)
            # Convert the CSV response into a pandas DataFrame
        
            test_data = pd.read_csv(io.StringIO(generated_text)) #type: ignore

        return test_data

    except Exception as e:
        st.error(f"Error generating test data: {e}")
        return None

# Function to create SDV metadata
def create_sdv_metadata(column_types):
    metadata = {
        "columns": {}
    }
    for column, column_type in column_types.items():
        metadata["columns"][column] = {
            "sdtype": column_type
        }
    return metadata

# function to insert test data incrementally into the replica database after generating test data
def insert_test_data_incrementally(test_data_df, replica_db="customer_testbed.db", table_name="customer"):
    try:
        if data_storage_option == "SQLite":
            # SQLite path remains unchanged
            conn_replica = sqlite3.connect(replica_db)

            cursor = conn_replica.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [col[1] for col in cursor.fetchall()]
            
            test_data_df = test_data_df[columns]  # Ensure the columns match the table schema
            
            # Fetch existing data from the replica table
            query = f"SELECT * FROM {table_name}"
            existing_data_df = pd.read_sql(query, conn_replica)

            # Find new rows to insert (rows not already in the replica table)
            new_data_df = pd.concat([test_data_df, existing_data_df]).drop_duplicates(keep=False)

            if new_data_df.empty:
                print("No new rows to insert.")
            else:
                # Insert the new rows into the replica table
                new_data_df.to_sql(table_name, conn_replica, if_exists="append", index=False)
                print(f"Inserted {len(new_data_df)} new rows into '{table_name}'.")

            # Close the connection
            conn_replica.close()
            return True
        else:
            # For Pandas option, still use SQLite as source of truth
            conn_replica = sqlite3.connect(replica_db)
            
            # Get existing data from SQLite
            existing_data_df = pd.read_sql(f"SELECT * FROM {table_name}", conn_replica)
            
            # Process as before
            test_data_df = test_data_df[existing_data_df.columns]
            new_data_df = pd.concat([test_data_df, existing_data_df]).drop_duplicates(keep=False)
            
            if not new_data_df.empty:
                # Insert new rows back to SQLite
                new_data_df.to_sql(table_name, conn_replica, if_exists="append", index=False)
                print(f"Inserted {len(new_data_df)} new rows into '{table_name}'.")
            
            conn_replica.close()
        return True
    except Exception as e:
        print(f"Error inserting test data incrementally: {e}")
        return False

# function to insert synthetic data into the replica database
def insert_synthetic_data_into_db(data: pd.DataFrame):
    try:
        # Insert the synthetic data into the replica database
        success = insert_test_data_incrementally(data)  # You can reuse your earlier insert function
        if success:
            st.success("Synthetic data inserted into replica database successfully!")
        else:
            st.error("Failed to insert synthetic data into the replica database.")
    except Exception as e:
        st.error(f"Error inserting synthetic data: {e}")
        
# fucntions for SDV 

# function to train ctgan model
def train_ctgan_model(data: pd.DataFrame,metadata):
    # Initialize CTGAN model
    #model = CTGAN()
    #model = GaussianCopula()   
    # Train the model
    # metadata= metadata.load_from_json('metadata\my_metadata_v1_1737839165.json')
    synthesizer = CTGANSynthesizer(metadata)
    logger.info(f"Synthesizer: {synthesizer}")
    # synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)
    logger.info(f"Synthesizer trained: {synthesizer}")
    return synthesizer

# autodetect data types from the dataframe for SDV training
def auto_detect_meta_data(data: pd.DataFrame):
    """Detect metadata from a single DataFrame"""
    metadata = Metadata.detect_from_dataframe(
        data=data,
        table_name='data'  # Provide a table name for the metadata
    )
    metadata.save_to_json('data_meta1.json')
    logger.info(f"Metadata: {metadata}")
    return metadata

# function to create meta file for SDV training
def create_meta_file(data: pd.DataFrame):
    meta_data = {
        "columns": [
            {"name": col, "type": "categorical" if data[col].dtype.name == 'category' else "numerical"}
            for col in data.columns
        ],
        "num_rows": len(data)
    }
    with open('data_meta.json', 'w') as meta_file:
        json.dump(meta_data, meta_file, indent=4)
    logger.info(f"Meta data: {meta_data}")
    return meta_data   
        
# generate synthetic data using ctgan model
def generate_synthetic_data(ctgan_model, num_rows) -> pd.DataFrame:
    synthetic_data = ctgan_model.sample(num_rows)
    logger.debug(f"Synthetic data: {synthetic_data}")
    return synthetic_data      

# function to clean sql query
def clean_sql(sql_text):
    """
    Removes Markdown-style formatting from an SQL query.
    Handles cases where the text starts with ```sql or just ```.
    """
    # Remove ```sql or ``` from the beginning
    if sql_text.startswith("```sql"):
        sql_text = sql_text[len("```sql"):].strip()
    elif sql_text.startswith("```"):
        sql_text = sql_text[len("```"):].strip()
    
    # Remove trailing ``` if present
    if sql_text.endswith("```"):
        sql_text = sql_text[:-len("```")].strip()
    logger.debug(f"Cleaned SQL: {sql_text}")
    return sql_text              

def clean_pandas_query(query_text):
    """
    Removes Markdown-style formatting from a Pandas query.
    Handles cases where the text starts with ```python or just ```.

    Args:
        query_text (str): The Pandas query text.

    Returns:
        str: The cleaned Pandas query.
    """
    # Remove ```python or ``` from the beginning
    if query_text.startswith("```python"):
        query_text = query_text[len("```python"):].strip()
    elif query_text.startswith("```"):
        query_text = query_text[len("```"):].strip()
    
    # Remove trailing ``` if present
    if query_text.endswith("```"):
        query_text = query_text[:-len("```")].strip()
    logger.debug(f"Cleaned Pandas query: {query_text}")
    return query_text

import re



# Set page configuration for better layout
st.set_page_config(page_title="CSV to DB Manager", layout="wide")

# Initialize Cohere API
# co = cohere.ClientV2('xz6VhvXnfldENCPxSIWa19qEumptDOjH2tPoXA1F')  # Replace with your actual API key

# Page Title
st.title("Intelligent Test Data Generator")

# Sidebar for actions
with st.sidebar:
    st.header("Workflow steps")
    
    st.markdown("1. Load the data into database")
    st.markdown("2. Generate SQL from user test condition – using LLM")
    st.markdown("3. Fetching data from TestBed db using generated SQL")
    st.markdown("4. If records found, fetch data from test-bed database.")
    st.markdown("5. If no records found, generate test data using LLM.")
    st.markdown("6. Incrementally Insert Test data into Test-Bed database.")
    st.markdown("**For high volume synthetic data**")
    st.markdown("7. Column Mapping – Mark the categorical and numerical columns.")
    st.markdown("8, Train the statiscal model using LLM generated test data")
    st.markdown("9. Generate Synthetic Data – based on the trained model.")
    st.markdown("10, Incrementally Insert Test data into Test-Bed database.")

# Add a toggle button to select between SQLite and Pandas DataFrame
st.sidebar.subheader("Select Data Storage Option")
data_storage_option = st.sidebar.radio(
    "Choose the data storage option:",
    ("SQLite", "Pandas DataFrame")
)

# Step 1: File Upload Section
st.subheader("Load data File")
uploaded_file = st.file_uploader("📤 Upload your data file here:", type=["csv"], help="Upload a data file to populate the database.")

if uploaded_file:
    # Upload and process the file
    st.write("**File Uploaded Successfully!**")
    table_name = "customer"
    table_name, column_info, data = upload_to_db(uploaded_file)
    st.session_state.column_info = column_info

    if table_name:
        st.success(f"Data successfully inserted into table: `{table_name}`.")
        st.markdown("### Table Preview")
        st.dataframe(
            data,
            column_config={
                "acct_num": st.column_config.NumberColumn(format="%d"),
                "zip4_code": st.column_config.NumberColumn(format="%d"),
                "govt_issued_id": st.column_config.NumberColumn(format="%d"),
                "phone_number": st.column_config.NumberColumn(format="%d"),
                "postal_code": st.column_config.NumberColumn(format="%d"),
                "index": st.column_config.NumberColumn(format="%d"),
            },
            hide_index=True,
        )  # Show a preview of the table
        if data_storage_option == "Pandas DataFrame":
            
            st.success("Data saved to sqlite file AND READY for test data generation using pandas!")
        else:
            st.error("Failed to upload and process the file.")

# Step 2: Create Replica DB
if data_storage_option == "SQLite":
    st.subheader("Create Test Bed Database")
    if st.button("Create Test Bed Database"):
        success = create_replica_db()
        if success:
            st.success("TestBed database `customer_testbed.db` created successfully!")
        else:
            st.error("Failed to create Test Bed database.")

# Step 3: Column Mapping and Metadata
# if uploaded_file:
#     st.subheader("Step 3: Define Column Types")
#     if not column_info.empty:
#         column_types = {}
#         cols = st.columns(4)  # Display 4 columns per row
#         for index, column_name in enumerate(column_info["name"]):
#             col = cols[index % 4]  # Assign to one of the 4 columns
#             with col:
#                 column_type = st.selectbox(
#                     f"Type for `{column_name}`",
#                     ["categorical", "numerical", "boolean", "ID"],
#                     key=column_name
#                 )
#                 column_types[column_name] = column_type

#         # Save metadata
#         if st.button("Save Column Metadata"):
#             metadata = create_sdv_metadata(column_types)
#             st.json(metadata)
#             st.success("Column metadata saved successfully!")

# Step 4: SQL Generation
st.subheader("Describe your test case scenario")

user_prompt = st.text_area(
    "🔍 Describe your test condition (e.g., 'find all records where ccid is same for different LOBs')",
    help="write in english natural language to generate data ."
)

if st.button("Generate test data"):
    if user_prompt:
        st.info("Generating Query, please wait...")
        table_name ="customer"
        if data_storage_option == "SQLite":
            query_text = call_llm_for_sql(user_prompt, table_name)
            query = clean_sql(query_text)
            # Ensure query is not None before concatenation
            if query is not None:
                print(query)
                st.toast("Generated SQL query: " + query)
            else:
                st.warning("No SQL query generated.")
        else:
            query_text = call_llm_for_pandas_query(user_prompt, table_name)
            st.info(f"Generated Pandas query: {query_text} ")
            if query_text:
                query = clean_pandas_query(query_text)
                # Ensure query is a string before concatenation
                if isinstance(query, str):
                    st.info("Generated Pandas query: " + query)
                else:
                    st.warning("Generated query is not a valid string.")
            else:
                query = None # or some default value
                st.error("Failed to generate Pandas query. Please try again.")
        
        if query:
            st.subheader("Generated Query")
            st.code(query, language="sql" if data_storage_option == "SQLite" else "python")
            
            # Execute the Query
            st.write("**Executing Query in test-bed DB...**")            
            if data_storage_option == "SQLite":
                results = execute_sql_on_db(query)
            else:
                results = execute_pandas_query(query)
                
            print("testing......")
            print(results)
            if results is not None and not results.empty:
                st.success("Records found!")
                st.subheader("Search Results")
                st.dataframe(
                    results,
                    column_config={
                        "acct_num": st.column_config.NumberColumn(format="%d"),
                        "zip4_code": st.column_config.NumberColumn(format="%d"),
                        "govt_issued_id": st.column_config.NumberColumn(format="%d"),
                        "phone_number": st.column_config.NumberColumn(format="%d"),
                        "index": st.column_config.NumberColumn(format="%d"),
                    },
                    hide_index=True,
                )  # Show a preview 
                st.session_state.test_data_from_llm = results
            else:
                st.warning("No test data found! ")
                with st.spinner("Generating test data using AI... This might take a moment."):
                    if "test_data_generated" not in st.session_state:
                        st.session_state.test_data_generated = False

                    test_data = call_llm_to_generate_test_data(user_prompt, table_name)

                if test_data is not None:
                    st.session_state.test_data_generated = True
                    st.session_state.test_data_from_llm= test_data
                    st.subheader("Generated Test Data")
                        
                    st.dataframe(
                        test_data,
                        column_config={
                            "acct_num": st.column_config.NumberColumn(format="%d"),
                            "zip4_code": st.column_config.NumberColumn(format="%d"),
                            "govt_issued_id": st.column_config.NumberColumn(format="%d"),
                            "phone_number": st.column_config.NumberColumn(format="%d"),
                            "index": st.column_config.NumberColumn(format="%d"),
                        },
                        hide_index=True,
                    )  # Show a preview 
                    def convert_df_to_csv(df):
                        return df.to_csv(index=False).encode('utf-8')

                    csv_data = convert_df_to_csv(test_data)
                    # print(csv_data)
                    #save csv_data temporarily
                    test_data.to_csv("temp_data.csv",index=False)
                    try:
                          
                        success = insert_test_data_incrementally(test_data)

                        if success:
                            st.success("Test data inserted into Test Bed database successfully!")

                            # Fetch and display the incremental data
                            st.write("**Displaying newly inserted incremental test data...**")
                            if data_storage_option == "SQLite":
                                with st.spinner("Fetching newly inserted records..."):
                                    conn_replica = sqlite3.connect("customer_testbed.db")
                                    # Fetch the latest records added to the table
                                    query = f"SELECT * FROM customer ORDER BY ROWID DESC"
                                    incremental_data = pd.read_sql(query, conn_replica)
                                    conn_replica.close()

                                if not incremental_data.empty:
                                    st.subheader("Newly Inserted Incremental Test Data")
                                    st.dataframe(
                                        incremental_data.head(500),
                                        column_config={
                                            "acct_num": st.column_config.NumberColumn(format="%d"),
                                            "zip4_code": st.column_config.NumberColumn(format="%d"),
                                            "govt_issued_id": st.column_config.NumberColumn(format="%d"),
                                            "phone_number": st.column_config.NumberColumn(format="%d"),
                                            "postal_code": st.column_config.NumberColumn(format="%d"),
                                            "index": st.column_config.NumberColumn(format="%d"),                                                                       },
                                        hide_index=True,
                                    )  # Show a preview
                                else:
                                    st.warning("No new data found!")
                            else:
                                #fetch and display the replica data using pandas
                                with st.spinner("Fetching newly inserted records..."):
                                    replica_df = load_from_sqlite()
                                    
                                    if replica_df is not None and not replica_df.empty:
                                        st.subheader("The TestBed Data with newly inserted records")
                                        st.dataframe(
                                            replica_df.head(500),
                                            column_config={
                                                "acct_num": st.column_config.NumberColumn(format="%d"),
                                                "zip4_code": st.column_config.NumberColumn(format="%d"),
                                                "govt_issued_id": st.column_config.NumberColumn(format="%d"),
                                                "phone_number": st.column_config.NumberColumn(format="%d"),
                                                "postal_code": st.column_config.NumberColumn(format="%d"),
                                                "index": st.column_config.NumberColumn(format="%d"),                                                                       },
                                            hide_index=True,
                                        )  # Show a preview
                                    else:
                                        st.warning("No new data found!")
                        else:
                            st.error("Failed to  insert test data into replica database.")
                    except Exception as e:
                        st.error(f"Error processing or inrting test data: {e}")   
                    # Generate more than using the LLM generated  test data by calling the statistical model method
                    
        else:
            st.error("Failed to generate test data. Please try again.")
    else:
        st.warning("Please provide a condition to generate data.")
if st.button("Generate high volume Synthetic data using statistical model", key="more_data") :
    try:
#st.dataframe(synthetic_data)
        print("\n----generating more data----\n")
        test_data = st.session_state.test_data_from_llm
        metafile = Metadata.load_from_json(filepath="data_meta.json")
        
        # metafile.save_to_json("metadata1.json")
        try:
            with st.spinner("Training statistical model with AI generated test data."):
                ctgan_model = train_ctgan_model(test_data,metafile)
                with st.spinner("Generating synthetic data..."):
                    synthetic_data = generate_synthetic_data(ctgan_model, num_rows=100)
                    st.session_state.synthetic_data = synthetic_data
            st.success("Synthetic data generated successfully!")
            #st.dataframe(synthetic_data)
            st.dataframe(
                synthetic_data.head(459),
                column_config={
                    "acct_num": st.column_config.NumberColumn(format="%d"),
                    "zip4_code": st.column_config.NumberColumn(format="%d"),
                    "govt_issued_id": st.column_config.NumberColumn(format="%d"),
                    "phone_number": st.column_config.NumberColumn(format="%d"),
                    "postal_code": st.column_config.NumberColumn(format="%d"),
                    "index": st.column_config.NumberColumn(format="%d"),
                },
                hide_index=True,
            )  # Show a preview 
        except Exception as e:
            st.error(f"Error training statistical model: {e}")
    except Exception as e:
        st.error(f"Error generating synthetic data: {e}")

# Display generated synthetic data
if "synthetic_data" in st.session_state:
    st.subheader("New Synthetic Data:")
    st.dataframe(st.session_state.synthetic_data)
    st.download_button(
        label="Download Generated Data",
        data=st.session_state.synthetic_data.to_csv(index=False),
        file_name=f"synthetic_data_{int(time.time())}.csv",
        mime="text/csv",
        key="download_more_data"
    )

# Footer
st.markdown("---")
st.caption("@Copyright - Tata Consultqancy Services | 2025")
