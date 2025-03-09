import pandas as pd
import sqlite3
import io

# functions related to working with sqlite database
def load_from_sqlite(table_name="customer", db_name="customer.db"):
    """
    Load data from SQLite database into a pandas DataFrame
    
    Args:
        table_name (str): Name of the table to load
        db_name (str): Name of the database file
        
    Returns:
        pd.DataFrame: Data from the table
    """
    conn = sqlite3.connect(db_name)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def save_to_sqlite(df, table_name="customer", db_name="customer.db", if_exists="replace"):
    """
    Save DataFrame to SQLite database
    
    Args:
        df (pd.DataFrame): DataFrame to save
        table_name (str): Name of the table
        db_name (str): Name of the database file
        if_exists (str): How to behave if the table already exists
        
    Returns:
        None
    """
    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    conn.close()
    
# function to create a replica of the original database
def create_replica_db(original_db="customer.db", replica_db="customer_testbed.db"):
    """
    Create a replica of the original database
    
    Args:
        original_db (str): Name of the original database file
        replica_db (str): Name of the replica database file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Always connect to both databases
        conn_original = sqlite3.connect(original_db)
        conn_replica = sqlite3.connect(replica_db)

        # Fetch all table names from the original database
        cursor_original = conn_original.cursor()
        cursor_original.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor_original.fetchall()

        # Copy data for each table
        for table_name_tuple in tables:
            table_name = table_name_tuple[0]

            # Read data from the original table
            data = pd.read_sql_query(f"SELECT * FROM {table_name}", conn_original)

            # Insert data into the replica database
            data.to_sql(table_name, conn_replica, if_exists="replace", index=False)

        # Close connections
        conn_original.close()
        conn_replica.close()
        return True
    except Exception as e:
        print(f"Error creating replica database: {e}")
        return False


def upload_to_db(file, db_name="customer.db"):
    """
    Upload CSV file to SQLite database
    
    Args:
        file: File-like object containing CSV data
        db_name (str): Name of the database file
        
    Returns:
        tuple: (table_name, column_info, data) if successful, (None, None, None) otherwise
    """
    try:
        # Read CSV file into DataFrame
        df = pd.read_csv(file)
        
        # Always connect to SQLite database regardless of option
        conn = sqlite3.connect(db_name)

        # Insert data into the database
        table_name = "customer"
        
        # Convert all string data to lowercase
        df = df.map(lambda x: x.lower() if isinstance(x, str) else x)
        
        df.to_sql(table_name, conn, if_exists="replace", index=False)

        # Fetch table info for display
        query = f"PRAGMA table_info({table_name})"
        column_info = pd.read_sql(query, conn)
        data = pd.read_sql(f"SELECT * FROM {table_name}", conn)

        conn.close()
        
        return table_name, column_info, data
    except Exception as e:
        print(f"Error uploading file to database: {e}")
        return None, None, None
    
# Function to execute SQL on the database
def execute_sql_on_db(sql_query, db_name="customer_testbed.db"):
    """
    Execute SQL query on the database
    
    Args:
        sql_query (str): SQL query to execute
        db_name (str): Name of the database file
        
    Returns:
        pd.DataFrame: Result of the query or None if error
    """
    try:
        conn = sqlite3.connect(db_name)
        results = pd.read_sql(sql_query, conn)
        conn.close()
        
        print(results)
        return results
    except Exception as e:
        print(f"Error executing SQL: {e}")
        return None


def execute_pandas_query(query_text, df=None):
    """
    Execute a pandas query on a DataFrame
    
    Args:
        query_text (str): Pandas query to execute
        df (pd.DataFrame, optional): DataFrame to execute query on. If None, loads from SQLite
        
    Returns:
        pd.DataFrame: Result of the query or None if error
    """
    try:
        # Load from SQLite instead of pickle
        df = load_from_sqlite("customer", "customer_testbed.db") if df is None else df
        
        # Debug print to see what data we're working with
        print("DataFrame loaded from database:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First row: {df.iloc[0].to_dict() if not df.empty else 'No data'}")
        
        if df is None or df.empty:
            print("No data found to execute the query.")
            return pd.DataFrame()
        
        # Make sure column names in query match the actual column names (lowercase)
        # Convert all column names to lowercase in the query
        lower_cols = {col: col.lower() for col in df.columns}
        query_text_lower = query_text
        
        # Replace column references with lowercase versions
        for col, lower_col in lower_cols.items():
            # Replace df['Column'] with df['column'] pattern
            query_text_lower = query_text_lower.replace(f"df['{col}']", f"df['{lower_col}']")
            query_text_lower = query_text_lower.replace(f"df[\"{col}\"]", f"df[\"{lower_col}\"]")
            # Replace df.Column with df.column pattern
            query_text_lower = query_text_lower.replace(f"df.{col}", f"df.{lower_col}")
        
        print(f"Original query: {query_text}")
        print(f"Modified query: {query_text_lower}")
        
        # Try to execute the modified query
        local_namespace = {"df": df}
        try:
            results = eval(query_text_lower, {}, local_namespace)
            
            # If results is a boolean mask, apply it to get the filtered DataFrame
            if isinstance(results, pd.Series) and results.dtype == bool:
                results = df[results]
                
            print(f"Query results: {results.shape if isinstance(results, pd.DataFrame) else results}")
            
            # If results modify the data, save back to SQLite
            if isinstance(results, pd.DataFrame) and results is not df:
                print(f"Saving results to SQLite: {results.shape}")
            
            return results if isinstance(results, pd.DataFrame) else None
            
        except Exception as inner_e:
            print(f"Error evaluating modified query: {inner_e}")
            # As a fallback, try to execute the original query
            try:
                results = eval(query_text, {}, local_namespace)
                
                # If results is a boolean mask, apply it to get the filtered DataFrame
                if isinstance(results, pd.Series) and results.dtype == bool:
                    results = df[results]
                    
                return results if isinstance(results, pd.DataFrame) else None
            except Exception as original_query_e:
                print(f"Error evaluating original query: {original_query_e}")
                return pd.DataFrame()
    
    except Exception as e:
        print(f"Error executing Pandas query: {e}")
        return pd.DataFrame()

# function to insert test data incrementally into the replica database after generating test data
def insert_test_data_incrementally(test_data_df, data_storage_option="SQLite", replica_db="customer_testbed.db", table_name="customer"):
    """
    Insert test data incrementally into the replica database
    
    Args:
        test_data_df (pd.DataFrame): Test data to insert
        data_storage_option (str): "SQLite" or "Pandas DataFrame"
        replica_db (str): Name of the replica database file
        table_name (str): Name of the table
        
    Returns:
        bool: True if successful, False otherwise
    """
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
def insert_synthetic_data_into_db(data: pd.DataFrame, data_storage_option="SQLite"):
    """
    Insert synthetic data into the replica database
    
    Args:
        data (pd.DataFrame): Synthetic data to insert
        data_storage_option (str): "SQLite" or "Pandas DataFrame"
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Insert the synthetic data into the replica database
        success = insert_test_data_incrementally(data, data_storage_option)
        return success
    except Exception as e:
        print(f"Error inserting synthetic data: {e}")
        return False
        
# function to generate data structure for test data generation
def generate_data_structure(table_name, db_name="customer.db"):
    """
    Generate data structure for test data generation
    
    Args:
        table_name (str): Name of the table
        db_name (str): Name of the database file
        
    Returns:
        list: List of strings describing the data structure
    """
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(db_name)
        query = f"PRAGMA table_info({table_name})"
        column_info = pd.read_sql(query, conn)
        conn.close()

        # Build the data structure
        data_structure = []
        for _, row in column_info.iterrows():
            column_name = row["name"]
            data_type = "String" if row["type"] in ["TEXT", "VARCHAR"] else \
                        "Integer" if row["type"] in ["INTEGER", "INT"] else \
                        "Boolean" if row["type"] in ["BOOLEAN"] else \
                        "Float" if row["type"] in ["REAL", "FLOAT"] else "String"
            example = "Example: " + ("Alice" if data_type == "String" else "25" if data_type == "Integer" else "True" if data_type == "Boolean" else "10.5")
            data_structure.append(f"- {column_name} ({data_type}, {example})")
        return data_structure
    except Exception as e:
        print(f"Error generating data structure: {e}")
        return None
    
import re
def extract_sql_query(response_text):
    # Define a regex pattern to match SQL queries
    pattern = re.compile(r"(SELECT\s.*?;)", re.IGNORECASE | re.DOTALL)
    match = pattern.search(response_text)
    if match:
        query = match.group(1).strip()
        # Add case-insensitive comparison for text fields
        query = query.replace("WHERE country = 'INDIA'", "WHERE LOWER(country) = 'india'")
        query = query.replace("WHERE country = 'India'", "WHERE LOWER(country) = 'india'")
        # Add more replacements for other common fields if needed
        return query
    else:
        return None

def delete_databases(db_names=["customer.db", "customer_testbed.db"]):
    """
    Delete specified databases
    
    Args:
        db_names (list): List of database file names to delete
        
    Returns:
        tuple: (success, message) where success is a boolean and message is a string
    """
    try:
        import os
        deleted = []
        not_found = []
        
        for db_name in db_names:
            if os.path.exists(db_name):
                try:
                    # Close any open connections first
                    conn = sqlite3.connect(db_name)
                    conn.close()
                except:
                    pass
                
                os.remove(db_name)
                deleted.append(db_name)
            else:
                not_found.append(db_name)
        
        if deleted and not not_found:
            return True, f"Successfully deleted: {', '.join(deleted)}"
        elif deleted and not_found:
            return True, f"Deleted: {', '.join(deleted)}. Not found: {', '.join(not_found)}"
        else:
            return False, f"Databases not found: {', '.join(not_found)}"
    except Exception as e:
        return False, f"Error deleting databases: {e}"