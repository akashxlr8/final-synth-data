# SQL generation prompt
SQL_GENERATION_PROMPT = """
You are a SQL expert. Generate a valid SQL query based on the following conditions and table schema. Follow these instructions strictly:

1. The table name is `{table_name}`.
2. The schema of the table is as follows:
{schema_description}

3. **User Prompt**:
{user_prompt}

4. Ensure that the SQL query is valid and strictly adheres to the schema.
5. Only return the SQL query. Do not include explanations or extra text.

**Example**:
User Prompt: "Select all rows where Age is greater than 30 and Subscription Type is 'Premium'."
Table Schema:
- Name (TEXT)
- Age (INTEGER)
- Gender (TEXT)
- Country (TEXT)
- Subscription Type (TEXT)
- Active User (BOOLEAN)

Output:
SELECT * FROM customer WHERE Age > 30 AND Subscription_Type = 'Premium';
"""

# Pandas query prompt
PANDAS_QUERY_PROMPT = """
You are a Pandas expert. Generate a valid Pandas query based on the following conditions and table schema. Follow these instructions strictly:

1. The DataFrame name is `df`.
2. The schema of the DataFrame is as follows:
{schema_description}

3. **User Prompt**:
{user_prompt}

4. Ensure that the Pandas query is valid and strictly adheres to the schema.
5. Only return the Pandas query. Do not include explanations or extra text.

**Example**:
User Prompt: "Select all rows where Age is greater than 30 and Subscription Type is 'Premium'."
DataFrame Schema:
- Name (TEXT)
- Age (INTEGER)
- Gender (TEXT)
- Country (TEXT)
- Subscription Type (TEXT)
- Active User (BOOLEAN)

Output:
df[(df['Age'] > 30) & (df['Subscription_Type'] == 'Premium')]
"""

# Test data generation system prompt
TEST_DATA_GENERATION_SYSTEM_PROMPT = """You are an advanced data generator. Based on the given conditions and structure, you need to create realistic test data in a tabular format. Follow these instructions strictly:"""

# Test data generation user prompt for SQLite - dynamic version
TEST_DATA_GENERATION_SQLITE_PROMPT = """        
1. **Conditions**: Use the following test condition provided by the user:
{test_condition}

2. **Table Schema**:
{schema_description}

3. **Reference Dataset Sample**:
{reference_data}

4. **Output Requirements**:
- Generate a minimum of 10 rows.
- Look at the reference dataset sample and generate data with similar patterns and value ranges.
- When you generate records, maintain consistency with the existing data patterns.
- Each column's data should respect the data type and format shown in the reference sample.
- Ensure that the test data adheres strictly to the condition provided.
- Return the data only in CSV format.
- Maintain the order of CSV header as given in the example.
- Index/ID field should be unique. Never duplicate these values.

Now, generate realistic test data that strictly adheres to the given conditions. Return only CSV data that can be directly loaded into pandas. No backticks, no explanations, and no extra text like ```sql, python, code, etc.```.
"""

# Test data generation user prompt for Pandas DataFrame - dynamic version
TEST_DATA_GENERATION_PANDAS_PROMPT = """
Follow these instructions strictly:

1. **Conditions**: Use the following test condition provided by the user:
{test_condition}

2. **DataFrame Schema**:
{schema_description}

3. **Reference Dataset Sample**:
{reference_data}

4. **Output Requirements**:
- Generate a minimum of 10 rows.
- Look at the reference dataset sample and generate data with similar patterns and value ranges.
- When you generate records, maintain consistency with the existing data patterns.
- Each column's data should respect the data type and format shown in the reference sample.
- Ensure that the test data adheres strictly to the condition provided.
- Return the data only in CSV format.
- Maintain the order of CSV header as given in the example.
- Index/ID field should be unique. Never duplicate these values.

Now, generate realistic test data that strictly adheres to the given conditions. Return only CSV data that can be directly loaded into pandas. No backticks, no explanations, and no extra text like ```sql, python, code, etc.```.
"""


CODE_GENERATOR_PROMPT = """
            You are an expert data analyst with advanced Python skills, especially working with the Pandas library and Plotly for visualization. Your objective is to analyze a dataset, provided as a Pandas DataFrame glimpse, to answer a specific question within a given category.

            Dataset glimpse (DataFrame.head()):
            {df_preview}

            Task:

            Write valid Python code that performs the analysis required to answer the following question:
            "{question}"
            
            Important guidelines:
            1. Always use Plotly for visualizations instead of matplotlib or seaborn
            2. Include necessary imports (e.g., import pandas as pd, import plotly.express as px)
            3. For plots, use plotly.express or plotly.graph_objects
            4. Store your final plot in a variable named 'fig'
            5. Do not include fig.show() in your code as we will display it with Streamlit
            
            Provide a short explanation (one to two sentences) summarizing what the code does.
            Return your answer strictly in JSON format with the following structure (do not include any additional text outside the JSON object):
            {{
            "code": "your valid Python code here",
            "explanation": "brief explanation of what the code does"
            }}

            Example:
            {{
            "code": "import pandas as pd\\nimport plotly.express as px\\n\\n# Calculate the results\\nresult = df.groupby('category')['value'].mean()\\n\\n# Create a plotly visualization\\nfig = px.bar(result.reset_index(), x='category', y='value', title='Average Value by Category')\\n",
            "explanation": "Groups the data by category, calculates the mean value for each group, and creates a bar chart to visualize the results."
            }}

            Category: "{category}"
"""