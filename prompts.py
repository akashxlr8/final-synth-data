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

# Test data generation user prompt for SQLite
TEST_DATA_GENERATION_SQLITE_PROMPT = """        
1. **Conditions**: Use the following test condition provided by the user:
{test_condition}

3. **reference dataset**:
index,acct_num,ccid,zip4_code,lobs,govt_issued_id,phone_number,first_name,last_name,address_line_one,city_name,state_code,postal_code,country
1234566,9100055032276,CC00000001,Null,RETAIL,3340783723534,2175557570,LOIS,WILLIAMS,936 CARDINAL LANE,CHAMPAIGN,IL,61820,UNITED STATES
1234567,9100050039685,CC00000002,56799,CARD,4413416413453,9185552699,JULIE,BELL,3714 HENRY FORD AVENUE,AFTON,OK,74331,UNITED STATES
1234568,5189413110708,CC00000003,Null,WEATH MANAGEMENT,9093134663453,5595550732,JESUS,MARESCOT,178 CARLING CT,SAN JOSE,CA,95111,UNITED STATES
1234569,54241810713641,CC00000004,Null,CORPORATE,4357890873453,Null,KING,MAKER,13TH ST,PLANO,TX,75074,UNITED STATES
1234570,517908006006,CC00000005,Null,CARD,50202221712312,7015555173,SUSAN,BLAKELY,3056 COURTRIGHT STREET,DICKINSON,ND,58601,UNITED STATES
1234571,5179080064702,CC00000006,67564,MORGAGE,5524154083242,45676549,ERICA,HWANG,486 DUCK CREEK ROAD,SAN FRANCISCO,CA,94107,UNITED STATES
1234572,5424181321715,CC00000007,9466,WEATH MANAGEMENT,3700047723423,78987675,MARCELLA,MCDONNELL,12242 E WHITMORE AVE,HUGHSON,CA,95326,UNITED STATES
1234573,5285001306647,CC00000008,2715,RETAIL,3423423422342,Null,SAM,SAMY,6500 CAMPUS CIRCLE DR E,IRVING,TX,75063,UNITED STATES
1234574,5424185127468,CC00000009,2127,CORPORATE,7467698866572,Null,NIH,FYTF HIUHIU,2887 MARK TWAIN DR,FARMERS BRANCH,TX,75234,UNITED STATES
1234575,9100055619353,CC00000010,Null,CARD,765220380223,4805550065,KRISTIE,COOPER,3924 ELMWOOD AVENUE,PHOENIX,AZ,85003,UNITED STATES
1234576,5179080561458,CC00000011,Null,WEATH MANAGEMENT,7510525336786,8085558139,SOO,CHRISTENSON,4996 ARRON SMITH DRIVE,HONOLULU,HI,96814,UNITED STATES
1234577,5189410433475,CC00000012,1910,CARD,9314259531231,Null,APRIL,SANCHEZ,10623 N OSCEOLA DR,WESTMINSTER,CO,80031,UNITED STATES
1234578,51790005549197,CC00000013,166,MORGAGE,34560000053,Null,SEE,SAMM,3323 BALCONES DR,IRVING,TX,75063,UNITED STATES
1234579,9100057952612,CC00000014,Null,CORPORATE,3685820105756,2485556850,VANESSA,WILLIAMSON,2967 CORPENING DRIVE,PONTIAC,MI,48342,UNITED STATES


4. **Output Requirements**:
- Generate a minimum of 10 rows.
- always look at the reference dataset and generate data within records value.
- when you generate records, you must maintain the same value for others fileds except some filds data which is given in user prompt
- learn the pattern and generate data as instructed.But maintain records data for CCID
- keep acct_num,govt_issued_id , zip4_code lenght and format same as exmple input
- Ensure that the test data adheres strictly to the condition provided.        
- Return the data only in CSV format.
- maintain the order of csv header as given example
- index is a unique fileds. never duplicate the value. keep maintain the exaple format data


5. **Example Output**:
index,acct_num,ccid,zip4_code,lobs,govt_issued_id,phone_number,first_name,last_name,address_line_one,city_name,state_code,postal_code,country
1234566,9100055032276,CC00000001,Null,RETAIL,3340783723534,2175557570,LOIS,WILLIAMS,936 CARDINAL LANE,CHAMPAIGN,IL,61820,UNITED STATES
1234567,9100050039685,CC00000002,56799,CARD,4413416413453,9185552699,JULIE,BELL,3714 HENRY FORD AVENUE,AFTON,OK,74331,UNITED STATES
1234568,5189413110708,CC00000003,Null,WEATH MANAGEMENT,9093134663453,5595550732,JESUS,MARESCOT,178 CARLING CT,SAN JOSE,CA,95111,UNITED STATES

Now, generate realistic test data that strictly adheres to the given conditions. Return only CSV data that can be directly loaded into pandas. No  backticks, no explanations, and no extra text like ```sql, python, code, etc.```.
"""

# Test data generation user prompt for Pandas DataFrame
TEST_DATA_GENERATION_PANDAS_PROMPT = """

    
    Follow these instructions strictly:
    1. **Conditions**: Use the following test condition provided by the user:
{test_condition}

    2. Generate a minimum of 10 rows.
    - Generate a minimum of 10 rows.
    - always look at the reference dataset and generate data within records value.
    - when you generate records, you must maintain the same value for others fileds except some filds data which is given in user prompt
    - learn the pattern and generate data as instructed.But maintain records data for CCID
    - keep acct_num,govt_issued_id , zip4_code lenght and format same as exmple input
    - Ensure that the test data adheres strictly to the condition provided.        
    - Return the data only in CSV format.
    - maintain the order of csv header as given example
    - index is a unique fileds. never duplicate the value. keep maintain the exaple format data

    {reference_dataset}
    Reference dataset:
    index,acct_num,ccid,zip4_code,lobs,govt_issued_id,phone_number,first_name,last_name,address_line_one,city_name,state_code,postal_code,country
    1234566,9100055032276,CC00000001,Null,RETAIL,3340783723534,2175557570,LOIS,WILLIAMS,936 CARDINAL LANE,CHAMPAIGN,IL,61820,UNITED STATES
    1234567,9100050039685,CC00000002,56799,CARD,4413416413453,9185552699,JULIE,BELL,3714 HENRY FORD AVENUE,AFTON,OK,74331,UNITED STATES
    1234568,5189413110708,CC00000003,Null,WEATH MANAGEMENT,9093134663453,5595550732,JESUS,MARESCOT,178 CARLING CT,SAN JOSE,CA,95111,UNITED STATES
   
    5. **Example Output**:
        index,acct_num,ccid,zip4_code,lobs,govt_issued_id,phone_number,first_name,last_name,address_line_one,city_name,state_code,postal_code,country
        1234566,9100055032276,CC00000001,Null,RETAIL,3340783723534,2175557570,LOIS,WILLIAMS,936 CARDINAL LANE,CHAMPAIGN,IL,61820,UNITED STATES
        1234567,9100050039685,CC00000002,56799,CARD,4413416413453,9185552699,JULIE,BELL,3714 HENRY FORD AVENUE,AFTON,OK,74331,UNITED STATES
        1234568,5189413110708,CC00000003,Null,WEATH MANAGEMENT,9093134663453,5595550732,JESUS,MARESCOT,178 CARLING CT,SAN JOSE,CA,95111,UNITED STATES
    
    Now, generate realistic test data that strictly adheres to the given conditions. Return only CSV data that can be directly loaded into pandas. No  backticks, no explanations, and no extra text like ```sql, python, code, etc.```.
"""