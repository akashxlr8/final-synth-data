from langchain_cohere import ChatCohere
# from langchain_openai import AzureChatOpenAI
# from langchain.tools import Tool
# from langchain_experimental.utilities import PythonREPL
from langchain.output_parsers import PydanticOutputParser

import os, json
import streamlit as st
import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional
from secrets_utils import get_cohere_api_key, get_azure_config

from logging_config import get_logger
logger = get_logger("llm_with_tool")

class CodeOutput(BaseModel):
    code: str = Field(description="Valid Python code to execute")
    explanation: Optional[str] = Field(description="Explanation of the code")

class CodeEnabledLLM:
    def __init__(self):
        # Initialize LLM with Streamlit secrets
        try:
            cohere_api_key = get_cohere_api_key()
            if cohere_api_key and cohere_api_key != "your_cohere_api_key_here":
                self.llm = ChatCohere(cohere_api_key=cohere_api_key)
                logger.info("Initialized ChatCohere with API key")
            else:
                # Create a mock LLM for demo purposes
                class MockLLM:
                    def invoke(self, messages):
                        from types import SimpleNamespace
                        import json
                        
                        # Mock code analysis response
                        response = {
                            "code": """import pandas as pd
import matplotlib.pyplot as plt

# Basic analysis
print("Dataset Info:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Simple visualization
if len(df.columns) > 0:
    first_col = df.columns[0]
    if df[first_col].dtype in ['int64', 'float64']:
        plt.figure(figsize=(8, 6))
        df[first_col].hist(bins=20)
        plt.title(f'Distribution of {first_col}')
        plt.show()
    else:
        print(f"Value counts for {first_col}:")
        print(df[first_col].value_counts().head())""",
                            "explanation": "This code performs basic exploratory data analysis including dataset overview and visualization based on column types."
                        }
                        
                        response_obj = SimpleNamespace()
                        response_obj.content = json.dumps(response)
                        return response_obj
                
                self.llm = MockLLM()
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
            self.llm = MockLLM()
            
        # Alternative Azure OpenAI implementation using secrets
        # try:
        #     azure_config = get_azure_config()
        #     if azure_config.get("openai_endpoint") and azure_config.get("openai_api_key"):
        #         self.llm = AzureChatOpenAI(
        #             azure_deployment=azure_config.get("deployment_name", "gpt-4o"),
        #             azure_endpoint=azure_config["openai_endpoint"],
        #             api_key=azure_config["openai_api_key"],
        #             api_version="2024-05-01-preview",
        #             temperature=0,
        #             max_tokens=None,
        #             timeout=None,
        #             max_retries=2,
        #         )
        # except Exception as e:
        #     logger.warning(f"Could not initialize Azure OpenAI: {e}")
            
        # Alternative Azure OpenAI implementation using secrets
        # try:
        #     azure_config = st.secrets.get("azure", {})
        #     if azure_config.get("openai_endpoint") and azure_config.get("openai_api_key"):
        #         self.llm = AzureChatOpenAI(
        #             azure_deployment="bfsi-genai-demo-gpt-4o",
        #             azure_endpoint=azure_config["openai_endpoint"],
        #             api_key=azure_config["openai_api_key"],
        #             api_version="2024-05-01-preview",
        #             temperature=0,
        #             max_tokens=None,
        #             timeout=None,
        #             max_retries=2,
        #         )
        # except Exception as e:
        #     logger.warning(f"Could not initialize Azure OpenAI: {e}")
        
        # self.llm=self.llm.with_structured_output(CodeOutput)
        
        # Initialize Python REPL tool
        # self.python_repl = PythonREPL()

        # # Define the tool
        # self.tools = [
        #     Tool(
        #         name="python_repl",
        #         func=self.python_repl.run,
        #         description="Useful for executing python code. Use this to perform data analysis, calculations, and any other tasks that require code execution. Input should be a valid python code snippet."
        #     )
        # ]

        # Initialize the parser
        self.parser = PydanticOutputParser(pydantic_object=CodeOutput)
        # self.llm=self.llm.bind_tools(tools=self.tools)
    def analyze_question(self, df: pd.DataFrame, question: str, category: str) -> CodeOutput:
        """Analyze a question by generating and executing Python code."""
        try:
            # Get a glimpse of the dataset (head)
            df_head = df.head().to_string()

            # Construct the prompt for the LLM
            prompt = f"""
            You are an expert data analyst with advanced Python skills, especially working with the Pandas library. Your objective is to analyze a dataset, provided as a Pandas DataFrame glimpse, to answer a specific question within a given category.

            Dataset glimpse (DataFrame.head()):
            {df.head().to_string()}

            Task:

            Write valid Python code that performs the analysis required to answer the following question:
            "{question}"
            Ensure your code includes any necessary imports (e.g., import pandas as pd) and is ready to run.
            Provide a short explanation (one to two sentences) summarizing what the code does.
            Return your answer strictly in JSON format with the following structure (do not include any additional text outside the JSON object):
            {{
            "code": "your valid Python code here",
            "explanation": "brief explanation of what the code does"
            }}

            Example:
            {{
            "code": "import pandas as pd\nresult = (df['col1'].mean() * df['col2'].sum())\nprint(result)",
            "explanation": "Calculates the mean of col1 and multiplies it by the sum of col2, then prints the result."
            }}

            Category: "{category}"
            """

            # Invoke the LLM with the prompt
            response = self.llm.invoke(prompt)

            print(response)
            # Parse the output using the PydanticOutputParser
            # parsed_output = self.parser.parse(str(response.content))
            # return parsed_output
            return response.content 
        except Exception as e:
            return CodeOutput(code="Error during analysis", explanation=str(e))

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    # Create a sample DataFrame
    data = {'col1': [1, 2, 3, 4, 5],
            'col2': [6, 7, 8, 9, 10]}
    df = pd.DataFrame(data)

    # Initialize the CodeEnabledLLM
    code_llm = CodeEnabledLLM()

    # Example question
    question = "What is the average of col1 multiplied by the sum of col2?"
    category = "Mathematical Calculation"

    # Analyze the question
    analysis_result = code_llm.analyze_question(df, question, category)
    print("Analysis Result:", analysis_result)
