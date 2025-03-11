from langchain_cohere import ChatCohere
# from langchain_openai import AzureChatOpenAI
# from langchain.tools import Tool
# from langchain_experimental.utilities import PythonREPL
from langchain.output_parsers import PydanticOutputParser

import os, json
from dotenv import load_dotenv
import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional

from logging_config import get_logger
logger = get_logger("llm_with_tool")
load_dotenv()

class CodeOutput(BaseModel):
    code: str = Field(description="Valid Python code to execute")
    explanation: Optional[str] = Field(description="Explanation of the code")

class CodeEnabledLLM:
    def __init__(self):
        # Initialize Azure OpenAI client
        # self.llm = AzureChatOpenAI(
        #     azure_deployment="bfsi-genai-demo-gpt-4o",
        #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        #     api_version="2024-05-01-preview",
        #     temperature=0,
        #     max_tokens=None,
        #     timeout=None,
        #     max_retries=2,
        # )

        self.llm = ChatCohere()
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
