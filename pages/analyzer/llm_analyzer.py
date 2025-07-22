from langchain_cohere import ChatCohere
# from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os
from typing import List
import pandas as pd
import streamlit as st
from secrets_utils import get_cohere_api_key, get_azure_config

from logging_config import get_logger
logger = get_logger("llm_analyzer")

class AnalyticalQuestion(BaseModel):
    question: str = Field(description="The analytical question about the dataset")
    category: str = Field(description="Category of the question (e.g., 'Trend Analysis', 'Correlation', 'Distribution', 'Outliers', 'Pattern Recognition')")
    reasoning: str = Field(description="Brief explanation of why this question is relevant")
    db_id: int = Field(default=None, description="Database ID for the question")

class DatasetQuestions(BaseModel):
    questions: List[AnalyticalQuestion] = Field(description="List of 5 analytical questions about the dataset")

class DatasetAnalyzer:
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
                        
                        # Mock analytical questions
                        mock_questions = {
                            "questions": [
                                {"question": "What is the distribution of values in the first column?", "category": "descriptive", "reasoning": "Understanding data distribution is fundamental"},
                                {"question": "Are there any missing values in the dataset?", "category": "quality", "reasoning": "Data quality assessment is crucial"},
                                {"question": "What are the key statistics for numerical columns?", "category": "descriptive", "reasoning": "Basic statistical overview"},
                                {"question": "Are there any outliers in the data?", "category": "quality", "reasoning": "Outlier detection helps identify data issues"},
                                {"question": "What patterns exist in the data?", "category": "exploratory", "reasoning": "Pattern identification for insights"}
                            ]
                        }
                        
                        response_obj = SimpleNamespace()
                        response_obj.content = json.dumps(mock_questions)
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
                    mock_questions = {"questions": [{"question": "Basic data analysis", "category": "general", "reasoning": "LLM not available"}]}
                    response_obj = SimpleNamespace()
                    response_obj.content = json.dumps(mock_questions)
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
        
        self.parser = PydanticOutputParser(pydantic_object=DatasetQuestions)
        
    def _create_dataset_summary(self, df: pd.DataFrame) -> str:
        """Create a summary of the dataset for the LLM"""
        summary = f"""
        Columns in the dataset: {', '.join(df.columns.tolist())}
        Number of rows: {df.shape[0]}
        
        Sample data (first 5 rows):
        {df.head().to_string()}
        
        Statistical summary:
        {df.describe().to_string()}
        
        Data types:
        {df.dtypes.to_string()}
        """
        logger.debug(f"Dataset summary: {summary}")
        return summary

    def generate_questions(self, df: pd.DataFrame) -> List[AnalyticalQuestion]:
        """Generate insightful questions based on the dataset"""
        
        dataset_summary = self._create_dataset_summary(df)
        
        prompt = ChatPromptTemplate.from_template("""
        You are a data analyst examining a dataset. Based on the following dataset summary, 
        generate 5 specific, analytical questions that can be answered using this data. 
        Focus on patterns, relationships, trends, and interesting insights that could be derived.

        Dataset Summary:
        {dataset_summary}

        {format_instructions}

        Generate exactly 5 questions, each with a different category from: 
        'Trend Analysis', 'Correlation', 'Distribution', 'Outliers', 'Pattern Recognition'.
        Make sure each question is specific to the data provided and includes clear reasoning.
        """)
        
        messages = prompt.format_messages(
            dataset_summary=dataset_summary,
            format_instructions=self.parser.get_format_instructions()
        )
        
        try:
            logger.info("Generating analytical questions...")
            output = self.llm.invoke(messages)
            
            
            parsed_output = self.parser.parse(str(output.content))
            
            logger.debug(f"Generated questions: {parsed_output.questions}")
            return parsed_output.questions
        except Exception as e:
            print(f"Error generating questions: {str(e)}")
            logger.error(f"Error generating questions: {str(e)}")
            return [AnalyticalQuestion(
                question="Error generating questions. Please try again.",
                category="Error",
                reasoning="An error occurred during processing"
            )]

    # Add a method to allow adding custom questions
    def add_custom_question(self, question: str, category: str, reasoning: str) -> AnalyticalQuestion:
        """Create a custom analytical question"""
        return AnalyticalQuestion(
            question=question,
            category=category,
            reasoning=reasoning
        )