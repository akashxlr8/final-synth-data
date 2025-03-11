import streamlit as st
import pandas as pd
from llm_analyzer import DatasetAnalyzer
from logging_config import get_logger
from llm_with_tool import CodeEnabledLLM
from db import AnalysisDatabase

logger = get_logger("app")

def main():
    st.title('CSV File Analyzer with AI Insights')
    
    # Initialize the analyzer
    analyzer = DatasetAnalyzer()
    
    # Initialize the database
    db = AnalysisDatabase()
    
    # File upload widget
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
    logger.info(f"Uploaded file: {uploaded_file.name}")
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # After reading the CSV file
            dataset_id = db.save_dataset(
                filename=uploaded_file.name,
                row_count=df.shape[0],
                column_count=df.shape[1],
                columns=df.columns.tolist()
            )
            st.session_state.current_dataset_id = dataset_id
            
            # Display basic information about the dataset
            st.subheader('Dataset Info')
            st.write(f'Number of rows: {df.shape[0]}')
            st.write(f'Number of columns: {df.shape[1]}')
            
            # Display column names
            st.subheader('Columns')
            st.write(df.columns.tolist())
            
            # Preview the data
            st.subheader('Data Preview')
            st.dataframe(df.head())
            
            # Basic statistics
            st.subheader('Statistical Summary')
            st.write(df.describe())
            
            # AI-Generated Questions
            st.subheader('AI-Generated Analysis Questions')
            with st.spinner('Generating questions about your dataset...'):
                questions = analyzer.generate_questions(df)
                for i, q in enumerate(questions, 1):
                    with st.expander(f"{i}. {q.question}"):
                        st.write(f"**Category:** {q.category}")
                        st.write(f"**Reasoning:** {q.reasoning}")
            
            # After generating questions
            for q in questions:
                question_id = db.save_question(
                    dataset_id=st.session_state.current_dataset_id,
                    question=q.question,
                    category=q.category,
                    reasoning=q.reasoning
                )
                # Store question_id for later use
                q.db_id = question_id
            
            # After showing the AI-generated questions, add a custom question input
            st.subheader('Add Your Own Analysis Question')
            custom_question = st.text_input("Enter your own analytical question:")
            custom_category = st.selectbox(
                "Select a category for your question:",
                ['Trend Analysis', 'Correlation', 'Distribution', 'Outliers', 'Pattern Recognition', 'Custom']
            )
            custom_reasoning = st.text_area("Explain your reasoning for this question:")
            add_question = st.button("Add Question")

            if add_question and custom_question:
                # Create a custom AnalyticalQuestion object
                from llm_analyzer import AnalyticalQuestion
                custom_q = AnalyticalQuestion(
                    question=custom_question,
                    category=custom_category or "Custom",
                    reasoning=custom_reasoning or "No Reasoning Provided"
                )
                # Append to the list of questions
                if 'questions' not in st.session_state:
                    st.session_state.questions = questions
                st.session_state.questions.append(custom_q)
                st.success("Question added successfully!")
                
                # When adding custom questions
                question_id = db.save_question(
                    dataset_id=st.session_state.current_dataset_id,
                    question=custom_question,
                    category=custom_category or "Category Not Specified",
                    reasoning=custom_reasoning,
                    is_custom=True
                )
                custom_q.db_id = question_id
                
            # Display all questions (including custom ones)
            st.subheader('All Analysis Questions')
            questions_to_display = st.session_state.get('questions', questions)
            for i, q in enumerate(questions_to_display, 1):
                with st.expander(f"{i}. {q.question}"):
                    st.write(f"**Category:** {q.category}")
                    st.write(f"**Reasoning:** {q.reasoning}")
            
            # After displaying all questions, add a button to analyze with LLM+Calculator
            st.subheader('Analyze Questions with AI')

            # Save current dataset ID in session state
            if 'current_dataset_id' not in st.session_state and dataset_id:
                st.session_state.current_dataset_id = dataset_id

            # Fetch questions from database
            try:
                # Load all questions for this dataset from DB
                db_questions = db.get_dataset_questions(st.session_state.current_dataset_id)
                
                # Convert DB questions to AnalyticalQuestion objects
                from llm_analyzer import AnalyticalQuestion
                questions_from_db = []
                for q_data in db_questions:
                    q_obj = AnalyticalQuestion(
                        question=q_data['question'],
                        category=q_data['category'],
                        reasoning=q_data['reasoning'],
                        db_id=q_data['id']
                    )
                    questions_from_db.append(q_obj)
                
                st.session_state.db_questions = questions_from_db
                
                # Define the dropdown and button
                selected_question = st.selectbox(
                    "Select a question to analyze with AI:",
                    [q.question for q in questions_from_db]
                )
                analyze_button = st.button("Analyze with Code-enabled AI")

                if analyze_button and selected_question:
                    try:
                        # Get the selected question object
                        selected_q = next(q for q in questions_from_db 
                                         if q.question == selected_question)
                        
                        # Initialize the Code-enabled LLM
                        calculator_llm = CodeEnabledLLM()
                        
                        with st.spinner("Analyzing with AI and calculator..."):
                            # Pass the dataset and question to the LLM with calculator
                            analysis = calculator_llm.analyze_question(df, selected_q.question, selected_q.category)
                            
                            # Display the analysis result in an expandable section
                            with st.expander("Analysis Result", expanded=True):
                                st.markdown(analysis)
                                st.info("This analysis was performed by an AI with access to calculation tools.")
                        
                        # After getting analysis result
                        if hasattr(analysis, 'code') and hasattr(analysis, 'explanation'):
                            db.save_code_analysis(
                                question_id=selected_q.db_id,
                                code=analysis.code,
                                explanation=analysis.explanation,
                                result=""
                            )
                        else:
                            # If analysis is a string or has a different structure
                            db.save_code_analysis(
                                question_id=selected_q.db_id,
                                code=str(analysis),
                                explanation="",
                                result=""
                            )
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
            except Exception as e:
                st.error(f"Error loading questions from database: {str(e)}")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == '__main__':
    main()
