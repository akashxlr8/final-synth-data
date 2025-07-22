# Intelligent Data Tools

This project is a Streamlit application that provides intelligent tools for working with your data. It leverages AI to generate test data and analyze datasets.

## Features

*   **Test Data Generator**: Generate synthetic test data based on user-defined scenarios. You can specify conditions in natural language, and the tool will generate relevant data. It supports both SQLite and Pandas DataFrames for data storage and manipulation.
*   **Data Analyzer**: Analyze your datasets to get insights. This tool provides basic statistics, visualizations, and an AI-powered analysis that can generate analytical questions and provide answers.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/intelligent-data-tools.git
    ```
2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Set up your API keys by creating a `.streamlit/secrets.toml` file with your credentials:
    ```toml
    [cohere]
    api_key = "your_actual_cohere_api_key"
    
    [azure]
    openai_endpoint = "https://your-resource.openai.azure.com/"
    openai_api_key = "your_actual_azure_api_key"
    deployment_name = "gpt-4o"
    
    [openai]
    api_key = "your_actual_openai_api_key"
    
    [app]
    debug_mode = false
    log_level = "INFO"
    ```
    
    **Important:** 
    - Replace placeholder values with your actual API keys
    - The `.streamlit/secrets.toml` file is already added to `.gitignore` for security
    - You only need to configure the API service you plan to use (Cohere, Azure OpenAI, or OpenAI)

## Usage

To run the application, use the following command:

```bash
streamlit run app.py
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
