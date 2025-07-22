import streamlit as st

# Set page configuration for better layout
st.set_page_config(page_title="AI Data Tools", layout="wide")

# Function to check if secrets are configured
def check_secrets_configuration():
    """Check if the required API keys are configured in secrets.toml"""
    try:
        cohere_key = st.secrets.get("cohere", {}).get("api_key", "")
        if cohere_key == "your_cohere_api_key_here" or not cohere_key:
            st.warning("⚠️ Please configure your API keys in `.streamlit/secrets.toml` for full functionality.")
            with st.expander("How to configure secrets"):
                st.markdown("""
                1. Create or edit `.streamlit/secrets.toml` in your project root
                2. Add your API keys:
                ```toml
                [cohere]
                api_key = "your_actual_cohere_api_key"
                
                [azure]
                openai_endpoint = "https://your-resource.openai.azure.com/"
                openai_api_key = "your_actual_azure_key"
                ```
                3. Restart the application
                """)
            return False
    except Exception:
        st.info("💡 Configure API keys in `.streamlit/secrets.toml` for AI features.")
        return False
    return True

# Main page content
st.title("Intelligent Data Tools")

# Check secrets configuration
check_secrets_configuration()

# Home page content
st.markdown("""
# Welcome to AI Data Tools

This application provides intelligent tools for working with your data:

### 📊 Test Data Generator
Generate test data using AI based on your specifications.

### 📈 Data Analyzer
Analyze your data and get insights.

Use the sidebar to navigate between different tools.
""")

# Footer
st.markdown("---")
st.caption("@Copyright - akashxlr8 | 2025")