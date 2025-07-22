"""
Utility functions for managing Streamlit secrets configuration
"""
import streamlit as st
from typing import Optional, Dict, Any
from logging_config import get_logger

logger = get_logger(__name__)

def get_secret(section: str, key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Safely get a secret value from Streamlit secrets.
    
    Args:
        section: The section name in secrets.toml (e.g., 'cohere', 'azure')
        key: The key name within the section (e.g., 'api_key')
        default: Default value if secret is not found
    
    Returns:
        The secret value or default if not found
    """
    try:
        return st.secrets.get(section, {}).get(key, default)
    except Exception as e:
        logger.warning(f"Could not access secret {section}.{key}: {e}")
        return default

def get_cohere_api_key() -> Optional[str]:
    """Get Cohere API key from secrets"""
    return get_secret("cohere", "api_key")

def get_azure_config() -> Dict[str, Any]:
    """Get Azure OpenAI configuration from secrets"""
    try:
        azure_config = st.secrets.get("azure", {})
        return {
            "openai_endpoint": azure_config.get("openai_endpoint"),
            "openai_api_key": azure_config.get("openai_api_key"),
            "deployment_name": azure_config.get("deployment_name", "gpt-4o")
        }
    except Exception as e:
        logger.warning(f"Could not access Azure config: {e}")
        return {}

def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from secrets"""
    return get_secret("openai", "api_key")

def validate_secrets() -> Dict[str, bool]:
    """
    Validate that required secrets are configured properly.
    
    Returns:
        Dictionary with validation results for each service
    """
    validation_results = {}
    
    # Check Cohere API key
    cohere_key = get_cohere_api_key()
    validation_results["cohere"] = (
        cohere_key is not None and 
        cohere_key != "your_cohere_api_key_here" and 
        len(cohere_key.strip()) > 0
    )
    
    # Check Azure configuration
    azure_config = get_azure_config()
    validation_results["azure"] = (
        azure_config.get("openai_endpoint") is not None and
        azure_config.get("openai_api_key") is not None and
        azure_config["openai_endpoint"] != "https://your-resource-name.openai.azure.com/" and
        azure_config["openai_api_key"] != "your_azure_openai_api_key_here"
    )
    
    # Check OpenAI API key
    openai_key = get_openai_api_key()
    validation_results["openai"] = (
        openai_key is not None and 
        openai_key != "your_openai_api_key_here" and 
        len(openai_key.strip()) > 0
    )
    
    return validation_results

def display_secrets_status():
    """Display the current status of secrets configuration in Streamlit"""
    validation_results = validate_secrets()
    
    st.subheader("🔐 API Configuration Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if validation_results["cohere"]:
            st.success("✅ Cohere API")
        else:
            st.error("❌ Cohere API")
    
    with col2:
        if validation_results["azure"]:
            st.success("✅ Azure OpenAI")
        else:
            st.error("❌ Azure OpenAI")
    
    with col3:
        if validation_results["openai"]:
            st.success("✅ OpenAI")
        else:
            st.error("❌ OpenAI")
    
    if not any(validation_results.values()):
        st.warning("⚠️ No API keys are configured. Please update `.streamlit/secrets.toml`")
        
        with st.expander("📝 How to configure secrets"):
            st.markdown("""
            1. Create or edit `.streamlit/secrets.toml` in your project root
            2. Add your API keys:
            ```toml
            [cohere]
            api_key = "your_actual_cohere_api_key"
            
            [azure]
            openai_endpoint = "https://your-resource.openai.azure.com/"
            openai_api_key = "your_actual_azure_key"
            deployment_name = "gpt-4o"
            
            [openai]
            api_key = "your_actual_openai_api_key"
            ```
            3. Restart the application
            """)
