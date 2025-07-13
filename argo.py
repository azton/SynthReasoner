# argo.py
import requests
import os
from typing import List, Optional, Dict
from smolagents import Model, ChatMessage, CodeAgent, tool
from datetime import datetime

# Set your ARGO_USER environment variable in a .env file or manually set it here
ARGO_USER = None

ARGO_MODELS = {
    'gpt4o': 'GPT-4o via Argo',
    'gpt4olatest': 'GPT-4o Latest via Argo',
    'gpto4mini': 'GPT-o4 Mini via Argo',
    'gpt41': 'GPT-4.1 via Argo',
    'claudeopus4': 'Claude Opus 4 via Argo', 
    'claudesonnet4': 'Claude Sonnet 4 via Argo',
    'claudesonnet37': 'Claude Sonnet 3.7 via Argo',
    'gemini25flash': 'Gemini 2.5 Flash via Argo',
    'gpt35': 'GPT-3.5 via Argo',
    'gpt35large': 'GPT-3.5 Large via Argo', 
    'gpt4': 'GPT-4 via Argo',
    'gpt4large': 'GPT-4 Large via Argo',
    'gpt4turbo': 'GPT-4 Turbo via Argo',
    'gpto1preview': 'GPT-o1 Preview via Argo',
    'gemini25pro': 'Gemini 2.5 Pro via Argo'
}



class ArgoModel(Model):
    """Simple wrapper for any Argo model."""
    
    def __init__(self, model_id: str, user_name: str = ARGO_USER, environment: str = "auto"):
        super().__init__()

        if user_name:
            self.user = user_name
        else:
            try:
                self.user = os.getenv("ARGO_USER")
            except:
                raise ValueError("ARGO_USER environment variable is not set. Please set it to your Argo username before using the ArgoModel. Example: export ARGO_USER=your_username")
            
        self.model_id = model_id
        # Simple environment detection
        if environment == "auto":
            # Production models
            if model_id in ["gpt35", "gpt35large", "gpt4", "gpt4large", "gpt4turbo", "gpt4o", "gpto1preview"]:
                self.url = "https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/"
            else:
                self.url = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"
        else:
            urls = {
                "prod": "https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/",
                "dev": "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/",
                "test": "https://apps-test.inside.anl.gov/argoapi/api/v1/resource/chat/"
            }
            self.url = urls[environment]

    def generate(self, messages: List[ChatMessage], stop_sequences: Optional[List[str]] = None, **kwargs) -> ChatMessage:
        # Convert messages
        argo_messages = []
        for msg in messages:
            role = getattr(msg, 'role', 'user')
            content = str(getattr(msg, 'content', msg))
            
            # Handle smolagents role conversions for Argo API compatibility
            if role == 'tool-call':
                # Convert tool-call to assistant since Argo API doesn't support tool-call
                role = 'assistant'
                content = f"Tool call: {content}"
            elif role == 'tool' or role == 'tool-response':
                # Convert tool/tool-response to user since Argo API expects user/assistant/system
                role = 'user'
                content = f"Tool result: {content}"
            
            # Handle o-series models that don't support system messages
            if role == 'system' and self.model_id in ["gpto1preview", "gpto1mini", "gpto1"]:
                role = 'user'
                content = f"System: {content}"
                
            argo_messages.append({"role": role, "content": content})
        
        # Build request
        data = {
            "user": self.user,
            "model": self.model_id,
            "messages": argo_messages,
            "stop": stop_sequences or []
        }
        
        # Add parameters based on model type
        if not self.model_id.startswith("gpt"):  # o-series models don't support these
            data.update({
                "temperature": kwargs.get("temperature", 0.1),
                "top_p": kwargs.get("top_p", 1.0)
            })
            
        # Claude models require these parameters to be set
        if self.model_id.startswith("claude"):
            data.update({
                "temperature": kwargs.get("temperature", 0.1),
                "top_p": kwargs.get("top_p", 1.0),
                "max_tokens": kwargs.get("max_tokens", 20000)  # Claude default
            })
        elif self.model_id.startswith("gpto") or self.model_id.startswith("gpt4"):
            data["max_completion_tokens"] = kwargs.get("max_tokens", 20000)
        else:
            data["max_tokens"] = kwargs.get("max_tokens", 20000)
        
        # Make request with comprehensive error handling
        try:
            if not self.user:
                error_msg = (
                    "ARGO_USER environment variable is not set. "
                    "Please set it to your Argo username before using the ArgoModel. "
                    "Example: export ARGO_USER=your_username"
                )
                return ChatMessage(role="assistant", content=f"Configuration Error: {error_msg}")
            
            response = requests.post(
                self.url,
                headers={"Content-Type": "application/json"},
                json=data,
                timeout=120
            )
            
            # Check HTTP status
            if response.status_code == 401:
                error_msg = (
                    f"Authentication failed (HTTP 401). "
                    f"Please check your ARGO_USER setting: '{self.user}'. "
                    f"Ensure you have access to the Argo API and the user is correctly configured."
                )
                return ChatMessage(role="assistant", content=f"Authentication Error: {error_msg}")
            elif response.status_code == 403:
                error_msg = (
                    f"Access forbidden (HTTP 403). "
                    f"User '{self.user}' may not have permission to access model '{self.model_id}' "
                    f"or the Argo API endpoint: {self.url}"
                )
                return ChatMessage(role="assistant", content=f"Permission Error: {error_msg}")
            elif response.status_code == 404:
                error_msg = (
                    f"Model or endpoint not found (HTTP 404). "
                    f"Model '{self.model_id}' may not exist or "
                    f"the API endpoint may be incorrect: {self.url}"
                )
                return ChatMessage(role="assistant", content=f"Not Found Error: {error_msg}")
            elif response.status_code == 500:
                error_msg = (
                    f"Internal server error (HTTP 500). "
                    f"The Argo API is experiencing issues. "
                    f"Please try again later or contact support."
                )
                return ChatMessage(role="assistant", content=f"Server Error: {error_msg}")
            elif response.status_code != 200:
                error_msg = (
                    f"HTTP {response.status_code} error. "
                    f"Response: {response.text[:200]}... "
                    f"URL: {self.url}, Model: {self.model_id}"
                )
                return ChatMessage(role="assistant", content=f"HTTP Error: {error_msg}")
            
            # Parse response
            try:
                result = response.json()
            except ValueError as json_error:
                error_msg = (
                    f"Failed to parse JSON response from Argo API. "
                    f"Response text (first 200 chars): {response.text[:200]}... "
                    f"JSON error: {str(json_error)}"
                )
                return ChatMessage(role="assistant", content=f"JSON Parse Error: {error_msg}")
            
            # Extract text from response
            if isinstance(result, dict):
                text = result.get("response")
                if text is None:
                    # Try alternative response keys
                    text = result.get("content") or result.get("message") or result.get("text")
                    if text is None:
                        error_msg = (
                            f"No 'response' field found in API response. "
                            f"Available keys: {list(result.keys())}. "
                            f"Full response: {str(result)[:300]}..."
                        )
                        return ChatMessage(role="assistant", content=f"Response Format Error: {error_msg}")
            else:
                text = str(result)
            
            # Ensure we have valid content
            if not text or not text.strip():
                error_msg = (
                    f"Model '{self.model_id}' returned empty response. "
                    f"This may indicate model configuration issues or prompt problems. "
                    f"Raw response: {str(result)}"
                )
                return ChatMessage(role="assistant", content=f"Empty Response Error: {error_msg}")
            
            # Wrap text in code tags if it doesn't already have them
            if '<code>' not in text and '</code>' not in text:
                text = f'<code>"""\n{text}\n"""</code>'
            
            return ChatMessage(role="assistant", content=text)
            
        except requests.exceptions.Timeout as timeout_error:
            error_msg = (
                f"Request timeout after 120 seconds when calling Argo API. "
                f"Model: {self.model_id}, URL: {self.url}. "
                f"The model may be overloaded or experiencing issues. "
                f"Try again later or use a different model."
            )
            return ChatMessage(role="assistant", content=f"Timeout Error: {error_msg}")
        except requests.exceptions.ConnectionError as conn_error:
            error_msg = (
                f"Connection error when calling Argo API: {str(conn_error)}. "
                f"URL: {self.url}. "
                f"Check your network connection and ensure the Argo API is accessible."
            )
            return ChatMessage(role="assistant", content=f"Connection Error: {error_msg}")
        except requests.exceptions.RequestException as req_error:
            error_msg = (
                f"Request error when calling Argo API: {str(req_error)}. "
                f"Model: {self.model_id}, URL: {self.url}. "
                f"This may be a temporary issue with the API."
            )
            return ChatMessage(role="assistant", content=f"Request Error: {error_msg}")
        except Exception as e:
            error_msg = (
                f"Unexpected error in ArgoModel.generate(): {type(e).__name__}: {str(e)}. "
                f"Model: {self.model_id}, URL: {self.url}. "
                f"Error occurred at line {e.__traceback__.tb_lineno if e.__traceback__ else 'unknown'}. "
                f"Please report this error if it persists."
            )
            return ChatMessage(role="assistant", content=f"Unexpected Error: {error_msg}")

    def __call__(self, messages: List[Dict[str, str]], **kwargs) -> ChatMessage:
        """
        Compatibility method for smolagents framework.
        Ensures we never return None under any circumstances.
        """
        try:
            # Convert dict messages to ChatMessage objects
            chat_messages = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                # Handle smolagents role conversions for Argo API compatibility
                if role == 'tool-call':
                    # Convert tool-call to assistant since Argo API doesn't support tool-call
                    role = 'assistant'
                    content = f"Tool call: {content}"
                elif role == 'tool' or role == 'tool-response':
                    # Convert tool/tool-response to user since Argo API expects user/assistant/system
                    role = 'user'
                    content = f"Tool result: {content}"
                
                # Handle content that might be a list (multi-modal)
                if isinstance(content, list):
                    # Extract text content from list
                    text_content = ""
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            text_content += item.get('text', '')
                        elif isinstance(item, str):
                            text_content += item
                    content = text_content
                
                chat_messages.append(ChatMessage(role=role, content=str(content)))
            
            # Call the main generate method
            result = self.generate(chat_messages, **kwargs)
            
            # Final safety check - ensure we never return None
            if result is None:
                error_msg = (
                    f"ArgoModel.generate() returned None unexpectedly. "
                    f"This should never happen. Model: {self.model_id}. "
                    f"Please report this as a bug."
                )
                return ChatMessage(role="assistant", content=f"Critical Error: {error_msg}")
            
            # Ensure the result has content
            if not hasattr(result, 'content') or result.content is None:
                error_msg = (
                    f"ArgoModel.generate() returned ChatMessage without content. "
                    f"Model: {self.model_id}. Result type: {type(result)}. "
                    f"This should never happen."
                )
                return ChatMessage(role="assistant", content=f"Content Error: {error_msg}")
            
            return result
            
        except Exception as e:
            # Ultimate fallback - catch any unexpected errors
            error_msg = (
                f"Critical error in ArgoModel.__call__(): {type(e).__name__}: {str(e)}. "
                f"Model: {self.model_id}. "
                f"Error occurred at line {e.__traceback__.tb_lineno if e.__traceback__ else 'unknown'}. "
                f"This is a fallback error handler to prevent None returns."
            )
            return ChatMessage(role="assistant", content=f"Fallback Error: {error_msg}")

# Simple test tool
@tool
def get_time() -> str:
    """Get current time"""
    return f"Current time: {datetime.now()}"

# Usage examples
if __name__ == "__main__":
    all_models = ["gpt4o", "claudeopus4", "claudesonnet4", "gemini25flash", "gpt35", "gpt35large", "gpt4", "gpt4large", "gpt4turbo", "gpto1preview"]
    
    for model_name in all_models:
        try:
            agent = CodeAgent(model=ArgoModel(model_name), tools=[get_time])
            result = agent.run("What time is it?")
            print(f"{model_name}: {result}")
        except Exception as e:
            print(f"{model_name}: Error - {e}")