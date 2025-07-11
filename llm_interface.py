#!/usr/bin/env python3
"""
Unified LLM interface supporting both Argo and Gemini
"""

import os
from typing import List, Dict, Any, Optional
try:
    from .gemini_client import GeminiClient
except ImportError:
    from gemini_client import GeminiClient

try:
    from argo import ArgoModel
    ARGO_AVAILABLE = True
except ImportError:
    ARGO_AVAILABLE = False


class UnifiedLLMClient:
    """Unified interface for both Argo and Gemini models"""
    
    def __init__(self, model_name: str, interface: str = "auto", **kwargs):
        """
        Initialize LLM client
        
        Args:
            model_name: Model name (e.g., "gemini-1.5-flash", "claudesonnet4")
            interface: "argo", "gemini", or "auto" (auto-detect)
            **kwargs: Additional parameters (api_key for Gemini, user_name for Argo)
        """
        self.model_name = model_name
        
        # Auto-detect interface if not specified
        if interface == "auto":
            interface = self._auto_detect_interface(model_name)
        
        self.interface = interface
        
        # Initialize appropriate client
        if interface == "gemini":
            api_key = kwargs.get("api_key") or os.getenv("GEMINI_TEST_KEY")
            self.client = GeminiClient(api_key=api_key, model=model_name)
        elif interface == "argo":
            if not ARGO_AVAILABLE:
                raise ImportError("Argo not available. Install argo.py or use Gemini interface.")
            user_name = kwargs.get("user_name") or os.getenv("ARGO_USER")
            self.client = ArgoModel(model_name, user_name)
        else:
            raise ValueError(f"Unsupported interface: {interface}")
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using the configured client"""
        
        if self.interface == "gemini":
            return self.client.generate(messages, **kwargs)
        elif self.interface == "argo":
            # Convert to ChatMessage format for Argo
            from smolagents import ChatMessage
            chat_messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in messages]
            response = self.client.generate(chat_messages, **kwargs)
            # Extract content from response
            if hasattr(response, 'content'):
                content = response.content
                # Remove code tags if present
                content = content.replace('<code>', '').replace('</code>', '')
                content = content.replace('"""', '')
                return content.strip()
            return str(response)
        else:
            raise ValueError(f"Unknown interface: {self.interface}")
    
    def _auto_detect_interface(self, model_name: str) -> str:
        """Auto-detect which interface to use based on model name"""
        
        # Gemini models
        if model_name.startswith("gemini"):
            return "gemini"
        
        # Argo models
        argo_models = {
            "gpt4o", "claudeopus4", "claudesonnet4", "gemini25flash", "gpt35", 
            "gpt35large", "gpt4", "gpt4large", "gpt4turbo", "gpto1preview", "gemini25pro"
        }
        
        if model_name in argo_models:
            if ARGO_AVAILABLE:
                return "argo"
            else:
                raise ImportError(f"Model {model_name} requires Argo, but argo.py is not available")
        
        # Default to Gemini for unknown models that look like Gemini
        if "gemini" in model_name.lower() or "flash" in model_name.lower():
            return "gemini"
        
        # Default to Argo for other models (if available)
        if ARGO_AVAILABLE:
            return "argo"
        else:
            return "gemini"


def get_available_models() -> Dict[str, List[str]]:
    """Get list of available models by interface"""
    
    models = {
        "gemini": [
            "gemini-2.5-flash",
            "gemini-1.5-flash", 
            "gemini-1.5-pro",
            "gemini-2.0-flash-exp"
        ]
    }
    
    if ARGO_AVAILABLE:
        models["argo"] = [
            "gpt4o", "claudeopus4", "claudesonnet4", "gemini25flash", 
            "gpt35", "gpt35large", "gpt4", "gpt4large", "gpt4turbo", 
            "gpto1preview", "gemini25pro"
        ]
    
    return models


def test_unified_interface():
    """Test the unified interface"""
    
    print("Testing Unified LLM Interface")
    print("=" * 40)
    
    # Test available models
    available = get_available_models()
    print("Available models:")
    for interface, models in available.items():
        print(f"  {interface}: {', '.join(models[:3])}{'...' if len(models) > 3 else ''}")
    
    # Test Gemini
    try:
        print("\\nTesting Gemini interface...")
        client = UnifiedLLMClient("gemini-2.5-flash", interface="gemini")
        response = client.generate([{"role": "user", "content": "What is 2+2? Just the number."}])
        print(f"Gemini response: {response}")
    except Exception as e:
        print(f"Gemini test failed: {e}")
    
    # Test Argo (if available)
    if ARGO_AVAILABLE:
        try:
            print("\\nTesting Argo interface...")
            client = UnifiedLLMClient("gpt4o", interface="argo") 
            response = client.generate([{"role": "user", "content": "What is 3+3? Just the number."}])
            print(f"Argo response: {response}")
        except Exception as e:
            print(f"Argo test failed: {e}")
    else:
        print("\\nArgo interface not available (argo.py not imported)")
    
    print("\\n✅ Interface tests complete")


if __name__ == "__main__":
    test_unified_interface()