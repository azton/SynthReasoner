#!/usr/bin/env python3
"""
Gemini client using OpenAI compatibility layer
"""

import os
import requests
import json
from typing import List, Dict, Any


class GeminiClient:
    """Gemini client using OpenAI-compatible API"""
    
    def __init__(self, api_key: str = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_TEST_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_TEST_KEY environment variable not set")
        
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
        
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using Gemini API"""
        
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 4000)
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                raise ValueError(f"Unexpected response format: {result}")
                
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Gemini API request failed: {e}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse Gemini API response: {e}")
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")


def test_gemini_client():
    """Test the Gemini client"""
    try:
        client = GeminiClient()
        
        messages = [
            {"role": "user", "content": "What is 2+2? Respond with just the number."}
        ]
        
        response = client.generate(messages)
        print(f"Gemini response: {response}")
        return True
        
    except Exception as e:
        print(f"Gemini test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_gemini_client()
    if success:
        print("✅ Gemini client working!")
    else:
        print("❌ Gemini client failed")