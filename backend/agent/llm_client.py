"""
LLM Client with Gemini (primary) and Ollama (fallback)
"""

import os
import logging
from typing import Optional, Dict, Any
import google.generativeai as genai
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified LLM client with automatic failover from Gemini to Ollama
    """
    
    def __init__(self):
        self.gemini_available = False
        self.ollama_available = False
        
        # Initialize Gemini (DISABLED - using Ollama only for now)
        # To re-enable Gemini, uncomment the code block below
        self.gemini_key = None
        self.gemini_available = False
        logger.info("⚠️  Gemini disabled - using Ollama only")
        
        # GEMINI INITIALIZATION CODE - COMMENTED OUT
        # self.gemini_key = os.getenv("GOOGLE_API_KEY")
        # if self.gemini_key:
        #     try:
        #         genai.configure(api_key=self.gemini_key)
        #         gemini_model_names = [
        #             'models/gemini-2.5-flash',
        #             'models/gemini-2.5-pro-preview-05-06',
        #             'models/gemini-1.5-pro',
        #             'models/gemini-1.5-flash',
        #             'models/gemini-pro'
        #         ]
        #         self.gemini_model = None
        #         last_error = None
        #         for model_name in gemini_model_names:
        #             try:
        #                 self.gemini_model = genai.GenerativeModel(model_name)
        #                 test_response = self.gemini_model.generate_content("test", generation_config=genai.types.GenerationConfig(max_output_tokens=10))
        #                 logger.info(f"✓ Gemini API initialized successfully with model: {model_name}")
        #                 self.gemini_available = True
        #                 break
        #             except Exception as model_error:
        #                 last_error = model_error
        #                 continue
        #         if not self.gemini_available:
        #             logger.warning(f"All Gemini model attempts failed. Last error: {last_error}")
        #     except Exception as e:
        #         logger.warning(f"Gemini API initialization failed: {e}")
        # END OF DISABLED GEMINI CODE
        
        # Initialize Ollama
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=2)
            if response.status_code == 200:
                self.ollama_available = True
                logger.info("✓ Ollama connection established")
        except Exception as e:
            logger.warning(f"Ollama connection failed: {e}")
        
        if not self.gemini_available and not self.ollama_available:
            logger.error("❌ No LLM available! Please configure Gemini API or Ollama")
    
    def generate(
        self, 
        prompt: str, 
        # max_tokens: int = 1024,
        max_tokens: int = 1000, 
        # temperature: float = 0.7,
        temperature: float = 0.3,
        force_ollama: bool = False
    ) -> str:
        """
        Generate text using available LLM (Gemini → Ollama fallback)
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            force_ollama: Force using Ollama instead of Gemini
            
        Returns:
            Generated text
        """
        # Skip Gemini (disabled) - use Ollama directly
        if False and self.gemini_available and not force_ollama:
            try:
                # Configure safety settings to be more permissive for legal content
                # Legal case discussions may trigger safety filters, so we use BLOCK_ONLY_HIGH
                safety_settings = [
                    {
                        "category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        "threshold": genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH
                    },
                    {
                        "category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        "threshold": genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH
                    },
                    {
                        "category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        "threshold": genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH
                    },
                    {
                        "category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        "threshold": genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH
                    }
                ]
                
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                    ),
                    safety_settings=safety_settings
                )
                text = self._extract_gemini_text(response)
                
                # Check finish reason - 2 means SAFETY (blocked by filters)
                finish_reason = None
                if response.candidates:
                    finish_reason = response.candidates[0].finish_reason
                
                if not text or finish_reason == 2:  # SAFETY
                    # If blocked by safety filters, retry with BLOCK_NONE for legal content
                    logger.info(f"Gemini response blocked by safety filters (finish_reason={finish_reason}), retrying with permissive settings for legal content")
                    try:
                        permissive_safety = [
                            {
                                "category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
                            },
                            {
                                "category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
                            },
                            {
                                "category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
                            },
                            {
                                "category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                "threshold": genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH
                            }
                        ]
                        response = self.gemini_model.generate_content(
                            prompt,
                            generation_config=genai.types.GenerationConfig(
                                max_output_tokens=max_tokens,
                                temperature=temperature,
                            ),
                            safety_settings=permissive_safety
                        )
                        text = self._extract_gemini_text(response)
                        if text:
                            logger.info("Gemini retry with permissive settings successful")
                            return text
                    except Exception as retry_error:
                        logger.debug(f"Gemini retry failed: {retry_error}")
                    
                    # Still empty after retry, raise to fallback
                    raise Exception(f"Empty Gemini response text (finish_reason={finish_reason} - SAFETY)")
                
                return text
            except Exception as e:
                logger.warning(f"Gemini generation failed: {e}, falling back to Ollama")
        
        # Fallback to Ollama
        if self.ollama_available:
            try:
                response = requests.post(
                    f"{self.ollama_host}/api/generate", 
                    json={
                        "model": "llama3.2",
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens
                        }
                    },
                    timeout=300
                )
                if response.status_code == 200:
                    return response.json()["response"]
                else:
                    raise Exception(f"Ollama returned status {response.status_code}")
            except Exception as e:
                logger.error(f"Ollama generation failed: {e}")
                raise Exception("All LLM backends failed")
        
        raise Exception("No LLM backend available")

    def _extract_gemini_text(self, response: Any) -> str:
        """Robustly extract text from Gemini responses.
        Falls back to concatenating candidate parts when response.text is unavailable.
        """
        try:
            if hasattr(response, "text") and response.text:
                return response.text
        except Exception:
            pass
        # Try candidates → content → parts → text
        try:
            texts = []
            for cand in (response.candidates or []):
                content = getattr(cand, "content", None)
                if not content:
                    continue
                parts = getattr(content, "parts", None)
                if not parts:
                    continue
                for p in parts:
                    t = getattr(p, "text", None)
                    if t:
                        texts.append(t)
            return "\n".join(texts).strip()
        except Exception:
            return ""
    
    def is_available(self) -> bool:
        """Check if any LLM backend is available"""
        return self.gemini_available or self.ollama_available
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all LLM backends"""
        return {
            "gemini": {
                "available": self.gemini_available,
                "configured": bool(self.gemini_key)
            },
            "ollama": {
                "available": self.ollama_available,
                "host": self.ollama_host
            },
            "any_available": self.is_available()
        }

