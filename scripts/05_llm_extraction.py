#!/usr/bin/env python3
"""
Phase 2 (Revised): LLM-based Information Extraction and Summarization
Uses LLMs to accurately extract structured data and generate English summaries.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import time
from dotenv import load_dotenv
import hashlib

# Load environment variables from .env file
load_dotenv()

# Paths
CASE_FILES_DIR = Path(r"D:FinalAI/Wakalat Sewa/CaseFiles")
GLOSSARY_DIR = Path(r"D:FinalAI/Wakalat Sewa/wakalt/tozip/glossary")
OUTPUT_DIR = Path(r"D:FinalAI/Wakalat Sewa/processed_llm")
OUTPUT_DIR.mkdir(exist_ok=True)

# LLM Configuration
# You can use: OpenAI (GPT-4), Anthropic (Claude), Google (Gemini), or local models (Ollama)
LLM_PROVIDER = "gemini"  # Options: "openai", "anthropic", "gemini", "ollama"


class LLMExtractor:
    """
    Uses LLMs to extract structured information from Nepali legal cases.
    """
    
    def __init__(self, provider="openai"):
        self.provider = provider
        self.setup_client()
    
    def setup_client(self):
        """Initialize the LLM client based on provider."""
        if self.provider == "openai":
            try:
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not found in environment variables")
                self.client = OpenAI(api_key=api_key)
                self.model = "gpt-4o-mini"  # Cost-effective, good for extraction
                print(f"✅ Using OpenAI GPT-4o-mini")
            except ImportError:
                raise ImportError("Please install: uv pip install openai")
        
        elif self.provider == "anthropic":  
            try:
                from anthropic import Anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
                self.client = Anthropic(api_key=api_key)
                self.model = "claude-3-haiku-20240307"  # Cost-effective
                print(f"✅ Using Anthropic Claude Haiku")
            except ImportError:
                raise ImportError("Please install: uv pip install anthropic")
        
        elif self.provider == "gemini":
            try:
                import google.generativeai as genai
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY not found in .env file")
                genai.configure(api_key=api_key)
                self.client = genai
                # Using Gemini 2.5 Flash - stable, fast, and very capable
                self.model = "gemini-2.5-flash"  # Stable and cost-effective
                # Alternative: "gemini-2.5-pro" for even better quality
                print(f"✅ Using Google Gemini 2.5 Flash")
                print(f"   API key loaded from .env file")
            except ImportError:
                raise ImportError("Please install: uv pip install google-generativeai")
        
        elif self.provider == "github":
            try:
                from openai import OpenAI
                token = os.getenv("GITHUB_TOKEN")
                if not token:
                    raise ValueError("GITHUB_TOKEN not found in .env file")
                # GitHub Models uses OpenAI-compatible API
                self.client = OpenAI(
                    base_url="https://models.inference.ai.azure.com",
                    api_key=token
                )
                # Using GPT-4o-mini for speed (5-10x faster than Llama 70B)
                # Alternative models: llama-3.3-70b-instruct, llama-3.1-8b-instant, mistral-large
                self.model = "gpt-4o-mini"
                print(f"✅ Using GitHub Models (FREE, 3000 req/day) with GPT-4o-mini")
                print(f"   ⚡ FAST MODE: ~5-10 seconds per case (vs 30-40s)")
                print(f"   GitHub token loaded from .env file")
            except ImportError:
                raise ImportError("Please install: uv pip install openai")
        
        elif self.provider == "groq":
            try:
                from groq import Groq
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY not found in .env file")
                self.client = Groq(api_key=api_key)
                self.model = "llama-3.3-70b-versatile"  # Current fast model (Jan 2025)
                # Alternative: "llama-3.1-8b-instant" for faster/cheaper
                print(f"✅ Using Groq (FREE, fast) with Llama 3.3 70B")
                print(f"   API key loaded from .env file")
            except ImportError:
                raise ImportError("Please install: uv pip install groq")
        
        elif self.provider == "ollama":
            # For local inference (free but requires local setup)
            import requests
            self.client = None
            self.model = "llama3.2"  # Or any model you have locally
            print(f"✅ Using Ollama (local) with {self.model}")
        
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    


    def create_extraction_prompt(self, case_text: str) -> str:
            """
            Create a prompt for the LLM to extract structured information in BOTH Nepali and English.
            """
            prompt = f"""You are an expert legal analyst specializing in Nepali law. Analyze this Nepali legal case document and extract information in BOTH Nepali (original) and English (translated) for bilingual search functionality.

    NEPALI LEGAL CASE DOCUMENT:
    {case_text[:8000]}  # Limit to first 8000 chars to fit in context

    EXTRACT THE FOLLOWING INFORMATION (DUAL LANGUAGE):

    1. case_number_nepali: Case number in Nepali numerals (e.g., "०७१-CI-०५२१")
    2. case_number_english: Case number in English numerals (e.g., "071-CI-0521")

    3. case_type_nepali: Type of case in Nepali (e.g., "निषेधाज्ञा", "कर्तव्य ज्यान")
    4. case_type_english: Type of case in English (e.g., "Injunction", "Murder")

    5. court_nepali: Court name in Nepali (e.g., "सर्वोच्च अदालत", "उच्च अदालत")
    6. court_english: Court name in English (e.g., "Supreme Court", "High Court")

    7. judges_nepali: List of judge names in Nepali (e.g., ["न्यायाधीश श्री ओमप्रकाश मिश्र"])
    8. judges_english: List of judge names in English (e.g., ["Justice Om Prakash Mishra"])

    9. plaintiff_nepali: Plaintiff/petitioner name in Nepali
    10. plaintiff_english: Plaintiff/petitioner name in English transliteration

    11. defendant_nepali: Defendant/respondent name in Nepali
    12. defendant_english: Defendant/respondent name in English transliteration

    13. plaintiff_lawyers_nepali: List of plaintiff's lawyers in Nepali
    14. plaintiff_lawyers_english: List of plaintiff's lawyers in English

    15. defendant_lawyers_nepali: List of defendant's lawyers in Nepali
    16. defendant_lawyers_english: List of defendant's lawyers in English

    17. decision_date_nepali: Decision date in Nepali format (e.g., "२०७३।०६।०४")
    18. decision_date_english: Decision date in English numerals (e.g., "2073.06.04")
    19. decision_date_gregorian: Convert to Gregorian date if possible (e.g., "2016-09-20")

    20. verdict_nepali: Court's verdict in Nepali (e.g., "खारेज", "सदर")
    21. verdict_english: Court's verdict in English (e.g., "dismissed", "upheld")

    22. legal_provisions_nepali: List of legal sections/acts in Nepali (e.g., ["न्याय प्रशासन ऐन, २०४८ को दफा ९"])
    23. legal_provisions_english: List of legal sections/acts in English (e.g., ["Justice Administration Act, 2048, Section 9"])

    24. key_facts: Brief description of key facts (3-4 sentences in English)
    25. legal_issue: Main legal question (2-3 sentences in English)
    26. court_reasoning: Summary of court's reasoning (3-4 sentences in English)
    27. final_order: Final order/decision (2-3 sentences in English)

    IMPORTANT GUIDELINES:
    - Extract BOTH Nepali original and English translation for each field
    - For names: Keep original Nepali, transliterate to English (not translate)
    - For dates: Provide Nepali calendar, English numerals, AND Gregorian if possible
    - If information is not found, use null
    - For lists, return empty array [] if not found
    - Be accurate and extract only what is clearly stated in the document
    - Maintain legal terminology accuracy in both languages

    Return ONLY valid JSON, no additional text or markdown formatting."""

            return prompt
        
    def create_summary_prompt(self, case_text: str) -> str:
        """
        Create a prompt for generating an English summary.
        """
        prompt = f"""You are an expert legal analyst. Read this Nepali legal case document and provide a comprehensive English summary.

    NEPALI LEGAL CASE DOCUMENT:
    {case_text[:12000]}

    PROVIDE A STRUCTURED ENGLISH SUMMARY WITH THE FOLLOWING SECTIONS:

    1. **Case Title**: Brief title describing the case
    2. **Parties**: Who is suing whom
    3. **Facts**: What happened (3-5 sentences)
    4. **Legal Issues**: What legal questions were raised (2-3 points)
    5. **Court's Analysis**: How the court analyzed the case (3-5 sentences)
    6. **Decision**: What the court decided (2-3 sentences)
    7. **Legal Significance**: Why this case matters (1-2 sentences)

    Keep the summary:
    - Professional and objective
    - Between 200-400 words
    - Focused on legal substance, not procedural details
    - Accessible to someone with basic legal knowledge

    Return the summary in clear, well-structured English."""

        return prompt
        
    def extract_with_openai(self, prompt: str, response_format="json") -> str:
        """Extract information using OpenAI."""
        try:
            if response_format == "json":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert legal analyst specializing in Nepali law. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,  # Low temperature for consistency
                    max_tokens=2000
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert legal analyst specializing in Nepali law."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=800
                )
            
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"Error with OpenAI: {e}")
            return None
        
    def extract_with_anthropic(self, prompt: str, response_format="json") -> str:
        """Extract information using Anthropic Claude."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.1 if response_format == "json" else 0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
        
        except Exception as e:
            print(f"Error with Anthropic: {e}")
            return None

    def extract_with_gemini(self, prompt: str, response_format="json") -> str:
        """Extract information using Google Gemini."""
        try:
            from google.generativeai.types import HarmCategory, HarmBlockThreshold
            
            # Configure safety settings to allow legal content
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            model = self.client.GenerativeModel(
                self.model,
                safety_settings=safety_settings
            )
            
            # Configure for JSON output if requested
            generation_config = {
                "temperature": 0.1 if response_format == "json" else 0.3,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
            
            if response_format == "json":
                # Add JSON instruction to prompt
                json_instruction = "\n\nIMPORTANT: Return ONLY valid JSON, no markdown code blocks (no ```json), no explanations."
                prompt = prompt + json_instruction
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Extract text from response
            if response.text:
                # Clean markdown code blocks if present
                text = response.text.strip()
                if text.startswith("```json"):
                    text = text[7:]  # Remove ```json
                if text.startswith("```"):
                    text = text[3:]  # Remove ```
                if text.endswith("```"):
                    text = text[:-3]  # Remove trailing ```
                return text.strip()
            else:
                print(f"Gemini returned empty response")
                return None
        
        except Exception as e:
            print(f"Error with Gemini: {e}")
            return None

    def extract_with_groq(self, prompt: str) -> str:
        """Extract information using Groq API (fast and free)."""
        try:
            # Add JSON instruction to prompt
            json_instruction = "\n\nIMPORTANT: Return ONLY valid JSON, no markdown code blocks (no ```json), no explanations."
            prompt_with_instruction = prompt + json_instruction
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a legal expert assistant that extracts structured information from Nepali court cases. You MUST respond with ONLY valid JSON, no markdown formatting, no code blocks, no explanations."
                    },
                    {
                        "role": "user",
                        "content": prompt_with_instruction
                    }
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            # Clean the response
            text = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if text.startswith("```json"):
                text = text[7:]  # Remove ```json
            if text.startswith("```"):
                text = text[3:]  # Remove ```
            if text.endswith("```"):
                text = text[:-3]  # Remove trailing ```
            
            return text.strip()
        
        except Exception as e:
            print(f"Error with Groq: {e}")
            return None
        
    def extract_with_ollama(self, prompt: str, response_format: str = "json") -> str:
        """Extract information using local Ollama.

        When response_format is "json", request structured JSON from the model
        and enable Ollama's JSON mode to ensure valid output.
        """
        try:
            import requests
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            if response_format == "json":
                # Nudge the model and enable JSON mode
                payload["format"] = "json"
                payload["prompt"] = prompt + "\n\nIMPORTANT: Respond with ONLY a single valid JSON object. No markdown, no code fences."
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload
            )
            
            if response.status_code == 200:
                text = response.json().get("response", "").strip()
                # Clean common formatting if any slipped through
                if text.startswith("```json"):
                    text = text[7:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                return text.strip()
            else:
                print(f"Ollama error: {response.status_code}")
                return None
        
        except Exception as e:
            print(f"Error with Ollama: {e}")
            return None
        
    def extract_structured_data(self, case_text: str, filename: str) -> Dict:
        """
        Extract structured data from a case using LLM.
        """
        prompt = self.create_extraction_prompt(case_text)
        
        # Call appropriate LLM
        if self.provider == "openai" or self.provider == "github":
            response = self.extract_with_openai(prompt, response_format="json")
        elif self.provider == "anthropic":
            response = self.extract_with_anthropic(prompt, response_format="json")
        elif self.provider == "gemini":
            response = self.extract_with_gemini(prompt, response_format="json")
        elif self.provider == "groq":
            response = self.extract_with_groq(prompt)
        elif self.provider == "ollama":
            response = self.extract_with_ollama(prompt, response_format="json")
        
        if not response:
            return {"filename": filename, "error": "LLM extraction failed"}
        
        try:
            # Parse JSON response
            extracted_data = json.loads(response)
            extracted_data["filename"] = filename
            extracted_data["extraction_model"] = f"{self.provider}:{self.model}"
            return extracted_data
        
        except json.JSONDecodeError as e:
            print(f"JSON parse error for {filename}: {e}")
            return {"filename": filename, "error": f"JSON parse error: {str(e)}", "raw_response": response}
        
    def generate_summary(self, case_text: str) -> str:
        """
        Generate an English summary of the case.
        """
        prompt = self.create_summary_prompt(case_text)
        
        # Call appropriate LLM
        if self.provider == "openai" or self.provider == "github":
            response = self.extract_with_openai(prompt, response_format="text")
        elif self.provider == "anthropic":
            response = self.extract_with_anthropic(prompt, response_format="text")
        elif self.provider == "gemini":
            response = self.extract_with_gemini(prompt, response_format="text")
        elif self.provider == "groq":
            response = self.extract_with_groq(prompt)
        elif self.provider == "ollama":
            response = self.extract_with_ollama(prompt, response_format="text")
        
        return response if response else "Summary generation failed"
        
    def process_case(self, filepath: Path) -> Dict:
        """
        Process a single case file: extract data + generate summary.
        """
        print(f"Processing file: {filepath.name}")  # Print the filename being processed
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                case_text = f.read()
            
            # Extract structured data
            extracted_data = self.extract_structured_data(case_text, filepath.name)
            
            # Generate English summary
            summary = self.generate_summary(case_text)
            extracted_data["english_summary"] = summary
            extracted_data["original_text"] = case_text  # Keep original for reference
            
            return extracted_data
        
        except Exception as e:
            return {
                "filename": filepath.name,
                "error": f"Processing error: {str(e)}"
            }


def process_cases_with_llm(
    sample_size: Optional[int] = None,
    provider: str = "openai",
    save_interval: int = 100
):
    """
    Process cases using LLM extraction with individual JSON files for resume capability.
    
    Args:
        sample_size: Number of cases to process (None = all)
        provider: LLM provider ("openai", "anthropic", "ollama", "github")
        save_interval: Combine all JSONs into main file every N cases
    """
    print("=" * 80)
    print("LLM-BASED CASE EXTRACTION & SUMMARIZATION (RESUMABLE)")
    print("=" * 80)
    print()
    
    # Create individual cases directory
    cases_dir = OUTPUT_DIR / "cases"
    cases_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize extractor
    extractor = LLMExtractor(provider=provider)
    
    # Get case files
    case_files = list(CASE_FILES_DIR.glob("*.txt"))
    
    if sample_size:
        case_files = case_files[:sample_size]
        print(f"🧪 Processing {sample_size} cases (sample mode)")
    else:
        print(f"📁 Processing all {len(case_files)} cases")
    
    print(f"🤖 Using {provider.upper()} for extraction")
    print(f"💾 Individual JSON files: {cases_dir}")
    print()
    
    # Check for already processed cases
    already_processed = []
    to_process = []
    
    for filepath in case_files:
        # Create JSON filename from case filename
        json_filename = safe_json_filename(filepath.name)
        json_path = cases_dir / json_filename
        
        if json_path.exists():
            already_processed.append(filepath.name)
        else:
            to_process.append(filepath)
    
    if already_processed:
        print(f"⏭️  Skipping {len(already_processed)} already processed cases")
        print(f"▶️  Processing {len(to_process)} remaining cases")
    else:
        print(f"▶️  Processing {len(to_process)} cases (fresh start)")
    
    print()
    
    # Process cases
    processed_count = 0
    error_count = 0
    
    for i, filepath in enumerate(tqdm(to_process, desc="Processing cases")):
        # Print file name being processed
        print(f"➡️  Now processing: {filepath.name}")
        try:
            # Process the case
            result = extractor.process_case(filepath)
            
            # Save individual JSON immediately
            json_filename = safe_json_filename(filepath.name)
            json_path = cases_dir / json_filename
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            if "error" in result and "extraction_model" not in result:
                error_count += 1
            else:
                processed_count += 1
            
            # Combine all JSONs into main files periodically
            if (i + 1) % save_interval == 0:
                combine_all_cases(cases_dir)
            
            # Rate limiting (avoid API limits)
            if provider in ["openai", "anthropic", "github"]:
                time.sleep(0.5)  # 2 requests/second
        
        except Exception as e:
            # Save error case
            json_filename = safe_json_filename(filepath.name)
            json_path = cases_dir / json_filename
            error_data = {"filename": filepath.name, "error": str(e), "status": "failed"}
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, ensure_ascii=False, indent=2)
            error_count += 1
    
    # Final combine
    combine_all_cases(cases_dir)
    
    # Statistics
    total_processed_now = processed_count + error_count
    total_in_system = len(already_processed) + processed_count
    
    print()
    print("=" * 80)
    print("📊 PROCESSING STATISTICS")
    print("=" * 80)
    print()
    print(f"Already processed (skipped): {len(already_processed)}")
    print(f"Newly processed (this run): {total_processed_now}")
    print(f"  ├─ Successful: {processed_count}")
    print(f"  └─ Errors: {error_count}")
    print(f"Total in database: {total_in_system}")
    if total_processed_now > 0:
        print(f"Success rate (this run): {processed_count / total_processed_now * 100:.1f}%")
    print()
    
    # Cost estimation (rough)
    if provider == "openai" or provider == "github":
        # GPT-4o-mini: ~$0.15/1M input tokens, ~$0.60/1M output tokens
        # Estimate: ~10k tokens per case (input + output)
        estimated_cost = processed_count * 10000 / 1_000_000 * 0.40  # Average
        if provider == "github":
            print(f"💰 Cost: FREE (GitHub Models)")
        else:
            print(f"💰 Estimated cost: ${estimated_cost:.2f}")
    
    print()
    print("✅ Processing complete!")
    print(f"📄 Individual JSONs: {cases_dir}")
    print(f"📄 Combined files: {OUTPUT_DIR}")
    print()


def combine_all_cases(cases_dir: Path):
    """
    Combine all individual JSON files into main database files.
    This is called periodically and at the end to create consolidated files.
    """
    print()
    print("📦 Combining all individual JSON files...")
    
    # Read all individual JSON files
    all_cases = []
    errors = []
    
    for json_file in sorted(cases_dir.glob("*.json")):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                case_data = json.load(f)
                
                if case_data.get("status") == "failed" or ("error" in case_data and "extraction_model" not in case_data):
                    errors.append(case_data)
                else:
                    all_cases.append(case_data)
        except Exception as e:
            print(f"⚠️  Error reading {json_file.name}: {e}")
    
    # Save combined JSON with all fields
    output_file = OUTPUT_DIR / "llm_extracted_cases.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)
    
    # Save CSV for easy viewing (without full text and summary)
    csv_data = []
    for case in all_cases:
        csv_case = {k: v for k, v in case.items() 
                   if k not in ['original_text', 'english_summary']}
        # Convert lists to strings for CSV
        for key, value in csv_case.items():
            if isinstance(value, list):
                csv_case[key] = '; '.join(str(v) for v in value)
        csv_data.append(csv_case)
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_file = OUTPUT_DIR / "llm_extracted_cases.csv"
        df.to_csv(csv_file, index=False)
    
    # Save summaries separately
    summaries = []
    for case in all_cases:
        summaries.append({
            'filename': case.get('filename'),
            'case_number_nepali': case.get('case_number_nepali'),
            'case_number_english': case.get('case_number_english'),
            'case_type_nepali': case.get('case_type_nepali'),
            'case_type_english': case.get('case_type_english'),
            'court_nepali': case.get('court_nepali'),
            'court_english': case.get('court_english'),
            'summary': case.get('english_summary', '')
        })
    
    summary_file = OUTPUT_DIR / "case_summaries.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    
    # Save errors
    if errors:
        error_file = OUTPUT_DIR / "extraction_errors.json"
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Combined {len(all_cases)} cases, {len(errors)} errors")
    print(f"   Saved to: {OUTPUT_DIR}")

def safe_json_filename(original_name: str, max_bytes: int = 200) -> str:
    """Return a filesystem-safe JSON filename preserving Unicode where possible.

    Truncates by UTF-8 byte length and appends a short hash for uniqueness.
    """
    stem = Path(original_name).stem.replace('/', '_').replace('\\', '_').strip()
    utf8_bytes = stem.encode('utf-8')
    if len(utf8_bytes) > max_bytes:
        utf8_bytes = utf8_bytes[:max_bytes]
        stem = utf8_bytes.decode('utf-8', errors='ignore')
    short_hash = hashlib.sha1(Path(original_name).stem.encode('utf-8')).hexdigest()[:10]
    return f"{stem}_{short_hash}.json"


if __name__ == "__main__":
    import sys
    
    # Configuration
    PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # Default to ollama (local, free)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--sample":
        sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        print(f"🧪 Running in SAMPLE mode ({sample_size} cases)")
        print()
        if PROVIDER == "ollama":
            print("🤖 Using Ollama (local, free, no API key needed)")
            print("   Make sure Ollama is running: ollama serve")
        else:
            print("⚠️  Make sure you have set your API key in .env file:")
            print("    GITHUB_TOKEN='your-token-here' (FREE, 3000 req/day - BEST!)")
            print("    OR GROQ_API_KEY='your-key-here' (FREE & FAST)")
            print("    OR GEMINI_API_KEY='your-key-here' (FREE but blocks criminal cases)")
            print("    OR OPENAI_API_KEY='your-key-here'")
            print("    OR ANTHROPIC_API_KEY='your-key-here'")
        print()
        process_cases_with_llm(sample_size=sample_size, provider=PROVIDER)
    else:
        print("🚀 Running in FULL mode (all cases)")
        print()
        print("⚠️  This will process all 10,357 cases and may cost money!")
        print("    Estimated cost with GPT-4o-mini: ~$40-50")
        print("    Estimated time: 2-3 hours")
        print()
        confirm = input("Continue? (yes/no): ")
        if confirm.lower() == 'yes':
            process_cases_with_llm(provider=PROVIDER)
        else:
            print("Cancelled.")

