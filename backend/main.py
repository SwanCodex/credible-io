from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
import traceback

# -----------------------------
# App initialization
# -----------------------------
load_dotenv()
app = FastAPI()

# -----------------------------
# Helper: Configure Gemini
# -----------------------------
def get_gemini_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Debug: No API Key found in env")
        raise HTTPException(status_code=400, detail="Gemini API Key not set. Please configure it in settings.")
    
    genai.configure(api_key=api_key)
    # Using 'gemini-3-flash-preview' as it's the available flash model
    return genai.GenerativeModel('gemini-3-flash-preview')

async def get_gemini_response(prompt: str, json_structure: dict = None):
    try:
        model = get_gemini_model()
        
        full_prompt = prompt
        if json_structure:
            full_prompt += f"\n\nOutput strictly in JSON format matching this structure:\n{json.dumps(json_structure, indent=2)}"
        
        # print("Debug: Sending prompt to Gemini...") 
        response = model.generate_content(full_prompt)
        text_response = response.text
        # print(f"Debug: Received response: {text_response[:100]}...")

        # Clean up code blocks if generic
        if text_response.startswith("```json"):
            text_response = text_response[7:-3]
        elif text_response.startswith("```"):
            text_response = text_response[3:-3]
            
        return json.loads(text_response) if json_structure else text_response
        
    except HTTPException as he:
        # Re-raise HTTP exceptions (like 400 for missing key) directly
        raise he
    except Exception as e:
        # Fallback or re-raise
        print(f"Gemini Error Details:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AI Analysis Failed: {str(e)}")

# -----------------------------
# CORS (for frontend + extension)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # OK for hackathon / local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Data models
# -----------------------------
class TextInput(BaseModel):
    content: str
    source_url: str | None = None


class ApiKeyInput(BaseModel):
    api_key: str


class RepairResponse(BaseModel):
    original_text: str
    repaired_text: str
    repair_explanation: str


# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
def root():
    return {"message": "Credible.io backend is running"}

@app.post("/settings/api-key")
def update_api_key(data: ApiKeyInput):
    """
    Update the Gemini API Key in .env file and runtime environment.
    """
    try:
        new_key = data.api_key.strip()
        if not new_key:
            raise HTTPException(status_code=400, detail="API Key cannot be empty")
            
        # Update .env file
        env_path = ".env"
        lines = []
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                lines = f.readlines()
        
        found = False
        with open(env_path, "w") as f:
            for line in lines:
                if line.startswith("GEMINI_API_KEY="):
                    f.write(f"GEMINI_API_KEY={new_key}\n")
                    found = True
                else:
                    f.write(line)
            if not found:
                f.write(f"\nGEMINI_API_KEY={new_key}\n")
        
        # Update runtime env
        os.environ["GEMINI_API_KEY"] = new_key
        
        return {"message": "API Key updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# -----------------------------
# Helper function
# -----------------------------
def generate_verification_suggestion(flags):
    if "Overconfident language" in flags:
        return "Check credible sources or studies that support this absolute claim."
    if "Insufficient context" in flags:
        return "Look for more detailed explanations or background information."
    return "Verify the claim using trusted academic or educational sources."


# -----------------------------
# Helper: Translation
# -----------------------------
async def translate_text(text: str, target_lang: str = "English") -> str:
    """Translates text to target language using Gemini."""
    prompt = f"Translate the following text to {target_lang}. Return ONLY the translated text, no explanation.\n\nText: {text}"
    return await get_gemini_response(prompt)

async def detect_language(text: str) -> str:
    """Detects the language of the text using Gemini."""
    prompt = f"Detect the language of the following text. Return ONLY the language name (e.g., 'English', 'Hindi', 'Spanish').\n\nText: {text}"
    return await get_gemini_response(prompt)


# =====================================================
# VERIFY ENDPOINT
# =====================================================
@app.post("/verify")
async def verify_text(data: TextInput):
    """
    Explainable credibility verification with Multilingual Support.
    """
    try:
        original_text = data.content
        
        # 1. Detect Language
        detected_lang = await detect_language(original_text)
        is_english = "english" in detected_lang.lower()
        
        # 2. Translate to English if needed (internally)
        text_to_analyze = original_text
        if not is_english:
            text_to_analyze = await translate_text(original_text, "English")
            
        # 3. Verify Facts (Using Gemini for trusted analysis)
        
        if is_english:
            prompt = f"""
            Analyze the credibility of the following educational text.
            Text: "{original_text}"
            
            Provide:
            1. A credibility score (0-100).
            2. A list of specific flags (e.g., "Overconfident language", "Lack of citation").
            3. A list of explanations.
            4. A final verdict ("Likely Reliable", "Needs Verification", etc.).
            5. A verification suggestion if the score is low.
            
            Output JSON strictly.
            """
        else:
             prompt = f"""
            Analyze the credibility of the educational text provided below.
            
            Original Text ({detected_lang}): "{original_text}"
            English Translation: "{text_to_analyze}"
            
            Tasks:
            1. Analyze the content for credibility issues, bias, or verification needs based on the English translation.
            2. Provide a credibility score (0-100).
            3. Identify specific flags (in English).
            4. Provide explanations. Each explanation MUST follow this format: 
               "'[Quote from Original Text in {detected_lang}]' - [Explanation in English]"
               Example: "'पृथ्वी सपाट आहे' - This claim contradicts scientific evidence."
            5. Provide a Final Verdict in {detected_lang}, followed by the English verdict in parentheses.
               Example: "{detected_lang} Verdict (English Verdict)"
            6. Provide a verification suggestion in English.
            
            Output JSON strictly.
            """
        
        json_structure = {
            "credibility_score": 0,
            "flags_detected": ["string"],
            "explanation": ["string"],
            "final_verdict": "string",
            "verification_suggestion": "string"
        }
        
        analysis_result = await get_gemini_response(prompt, json_structure)
        
        # 4. (Step removed) No post-translation needed as prompts handle the formatting directly.
                
        return analysis_result

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# LIVE MISINFORMATION REPAIR ENDPOINT
# =====================================================
@app.post("/repair", response_model=RepairResponse)
def repair_text(data: TextInput):
    """
    Live Misinformation Repair (MVP)

    - Does NOT claim absolute truth
    - Rewrites only risky phrasing
    - Preserves original meaning
    - Fully explainable
    """
    try:
        original_text = data.content
        text_lower = original_text.lower()

        needs_repair = False
        reasons = []

        # --- Reuse same risk logic ---
        if "definitely" in text_lower or "always" in text_lower:
            needs_repair = True
            reasons.append(
                "Replaced absolute language with evidence-aware phrasing."
            )

        if len(original_text.split()) < 10:
            needs_repair = True
            reasons.append(
                "Clarified content due to insufficient context."
            )

        # --- If no repair needed ---
        if not needs_repair:
            return {
                "original_text": original_text,
                "repaired_text": original_text,
                "repair_explanation": (
                    "No risky or misleading claims detected. No repair required."
                )
            }

        # --- Conservative repair logic (MVP-safe) ---
        repaired_text = original_text

        if "definitely" in text_lower:
            repaired_text = repaired_text.replace(
                "definitely",
                "according to available evidence"
            )

        if "always" in text_lower:
            repaired_text = repaired_text.replace(
                "always",
                "in most documented cases"
            )

        explanation = " ".join(reasons)

        return {
            "original_text": original_text,
            "repaired_text": repaired_text,
            "repair_explanation": explanation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# BIAS & PERSPECTIVE ANALYZER ENDPOINT
# =====================================================
@app.post("/analyze-bias")
async def analyze_bias(data: TextInput):
    """
    Bias & Perspective Analyzer using Gemini 1.5 Flash
    """
    try:
        text = data.content
        
        prompt = f"""
        Analyze the following text for bias and perspective. 
        Text: "{text}"
        
        Provide:
        1. A bias score (0-100, where 0 is neutral/unbiased and 100 is extremely biased).
        2. A bias level (Minimal, Low, Moderate, High, Severe).
        3. An overall assessment of the bias.
        4. Specific flags for bias (political, ideological, etc.).
        5. Issues with perspective (one-sided, lack of alternative views).
        6. Detected emotional or loaded language.
        7. Recommendations for improving objectivity.
        8. An objectivity rating (0-100).
        """
        
        json_structure = {
            "bias_score": 0,
            "bias_level": "string",
            "overall_assessment": "string",
            "bias_flags": ["string"],
            "perspective_issues": ["string"],
            "emotional_language_detected": ["string"],
            "recommendations": ["string"],
            "objectivity_rating": 0
        }
        
        return await get_gemini_response(prompt, json_structure)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# =====================================================
# XAI & INTUITIVE INFO ENDPOINT
# =====================================================
@app.post("/xai-info")
async def xai_info(data: TextInput):
    """
    XAI & Intuitive Info (Explainable AI) using Gemini 1.5 Flash
    """
    try:
        text = data.content
        
        prompt = f"""
        Provide Explainable AI (XAI) insights for the credibility analysis of the text below. 
        Text: "{text}"
        
        Explain:
        1. How confident the AI is in analyzing this text.
        2. Breakdown of confidence factors (source, length, etc.).
        3. Transparency about what methods were used (Generative AI logic).
        4. Key decision factors in the analysis.
        5. Known limitations of this specific analysis.
        6. Alternative interpretations of the text.
        7. Guidance for the user on how to interpret this analysis.
        8. A transparency note.
        """
        
        json_structure = {
            "overall_confidence": 0,
            "confidence_level": "string",
            "confidence_explanation": "string",
            "confidence_breakdown": {
                "text_length": 0,
                "source_available": 0,
                "clear_structure": 0,
                "technical_terms": 0
            },
            "analysis_methods_used": ["string"],
            "decision_factors": {
                "text_length_weight": "string",
                "source_credibility_weight": "string",
                "structural_clarity_weight": "string",
                "technical_vocabulary_weight": "string"
            },
            "known_limitations": ["string"],
            "alternative_interpretations": ["string"],
            "user_guidance": ["string"],
            "transparency_note": "string"
        }
        
        return await get_gemini_response(prompt, json_structure)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# =====================================================
# KNOWLEDGE GAP MAPPING ENDPOINT
# =====================================================
@app.post("/knowledge-gaps")
async def knowledge_gaps(data: TextInput):
    """
    Knowledge Gap Mapping using Gemini 1.5 Flash
    """
    try:
        text = data.content
        
        prompt = f"""
        Identify knowledge gaps in the following educational text.
        Text: "{text}"
        
        Provide:
        1. A completeness score (0-100).
        2. A qualitative assessment of completeness.
        3. Specific knowledge gaps or missing information.
        4. Missing context that would aid understanding.
        5. Prerequisite knowledge required to understand this text.
        6. Related topics that the user should explore.
        7. An analysis of the depth of the content (Surface, Moderate, Deep) with a score (0-100) and description.
        8. Coverage gaps (e.g., missing perspectives, examples).
        9. A recommended learning path (steps).
        10. Actionable recommendations for the learner.
        """
        
        json_structure = {
            "completeness_score": 0,
            "completeness_assessment": "string",
            "knowledge_gaps_identified": ["string"],
            "missing_context": ["string"],
            "prerequisite_knowledge": ["string"],
            "related_topics_to_explore": ["string"],
            "depth_analysis": {
                "level": "string",
                "score": 0.0,
                "description": "string"
            },
            "coverage_gaps": ["string"],
            "learning_path_recommendations": [
                { "step": 1, "focus": "string", "action": "string" }
            ],
            "actionable_recommendations": ["string"]
        }
        
        return await get_gemini_response(prompt, json_structure)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

