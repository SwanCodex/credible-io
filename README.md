# Credible.io – Explainable Credibility Analysis and Live Misinformation Repair

Credible.io is an AI-assisted system designed to combat misinformation in educational content by providing explainable credibility analysis and transparent, ethical misinformation repair.

This project was developed as part of **TechFiesta’26 – International Hackathon**.

---

## Problem Statement

Students increasingly rely on online resources and AI-generated content for learning.  
However, such content may contain hallucinations, misleading claims, overconfident language, or lack sufficient context.

Credible.io addresses this problem by:
- Evaluating the credibility of educational content
- Explaining why content may be risky
- Providing safe, explainable corrections without silently altering original material

---

## Key Features

### Multilingual Verification & Translation
- **Automatic Language Detection**: Instantly identifies the language of input text (e.g., Hindi, Marathi, Spanish).
- **Cross-Lingual Fact-Checking**: Translates content internally to English for deep verification against trusted sources.
- **Dual-Language Reporting**: 
    - Display verdicts in both the original language and English.
    - Provides specific explanations with **quoted text from the original language** alongside English reasoning.

### Advanced AI Analysis (Powered by Gemini)
- **Bias & Perspective Analysis**: Detects emotional language, political leaning, and potential objectivity issues.
- **Knowledge Gap Mapping**: Identifies missing context, prerequisites, and suggests a learning path.
- **XAI (Explainable AI)**: breaks down why the AI made a decision, boosting transparency.

### Explainable Credibility Verification
- Generates a credibility score based on factual accuracy and linguistic patterns.
- Detects risk patterns such as:
  - Overconfident or absolute language
  - Insufficient contextual information
- Provides clear explanations instead of black-box decisions.

### Settings & Configuration
- **Dynamic API Key Management**: Securely configure your Gemini API Key directly from the frontend settings.

### Browser Extension (Manifest V3)
- Extracts educational content directly from webpages
- Uses CSP-safe, user-triggered analysis
- Works on real-world websites without modifying page content

### Live Misinformation Repair
- Identifies risky or misleading phrasing
- Rewrites only the problematic portions of text
- Preserves original meaning and context
- Displays a before-and-after comparison using a non-intrusive overlay
- Ensures transparency by explaining every repair

---

## System Architecture

Browser Extension / Frontend
|
v
FastAPI Backend (Language Detection)
|
v
Google Gemini API (Translation & Verification)
|
v
Response Formatting (Dual-Language Explanations)
|
v
User Interface

---

## Technology Stack

- Backend: Python, FastAPI, Google Generative AI (Gemini)
- Frontend (Web): HTML, JavaScript
- Browser Extension: Chrome Manifest V3
- AI Logic: Gemini 3.0 Flash Preview / 1.5 Flash
- Architecture: Modular and extensible

---

## How to Run the Project

### Backend Setup

1. **Install Dependencies**:
```bash
pip install fastapi uvicorn google-generativeai python-dotenv
```

2. **Run the Server**:
```bash
uvicorn backend.main:app --reload
```

3. **Configure API Key**:
   - Open `frontend/index.html` in your browser.
   - Scroll to the **Settings** section.
   - Enter your Google Gemini API Key and click **Save**.

4. **Verify**:
   - Backend Runs at: http://127.0.0.1:8000
   - Swagger Documentation: http://127.0.0.1:8000/docs

## Project Status

- MVP completed
- Browser extension and backend fully integrated
- Live misinformation repair implemented
- Future work planned for claim-level fact verification using external sources

## Hackathon Highlights
- Real-time educational content analysis
- Explainable AI instead of opaque scoring
- Ethical, non-destructive misinformation repair
- Designed to support learning and critical thinking

## License
This project is licensed under the MIT License.
