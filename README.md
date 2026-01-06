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

### Explainable Credibility Verification
- Generates a heuristic credibility score
- Detects linguistic risk patterns such as:
  - Overconfident or absolute language
  - Insufficient contextual information
- Provides clear explanations instead of black-box decisions

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

### Ethical and Transparent AI Design
- No silent rewriting of content
- No blind trust in sources
- Users remain in control of what they read and reuse

---

## System Architecture

Browser Extension
|
v
Content Script (Text Extraction)
|
v
Background Service Worker
|
v
FastAPI Backend
| |
/verify /repair

---

## Technology Stack

- Backend: Python, FastAPI
- Frontend (Web): HTML, JavaScript
- Browser Extension: Chrome Manifest V3
- AI Logic: Explainable, rule-based heuristics (MVP)
- Architecture: Modular and extensible

---

## How to Run the Project

### Backend Setup

```bash
pip install fastapi uvicorn
uvicorn main:app --reload

Backend Runs at: http://127.0.0.1:8000
Interactive API documentation: http://127.0.0.1:8000/docs

## Project Status

- MVP completed
- Browser extension and backend fully integrated
- Live misinformation repair implemented
- Future work planned for claim-level fact verification using external sources
- Hackathon Highlights
- Real-time educational content analysis
- Explainable AI instead of opaque scoring
- Ethical, non-destructive misinformation repair
- Designed to support learning and critical thinking

## License
This project is licensed under the MIT License.
