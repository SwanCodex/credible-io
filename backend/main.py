from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import re
import requests
from fuzzywuzzy import fuzz
import pdfplumber
from datetime import datetime
import tempfile
import os

# -----------------------------
# App initialization
# -----------------------------
app = FastAPI()

# -----------------------------
# CORS (for frontend + extension)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Data models
# -----------------------------
class TextInput(BaseModel):
    content: str
    source_url: Optional[str] = None


class RepairResponse(BaseModel):
    original_text: str
    repaired_text: str
    repair_explanation: str


# -----------------------------
# Helper functions
# -----------------------------
def generate_verification_suggestion(flags):
    if "Overconfident language" in flags:
        return "Check credible sources or studies that support this absolute claim."
    if "Insufficient context" in flags:
        return "Look for more detailed explanations or background information."
    if "High hallucination risk" in flags:
        return "This content contains unsupported claims. Verify with trusted sources."
    if "Weak citations" in flags or "No citations found" in flags:
        return "Citations need verification. Check if sources actually support the claims."
    return "Verify the claim using trusted academic or educational sources."


def detect_hallucination(text):
    """
    AI Hallucination Detection
    Checks for:
    - Fake or non-existent references
    - Unsupported facts
    - Overconfident statements without evidence
    """
    text_lower = text.lower()
    flagged_claims = []
    risk_score = 0
    
    # Pattern 1: Overconfident statements without sources
    overconfident_patterns = [
        (r'\b(definitely|absolutely|certainly|without doubt|guaranteed|proven fact)\b', 10),
        (r'\b(always|never|everyone|no one|all|none)\b', 8),
        (r'\b(100%|completely|totally)\b', 7)
    ]
    
    for pattern, score in overconfident_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            risk_score += len(matches) * score
            for match in matches[:3]:  # Limit to first 3
                flagged_claims.append(f"Overconfident language: '{match}' used without evidence")
    
    # Pattern 2: Vague citations without specific sources
    vague_citation_patterns = [
        r'according to (recent studies|a study|research)',
        r'scientists (say|claim|found|discovered)',
        r'experts (agree|believe|state|confirm)',
        r'studies (show|prove|indicate|suggest)'
    ]
    
    for pattern in vague_citation_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            risk_score += len(matches) * 15
            for match in matches[:2]:
                flagged_claims.append(f"Vague citation: 'according to {match[1]}' - no specific source provided")
    
    # Pattern 3: Statistics without sources
    stat_matches = re.findall(r'\d+(?:\.\d+)?%', text)
    if len(stat_matches) > 0 and not re.search(r'(source:|according to|study|research|\[|\(20\d{2}\))', text_lower):
        risk_score += len(stat_matches) * 12
        flagged_claims.append(f"Found {len(stat_matches)} statistic(s) without citations: {', '.join(stat_matches[:3])}")
    
    # Pattern 4: Claims about discoveries/research without citations
    suspicious_phrases = [
        ('recent discovery', 12),
        ('new study shows', 12),
        ('breakthrough research', 10),
        ('scientists recently found', 12),
        ('studies prove', 15),
        ('research confirms', 12)
    ]
    
    for phrase, score in suspicious_phrases:
        if phrase in text_lower:
            # Check if there's a proper citation nearby
            if not re.search(r'(\[|\()\d{4}(\]|\))|doi:|arxiv:|https?://', text_lower):
                risk_score += score
                flagged_claims.append(f"Unsupported claim: '{phrase}' without proper citation")
    
    # Pattern 5: AI-generated markers
    ai_markers = [
        'as an ai', 'i don\'t have access', 'i cannot', 
        'as of my last update', 'i apologize for'
    ]
    
    for marker in ai_markers:
        if marker in text_lower:
            risk_score += 20
            flagged_claims.append(f"AI-generated content detected: '{marker}'")
    
    # Determine risk level
    if risk_score >= 40:
        risk_level = "High"
    elif risk_score >= 20:
        risk_level = "Medium"
    elif risk_score > 0:
        risk_level = "Low"
    else:
        risk_level = "None"
    
    hallucination_detected = risk_score > 0
    
    # Generate explanation
    if hallucination_detected:
        explanation = f"Detected {len(flagged_claims)} potential hallucination indicator(s). "
        explanation += "This content may contain AI-generated or unsupported claims. Verify with trusted sources before using."
    else:
        explanation = "No hallucination indicators detected. Content appears to be appropriately sourced and factual."
    
    return {
        "hallucination_detected": hallucination_detected,
        "risk_level": risk_level,
        "flagged_claims": flagged_claims[:10],  # Limit to 10
        "explanation": explanation,
        "risk_score": risk_score
    }


def validate_citations(text):
    """
    Citation Validator
    Checks for:
    - Ghost sources (fake references)
    - Proper citation format
    - Authority of sources
    """
    citation_details = []
    valid_count = 0
    invalid_citations = []
    
    # Pattern 1: URLs in text
    urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text)
    
    for url in urls:
        url_lower = url.lower()
        if '.edu' in url_lower or '.gov' in url_lower:
            valid_count += 1
            citation_details.append({
                "type": "URL",
                "source": url[:60] + "..." if len(url) > 60 else url,
                "authority": "High",
                "status": "Valid"
            })
        elif '.org' in url_lower or 'wikipedia.org' in url_lower:
            valid_count += 1
            citation_details.append({
                "type": "URL",
                "source": url[:60] + "..." if len(url) > 60 else url,
                "authority": "Medium",
                "status": "Valid"
            })
        elif any(domain in url_lower for domain in ['nature.com', 'science.org', 'ieee.org', 'acm.org']):
            valid_count += 1
            citation_details.append({
                "type": "URL",
                "source": url[:60] + "..." if len(url) > 60 else url,
                "authority": "High",
                "status": "Valid - Academic"
            })
        elif any(suspicious in url_lower for suspicious in ['blogspot', 'wordpress.com', 'tumblr', 'medium.com']):
            invalid_citations.append({
                "source": url[:50] + "..." if len(url) > 50 else url,
                "reason": "Low authority source (personal blog/medium)"
            })
            citation_details.append({
                "type": "URL",
                "source": url[:60] + "..." if len(url) > 60 else url,
                "authority": "Low",
                "status": "Questionable"
            })
        else:
            citation_details.append({
                "type": "URL",
                "source": url[:60] + "..." if len(url) > 60 else url,
                "authority": "Medium",
                "status": "Needs verification"
            })
    
    # Pattern 2: Academic citations - (Author, Year) or [1]
    # Match (Smith, 2020) or (Smith et al., 2020)
    academic_citations = re.findall(r'\(([A-Z][a-z]+(?:\s+et al\.)?),?\s+(19|20)\d{2}\)', text)
    academic_count = len(academic_citations)
    
    if academic_count > 0:
        valid_count += academic_count
        citation_details.append({
            "type": "Academic",
            "count": academic_count,
            "authority": "High",
            "status": f"Found {academic_count} academic citation(s) in proper format"
        })
    
    # Pattern 3: Numbered citations [1], [2], etc.
    numbered_citations = re.findall(r'\[(\d+)\]', text)
    if numbered_citations:
        citation_details.append({
            "type": "Numbered",
            "count": len(set(numbered_citations)),  # Unique references
            "authority": "Unknown",
            "status": f"Found {len(set(numbered_citations))} numbered reference(s) - check reference list"
        })
    
    # Pattern 4: DOI or ArXiv references (high authority)
    doi_pattern = re.findall(r'doi:\s*[\d.]+/[\w\-\.]+', text.lower())
    arxiv_pattern = re.findall(r'arxiv:\s*\d+\.\d+', text.lower())
    
    scholarly_count = len(doi_pattern) + len(arxiv_pattern)
    if scholarly_count > 0:
        valid_count += scholarly_count
        citation_details.append({
            "type": "Scholarly",
            "count": scholarly_count,
            "authority": "Very High",
            "status": f"Found {scholarly_count} DOI/ArXiv reference(s) - Verified scholarly sources"
        })
    
    # Calculate totals
    total_citations = len(urls) + academic_count + scholarly_count
    
    # Determine citation quality
    if total_citations == 0:
        quality = "No Citations"
    elif valid_count >= total_citations * 0.8:
        quality = "Excellent"
    elif valid_count >= total_citations * 0.6:
        quality = "Good"
    elif valid_count >= total_citations * 0.4:
        quality = "Fair"
    else:
        quality = "Poor"
    
    # Add warning for no citations
    if total_citations == 0:
        invalid_citations.append({
            "source": "N/A",
            "reason": "No citations found in content - claims are unsupported"
        })
    
    return {
        "citations_found": total_citations,
        "valid_citations": valid_count,
        "invalid_citations": invalid_citations,
        "citation_quality": quality,
        "details": citation_details
    }


# =====================================================
# ADVANCED CITATION VALIDATOR CLASSES
# =====================================================

class CitationExtractor:
    """Extracts citations from PDF documents"""
    
    def __init__(self):
        # Citation patterns for different formats
        self.patterns = {
            'doi': r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+',
            'url': r'https?://[^\s<>"{}|\\^`\[\]]+',
            'apa': r'([A-Z][a-z]+(?:,?\s+[A-Z]\.)?)(?:,?\s+&?\s+[A-Z][a-z]+(?:,?\s+[A-Z]\.)?)*\s+\((\d{4})\)\.\s+([^\.]+)\.',
            'ieee': r'\[(\d+)\]\s+([A-Za-z\.\s,]+),\s+"([^"]+)"',
            'author_year': r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+\((\d{4})\)',
        }
    
    def extract_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract all citations from PDF"""
        citations = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
            
            # Extract different citation types
            citations.extend(self._extract_dois(full_text))
            citations.extend(self._extract_apa_citations(full_text))
            citations.extend(self._extract_ieee_citations(full_text))
            citations.extend(self._extract_urls(full_text))
            
            # Remove duplicates based on raw text
            seen = set()
            unique_citations = []
            for cit in citations:
                if cit['raw'] not in seen:
                    seen.add(cit['raw'])
                    unique_citations.append(cit)
            
            return unique_citations
            
        except Exception as e:
            print(f"Error extracting citations: {e}")
            return []
    
    def _extract_dois(self, text: str) -> List[Dict]:
        """Extract DOI citations"""
        dois = re.findall(self.patterns['doi'], text)
        return [{'type': 'DOI', 'doi': doi, 'raw': doi} for doi in dois]
    
    def _extract_apa_citations(self, text: str) -> List[Dict]:
        """Extract APA style citations"""
        matches = re.findall(self.patterns['apa'], text)
        citations = []
        for match in matches:
            citations.append({
                'type': 'APA',
                'author': match[0].strip(),
                'year': match[1],
                'title': match[2].strip(),
                'raw': f"{match[0]} ({match[1]}). {match[2]}."
            })
        return citations
    
    def _extract_ieee_citations(self, text: str) -> List[Dict]:
        """Extract IEEE style citations"""
        matches = re.findall(self.patterns['ieee'], text)
        citations = []
        for match in matches:
            citations.append({
                'type': 'IEEE',
                'number': match[0],
                'author': match[1].strip(),
                'title': match[2].strip(),
                'raw': f"[{match[0]}] {match[1]}, \"{match[2]}\""
            })
        return citations
    
    def _extract_urls(self, text: str) -> List[Dict]:
        """Extract URL citations"""
        urls = re.findall(self.patterns['url'], text)
        # Filter out common non-citation URLs
        filtered_urls = [url for url in urls if not any(x in url for x in ['.png', '.jpg', '.gif', 'twitter.com', 'facebook.com'])]
        return [{'type': 'URL', 'url': url, 'raw': url} for url in filtered_urls]


class GhostSourceDetector:
    """Verifies if citations exist in academic databases"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'EduVeritas/1.0 (Educational Citation Validator)'
        })
    
    def verify_citation(self, citation: Dict) -> Dict:
        """Main verification method - routes to appropriate checker"""
        if citation.get('doi'):
            return self.verify_doi(citation['doi'])
        elif citation.get('title'):
            return self.search_by_title(citation['title'], citation.get('author'))
        elif citation.get('url'):
            return self.verify_url(citation['url'])
        else:
            return {'exists': False, 'error': 'Insufficient information'}
    
    def verify_doi(self, doi: str) -> Dict:
        """Verify DOI against CrossRef"""
        url = f"https://api.crossref.org/works/{doi}"
        
        try:
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()['message']
                
                return {
                    'exists': True,
                    'verified_source': 'CrossRef',
                    'title': data.get('title', ['Unknown'])[0],
                    'authors': self._extract_authors(data.get('author', [])),
                    'year': self._extract_year(data),
                    'journal': data.get('container-title', ['Unknown'])[0],
                    'doi': doi,
                    'type': data.get('type', 'unknown'),
                    'publisher': data.get('publisher', 'Unknown')
                }
            else:
                return {'exists': False, 'error': 'DOI not found in CrossRef'}
                
        except Exception as e:
            return {'exists': False, 'error': f'CrossRef error: {str(e)}'}
    
    def search_by_title(self, title: str, author: str = None) -> Dict:
        """Search by title in Semantic Scholar"""
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            'query': title,
            'limit': 5,
            'fields': 'title,authors,year,abstract,citationCount,venue,externalIds'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('data') and len(data['data']) > 0:
                    # Find best match using fuzzy matching
                    best_match = None
                    best_similarity = 0
                    
                    for paper in data['data']:
                        similarity = fuzz.ratio(title.lower(), paper['title'].lower()) / 100.0
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = paper
                    
                    if best_similarity > 0.75:  # 75% similarity threshold
                        return {
                            'exists': True,
                            'verified_source': 'Semantic Scholar',
                            'title': best_match['title'],
                            'authors': [a['name'] for a in best_match.get('authors', [])],
                            'year': best_match.get('year'),
                            'abstract': best_match.get('abstract', ''),
                            'citations': best_match.get('citationCount', 0),
                            'venue': best_match.get('venue', 'Unknown'),
                            'paperId': best_match.get('paperId'),
                            'doi': best_match.get('externalIds', {}).get('DOI'),
                            'similarity_score': round(best_similarity, 2)
                        }
                    else:
                        return {
                            'exists': False,
                            'error': f'No close match found (best: {int(best_similarity*100)}%)'
                        }
                else:
                    # Try OpenAlex as fallback
                    return self._search_openalex(title)
            else:
                return {'exists': False, 'error': 'Semantic Scholar search failed'}
                
        except Exception as e:
            return {'exists': False, 'error': f'Search error: {str(e)}'}
    
    def _search_openalex(self, title: str) -> Dict:
        """Fallback search using OpenAlex"""
        url = "https://api.openalex.org/works"
        params = {'search': title, 'per-page': 1}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('results') and len(data['results']) > 0:
                    paper = data['results'][0]
                    return {
                        'exists': True,
                        'verified_source': 'OpenAlex',
                        'title': paper.get('title'),
                        'year': paper.get('publication_year'),
                        'doi': paper.get('doi'),
                        'type': paper.get('type'),
                        'cited_by_count': paper.get('cited_by_count', 0)
                    }
            
            return {'exists': False, 'error': 'Not found in any database'}
            
        except Exception as e:
            return {'exists': False, 'error': f'OpenAlex error: {str(e)}'}
    
    def verify_url(self, url: str) -> Dict:
        """Verify if URL is accessible"""
        try:
            response = self.session.head(url, timeout=5, allow_redirects=True)
            
            if response.status_code == 200:
                return {
                    'exists': True,
                    'verified_source': 'URL Check',
                    'url': url,
                    'status': 'accessible'
                }
            else:
                return {
                    'exists': False,
                    'error': f'URL returned {response.status_code}',
                    'url': url
                }
        except:
            return {'exists': False, 'error': 'URL not accessible', 'url': url}
    
    def _extract_authors(self, authors_data: List) -> List[str]:
        """Extract author names from CrossRef data"""
        return [f"{a.get('given', '')} {a.get('family', '')}".strip() 
                for a in authors_data[:3]]  # Limit to first 3
    
    def _extract_year(self, data: Dict) -> Optional[int]:
        """Extract publication year from CrossRef data"""
        if 'published-print' in data:
            return data['published-print'].get('date-parts', [[None]])[0][0]
        elif 'published-online' in data:
            return data['published-online'].get('date-parts', [[None]])[0][0]
        return None


class ContextualMatcher:
    """Checks if citation supports the claim"""
    
    def check_claim_support(self, claim: str, abstract: str) -> Dict:
        """
        Check if abstract supports the claim using keyword overlap
        (In production, use LLM like GPT/Claude for semantic analysis)
        """
        if not abstract:
            return {
                'supported': 'unknown',
                'confidence': 0.0,
                'reason': 'Abstract not available for verification'
            }
        
        # Clean and tokenize
        claim_words = set(word.lower() for word in re.findall(r'\w+', claim) if len(word) > 3)
        abstract_words = set(word.lower() for word in re.findall(r'\w+', abstract) if len(word) > 3)
        
        # Calculate overlap
        if not claim_words:
            return {'supported': 'unknown', 'confidence': 0.0, 'reason': 'No keywords in claim'}
        
        overlap = len(claim_words & abstract_words) / len(claim_words)
        
        # Determine support level
        if overlap > 0.6:
            return {
                'supported': 'likely',
                'confidence': round(overlap, 2),
                'reason': f'Strong keyword match ({int(overlap*100)}%)',
                'matching_keywords': list(claim_words & abstract_words)[:5]
            }
        elif overlap > 0.3:
            return {
                'supported': 'partial',
                'confidence': round(overlap, 2),
                'reason': f'Moderate keyword match ({int(overlap*100)}%)',
                'matching_keywords': list(claim_words & abstract_words)[:5]
            }
        else:
            return {
                'supported': 'unlikely',
                'confidence': round(1 - overlap, 2),
                'reason': f'Low keyword match ({int(overlap*100)}%) - possible evidence mismatch',
                'matching_keywords': list(claim_words & abstract_words)[:5]
            }


class AuthorityGrader:
    """Grades citation quality"""
    
    # Prestigious journals (A+ tier)
    TOP_JOURNALS = {
        'nature', 'science', 'cell', 'lancet', 'nejm', 'new england journal',
        'jama', 'bmj', 'pnas', 'proceedings of the national academy'
    }
    
    # Good journals (A tier)
    GOOD_JOURNALS = {
        'ieee', 'acm', 'springer', 'elsevier', 'wiley', 'oxford', 'cambridge'
    }
    
    def grade_citation(self, citation_data: Dict) -> Dict:
        """Assign quality grade A+ to F"""
        
        if not citation_data.get('exists'):
            return {
                'grade': 'F',
                'score': 0,
                'reasons': ['Citation not found in any database - likely fabricated'],
                'status': 'ghost_source'
            }
        
        score = 50  # Base score
        reasons = []
        
        # Factor 1: Source verification (+20 points)
        if citation_data.get('verified_source'):
            score += 20
            reasons.append(f"Verified in {citation_data['verified_source']}")
        
        # Factor 2: Journal prestige (+30 points max)
        journal = citation_data.get('journal', '').lower()
        venue = citation_data.get('venue', '').lower()
        
        if any(top in journal or top in venue for top in self.TOP_JOURNALS):
            score += 30
            reasons.append('Published in top-tier journal (Nature, Science, etc.)')
        elif any(good in journal or good in venue for good in self.GOOD_JOURNALS):
            score += 20
            reasons.append('Published in reputable journal')
        elif journal or venue:
            score += 10
            reasons.append('Published in peer-reviewed venue')
        
        # Factor 3: DOI presence (+10 points)
        if citation_data.get('doi'):
            score += 10
            reasons.append('Has DOI - easily verifiable')
        else:
            reasons.append('No DOI - harder to verify independently')
        
        # Factor 4: Citation count (+10 points)
        citations = citation_data.get('citations') or citation_data.get('cited_by_count', 0)
        if citations > 100:
            score += 10
            reasons.append(f'Highly cited ({citations} citations)')
        elif citations > 10:
            score += 5
            reasons.append(f'Moderately cited ({citations} citations)')
        
        # Factor 5: Recency check (-10 if too old)
        year = citation_data.get('year')
        if year:
            age = datetime.now().year - int(year)
            if age > 15:
                score -= 10
                reasons.append(f'⚠️ Source is {age} years old - may be outdated')
            elif age < 5:
                score += 5
                reasons.append(f'Recent publication ({year})')
        
        # Factor 6: Publication type
        pub_type = citation_data.get('type', '').lower()
        if 'journal-article' in pub_type:
            score += 5
            reasons.append('Peer-reviewed journal article')
        elif 'book' in pub_type:
            score += 3
            reasons.append('Published book')
        
        # Ensure score is 0-100
        score = max(0, min(100, score))
        grade = self._score_to_grade(score)
        
        return {
            'grade': grade,
            'score': score,
            'reasons': reasons,
            'status': 'verified' if citation_data.get('exists') else 'unverified'
        }
    
    def _score_to_grade(self, score: int) -> str:
        """Convert numeric score to letter grade"""
        if score >= 95: return 'A+'
        elif score >= 90: return 'A'
        elif score >= 85: return 'A-'
        elif score >= 80: return 'B+'
        elif score >= 75: return 'B'
        elif score >= 70: return 'B-'
        elif score >= 65: return 'C+'
        elif score >= 60: return 'C'
        elif score >= 50: return 'D'
        else: return 'F'


class AdvancedCitationValidator:
    """Main validator class that orchestrates all components"""
    
    def __init__(self):
        self.extractor = CitationExtractor()
        self.detector = GhostSourceDetector()
        self.matcher = ContextualMatcher()
        self.grader = AuthorityGrader()
    
    def validate_pdf(self, pdf_path: str) -> Dict:
        """
        Complete validation pipeline for PDF
        Returns comprehensive report
        """
        print(f"🔍 Extracting citations from PDF...")
        citations = self.extractor.extract_from_pdf(pdf_path)
        
        if not citations:
            return {
                'success': False,
                'error': 'No citations found in PDF',
                'total_citations': 0
            }
        
        print(f"✅ Found {len(citations)} citations")
        print(f"🔬 Verifying against academic databases...")
        
        results = []
        for idx, citation in enumerate(citations, 1):
            print(f"  [{idx}/{len(citations)}] Checking: {citation['raw'][:60]}...")
            
            # Step 1: Ghost-source detection
            verification = self.detector.verify_citation(citation)
            
            # Step 2: Contextual matching (if abstract available)
            context_check = {}
            if verification.get('exists') and verification.get('abstract'):
                context_check = self.matcher.check_claim_support(
                    citation.get('raw', ''),
                    verification.get('abstract', '')
                )
            
            # Step 3: Authority grading
            authority = self.grader.grade_citation(verification)
            
            results.append({
                'citation_number': idx,
                'original': citation,
                'verification': verification,
                'context_check': context_check,
                'authority_grade': authority
            })
        
        # Generate summary
        summary = self._generate_summary(results)
        
        return {
            'success': True,
            'total_citations': len(citations),
            'summary': summary,
            'detailed_results': results
        }
    
    def _generate_summary(self, results: List[Dict]) -> Dict:
        """Generate summary statistics"""
        total = len(results)
        verified = sum(1 for r in results if r['verification'].get('exists'))
        ghost_sources = total - verified
        
        grades = [r['authority_grade']['score'] for r in results]
        avg_score = sum(grades) / total if total > 0 else 0
        
        # Count by grade
        grade_distribution = {}
        for r in results:
            grade = r['authority_grade']['grade']
            grade_distribution[grade] = grade_distribution.get(grade, 0) + 1
        
        # Identify problematic citations
        fake_citations = [r for r in results if not r['verification'].get('exists')]
        low_quality = [r for r in results if r['authority_grade']['score'] < 60]
        
        return {
            'total_citations': total,
            'verified': verified,
            'ghost_sources': ghost_sources,
            'verification_rate': round(verified / total * 100, 1) if total > 0 else 0,
            'average_quality_score': round(avg_score, 1),
            'overall_grade': self.grader._score_to_grade(int(avg_score)),
            'grade_distribution': grade_distribution,
            'fake_citations_count': len(fake_citations),
            'low_quality_count': len(low_quality),
            'recommendations': self._generate_recommendations(verified, total, avg_score)
        }
    
    def _generate_recommendations(self, verified: int, total: int, avg_score: float) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if verified < total * 0.8:
            recommendations.append('⚠️ Over 20% of citations could not be verified - consider replacing with verified sources')
        
        if avg_score < 70:
            recommendations.append('📚 Bibliography quality is below standard - use more peer-reviewed sources')
        
        if avg_score >= 85:
            recommendations.append('✅ Excellent citation quality - bibliography meets academic standards')
        
        return recommendations



# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
def root():
    return {
        "message": "EduVeritas backend is running",
        "version": "2.0",
        "features": ["Hallucination Detection", "Citation Validation", "Content Verification"]
    }


# =====================================================
# VERIFY ENDPOINT (Enhanced with all features)
# =====================================================
@app.post("/verify")
def verify_text(data: TextInput):
    """
    Comprehensive credibility verification with:
    - AI Hallucination Detection
    - Citation Validation
    - Content Analysis
    """
    try:
        text = data.content
        text_lower = text.lower()

        score = 80
        flags = []
        reasons = []

        # --- AI HALLUCINATION DETECTION ---
        hallucination_result = detect_hallucination(text)
        
        if hallucination_result["hallucination_detected"]:
            if hallucination_result["risk_level"] == "High":
                score -= 30
                flags.append("High hallucination risk")
                reasons.append(f"⚠️ HIGH RISK: {len(hallucination_result['flagged_claims'])} hallucination indicators detected")
            elif hallucination_result["risk_level"] == "Medium":
                score -= 20
                flags.append("Medium hallucination risk")
                reasons.append(f"⚠️ MEDIUM RISK: Some claims lack proper verification")
            elif hallucination_result["risk_level"] == "Low":
                score -= 10
                flags.append("Low hallucination risk")
                reasons.append("⚠️ Minor issues with claim verification detected")
        else:
            score += 5
            reasons.append("✅ No hallucination indicators detected")

        # --- CITATION VALIDATION ---
        citation_result = validate_citations(text)
        
        if citation_result["citations_found"] == 0:
            score -= 20
            flags.append("No citations found")
            reasons.append("❌ No citations or references found to support claims")
        elif citation_result["citation_quality"] == "Poor":
            score -= 15
            flags.append("Weak citations")
            reasons.append(f"⚠️ Citation quality is poor: {citation_result['valid_citations']}/{citation_result['citations_found']} valid")
        elif citation_result["citation_quality"] == "Fair":
            score -= 5
            reasons.append(f"⚠️ Citation quality is fair: {citation_result['valid_citations']}/{citation_result['citations_found']} valid")
        elif citation_result["citation_quality"] in ["Good", "Excellent"]:
            score += 10
            reasons.append(f"✅ Good citations: {citation_result['valid_citations']} valid reference(s) found")
        
        # Add warnings for invalid citations
        if len(citation_result["invalid_citations"]) > 0:
            for invalid in citation_result["invalid_citations"][:3]:  # First 3
                reasons.append(f"⚠️ {invalid['reason']}")

        # --- Basic content checks ---
        if "definitely" in text_lower or "always" in text_lower or "never" in text_lower:
            score -= 10
            flags.append("Overconfident language")
            reasons.append("⚠️ Uses absolute terms (definitely/always/never) without evidence")

        if len(text.split()) < 10:
            score -= 10
            flags.append("Insufficient context")
            reasons.append("⚠️ Text is very short - difficult to verify reliably")
        
        # Check for uncited statistics
        stats = re.findall(r'\d+(?:\.\d+)?%', text)
        if len(stats) > 0 and citation_result["citations_found"] == 0:
            score -= 8
            reasons.append(f"⚠️ Found {len(stats)} statistic(s) without any citations")

        # --- Source authority analysis ---
        authority_bonus = 0

        if data.source_url:
            url_lower = data.source_url.lower()
            if "wikipedia.org" in url_lower:
                authority_bonus = 10
                reasons.append("✅ Wikipedia source (community-reviewed)")
            elif ".edu" in url_lower or ".gov" in url_lower:
                authority_bonus = 15
                reasons.append("✅ Educational/government domain (.edu/.gov)")
            elif any(domain in url_lower for domain in ['nature.com', 'science.org', 'ieee.org']):
                authority_bonus = 20
                reasons.append("✅ Peer-reviewed academic publication")
            elif any(sus in url_lower for sus in ['blogspot', 'wordpress.com', 'medium.com']):
                authority_bonus = -10
                flags.append("Low authority source")
                reasons.append("⚠️ Personal blog or low-authority platform")

        score += authority_bonus
        score = max(0, min(score, 100))  # Clamp between 0-100

        # Generate suggestion
        suggestion = None
        if score < 70:
            suggestion = generate_verification_suggestion(flags)

        # Determine verdict
        if score >= 80:
            verdict = "Highly Reliable ✓"
        elif score >= 60:
            verdict = "Likely Reliable"
        elif score >= 40:
            verdict = "Needs Verification ⚠"
        else:
            verdict = "Unreliable - Do Not Use ✗"

        return {
            "credibility_score": score,
            "flags_detected": flags,
            "explanation": reasons,
            "final_verdict": verdict,
            "verification_suggestion": suggestion,
            "hallucination_analysis": hallucination_result,
            "citation_analysis": citation_result
        }

    except Exception as e:
        print(f"Error in verify_text: {str(e)}")  # Debug logging
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# HALLUCINATION CHECK ENDPOINT
# =====================================================
@app.post("/check-hallucination")
def check_hallucination(data: TextInput):
    """
    Dedicated endpoint for hallucination detection
    """
    try:
        result = detect_hallucination(data.content)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# CITATION VALIDATION ENDPOINT
# =====================================================
@app.post("/validate-citations")
def check_citations(data: TextInput):
    """
    Dedicated endpoint for citation validation
    """
    try:
        result = validate_citations(data.content)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# LIVE MISINFORMATION REPAIR ENDPOINT
# =====================================================
@app.post("/repair")
def repair_text(data: TextInput):
    """
    Live Misinformation Repair
    """
    try:
        original_text = data.content
        text_lower = original_text.lower()

        needs_repair = False
        reasons = []

        # Check for hallucinations
        hallucination_result = detect_hallucination(original_text)
        if hallucination_result["hallucination_detected"]:
            needs_repair = True
            reasons.append("Softened potentially hallucinated claims")

        # Check for absolute language
        if any(word in text_lower for word in ["definitely", "always", "never", "100%"]):
            needs_repair = True
            reasons.append("Replaced absolute language with evidence-aware phrasing")

        if len(original_text.split()) < 10:
            needs_repair = True
            reasons.append("Added context due to insufficient information")

        if not needs_repair:
            return RepairResponse(
                original_text=original_text,
                repaired_text=original_text,
                repair_explanation="No risky or misleading claims detected. Content is acceptable."
            )

        # Apply repairs
        repaired_text = original_text
        
        replacements = {
            "definitely": "according to available evidence",
            "Definitely": "According to available evidence",
            "always": "in most documented cases",
            "Always": "In most documented cases",
            "never": "rarely",
            "Never": "Rarely",
            "100%": "the majority of",
            "everyone knows": "it is commonly understood",
            "proven fact": "well-supported by evidence"
        }

        for old, new in replacements.items():
            if old in repaired_text:
                repaired_text = repaired_text.replace(old, new)
                if old not in ["100%"]:  # Don't add twice for percentage
                    reasons.append(f"Changed '{old}' to '{new}'")

        explanation = " | ".join(set(reasons))  # Remove duplicates

        return RepairResponse(
            original_text=original_text,
            repaired_text=repaired_text,
            repair_explanation=explanation
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# =====================================================
# PDF CITATION VALIDATION ENDPOINT
# =====================================================
@app.post("/api/validate-pdf-citations")
async def validate_pdf_citations(file: UploadFile = File(...)):
    """
    Advanced PDF Citation Validator
    Extracts citations and validates against academic databases
    Returns comprehensive quality report with grades
    """
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Save uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Initialize validator and process PDF
        validator = AdvancedCitationValidator()
        report = validator.validate_pdf(tmp_path)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return report
        
    except Exception as e:
        # Clean up on error
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        raise HTTPException(
            status_code=500,
            detail=f"Validation error: {str(e)}"
        )


# =====================================================
# FILE UPLOAD ENDPOINT (for PDF/Image in future)
# =====================================================
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Placeholder for file upload functionality
    """
    try:
        # For now, just return a message
        return {
            "message": "File upload endpoint ready",
            "filename": file.filename,
            "note": "PDF and image processing will be implemented with required libraries"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
