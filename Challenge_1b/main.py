import json
import os
import datetime
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import torch
import re
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string

# Initialize NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# --- Configuration ---
DEVICE = "cpu"
MODEL_NAME = 'all-MiniLM-L6-v2'
MAX_SECTIONS = 5  # Maximum number of sections to return
MIN_SECTION_LENGTH = 50  # Minimum characters for a section to be considered
MAX_SUBSECTION_LENGTH = 1500  # Character limit for refined text
REJECTED_TITLES = {
    "introduction", "conclusion", "contents", "table of contents",
    "references", "bibliography", "appendix", "acknowledgements", "index"
}

class DocumentAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)
    
    def is_bold(self, span: dict) -> bool:
        """Checks if the font flags indicate a bold style."""
        return span['flags'] & 16 > 0
    
    def get_document_styles(self, doc: fitz.Document) -> dict:
        """Analyzes the document to find common text styles."""
        styles = defaultdict(int)
        for page in doc:
            blocks = page.get_text("dict").get('blocks', [])
            for block in blocks:
                if 'lines' in block:
                    for line in block['lines']:
                        for span in line['spans']:
                            style_key = (round(span['size']), self.is_bold(span), span['font'])
                            styles[style_key] += len(span['text'])
        return styles
    
    def clean_text(self, text: str) -> str:
        """Cleans text by removing excessive whitespace and special characters."""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s\-.,;:!?]', '', text)
        return text
    
    def parse_pdf_sections(self, pdf_path: str) -> List[Dict]:
        """Robust PDF parser using style analysis."""
        doc = fitz.open(pdf_path)
        if not doc or doc.page_count == 0:
            return []
        
        styles = self.get_document_styles(doc)
        if not styles:
            return []
        
        # Determine body text style (most common)
        body_style = max(styles.items(), key=lambda x: x[1])[0]
        body_size, body_is_bold, _ = body_style
        
        sections = []
        current_title = "Document"
        current_content = ""
        section_start_page = 1

        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("dict").get('blocks', [])
            for block in blocks:
                if 'lines' not in block:
                    continue
                    
                block_text = " ".join(" ".join(span['text'] for span in line['spans']) 
                            for line in block['lines']).strip()
                block_text = self.clean_text(block_text)
                if not block_text:
                    continue

                # Determine if this is a header
                is_header = False
                try:
                    first_span = block['lines'][0]['spans'][0]
                    span_size = round(first_span['size'])
                    span_is_bold = self.is_bold(first_span)
                    
                    # Header detection logic
                    header_score = 0
                    if span_size > body_size:
                        header_score += 2
                    if span_is_bold and not body_is_bold:
                        header_score += 2
                    if len(block_text.split()) < 10 and "\n" not in block_text:
                        header_score += 1
                    if header_score >= 3:
                        is_header = True
                except (IndexError, KeyError):
                    pass

                if is_header:
                    if current_content.strip() and len(current_content) >= MIN_SECTION_LENGTH:
                        sections.append({
                            "document": os.path.basename(pdf_path),
                            "page_number": section_start_page,
                            "section_title": current_title,
                            "content": current_content
                        })
                    current_title = block_text
                    current_content = ""
                    section_start_page = page_num
                else:
                    current_content += " " + block_text if current_content else block_text

        if current_content.strip() and len(current_content) >= MIN_SECTION_LENGTH:
            sections.append({
                "document": os.path.basename(pdf_path),
                "page_number": section_start_page,
                "section_title": current_title,
                "content": current_content
            })
        
        doc.close()
        return sections
    
    def extract_keyphrases(self, text: str, n: int = 5) -> List[str]:
        """Extracts keyphrases from text using simple frequency analysis."""
        words = [w.lower() for w in word_tokenize(text) 
                if w.lower() not in self.stop_words and w not in self.punctuation]
        
        # Create 2-word phrases
        phrases = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        
        # Count and return most common
        freq_dist = nltk.FreqDist(phrases + words)
        return [item for item, _ in freq_dist.most_common(n)]
    
    def extract_exclusion_terms(self, job_task: str) -> List[str]:
        """Identifies terms that should be excluded based on job requirements."""
        exclusion_terms = []
        job_lower = job_task.lower()
        
        # Common exclusion patterns
        if "vegetarian" in job_lower:
            exclusion_terms.extend(["beef", "pork", "chicken", "meat", "fish", "seafood"])
        if "gluten-free" in job_lower:
            exclusion_terms.extend(["wheat", "flour", "bread", "pasta", "barley", "rye"])
        if "vegan" in job_lower:
            exclusion_terms.extend(["dairy", "cheese", "milk", "egg", "butter", "honey"])
        
        # Add domain-specific exclusions
        if "form" in job_lower or "hr" in job_lower:
            exclusion_terms.extend(["image", "photo", "picture", "illustration"])
            
        return list(set(exclusion_terms))  # Remove duplicates
    
    def should_exclude(self, text: str, exclusion_terms: List[str]) -> bool:
        """Determines if text should be excluded based on exclusion terms."""
        text_lower = text.lower()
        return any(term in text_lower for term in exclusion_terms)
    
    def is_relevant_section(self, section: Dict, query_embedding: torch.Tensor, 
                          keyphrases: List[str], exclusion_terms: List[str]) -> bool:
        """
        Determines if a section is relevant using combined criteria.
        """
        # Check for excluded terms first
        if self.should_exclude(section['content'], exclusion_terms):
            return False
            
        # Check for explicit negative terms in title
        title_lower = section['section_title'].lower()
        if any(term in title_lower for term in REJECTED_TITLES):
            return False
            
        # Check for keyphrase matches in content
        content_lower = section['content'].lower()
        keyword_matches = sum(1 for phrase in keyphrases if phrase in content_lower)
        
        # Calculate semantic similarity
        content_embedding = self.model.encode(section['content'], convert_to_tensor=True, device=DEVICE)
        similarity = util.cos_sim(query_embedding, content_embedding).item()
        
        # Combined relevance score
        relevance_score = (0.4 * keyword_matches/len(keyphrases)) + (0.6 * similarity)
        return relevance_score >= 0.5
    
    def rank_sections(self, sections: List[Dict], query_embedding: torch.Tensor, 
                     keyphrases: List[str]) -> List[Tuple[float, Dict]]:
        """
        Ranks sections by relevance using title and content analysis.
        """
        ranked = []
        
        for section in sections:
            # Title relevance
            title_embedding = self.model.encode(section['section_title'], convert_to_tensor=True, device=DEVICE)
            title_score = util.cos_sim(query_embedding, title_embedding).item()
            
            # Content relevance
            content_embedding = self.model.encode(section['content'], convert_to_tensor=True, device=DEVICE)
            content_score = util.cos_sim(query_embedding, content_embedding).item()
            
            # Keyword matches
            content_lower = section['content'].lower()
            keyword_matches = sum(1 for phrase in keyphrases if phrase in content_lower)
            
            # Combined score (weighted)
            combined_score = (0.4 * title_score) + (0.4 * content_score) + (0.2 * keyword_matches/len(keyphrases))
            ranked.append((combined_score, section))
        
        # Sort by score descending
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked
    
    def extract_relevant_content(self, text: str, query_embedding: torch.Tensor, 
                               keyphrases: List[str], exclusion_terms: List[str]) -> str:
        """
        Extracts the most relevant portions of text that match criteria.
        """
        sentences = sent_tokenize(text)
        if not sentences:
            return text[:MAX_SUBSECTION_LENGTH]
        
        # Score each sentence
        scored_sentences = []
        for sent in sentences:
            if len(sent) < 20:
                continue
                
            # Check for excluded terms
            if self.should_exclude(sent, exclusion_terms):
                continue
                
            # Semantic similarity
            sent_embedding = self.model.encode(sent, convert_to_tensor=True, device=DEVICE)
            similarity = util.cos_sim(query_embedding, sent_embedding).item()
            
            # Keyword matches
            sent_lower = sent.lower()
            keyword_matches = sum(1 for phrase in keyphrases if phrase in sent_lower)
            
            # Combined score
            score = (0.7 * similarity) + (0.3 * keyword_matches/len(keyphrases))
            scored_sentences.append((score, sent))
        
        if not scored_sentences:
            return text[:MAX_SUBSECTION_LENGTH]
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [sent for score, sent in scored_sentences[:5]]
        
        # Return as single string
        result = ' '.join(top_sentences)
        return result[:MAX_SUBSECTION_LENGTH]
    
    def analyze_documents(self, input_data: Dict, collection_path: str=None) -> Dict:
        """Main analysis pipeline with proper exclusion criteria."""
        persona = input_data["persona"]["role"]
        job_task = input_data["job_to_be_done"]["task"]
        
        # Create query and extract keyphrases
        query = f"{persona} needs to {job_task}"
        keyphrases = self.extract_keyphrases(job_task)
        exclusion_terms = self.extract_exclusion_terms(job_task)
        query_embedding = self.model.encode(query, convert_to_tensor=True, device=DEVICE)
        
        print(f"INFO: Processing query: '{query}'")
        print(f"INFO: Keyphrases: {keyphrases}")
        print(f"INFO: Exclusion terms: {exclusion_terms}")

        # Parse all documents
        all_sections = []
        pdf_folder = os.path.join(collection_path, "PDFs")
        print(f"DEBUG: Looking for PDFs in: {pdf_folder}")
        print(f"DEBUG: Documents to process: {[doc['filename'] for doc in input_data['documents']]}")
        
        for doc in input_data["documents"]:
            pdf_path = os.path.join(pdf_folder, doc["filename"])
            print(f"DEBUG: Processing document: {doc['filename']}")
            print(f"DEBUG: Full path: {pdf_path}")
            
            if os.path.exists(pdf_path):
                sections = self.parse_pdf_sections(pdf_path)
                all_sections.extend(sections)
                print(f"DEBUG: Added {len(sections)} sections from {doc['filename']}")
            else:
                print(f"WARNING: PDF not found: {pdf_path}")
        
        print(f"DEBUG: Total sections extracted: {len(all_sections)}")
        
        if not all_sections:
            # If still no sections, create a more detailed error message
            available_files = []
            if os.path.exists(pdf_folder):
                available_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
            
            error_msg = f"No sections found in documents. "
            error_msg += f"PDF folder: {pdf_folder}, "
            error_msg += f"Available PDFs: {available_files}, "
            error_msg += f"Requested files: {[doc['filename'] for doc in input_data['documents']]}"
            
            raise ValueError(error_msg)
        
        # Filter sections by relevance and exclusions
        relevant_sections = [
            section for section in all_sections 
            if self.is_relevant_section(section, query_embedding, keyphrases, exclusion_terms)
        ]
        
        if not relevant_sections:
            print("WARNING: No sections matched strict criteria, relaxing constraints...")
            # Fallback to sections that at least don't contain excluded terms
            relevant_sections = [
                section for section in all_sections
                if not self.should_exclude(section['content'], exclusion_terms)
            ]
        
        ranked_sections = self.rank_sections(relevant_sections, query_embedding, keyphrases)
        
        # Prepare output in required format
        output = {
            "metadata": {
                "input_documents": [doc["filename"] for doc in input_data["documents"]],
                "persona": persona,
                "job_to_be_done": job_task,
                "processing_timestamp": datetime.datetime.now().isoformat()
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }
        
        # Add top sections while maintaining required format
        for rank, (score, section) in enumerate(ranked_sections[:MAX_SECTIONS], start=1):
            output["extracted_sections"].append({
                "document": section["document"],
                "section_title": section["section_title"],
                "importance_rank": rank,
                "page_number": section["page_number"]
            })
            
            refined_content = self.extract_relevant_content(
                section["content"], query_embedding, keyphrases, exclusion_terms
            )
            output["subsection_analysis"].append({
                "document": section["document"],
                "refined_text": refined_content,
                "page_number": section["page_number"]
            })
        
        return output

def main():
    parser = argparse.ArgumentParser(
        description="Document intelligence system for Challenge 1B"
    )
    parser.add_argument("--collection", type=str, required=True,
                      help="Path to collection directory containing input JSON and PDFs")
    args = parser.parse_args()
    
    try:
        # Normalize the collection path
        collection_path = os.path.normpath(args.collection)
        
        # Build input path
        input_path = os.path.join(collection_path, "challenge1b_input.json")
        
        with open(input_path, 'r') as f:
            input_data = json.load(f)
        
        analyzer = DocumentAnalyzer()
        results = analyzer.analyze_documents(input_data, collection_path)
        
        # Build output path
        output_path = os.path.join(collection_path, "challenge1b_output.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"SUCCESS: Analysis complete. Results saved to {output_path}")
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise


if __name__ == "__main__":
    main()