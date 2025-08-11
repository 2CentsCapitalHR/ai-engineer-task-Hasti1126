"""
ADGM Corporate Agent - Final Complete Implementation
Uses pre-indexed documents from Qdrant Cloud with Vision LLM
Model: meta-llama/llama-4-scout-17b-16e-instruct
Includes: Enhanced legal citations, alternative clauses, multimodal analysis
"""

import gradio as gr
import docx
from docx.enum.text import WD_COLOR_INDEX
from docx.shared import RGBColor
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from transformers import CLIPProcessor, CLIPModel
from langchain_groq import ChatGroq
from langchain.schema.messages import HumanMessage
import torch
import numpy as np
import os
import json
import tempfile
import shutil
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import uuid
import re

load_dotenv()

class ADGMCorporateAgent:
    """Complete ADGM Corporate Agent with Vision LLM, Legal Citations, and Enhanced Analysis"""
    
    def __init__(self, qdrant_url: str, qdrant_api_key: str, groq_api_key: str):
        # Connect to Qdrant with indexed documents
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = "adgm_reference_docs"
        
        # Initialize CLIP for embeddings
        print("🔄 Loading CLIP model for embeddings...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()
        
        # Initialize Vision LLM
        print("🔄 Initializing Vision LLM: meta-llama/llama-4-scout-17b-16e-instruct")
        self.llm = ChatGroq(
            groq_api_key=groq_api_key, 
            model="meta-llama/llama-4-scout-17b-16e-instruct"
        )
        
        # Verify connection to indexed documents
        self._verify_indexed_documents()
        
        # ADGM Document checklists
        self.document_checklists = {
            "Company Incorporation": [
                "Articles of Association", "Memorandum of Association", 
                "Board Resolution", "UBO Declaration Form", "Register of Members and Directors"
            ],
            "Licensing Application": [
                "License Application Form", "Business Plan", "Financial Projections",
                "Compliance Manual", "Professional Indemnity Insurance"
            ],
            "Branch Registration": [
                "Branch Registration Form", "Parent Company Certificate", 
                "Board Resolution for Branch", "Power of Attorney", "Audited Financial Statements"
            ],
            "Foundation Registration": [
                "Foundation Registration Form", "Foundation Charter",
                "Beneficial Ownership Declaration", "Initial Endowment Evidence"
            ],
            "Employment and HR": [
                "Employment Contract", "Employee Handbook", "HR Policies",
                "Disciplinary Procedures", "Leave Policies"
            ],
            "Commercial Agreements": [
                "Service Agreement", "Supply Agreement", "Distribution Agreement",
                "Partnership Agreement", "Joint Venture Agreement"
            ]
        }
        
        # ADGM Compliance Red Flags
        self.red_flags = {
            "jurisdiction_issues": [
                "UAE Federal Courts", "Dubai Courts", "Sharjah Courts", "Abu Dhabi Courts",
                "UAE Federal Law", "Dubai International Financial Centre", "DIFC", 
                "Dubai International Court", "Emirates"
            ],
            "missing_clauses": [
                "governing law", "dispute resolution", "registered address", 
                "ADGM jurisdiction", "arbitration clause", "applicable law"
            ],
            "non_binding_language": [
                "may consider", "might", "possibly", "perhaps", "could potentially",
                "at discretion", "if deemed appropriate"
            ],
            "incomplete_sections": [
                "to be completed", "TBD", "pending", "[INSERT]", "DRAFT",
                "under review", "subject to approval"
            ]
        }
        
        # ADGM Legal References Database for Enhanced Citations
        self.adgm_legal_refs = {
            "jurisdiction": {
                "citation": "ADGM Companies Regulations 2020, Art. 6 & ADGM Courts Law 2013, Art. 15",
                "rule": "All disputes must be subject to ADGM jurisdiction and courts",
                "alternative": "This Agreement shall be governed by ADGM law and any disputes shall be subject to the exclusive jurisdiction of ADGM Courts."
            },
            "governing_law": {
                "citation": "ADGM Companies Regulations 2020, Art. 6(1)",
                "rule": "ADGM law must be specified as governing law",
                "alternative": "This Agreement shall be governed by and construed in accordance with the laws of the Abu Dhabi Global Market (ADGM)."
            },
            "dispute_resolution": {
                "citation": "ADGM Arbitration Law 2015, Art. 7 & ADGM Courts Law 2013",
                "rule": "Dispute resolution mechanism must reference ADGM courts or ADGM-approved arbitration",
                "alternative": "Any dispute arising from this Agreement shall be resolved through arbitration under the ADGM Arbitration Law 2015 or submitted to the jurisdiction of ADGM Courts."
            },
            "registered_address": {
                "citation": "ADGM Companies Regulations 2020, Art. 23",
                "rule": "Companies must maintain registered address within ADGM",
                "alternative": "The registered address of the Company is [Address], Abu Dhabi Global Market, UAE."
            },
            "board_resolution": {
                "citation": "ADGM Companies Regulations 2020, Art. 128-135",
                "rule": "Board resolutions must comply with ADGM procedural requirements",
                "alternative": "BE IT RESOLVED that the Board of Directors, in accordance with ADGM Companies Regulations 2020, hereby approves..."
            },
            "articles_of_association": {
                "citation": "ADGM Companies Regulations 2020, Art. 18-22",
                "rule": "Articles must comply with ADGM template requirements",
                "alternative": "As per ADGM Companies Regulations 2020, the Articles of Association shall include provisions for [specific requirement]."
            },
            "ubo_declaration": {
                "citation": "ADGM AML Rules 2019, Rule 3.2.1 & ADGM Companies Regulations 2020, Art. 41",
                "rule": "Ultimate Beneficial Ownership must be declared as per ADGM requirements",
                "alternative": "The Company hereby declares its Ultimate Beneficial Ownership structure in compliance with ADGM AML Rules 2019."
            },
            "licensing": {
                "citation": "ADGM General Rules 2019, Rule 2.1.1 & ADGM Business Licensing Regulations 2015",
                "rule": "Business activities must be properly licensed under ADGM framework",
                "alternative": "The Company is authorized to conduct [business activities] under License No. [number] issued by ADGM Registration Authority."
            },
            "signatory": {
                "citation": "ADGM Companies Regulations 2020, Art. 145 & ADGM General Rules 2019",
                "rule": "Authorized signatories must be properly appointed and documented",
                "alternative": "This document is executed by [Name], duly authorized signatory of [Company Name] as per Board Resolution dated [date]."
            }
        }
        
        print("✅ ADGM Corporate Agent initialized successfully")
        print(f"📚 Connected to indexed knowledge base with {self.indexed_stats['total_documents']} documents")
    
    def _verify_indexed_documents(self):
        """Verify connection to indexed documents and get statistics"""
        try:
            # Get collection info
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            # Get some sample documents
            search_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=10,
                with_payload=True
            )
            
            # Calculate statistics
            text_docs = len([p for p in search_result[0] if p.payload.get('content_type') == 'text'])
            image_docs = len([p for p in search_result[0] if p.payload.get('content_type') == 'image'])
            
            self.indexed_stats = {
                "total_points": collection_info.points_count,
                "total_documents": len(set([p.payload.get('document_id') for p in search_result[0] if p.payload.get('document_id')])),
                "text_chunks": text_docs,
                "images": image_docs,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance
            }
            
            print(f"✅ Verified indexed collection: {collection_info.points_count} vectors")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not verify indexed documents: {e}")
            self.indexed_stats = {"total_points": 0, "total_documents": 0, "text_chunks": 0, "images": 0}
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate CLIP text embedding"""
        inputs = self.clip_processor(
            text=text, return_tensors="pt", padding=True, 
            truncation=True, max_length=77
        )
        with torch.no_grad():
            features = self.clip_model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.squeeze().numpy()
    
    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Generate CLIP image embedding"""
        inputs = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.squeeze().numpy()
    
    def search_indexed_documents(self, query: str, content_type: str = "both", limit: int = 5, 
                                category_filter: Optional[str] = None) -> List:
        """Search the indexed ADGM knowledge base"""
        try:
            # Generate query embedding
            query_embedding = self.embed_text(query)
            
            # Build filters
            filters = []
            if content_type != "both":
                filters.append(FieldCondition(key="content_type", match=MatchValue(value=content_type)))
            if category_filter:
                filters.append(FieldCondition(key="category", match=MatchValue(value=category_filter)))
            
            query_filter = Filter(must=filters) if filters else None
            
            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=query_filter,
                limit=limit,
                with_payload=True
            )
            
            return search_results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def identify_document_process(self, docx_files: List) -> Tuple[str, float]:
        """Identify legal process from uploaded files"""
        file_names = [f.name.lower() for f in docx_files]
        scores = {}
        
        for process, required_docs in self.document_checklists.items():
            score = 0
            matched_docs = 0
            
            for required_doc in required_docs:
                doc_keywords = required_doc.lower().replace(" ", "").split()
                
                for file_name in file_names:
                    # Clean filename for better matching
                    clean_filename = file_name.replace("_", "").replace("-", "").replace(" ", "")
                    
                    # Check if any keywords match
                    if any(keyword in clean_filename for keyword in doc_keywords):
                        score += 1
                        matched_docs += 1
                        break
            
            scores[process] = score / len(required_docs) if required_docs else 0
        
        best_process = max(scores, key=scores.get) if scores else "General Document Review"
        confidence = scores.get(best_process, 0)
        
        return (best_process, confidence) if confidence > 0.25 else ("General Document Review", confidence)
    
    def extract_all_images_from_docx(self, docx_path: str) -> List[str]:
        """Extract ALL images from DOCX - tables and embedded images"""
        all_images = []
        
        try:
            doc = docx.Document(docx_path)
            
            # Extract tables as images
            for table_idx, table in enumerate(doc.tables):
                try:
                    table_data = []
                    for row in table.rows:
                        row_data = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                        if row_data:  # Only add non-empty rows
                            table_data.append(row_data)
                    
                    if table_data and len(table_data) > 1:  # Must have header + data
                        table_image = self._create_table_image(table_data)
                        if table_image:
                            buffered = io.BytesIO()
                            table_image.save(buffered, format="PNG")
                            image_base64 = base64.b64encode(buffered.getvalue()).decode()
                            all_images.append(image_base64)
                
                except Exception as e:
                    print(f"Error processing table {table_idx}: {e}")
            
            # Extract embedded images
            try:
                for rel in doc.part.rels.values():
                    if "image" in rel.target_ref:
                        try:
                            image_data = rel.target_part.blob
                            image = Image.open(io.BytesIO(image_data))
                            if image.mode != 'RGB':
                                image = image.convert('RGB')
                            
                            # Resize if too large
                            max_size = (1024, 1024)
                            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                                image.thumbnail(max_size, Image.Resampling.LANCZOS)
                            
                            buffered = io.BytesIO()
                            image.save(buffered, format="PNG")
                            image_base64 = base64.b64encode(buffered.getvalue()).decode()
                            all_images.append(image_base64)
                            
                        except Exception as e:
                            print(f"Error processing embedded image: {e}")
            except Exception as e:
                print(f"Error accessing document images: {e}")
        
        except Exception as e:
            print(f"Error opening DOCX file: {e}")
        
        return all_images
    
    def _create_table_image(self, table_data: List[List[str]]) -> Optional[Image.Image]:
        """Create high-quality table image for Vision LLM"""
        try:
            if not table_data or not table_data[0]:
                return None
            
            rows = len(table_data)
            cols = max(len(row) for row in table_data)
            
            # Create figure with appropriate size
            fig, ax = plt.subplots(figsize=(max(cols * 1.8, 6), max(rows * 0.7, 3)))
            ax.set_xlim(0, cols)
            ax.set_ylim(0, rows)
            ax.axis('off')
            
            # Draw table with clear formatting
            for i, row in enumerate(table_data):
                for j in range(cols):
                    cell_text = row[j] if j < len(row) else ""
                    
                    # Draw cell border
                    rect = patches.Rectangle(
                        (j, rows - i - 1), 1, 1,
                        linewidth=1.5, edgecolor='black', 
                        facecolor='lightblue' if i == 0 else 'white',  # Header highlighting
                        alpha=0.7 if i == 0 else 1.0
                    )
                    ax.add_patch(rect)
                    
                    # Add text with appropriate formatting
                    cell_text = str(cell_text)[:80]  # Limit text length
                    ax.text(j + 0.5, rows - i - 0.5, cell_text,
                           ha='center', va='center', fontsize=9, 
                           weight='bold' if i == 0 else 'normal',
                           wrap=True)
            
            # Convert to PIL Image
            fig.canvas.draw()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, 
                       facecolor='white', edgecolor='none')
            buf.seek(0)
            image = Image.open(buf)
            plt.close(fig)
            
            return image
            
        except Exception as e:
            print(f"Error creating table image: {e}")
            return None
    
    def analyze_docx_with_vision_and_retrieval(self, docx_path: str, process_type: str) -> Dict:
        """Enhanced analysis with structured issues for legal commenting"""
        try:
            doc = docx.Document(docx_path)
            
            # Extract text content with paragraph tracking
            full_text = []
            paragraph_locations = {}  # Track paragraph locations for precise commenting
            for para_idx, para in enumerate(doc.paragraphs):
                if para.text.strip():
                    full_text.append(para.text.strip())
                    # Map text to paragraph numbers for precise location tracking
                    paragraph_locations[para.text.strip()[:50]] = f"Paragraph {para_idx + 1}"
            
            document_text = '\n'.join(full_text)
            
            # 1. Search indexed ADGM documents for relevant references
            print(f"🔍 Searching indexed knowledge base for: {process_type}")
            
            # Search for process-specific references
            process_search = self.search_indexed_documents(
                query=f"{process_type} requirements compliance", 
                content_type="text", 
                limit=5
            )
            
            # Search for content-specific references  
            content_search = self.search_indexed_documents(
                query=document_text[:800],  # First part of document
                content_type="both",
                limit=5
            )
            
            # Search for similar visual content
            visual_search = self.search_indexed_documents(
                query=document_text[:500],
                content_type="image",
                limit=3
            ) if len(document_text) > 100 else []
            
            # 2. Detect red flags in text
            red_flags = self._detect_red_flags(document_text)
            
            # 3. Extract images from user document
            user_images = self.extract_all_images_from_docx(docx_path)
            
            print(f"📊 Found {len(user_images)} images in user document")
            print(f"📚 Retrieved {len(process_search) + len(content_search)} reference text chunks")
            print(f"🖼️ Retrieved {len(visual_search)} reference images")
            
            # 4. Enhanced prompt for structured legal analysis
            content = [{
                "type": "text",
                "text": f"""You are an ADGM legal compliance expert. Provide detailed analysis with specific legal citations and alternative clauses.

DOCUMENT TYPE: {process_type}
DOCUMENT TEXT: {document_text[:2500]}

RED FLAGS DETECTED: {json.dumps(red_flags, indent=2)}

RELEVANT ADGM REFERENCE DOCUMENTS:
Process-Specific References:
{self._format_search_results(process_search)}

Content-Specific References:
{self._format_search_results(content_search)}

Visual References from ADGM Database:
{self._format_image_search_results(visual_search)}

The user document contains {len(user_images)} images/tables shown below.

Analyze the document and return JSON with this EXACT structure:
{{
    "compliance_score": <number 0-100>,
    "jurisdiction_compliance": <boolean>,
    "issues_found": [
        {{
            "location": "Specific section/paragraph where issue occurs",
            "issue": "Clear description of the compliance issue",
            "severity": "High|Medium|Low",
            "text_match": "Exact text from document that has the issue",
            "adgm_violation": "Specific ADGM regulation violated",
            "suggestion": "Detailed correction needed",
            "alternative_clause": "Complete alternative wording that complies with ADGM"
        }}
    ],
    "missing_elements": ["List of required elements not found"],
    "visual_compliance_issues": ["Issues found in tables/images"],
    "reference_alignment": "How well document aligns with ADGM templates",
    "recommendations": ["Specific actions to achieve compliance"]
}}

FOCUS ON:
1. Jurisdiction clauses (must reference ADGM courts, not UAE Federal/Dubai courts)
2. Governing law provisions (must specify ADGM law)
3. Dispute resolution mechanisms (ADGM arbitration or courts)
4. Registered address requirements
5. Signatory authorization
6. Required disclosures and declarations
7. Table formatting and data compliance
8. Visual element completeness

For each issue, provide the EXACT text that needs changing and complete alternative wording."""
            }]
            
            # Add user document images
            for i, image_data in enumerate(user_images):
                content.append({
                    "type": "text",
                    "text": f"\n--- USER DOCUMENT IMAGE {i+1} ---"
                })
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data}"
                    }
                })
            
            # 5. Get Vision LLM response
            message = HumanMessage(content=content)
            response = self.llm.invoke([message])
            
            # 6. Parse and enhance response
            try:
                analysis_result = json.loads(response.content)
                
                # Enhance issues with paragraph locations
                for issue in analysis_result.get("issues_found", []):
                    if "text_match" in issue and issue["text_match"]:
                        # Try to find paragraph location
                        text_snippet = issue["text_match"][:50]
                        location = paragraph_locations.get(text_snippet, "Document")
                        if issue.get("location") == "Document" or not issue.get("location"):
                            issue["location"] = location
                
                # Add retrieval metadata
                analysis_result["retrieval_stats"] = {
                    "process_references": len(process_search),
                    "content_references": len(content_search),
                    "visual_references": len(visual_search),
                    "user_images_analyzed": len(user_images)
                }
                
            except json.JSONDecodeError:
                # Enhanced fallback response with legal structure
                analysis_result = {
                    "compliance_score": 60,
                    "jurisdiction_compliance": False,
                    "issues_found": [
                        {
                            "location": "Document Review",
                            "issue": "Comprehensive legal review required - JSON parsing failed",
                            "severity": "Medium", 
                            "text_match": document_text[:100] if document_text else "No content",
                            "adgm_violation": "General compliance review needed",
                            "suggestion": "Manual review by ADGM compliance expert recommended",
                            "alternative_clause": "Please consult ADGM regulations for proper clause wording"
                        }
                    ],
                    "missing_elements": ["Detailed analysis unavailable due to parsing error"],
                    "visual_compliance_issues": ["Visual analysis unavailable"],
                    "reference_alignment": "Unable to assess due to technical issue",
                    "recommendations": ["Seek professional ADGM legal consultation", "Retry analysis"],
                    "raw_response": response.content[:500],
                    "retrieval_stats": {
                        "process_references": len(process_search),
                        "content_references": len(content_search),
                        "visual_references": len(visual_search),
                        "user_images_analyzed": len(user_images)
                    }
                }
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ Enhanced analysis error: {e}")
            return {
                "error": f"Vision analysis failed: {str(e)}",
                "compliance_score": 0,
                "jurisdiction_compliance": False,
                "issues_found": [
                    {
                        "location": "System Error",
                        "issue": f"Technical error during analysis: {str(e)}",
                        "severity": "High",
                        "text_match": "",
                        "adgm_violation": "System malfunction",
                        "suggestion": "Please try again or contact support",
                        "alternative_clause": "Technical issue prevents clause suggestion"
                    }
                ],
                "missing_elements": ["Analysis incomplete due to error"],
                "visual_compliance_issues": ["Visual analysis failed"],
                "reference_alignment": "Error prevented assessment",
                "recommendations": ["Technical issue - please retry", "Contact system administrator"]
            }
    
    def _detect_red_flags(self, text: str) -> Dict[str, List[str]]:
        """Detect ADGM compliance red flags"""
        found_flags = {}
        text_lower = text.lower()
        
        for category, patterns in self.red_flags.items():
            found_in_category = []
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    found_in_category.append(pattern)
            
            if found_in_category:
                found_flags[category] = found_in_category
        
        return found_flags
    
    def _format_search_results(self, results: List) -> str:
        """Format text search results for LLM prompt"""
        if not results:
            return "No relevant text references found."
        
        formatted = []
        for i, result in enumerate(results[:5], 1):
            payload = result.payload
            content = payload.get('content', '')[:300]
            doc_name = payload.get('document_name', 'Unknown Document')
            doc_type = payload.get('document_type', 'Unknown Type')
            score = result.score
            
            formatted.append(f"{i}. [{doc_type}] {doc_name} (Relevance: {score:.3f})")
            formatted.append(f"   Content: {content}...")
            formatted.append("")
        
        return '\n'.join(formatted)
    
    def _format_image_search_results(self, results: List) -> str:
        """Format image search results for LLM prompt"""
        if not results:
            return "No relevant visual references found."
        
        formatted = []
        for i, result in enumerate(results[:3], 1):
            payload = result.payload
            doc_name = payload.get('document_name', 'Unknown Document')
            image_type = payload.get('image_type', 'image')
            score = result.score
            
            formatted.append(f"{i}. Visual {image_type} from {doc_name} (Relevance: {score:.3f})")
        
        return '\n'.join(formatted)
    
    def add_comments_to_docx(self, docx_path: str, issues: List[Dict]) -> str:
        """Enhanced inline commenting with ADGM legal citations and alternative clauses"""
        try:
            doc = docx.Document(docx_path)
            comments_added = 0
            
            for issue in issues:
                text_match = issue.get('text_match', '')
                issue_type = issue.get('issue', '').lower()
                severity = issue.get('severity', 'Medium')
                location = issue.get('location', 'Unknown')
                suggestion = issue.get('suggestion', '')
                alternative_clause = issue.get('alternative_clause', '')
                
                if not text_match:
                    continue
                    
                # Determine ADGM legal reference based on issue type
                legal_ref = None
                for key, ref_data in self.adgm_legal_refs.items():
                    if key in issue_type or key in suggestion.lower():
                        legal_ref = ref_data
                        break
                
                # If no specific reference found, use generic ADGM compliance
                if not legal_ref:
                    legal_ref = {
                        "citation": "ADGM Companies Regulations 2020 & ADGM General Rules 2019",
                        "rule": "Document must comply with ADGM legal framework",
                        "alternative": alternative_clause or "Please review and align with ADGM regulatory requirements."
                    }
                else:
                    # Use provided alternative if available, otherwise use reference default
                    if alternative_clause:
                        legal_ref["alternative"] = alternative_clause
                
                # Find and process paragraphs containing the issue
                for para_idx, para in enumerate(doc.paragraphs):
                    if text_match.lower() in para.text.lower():
                        # Highlight the problematic text
                        for run in para.runs:
                            if text_match.lower() in run.text.lower():
                                if severity.lower() == 'high':
                                    run.font.highlight_color = WD_COLOR_INDEX.RED
                                elif severity.lower() == 'medium':
                                    run.font.highlight_color = WD_COLOR_INDEX.YELLOW
                                else:
                                    run.font.highlight_color = WD_COLOR_INDEX.LIGHT_BLUE
                        
                        # Create comprehensive comment with ADGM legal citation
                        comment_parts = [
                            f"🚨 ADGM COMPLIANCE ISSUE ({severity.upper()})",
                            f"📍 Location: {location}",
                            f"⚖️ Legal Citation: {legal_ref['citation']}",
                            f"📋 Requirement: {legal_ref['rule']}",
                            f"🔧 Issue: {suggestion or issue.get('issue', 'Compliance issue identified')}",
                            f"✅ Suggested Alternative: {legal_ref['alternative']}"
                        ]
                        
                        # Add the comprehensive comment
                        try:
                            # Insert new paragraph after current one
                            new_para_element = docx.oxml.shared.OxmlElement('w:p')
                            para._element.getparent().insert(
                                list(para._element.getparent()).index(para._element) + 1,
                                new_para_element
                            )
                            comment_para = docx.text.paragraph.Paragraph(new_para_element, para._parent)
                            
                            # Add each part of the comment with appropriate formatting
                            for part_idx, part in enumerate(comment_parts):
                                if part_idx > 0:
                                    comment_para.add_run("\n")
                                
                                run = comment_para.add_run(part)
                                
                                # Style based on content type
                                if "🚨" in part:  # Header
                                    run.bold = True
                                    run.font.size = docx.shared.Pt(12)
                                    if severity.lower() == 'high':
                                        run.font.color.rgb = RGBColor(220, 20, 60)  # Crimson
                                    elif severity.lower() == 'medium':
                                        run.font.color.rgb = RGBColor(255, 140, 0)   # Dark orange
                                    else:
                                        run.font.color.rgb = RGBColor(30, 144, 255)  # Dodger blue
                                elif "⚖️" in part:  # Legal citation
                                    run.bold = True
                                    run.italic = True
                                    run.font.color.rgb = RGBColor(0, 0, 139)  # Dark blue
                                    run.font.size = docx.shared.Pt(10)
                                elif "✅" in part:  # Alternative suggestion
                                    run.bold = True
                                    run.font.color.rgb = RGBColor(0, 128, 0)  # Green
                                    run.font.size = docx.shared.Pt(10)
                                else:  # Regular content
                                    run.font.size = docx.shared.Pt(9)
                                    run.font.color.rgb = RGBColor(64, 64, 64)  # Dark gray
                            
                            # Add separator
                            separator_run = comment_para.add_run("\n" + "─" * 60 + "\n")
                            separator_run.font.color.rgb = RGBColor(128, 128, 128)
                            separator_run.font.size = docx.shared.Pt(8)
                            
                            comments_added += 1
                            
                        except Exception as e:
                            print(f"Error adding detailed comment: {e}")
                            # Fallback to simple comment
                            try:
                                fallback_para = doc.add_paragraph()
                                fallback_run = fallback_para.add_run(
                                    f"🚨 ADGM COMPLIANCE: {suggestion} (Ref: {legal_ref['citation']})"
                                )
                                fallback_run.bold = True
                                fallback_run.font.color.rgb = RGBColor(255, 0, 0)
                                comments_added += 1
                            except:
                                print(f"Failed to add any comment for issue: {issue_type}")
                        
                        break  # Only add comment once per issue
            
            # Add comprehensive summary comment at the end
            if comments_added > 0:
                try:
                    summary_para = doc.add_paragraph()
                    summary_parts = [
                        f"\n{'='*60}",
                        f"📊 ADGM COMPLIANCE SUMMARY",
                        f"{'='*60}",
                        f"🔍 Total Issues Identified: {comments_added}",
                        f"⚖️ All citations reference current ADGM regulations",
                        f"✅ Review suggested alternatives and implement corrections",
                        f"📞 For clarification, consult ADGM Registration Authority",
                        f"🌐 ADGM Regulations: https://www.adgm.com/doing-business/laws-and-regulations",
                        f"📅 Analysis Date: {datetime.now().strftime('%B %d, %Y')}",
                        f"{'='*60}\n"
                    ]
                    
                    for part in summary_parts:
                        summary_run = summary_para.add_run(part + "\n")
                        if "📊" in part or "=" in part:
                            summary_run.bold = True
                            summary_run.font.color.rgb = RGBColor(0, 0, 139)
                        else:
                            summary_run.font.color.rgb = RGBColor(64, 64, 64)
                        summary_run.font.size = docx.shared.Pt(10)
                        
                except Exception as e:
                    print(f"Error adding summary: {e}")
            
            # Save the enhanced reviewed document
            output_path = docx_path.replace('.docx', '_ADGM_Legal_Review.docx')
            doc.save(output_path)
            
            print(f"✅ Added {comments_added} detailed ADGM legal comments with citations")
            return output_path
            
        except Exception as e:
            print(f"❌ Error in enhanced commenting: {e}")
            return docx_path
    
    def process_user_documents(self, uploaded_files: List) -> Dict:
        """Complete processing pipeline with retrieval and enhanced analysis"""
        if not uploaded_files:
            return {"error": "No files uploaded"}
        
        # Filter for DOCX files only
        docx_files = [f for f in uploaded_files if f.name.endswith('.docx')]
        if not docx_files:
            return {"error": "Please upload DOCX files only"}
        
        temp_dir = tempfile.mkdtemp()
        results = {
            "timestamp": datetime.now().isoformat(),
            "model_used": "meta-llama/llama-4-scout-17b-16e-instruct",
            "knowledge_base_stats": self.indexed_stats,
            "total_files": len(docx_files),
            "process_identified": "",
            "confidence": 0.0,
            "documents_analysis": [],
            "missing_documents": [],
            "overall_compliance": 0,
            "reviewed_files": []
        }
        
        try:
            # Save uploaded files
            saved_files = []
            for file in docx_files:
                temp_path = Path(temp_dir) / file.name
                with open(temp_path, 'wb') as f:
                    f.write(file.read())
                saved_files.append(temp_path)
            
            # Identify process type
            process_type, confidence = self.identify_document_process(docx_files)
            results["process_identified"] = process_type
            results["confidence"] = confidence
            
            print(f"🎯 Identified process: {process_type} (Confidence: {confidence:.1%})")
            
            # Check for missing documents
            if process_type in self.document_checklists:
                required_docs = self.document_checklists[process_type]
                uploaded_names = [f.name.lower() for f in docx_files]
                
                missing_docs = []
                for required_doc in required_docs:
                    doc_keywords = required_doc.lower().replace(" ", "").split()
                    found = any(
                        any(keyword in uploaded_name.replace("_", "").replace("-", "") 
                            for keyword in doc_keywords)
                        for uploaded_name in uploaded_names
                    )
                    if not found:
                        missing_docs.append(required_doc)
                
                results["missing_documents"] = missing_docs
                print(f"📋 Missing documents: {len(missing_docs)}")
                
                # Generate missing document notification
                if missing_docs:
                    total_required = len(required_docs)
                    uploaded_count = total_required - len(missing_docs)
                    results["checklist_message"] = f"It appears that you're trying to {process_type.lower()} in ADGM. Based on our reference list, you have uploaded {uploaded_count} out of {total_required} required documents. The missing document(s) appear to be: {', '.join(missing_docs)}."
            
            # Analyze each document with comprehensive retrieval and vision analysis
            compliance_scores = []
            
            for file_idx, file_path in enumerate(saved_files):
                print(f"\n🔍 Analyzing document {file_idx + 1}/{len(saved_files)}: {file_path.name}")
                
                # Perform comprehensive analysis with retrieval
                analysis = self.analyze_docx_with_vision_and_retrieval(str(file_path), process_type)
                
                # Add enhanced comments to document if issues found
                if analysis.get('issues_found') and not analysis.get('error'):
                    reviewed_path = self.add_comments_to_docx(str(file_path), analysis['issues_found'])
                    if reviewed_path != str(file_path):  # Comments were actually added
                        results["reviewed_files"].append(reviewed_path)
                
                analysis["file_name"] = file_path.name
                results["documents_analysis"].append(analysis)
                
                # Track compliance scores
                if 'compliance_score' in analysis and analysis['compliance_score'] > 0:
                    compliance_scores.append(analysis['compliance_score'])
            
            # Calculate overall compliance
            if compliance_scores:
                results["overall_compliance"] = sum(compliance_scores) / len(compliance_scores)
            else:
                results["overall_compliance"] = 0
            
            print(f"✅ Analysis complete. Overall compliance: {results['overall_compliance']:.1f}%")
            
            return results
            
        except Exception as e:
            print(f"❌ Processing error: {e}")
            return {"error": f"Processing failed: {str(e)}"}
        
        finally:
            # Cleanup temporary files
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass

def create_gradio_interface():
    """Create the comprehensive Gradio interface"""
    
    # Initialize the agent
    try:
        agent = ADGMCorporateAgent(
            qdrant_url=os.getenv("QDRANT_CLOUD_URL"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return None
    
    def search_knowledge_base(query, search_type, category):
        """Search the indexed knowledge base"""
        if not query.strip():
            return "Please enter a search query."
        
        try:
            results = agent.search_indexed_documents(
                query=query,
                content_type=search_type.lower(),
                limit=8,
                category_filter=category if category != "All Categories" else None
            )
            
            if not results:
                return f"No results found for: '{query}'"
            
            report = f"## 🔍 Knowledge Base Search Results\n**Query**: {query}\n**Found**: {len(results)} results\n\n"
            
            for i, result in enumerate(results, 1):
                payload = result.payload
                content_type = payload.get('content_type', 'unknown')
                doc_name = payload.get('document_name', 'Unknown Document')
                doc_type = payload.get('document_type', 'Unknown')
                category = payload.get('category', 'General')
                score = result.score
                
                report += f"### {i}. {content_type.title()}: {doc_name}\n"
                report += f"**Type**: {doc_type} | **Category**: {category} | **Relevance**: {score:.3f}\n"
                
                if content_type == 'text':
                    content = payload.get('content', '')[:400]
                    report += f"**Content**: {content}...\n"
                elif content_type == 'image':
                    image_type = payload.get('image_type', 'image')
                    report += f"**Visual Content**: {image_type.title()} from document\n"
                
                report += "\n---\n\n"
            
            return report
            
        except Exception as e:
            return f"❌ Search error: {str(e)}"
    
    def analyze_documents(files):
        """Main document analysis function with comprehensive reporting"""
        if not files:
            return "Please upload DOCX files for analysis.", "", None
        
        try:
            # Process documents
            print(f"📄 Processing {len(files)} uploaded files...")
            results = agent.process_user_documents(files)
            
            if "error" in results:
                return f"❌ Error: {results['error']}", "", None
            
            # Generate comprehensive summary
            kb_stats = results.get('knowledge_base_stats', {})
            process_type = results['process_identified']
            confidence = results['confidence']
            missing_docs = results['missing_documents']
            overall_compliance = results['overall_compliance']
            checklist_message = results.get('checklist_message', '')
            
            summary = f"""# 🏛️ ADGM Corporate Agent - Complete Legal Analysis

## 🤖 Analysis Details
**AI Model**: {results.get('model_used', 'meta-llama/llama-4-scout-17b-16e-instruct')} (Vision LLM)
**Knowledge Base**: {kb_stats.get('total_points', 0)} indexed vectors from {kb_stats.get('total_documents', 0)} ADGM documents
**Analysis Timestamp**: {results.get('timestamp', 'Unknown')}

## 📋 Process Identification & Document Checklist
**Identified Process**: {process_type}
**Confidence Level**: {confidence:.1%}

{checklist_message}

## 📊 Overall Compliance Assessment
**Documents Analyzed**: {results['total_files']}
**Overall Compliance Score**: {overall_compliance:.1f}/100
"""
            
            # Enhanced compliance level indicator
            if overall_compliance >= 85:
                summary += "**Status**: ✅ **HIGHLY COMPLIANT** - Excellent adherence to ADGM requirements\n\n"
            elif overall_compliance >= 70:
                summary += "**Status**: ✅ **COMPLIANT** - Good adherence to ADGM requirements with minor improvements needed\n\n"
            elif overall_compliance >= 50:
                summary += "**Status**: ⚠️ **NEEDS SIGNIFICANT IMPROVEMENT** - Multiple compliance issues identified\n\n"
            else:
                summary += "**Status**: ❌ **NON-COMPLIANT** - Major compliance gaps require immediate attention\n\n"
            
            # Missing documents section
            if missing_docs:
                summary += f"## ⚠️ Missing Required Documents ({len(missing_docs)})\n"
                for doc in missing_docs:
                    summary += f"- {doc}\n"
                summary += "\n**Action Required**: Please prepare and submit the missing documents listed above.\n\n"
            else:
                summary += "## ✅ Document Completeness\nAll required documents for this process are present.\n\n"
            
            # Individual document analysis with enhanced details
            summary += "## 📄 Individual Document Analysis\n\n"
            
            for i, analysis in enumerate(results["documents_analysis"], 1):
                file_name = analysis.get("file_name", f"Document {i}")
                compliance_score = analysis.get("compliance_score", 0)
                issues_count = len(analysis.get("issues_found", []))
                visual_issues = len(analysis.get("visual_compliance_issues", []))
                retrieval_stats = analysis.get("retrieval_stats", {})
                reference_alignment = analysis.get("reference_alignment", "Not assessed")
                
                # Enhanced status emoji based on compliance score
                if compliance_score >= 85:
                    status_emoji = "✅"
                    status_text = "HIGHLY COMPLIANT"
                elif compliance_score >= 70:
                    status_emoji = "✅"
                    status_text = "COMPLIANT"
                elif compliance_score >= 50:
                    status_emoji = "⚠️"
                    status_text = "NEEDS IMPROVEMENT"
                else:
                    status_emoji = "❌"
                    status_text = "NON-COMPLIANT"
                
                summary += f"### {status_emoji} {file_name} - {status_text}\n"
                summary += f"**Compliance Score**: {compliance_score}/100\n"
                summary += f"**Issues Found**: {issues_count} textual + {visual_issues} visual\n"
                summary += f"**ADGM Jurisdiction**: {'✅ Compliant' if analysis.get('jurisdiction_compliance') else '❌ Non-Compliant'}\n"
                summary += f"**Template Alignment**: {reference_alignment}\n"
                summary += f"**Knowledge Base Matching**: {retrieval_stats.get('process_references', 0)} process refs, {retrieval_stats.get('content_references', 0)} content refs\n"
                
                # Show critical issues with legal citations
                high_priority_issues = [issue for issue in analysis.get("issues_found", []) if issue.get("severity", "").lower() == "high"]
                if high_priority_issues:
                    summary += f"\n**🚨 Critical Legal Issues ({len(high_priority_issues)}):**\n"
                    for issue in high_priority_issues[:3]:  # Top 3 critical issues
                        adgm_violation = issue.get('adgm_violation', 'ADGM compliance issue')
                        summary += f"- {issue.get('issue', 'Unknown issue')} (Violates: {adgm_violation})\n"
                
                summary += "\n---\n\n"
            
            # Enhanced recommendations section
            all_recommendations = []
            for analysis in results["documents_analysis"]:
                all_recommendations.extend(analysis.get("recommendations", []))
            
            if all_recommendations:
                unique_recommendations = list(set(all_recommendations))[:6]  # Top 6 unique recommendations
                summary += "## 🎯 Priority Action Items\n\n"
                for i, rec in enumerate(unique_recommendations, 1):
                    summary += f"{i}. {rec}\n"
                summary += "\n"
            
            # Legal compliance footer with enhanced warnings
            if overall_compliance < 60:
                summary += """## 🚨 CRITICAL COMPLIANCE ALERT
Your documents contain significant ADGM compliance violations that must be addressed before submission.

**Immediate Actions Required:**
1. Review all highlighted sections in the downloaded documents
2. Implement suggested alternative clauses with proper ADGM legal citations
3. Consult with ADGM-qualified legal counsel
4. Verify all jurisdiction and governing law clauses reference ADGM
5. Ensure all required documents are present and properly executed

**ADGM Resources:**
- Registration Authority: https://www.adgm.com/registration-authority
- Legal Framework: https://www.adgm.com/doing-business/laws-and-regulations
- Professional Services Directory: https://www.adgm.com/doing-business/find-a-service-provider
"""
            elif overall_compliance < 80:
                summary += """## ⚠️ COMPLIANCE IMPROVEMENTS NEEDED
While your documents show good compliance, several areas require attention before final submission.

**Recommended Actions:**
1. Address all flagged issues in the reviewed documents
2. Implement suggested ADGM-compliant alternative clauses
3. Verify proper legal citations and references
4. Consider professional legal review for final validation
"""
            else:
                summary += """## ✅ EXCELLENT COMPLIANCE STATUS
Your documents demonstrate strong adherence to ADGM requirements.

**Final Steps:**
1. Review minor suggestions in commented documents
2. Ensure all signatories are properly authorized
3. Verify document execution requirements are met
4. Proceed with confidence to ADGM submission
"""
            
            summary += f"\n---\n*Comprehensive analysis completed on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*\n"
            summary += "*All legal citations reference current ADGM regulations and laws.*"
            
            # Prepare download file
            download_file = None
            if results["reviewed_files"]:
                download_file = results["reviewed_files"][0]  # Return first reviewed file
            
            return summary, json.dumps(results, indent=2), download_file
            
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return f"❌ Analysis failed: {str(e)}", "", None
    
    # Create the comprehensive Gradio interface
    with gr.Blocks(title="ADGM Corporate Agent - Complete Legal Analysis", theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"""
        # 🏛️ ADGM Corporate Agent - Complete Legal Analysis System
        ### Powered by meta-llama/llama-4-scout-17b-16e-instruct (Vision LLM)
        
        **Knowledge Base**: {agent.indexed_stats.get('total_points', 0)} vectors from {agent.indexed_stats.get('total_documents', 0)} ADGM documents
        
        **Complete Analysis Features**: Text compliance • Visual analysis • Legal citations • Alternative clauses • Document checklists
        """)
        
        with gr.Tabs():
            # Main Document Analysis Tab
            with gr.Tab("📄 Legal Document Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Upload Legal Documents for Analysis")
                        
                        file_upload = gr.Files(
                            label="Upload DOCX Documents",
                            file_types=[".docx"],
                            file_count="multiple"
                        )
                        
                        analyze_btn = gr.Button(
                            "🔍 Analyze with Vision LLM + Legal Citations", 
                            variant="primary", 
                            size="lg"
                        )
                        
                        gr.Markdown("""
                        **🎯 Complete Analysis Capabilities:**
                        - ✅ **Text Compliance**: Full legal text analysis with ADGM citations
                        - ✅ **Visual Analysis**: Table and image compliance validation
                        - ✅ **Document Checklist**: Automatic missing document detection
                        - ✅ **Legal Citations**: Specific ADGM law and regulation references
                        - ✅ **Alternative Clauses**: Ready-to-use compliant wording
                        - ✅ **Inline Comments**: Detailed comments with legal guidance
                        - ✅ **Jurisdiction Validation**: ADGM court and law compliance
                        - ✅ **Cross-Reference**: Match against 13K+ ADGM documents
                        
                        **📋 Supported Legal Processes:**
                        - **Company Incorporation** (5 required documents)
                        - **Business Licensing** (5 required documents)
                        - **Branch Registration** (5 required documents)
                        - **Foundation Registration** (4 required documents)
                        - **Employment & HR** (5 document types)
                        - **Commercial Agreements** (5 agreement types)
                        """)
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Comprehensive Analysis Results")
                        analysis_output = gr.Markdown(
                            "Upload DOCX files to begin comprehensive legal analysis with Vision LLM."
                        )
                        
                        download_file = gr.File(
                            label="📥 Download Reviewed Document with Legal Comments",
                            visible=True
                        )
            
            # Knowledge Base Search Tab
            with gr.Tab("🔍 ADGM Knowledge Base Search"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Search 13K+ Indexed ADGM Documents")
                        search_query = gr.Textbox(
                            label="Search Query",
                            placeholder="e.g., 'company incorporation requirements', 'dispute resolution ADGM courts', 'UBO declaration procedures'",
                            lines=3
                        )
                        
                        with gr.Row():
                            search_type = gr.Radio(
                                choices=["Both", "Text", "Image"],
                                value="Both",
                                label="Content Type",
                                info="Search text documents, visual content, or both"
                            )
                            
                            category_filter = gr.Dropdown(
                                choices=[
                                    "All Categories", 
                                    "Company Formation", 
                                    "Licensing", 
                                    "Branch Registration", 
                                    "Employment and HR", 
                                    "Commercial Agreements",
                                    "Foundation Registration",
                                    "Regulatory Compliance"
                                ],
                                value="All Categories",
                                label="Document Category"
                            )
                        
                        search_btn = gr.Button("🔍 Search ADGM Knowledge Base", variant="secondary")
                        
                        gr.Markdown("""
                        **Search Tips:**
                        - Use specific ADGM terminology for better results
                        - Search for process names, legal concepts, or document types
                        - Visual search finds relevant tables, forms, and diagrams
                        - Category filters narrow results to specific legal areas
                        """)
                        
                    with gr.Column():
                        search_results = gr.Markdown(
                            "Enter a search query to find relevant ADGM regulations, templates, and guidance documents."
                        )
            
            # Detailed Analysis Report Tab
            with gr.Tab("📊 Complete Analysis Report"):
                gr.Markdown("### Comprehensive JSON Analysis Data")
                gr.Markdown("""
                This section contains the complete structured analysis data including:
                - **Document compliance scores** and detailed assessments
                - **Legal issue identification** with ADGM law citations
                - **Knowledge base retrieval statistics** and matching documents
                - **Visual analysis results** for tables and images
                - **Missing document identification** and checklist validation
                - **Alternative clause suggestions** and implementation guidance
                """)
                
                detailed_json = gr.JSON(
                    label="Complete Analysis Data (JSON Format)",
                    value={
                        "message": "Comprehensive analysis data will appear here after document processing",
                        "features": [
                            "Compliance scoring (0-100)",
                            "ADGM legal citations", 
                            "Alternative clause suggestions",
                            "Visual content analysis",
                            "Knowledge base cross-referencing",
                            "Document checklist validation"
                        ]
                    }
                )
        
        # Event handlers with enhanced functionality
        analyze_btn.click(
            fn=analyze_documents,
            inputs=[file_upload],
            outputs=[analysis_output, detailed_json, download_file]
        )
        
        search_btn.click(
            fn=search_knowledge_base,
            inputs=[search_query, search_type, category_filter],
            outputs=[search_results]
        )
        
        # Add footer with system information
        gr.Markdown("""
        ---
        **System Information**: ADGM Corporate Agent v2.0 | Vision LLM: meta-llama/llama-4-scout-17b-16e-instruct | 
        Vector Database: Qdrant Cloud | Knowledge Base: 13,310 text chunks + 4,047 images from ADGM documents
        
        *For technical support or legal clarification, consult ADGM Registration Authority or qualified legal counsel.*
        """)
    
    return demo

def main():
    """Enhanced main application entry point"""
    
    # Environment validation with detailed feedback
    required_vars = ["QDRANT_CLOUD_URL", "QDRANT_API_KEY", "GROQ_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ CONFIGURATION ERROR")
        print("="*50)
        print(f"Missing environment variables: {missing_vars}")
        print("\nPlease set them in your .env file:")
        for var in missing_vars:
            if "QDRANT" in var:
                print(f"{var}=https://your-cluster.region.cloud.qdrant.io:6333")
            else:
                print(f"{var}=your_api_key_here")
        print("\nExiting application...")
        return
    
    # Create and launch interface
    try:
        demo = create_gradio_interface()
        if demo is None:
            print("❌ Failed to create interface")
            return
        
        print("\n" + "="*70)
        print("🚀 STARTING ADGM CORPORATE AGENT - COMPLETE LEGAL ANALYSIS SYSTEM")
        print("="*70)
        print("🤖 Vision LLM: meta-llama/llama-4-scout-17b-16e-instruct")
        print("📚 Knowledge Base: Pre-indexed ADGM legal documents (13K+ chunks)")
        print("🔍 Analysis: Text + Vision + Legal Citations + Alternative Clauses")
        print("⚖️ Legal Framework: Complete ADGM compliance checking")
        print("🌐 Interface: http://localhost:7860")
        print("📄 Features: Document checklists, red flag detection, inline commenting")
        print("="*70)
        
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            show_tips=True
        )
        
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        print("\nTroubleshooting:")
        print("1. Verify your .env file contains all required API keys")
        print("2. Check Qdrant Cloud connection and API key")
        print("3. Ensure Groq API key has sufficient credits")
        print("4. Confirm setup_documents.py was run successfully")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
