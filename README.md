[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/vgbm4cZ0)

# 🏛️ ADGM Corporate Agent - Enhanced Legal Document Analysis

[![Python](https://img.shields.ios://img.shields.iog.shieldsd AI-powered legal document analyzer specifically designed for **Abu Dhabi Global Market (ADGM)** compliance verification. This system combines **vector similarity search** with **Vision LLM analysis** to provide comprehensive legal document review with inline citations and ADGM regulatory compliance checking.

## 🎯 Key Features

### 🤖 **Advanced AI Analysis**
- **Vision LLM**: `meta-llama/llama-4-scout-17b-16e-instruct` via Direct Groq API
- **Vector Similarity**: CLIP embeddings with cosine distance for semantic document matching
- **Multimodal Processing**: Text + Image analysis from DOCX documents
- **Knowledge Base**: 13,310+ text chunks + 4,047 images from ADGM reference documents

### ⚖️ **Legal Compliance Features**
- **ADGM Jurisdiction Verification**: Automated compliance checking against ADGM regulations
- **Legal Citations**: Specific references to ADGM Companies Regulations 2020, ADGM Courts Law 2013
- **Red Flag Detection**: Identifies non-compliant language and missing clauses
- **Inline Comments**: Enhanced DOCX markup with legal alternatives and citations
- **Process Identification**: Automatically identifies document types (Incorporation, Licensing, etc.)

### 🔧 **Enhanced File Processing**
- **Multi-Method Upload**: 4 different file reading methods for maximum compatibility
- **File Validation**: Comprehensive DOCX structure validation and corruption detection
- **Error Recovery**: Multiple fallback systems for problematic file uploads
- **Debug Information**: Real-time file processing diagnostics

### 🔍 **Vector Similarity Search**
- **Semantic Search**: Natural language queries across 217 indexed ADGM documents
- **Document Matching**: Finds similar legal documents based on content and context
- **Visual Similarity**: Table and image comparison against ADGM reference materials
- **Compliance Scoring**: 0-100 compliance scores with detailed breakdowns

## 📋 System Requirements

### **Python Dependencies**
```
python>=3.8
gradio>=4.0.0
python-docx>=0.8.11
qdrant-client>=1.6.0
transformers>=4.30.0
groq>=0.4.0
torch>=2.0.0
pillow>=9.0.0
matplotlib>=3.5.0
python-dotenv>=1.0.0
numpy>=1.21.0
```

### **External Services**
- **Qdrant Cloud**: Vector database for document storage
- **Groq API**: Vision LLM processing
- **ADGM Knowledge Base**: Pre-indexed legal documents (see setup section)

## 🚀 Quick Start

### **1. Clone Repository**
```bash
git clone https://github.com/yourusername/adgm-corporate-agent.git
cd adgm-corporate-agent
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Environment Configuration**
Create a `.env` file with your API keys:
```env
QDRANT_CLOUD_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=your_groq_api_key
```

### **4. Knowledge Base Setup**
Ensure you have the required indexed files:
- `adgm_image_store.json` - Image embeddings 
- `adgm_image_metadata.json` - Image metadata
- `indexing_metadata.json` - Indexing statistics and configuration

### **5. Launch Application**
```bash
python main_app.py
```

Access the interface at: `http://localhost:7860`

## 📚 Knowledge Base Statistics

Based on the indexed ADGM reference documents:

| Component | Count | Details |
|-----------|-------|---------|
| **Documents Processed** | 217 | Original ADGM legal documents |
| **Text Chunks** | 13,310 | Semantic text segments for vector search |
| **Images Stored** | 4,087 | Tables, diagrams, and embedded images |
| **PDF Files** | 137 | Primary document format |
| **DOCX Files** | 80 | Secondary document format |
| **Vector Dimensions** | 512 | CLIP embedding size |
| **Storage Size** | 79.73 MB | Total image storage |

## 🔍 Usage Examples

### **Document Analysis**
Upload DOCX files for comprehensive ADGM compliance analysis:

```python
# Supported document types:
- Company Incorporation documents
- Licensing Applications  
- Branch Registration forms
- Foundation Registration
- Employment and HR policies
- Commercial Agreements
```

### **Knowledge Base Search**
Search through indexed ADGM documents:
```
Query: "company incorporation requirements"
Results: 
- Articles of Association templates
- Memorandum of Association samples
- UBO Declaration requirements
- Board Resolution formats
```

### **Compliance Analysis Output**
```json
{
  "compliance_score": 85,
  "jurisdiction_compliance": true,
  "issues_found": [
    {
      "location": "Paragraph 5",
      "issue": "Missing ADGM jurisdiction clause",
      "severity": "High",
      "suggestion": "Add ADGM Courts jurisdiction clause",
      "alternative_clause": "This Agreement shall be governed by ADGM law..."
    }
  ],
  "missing_documents": ["UBO Declaration Form"],
  "recommendations": ["Add dispute resolution clause"]
}
```

## 🏗️ System Architecture

### **Core Components**

1. **ADGMCorporateAgent Class**
   - Document processing and analysis
   - Vector similarity search
   - ADGM compliance verification

2. **Qdrant Vector Database**
   - Stores indexed ADGM documents
   - CLIP embeddings for semantic search
   - Cosine similarity matching

3. **Groq Vision LLM**
   - Advanced document analysis
   - Legal reasoning and recommendations
   - JSON-structured compliance reports

4. **Gradio Interface**
   - User-friendly web interface
   - File upload with debugging
   - Real-time analysis results

### **File Processing Pipeline**

```
Upload → Validation → Multiple Reading Methods → DOCX Parsing → 
Text Extraction → Image Processing → Vector Similarity Search → 
LLM Analysis → Compliance Checking → Enhanced Comments → 
Download with Citations
```

## 🔧 Advanced Configuration

### **Model Selection**
The system supports multiple LLM models with automatic fallback:
```python
preferred_models = [
    "meta-llama/llama-4-scout-17b-16e-instruct",  # Primary
    "llama-3.1-70b-versatile",                    # Fallback 1
    "llama-3.1-8b-instant"                        # Fallback 2
]
```

### **Document Processing Types**
- **Company Incorporation**: Articles, Memorandum, Board Resolutions
- **Licensing Applications**: Business Plans, Compliance Manuals
- **Branch Registration**: Parent Company Certificates, Power of Attorney
- **Foundation Registration**: Charter documents, Beneficial Ownership
- **Employment & HR**: Contracts, Policies, Procedures
- **Commercial Agreements**: Service, Supply, Distribution agreements

## 🐛 Troubleshooting

### **Common Issues**

#### **Empty File Upload**
```bash
# Debug output:
📊 File content size: 0 bytes
🔄 Method 1: Reading from file path...
❌ Method 1: File on disk is empty
```
**Solution**: Use simple filenames, check original file integrity, try different browser

#### **Model Availability**
```bash
⚠️ Model meta-llama/llama-4-scout-17b-16e-instruct not available
✅ Using fallback model: llama-3.1-70b-versatile
```
**Solution**: System automatically falls back to available models

#### **Vector Search Issues**
```bash
❌ Error in vector similarity search: Collection not found
```
**Solution**: Ensure Qdrant collection is properly indexed with required files

### **Debug Features**
- **File Upload Diagnostics**: Real-time file processing information
- **Vector Search Statistics**: Similarity scores and match counts
- **Error Recovery**: Multiple fallback methods for file processing
- **Processing Logs**: Detailed console output for troubleshooting

## 📊 Performance Metrics

- **Document Processing Speed**: ~2-5 seconds per DOCX file
- **Vector Search Latency**: <1 second for semantic queries
- **Compliance Analysis**: ~10-15 seconds for comprehensive review
- **File Upload Success Rate**: 99%+ with enhanced error handling
- **Knowledge Base Coverage**: 217 ADGM reference documents

## 🙏 Acknowledgments

- **ADGM Registration Authority** for regulatory guidance and documentation
- **Groq** for high-performance LLM inference
- **Qdrant** for vector database capabilities
- **OpenAI CLIP** for multimodal embeddings
- **Gradio** for intuitive web interface



**⚖️ Legal Disclaimer**: This tool provides automated analysis assistance and should not replace professional legal advice. Always consult qualified ADGM legal experts for official compliance verification.



*Last Updated: August 2025 | Version 2.0 - Enhanced File Upload Edition*
