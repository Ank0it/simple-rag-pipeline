# 🚀 Simple RAG (Retrieval-Augmented Generation) System

A powerful and easy-to-use RAG implementation using LangChain, enabling semantic search across multiple document types including text files, PDFs, and web pages.

## 📋 Overview

This project demonstrates a complete RAG pipeline that:
- **Ingests** documents from multiple sources (text files, PDFs, web pages)
- **Processes** documents into manageable chunks
- **Embeds** text into vector representations
- **Stores** embeddings in vector databases (Chroma & FAISS)
- **Retrieves** relevant information through semantic search

## ✨ Features

- 📄 **Multiple Document Loaders**
  - Text files (.txt)
  - PDF documents
  - Web pages (with BeautifulSoup parsing)

- 🔍 **Advanced Text Processing**
  - Recursive character text splitting
  - Configurable chunk size and overlap
  - Preserves document context

- 🧠 **Flexible Embedding Options**
  - HuggingFace Embeddings (free, local)
  - Google Generative AI Embeddings
  - Easy to switch between models

- 💾 **Multiple Vector Stores**
  - **Chroma DB** - Persistent vector storage
  - **FAISS** - High-performance similarity search

- 🎯 **Semantic Search**
  - Find relevant documents based on meaning
  - Ranked similarity results
  - Fast query processing

## 📁 Project Structure

```
Langchain/
├── rag/
│   ├── simplerag.ipynb      # Main notebook
│   ├── speech.txt           # Sample text file
│   └── sample.pdf           # Sample PDF file
├── .env                     # Environment variables
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.13.2 or higher
- Virtual environment (recommended)

### Setup Steps

1. **Clone or download the repository**
   ```bash
   cd C:\.Agenx\Langchain
   ```

2. **Create and activate virtual environment**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

5. **Launch Jupyter Notebook**
   ```powershell
   jupyter notebook
   ```

## 📦 Dependencies

See `requirements.txt` for complete list. Key packages:
- `langchain` - Core LangChain framework
- `langchain-community` - Community integrations
- `langchain-google-genai` - Google AI integration
- `chromadb` - Chroma vector database
- `faiss-cpu` - FAISS similarity search
- `sentence-transformers` - HuggingFace embeddings

## 🚀 Quick Start

### 1. Data Ingestion

**Load Text Files:**
```python
from langchain_community.document_loaders import TextLoader 
loader = TextLoader('speech.txt')
text_documents = loader.load()
```

**Load PDF Files:**
```python
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("sample.pdf")
text_docs = loader.load()
```

**Load Web Pages:**
```python
from langchain_community.document_loaders import WebBaseLoader
import bs4

loader = WebBaseLoader(
    web_paths=("https://example.com",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(
        class_=("post-title", "post-content")
    ))
)
text_documents = loader.load()
```

### 2. Text Splitting

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
documents = text_splitter.split_documents(text_docs)
```

### 3. Create Embeddings & Vector Store

**Using HuggingFace (Free, Local):**
```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma 

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = Chroma.from_documents(documents[:20], embeddings)
```

**Using Google Gemini (Requires API Key):**
```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)
db = Chroma.from_documents(documents[:20], embeddings)
```

**Using FAISS:**
```python
from langchain_community.vectorstores import FAISS

db_faiss = FAISS.from_documents(documents[:20], embeddings)
```

### 4. Query the Vector Store

```python
query = "What is clean code?"
results = db.similarity_search(query)
print(results[0].page_content)
```

## 💡 Usage Examples

### Example 1: Search PDF Content
```python
query = "What are error handling best practices?"
result = db.similarity_search(query)
print(result[0].page_content)
```

### Example 2: Multi-Result Search
```python
query = "Why is clean code important?"
results = db.similarity_search(query, k=3)  # Get top 3 results
for i, doc in enumerate(results):
    print(f"Result {i+1}:\n{doc.page_content}\n")
```

### Example 3: Search with Score
```python
query = "Code formatting purpose"
results = db.similarity_search_with_score(query)
for doc, score in results:
    print(f"Score: {score}\n{doc.page_content}\n")
```

## 🔧 Configuration

### Chunk Size Tuning
Adjust based on your use case:
- **Small chunks (200-500)**: Better for precise answers
- **Medium chunks (500-1000)**: Balanced approach (default)
- **Large chunks (1000-2000)**: Better for context preservation

### Embedding Models

**HuggingFace Options:**
- `sentence-transformers/all-MiniLM-L6-v2` (fast, lightweight)
- `sentence-transformers/all-mpnet-base-v2` (more accurate)
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingual)

**Google Options:**
- `models/embedding-001` (default)
- Requires valid Google API key

## 🐛 Troubleshooting

### Issue: Module Not Found
```powershell
pip install --upgrade langchain langchain-community langchain-core
```

### Issue: Google API Quota Exceeded
- Wait 24 hours for quota reset
- Or switch to HuggingFace embeddings (no quota limits)

### Issue: FAISS Installation Error
```powershell
pip install faiss-cpu --no-cache-dir
```

### Issue: Virtual Environment Corrupted
```powershell
Remove-Item -Recurse -Force .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 📊 Performance Tips

1. **Batch Processing**: Process documents in batches for large datasets
2. **Persistent Storage**: Save vector stores to disk to avoid re-embedding
3. **Index Optimization**: Use FAISS for production with large document collections
4. **Chunk Overlap**: Use 10-20% overlap for better context preservation

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Add more document loaders
- Implement conversation memory
- Add LLM integration for answer generation
- Create CLI interface
- Add unit tests

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - Framework foundation
- [HuggingFace](https://huggingface.co/) - Embedding models
- [Chroma](https://www.trychroma.com/) - Vector database
- [FAISS](https://github.com/facebookresearch/faiss) - Similarity search

## 📧 Contact

For questions or support, please open an issue in the repository.

---

**Built with ❤️ using LangChain**