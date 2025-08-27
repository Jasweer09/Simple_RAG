# 📝 RAG Experimentation Project: Data Loaders, Chunking, and Custom Embeddings

Welcome to my **Retrieval-Augmented Generation (RAG)** experimentation project! This project explores building a RAG pipeline using **LangChain** for data ingestion, chunking, and vector storage, with a focus on creating a custom embeddings wrapper and evaluating different vector stores. Below, I outline the experiments, key learnings, and reasons for specific design choices, including why I opted for a custom embeddings wrapper instead of built-in options.

---

## 🌟 Project Overview

This project demonstrates the implementation of a RAG system to process and query documents (text and PDF) using LangChain's data loaders, text splitting, and vector stores. I ingested a text file (`Speech.txt`) and a PDF resume, performed chunking, created a custom embeddings wrapper for the Gemma-2b model, and experimented with **Chroma** and **FAISS** vector stores for efficient similarity searches. The goal was to understand the RAG pipeline's components and optimize retrieval for downstream tasks like fraud detection queries.

---

## 🛠️ Project Structure

- **`simplerag.ipynb`**: Jupyter notebook containing the RAG pipeline implementation, including data ingestion, chunking, custom embeddings, and vector store experiments.
- **`Speech.txt`**: Sample text file with news snippets for ingestion and testing.
- **PDF Resume**: A resume document used to test PDF ingestion and querying (e.g., fraud detection projects).
- **Dependencies**: LangChain, Sentence Transformers, Chroma, FAISS, and related libraries.

---

## 🎯 Experiments and Key Learnings

### 1. Data Ingestion with LangChain Loaders
- **Experiment**: Used `TextLoader` for `Speech.txt` and a PDF loader (implied in the notebook) for the resume. Also explored `WebBaseLoader` with BeautifulSoup for web content extraction.
- **Learnings**:
  - `TextLoader` efficiently handles plain text, preserving metadata like source file names.
  - `WebBaseLoader` with `bs4.SoupStrainer` allows targeted scraping of web content (e.g., specific HTML classes), reducing noise.
  - PDF ingestion required careful preprocessing to handle formatting issues and extract structured text.

### 2. Text Chunking
- **Experiment**: Applied `RecursiveCharacterTextSplitter` with a chunk size of 1000 and 200 overlap to split documents into manageable pieces.
- **Learnings**:
  - Chunking ensures compatibility with embedding models’ input limits and improves retrieval relevance.
  - Overlap (200 characters) preserves context across chunks, critical for coherent retrieval in RAG.
  - Balancing chunk size is key: too large risks losing granularity, while too small fragments context.

### 3. Custom Embeddings Wrapper
- **Experiment**: Created a `GemmaEmbeddings` wrapper class for the `Jaume/gemma-2b-embeddings` model, implementing LangChain’s `Embeddings` interface (`embed_documents` and `embed_query`).
- **Why Not Built-In Embeddings?**:
  - **Specific Model Choice**: The Gemma-2b model isn’t natively supported by LangChain’s built-in embeddings (e.g., `OllamaEmbeddings`). A custom wrapper was necessary to integrate this specific Hugging Face model.
  - **Flexibility**: The wrapper allows fine-grained control over embedding generation, enabling future modifications (e.g., batch processing optimizations).
  - **Portability**: Custom implementation ensures compatibility across different environments without relying on LangChain’s predefined integrations, which may lag for newer models.
- **Learnings**:
  - Wrapping Sentence Transformers with LangChain’s `Embeddings` interface is straightforward but requires handling list-to-list conversions for compatibility.
  - Gemma-2b embeddings provided robust semantic representations, improving retrieval accuracy for queries like “fraud detection project.”

### 4. Vector Stores: Chroma vs. FAISS
- **Experiment**: Stored document embeddings in both **Chroma** and **FAISS** vector stores and performed similarity searches (e.g., querying for fraud detection projects).
- **Learnings**:
  - **Chroma**:
    - Easy to use with LangChain, offering persistent storage and simple setup for small-scale projects.
    - Suitable for rapid prototyping but slower for large-scale datasets due to less optimized indexing.
  - **FAISS**:
    - Highly efficient for large datasets, leveraging optimized vector quantization (e.g., Gaussian clustering) for fast similarity searches.
    - Better suited for production environments requiring high-speed retrieval.
    - Requires manual setup for persistence compared to Chroma’s out-of-the-box solution.
  - Performed `similarity_search`, `similarity_search_by_vector`, and `similarity_search_with_score` to compare retrieval performance, confirming FAISS’s edge in speed and scalability.

### 5. Querying and Retrieval
- **Experiment**: Queried the vector stores for details on a fraud detection project, retrieving relevant resume chunks.
- **Learnings**:
  - FAISS returned more precise results with scores, enabling fine-tuned ranking of retrieved documents.
  - Custom embeddings ensured semantic alignment between queries and document chunks, critical for accurate retrieval.
  - Metadata (e.g., source file, page number) preserved during ingestion aided in tracing results back to original documents.

---

## 🚀 How to Use the Project

### Prerequisites
- Python 3.8+
- Install dependencies: `pip install langchain langchain-community sentence-transformers chromadb faiss-cpu python-dotenv bs4`
- Install Hugging Face’s Sentence Transformers: `pip install sentence-transformers`
- Access to the Gemma-2b embeddings model (`Jaume/gemma-2b-embeddings`) via Hugging Face.

### Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data**:
   - Place `Speech.txt` and any PDF files in the project directory.
   - Update file paths in `simplerag.ipynb` if needed.

4. **Run the Notebook**:
   - Launch Jupyter: `jupyter notebook simplerag.ipynb`
   - Execute cells sequentially to ingest data, chunk documents, generate embeddings, and query the vector stores.

5. **Query Example**:
   - Run a query like `"Tell me Jasweers Fraud Detection Project?"` to retrieve relevant document chunks.
   - Compare results from Chroma and FAISS vector stores.

---

## 🔧 Customizing the Project

1. **Try Different Embedding Models**:
   - Modify the `GemmaEmbeddings` class to use other Hugging Face models (e.g., `sentence-transformers/all-MiniLM-L6-v2`) for comparison.
   - Update `model = SentenceTransformer("new-model-name")` in the notebook.

2. **Adjust Chunking Parameters**:
   - Experiment with `chunk_size` and `chunk_overlap` in `RecursiveCharacterTextSplitter` to optimize retrieval for your use case.

3. **Add New Data Sources**:
   - Extend the pipeline to ingest CSVs, JSON, or additional web pages using LangChain’s loaders.
   - Example: Add a CSV loader with `from langchain_community.document_loaders import CSVLoader`.

4. **Explore Other Vector Stores**:
   - Integrate vector stores like **Pinecone** or **Weaviate** for cloud-based scalability.
   - Update the vector store initialization in the notebook (e.g., `Pinecone.from_documents`).

5. **Enhance Queries**:
   - Add a prompt template to refine queries for specific tasks (e.g., summarizing fraud detection details).
   - Integrate an LLM (e.g., via LangChain’s `ChatOpenAI`) for generative responses based on retrieved documents.

---

## 🧠 Why This Approach?

- **Custom Embeddings Wrapper**: Built-in LangChain embeddings (e.g., `OllamaEmbeddings`) didn’t support Gemma-2b, and a custom wrapper provided flexibility for future model updates and optimizations.
- **Chroma vs. FAISS**: Chroma is great for quick prototyping, while FAISS excels in performance for large-scale applications, allowing me to evaluate trade-offs.
- **Chunking Strategy**: Recursive splitting with overlap ensured context preservation, critical for RAG’s retrieval accuracy.
- **Data Loaders**: Using multiple loaders (text, PDF, web) demonstrated LangChain’s versatility in handling diverse data sources.

---

## 🎉 Key Takeaways

This project deepened my understanding of RAG pipelines, from data ingestion to retrieval. Key insights:
- Custom embeddings wrappers unlock flexibility for non-standard models.
- FAISS outperforms Chroma for speed and scalability in large datasets.
- Effective chunking is crucial for balancing context and retrieval granularity.
- LangChain’s loaders simplify multi-source data ingestion, making RAG pipelines robust.

Check out the code on my GitHub: [https://github.com/Jasweer09](https://github.com/Jasweer09). Let’s connect to discuss AI, RAG, or cool projects! 🚀