# Quick Start Guide

Get up and running with the Agentic RAG Workflow Engine in minutes.

## 30-Second Setup

```bash
# 1. Clone and install
git clone https://github.com/your-org/agentic-rag-engine.git
cd agentic-rag-engine
pip install -r requirements.txt

# 2. Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# 3. Add documents and start querying
cp your-documents.pdf data/project_docs/
python src/cli.py ingest data/ --recursive
python src/cli.py query "What are the main topics in my documents?"
```

## Step-by-Step Tutorial

### 1. Verify Installation

Check that everything is working:

```bash
cd src
python cli.py status
```

Expected output:
```
Vector Database: Connected
Documents indexed: 0
LLM Model: gpt-4o
Embedding Model: text-embedding-3-large
OpenAI API Key: Configured
```

### 2. Add Your First Document

Place a document in the data directory:

```bash
# Example: Add a project README
cp ../README.md data/project_docs/
```

Or create a sample document:

```bash
echo "# Sample Document
This is a test document for the RAG system.
It contains information about artificial intelligence and machine learning." > data/project_docs/sample.md
```

### 3. Ingest Documents

Index your documents into the vector database:

```bash
python src/cli.py ingest data/project_docs --recursive
```

You'll see progress output:
```
Found 1 documents to process
Processing sample.md...
Successfully processed 1/1 documents
```

### 4. Ask Your First Question

Query the system:

```bash
python src/cli.py query "What information is available about AI?"
```

Example response:
```
Generated Answer:
Based on the available documents, artificial intelligence and machine learning 
are mentioned as key topics [Source 1]. The system contains information about 
these technologies and their applications.

Sources:
1. data/project_docs/sample.md
   Preview: This is a test document for the RAG system. It contains information...
```

### 5. Try More Complex Queries

Test the multi-agent capabilities:

```bash
# Complex analysis query
python src/cli.py query "Compare the different approaches mentioned in the documents"

# Entity-focused query  
python src/cli.py query "What are the main technical requirements?"

# Multi-part question
python src/cli.py query "How does the system work and what are its benefits?"
```

## Interactive Exploration

Use Jupyter notebooks for interactive exploration:

```bash
# Install Jupyter if not already installed
pip install jupyter

# Start Jupyter
jupyter notebook notebooks/

# Open and run:
# - 01_data_ingestion_test.ipynb
# - 02_retrieval_test.ipynb
# - 03_full_workflow_demo.ipynb
```

## Common Use Cases

### Research Assistant
```bash
# Add research papers
cp research-papers/*.pdf data/research_papers/
python src/cli.py ingest data/research_papers --recursive

# Query across papers
python src/cli.py query "What are the latest findings on neural networks?"
```

### Documentation Search
```bash
# Add project documentation
cp docs/*.md data/project_docs/
python src/cli.py ingest data/project_docs --recursive

# Find specific information
python src/cli.py query "How do I configure the authentication system?"
```

### Knowledge Base
```bash
# Add various document types
cp knowledge-base/*.{pdf,md,txt} data/
python src/cli.py ingest data/ --recursive

# Cross-reference information
python src/cli.py query "What are the connections between these topics?"
```

## Command Reference

| Command | Description | Example |
|---------|-------------|---------|
| `status` | Show system status | `python src/cli.py status` |
| `ingest` | Add documents to database | `python src/cli.py ingest data/ -r` |
| `query` | Ask questions | `python src/cli.py query "your question"` |

### Ingestion Options
- `--recursive, -r`: Process subdirectories
- `--config, -c`: Use custom config file

### Query Options
- `--verbose, -v`: Show detailed workflow steps
- `--config, -c`: Use custom config file

## Understanding the Output

### Query Response Structure
```
Generated Answer: [Main response with citations]

Sources:
1. filename.pdf
   Page: 5
   Preview: [First 100 characters of relevant text]

Processing time: 3.45s
```

### Workflow Steps (with --verbose)
```
Query Analyzer: Analyzed query structure and extracted key entities
  Query type: complex, Complexity: 4/5, Key entities: AI, machine learning

Retriever: Retrieved 5 relevant document chunks  
  Sources: sample.md, research-paper.pdf

Synthesizer: Generated comprehensive answer with source citations
  Confidence: 0.85, Sources cited: 2
```

## Tips for Better Results

1. **Use descriptive questions**: Instead of "What is this?", ask "What are the main features of the neural network architecture?"

2. **Be specific**: "How does the authentication work?" vs "Tell me about authentication"

3. **Ask follow-up questions**: The system maintains context for related queries

4. **Check sources**: Always review the cited sources to verify accuracy

5. **Use verbose mode**: Add `-v` to understand how the system processes your query

## Next Steps

- Read the [Architecture Overview](Architecture-Overview) to understand how the system works
- Learn about [Document Processing](Document-Processing) for optimal document preparation
- Explore [Configuration](Configuration) options to customize behavior
- Check [Performance Tuning](Performance-Tuning) for optimization tips

## Troubleshooting Quick Fixes

**No documents found**: Ensure files are in `data/` subdirectories with supported extensions (.pdf, .md, .txt)

**API errors**: Verify your OpenAI API key is set and has sufficient credits

**Slow responses**: Check your internet connection and consider using fewer documents initially

**Empty responses**: Try rephrasing your question or checking if documents were properly ingested