# Agentic RAG Workflow Engine

An intelligent multi-agent Retrieval-Augmented Generation (RAG) system that uses specialized agents to analyze queries, retrieve relevant documents, and synthesize comprehensive answers with source citations.

## Features

- **Intelligent Query Analysis**: Break down complex queries and identify key entities
- **Multi-Agent Architecture**: Specialized agents for analysis, retrieval, and synthesis
- **Semantic Document Chunking**: Keep related content together for better retrieval
- **Source Citation**: All answers include citations to source documents
- **Modular Design**: Easy to swap LLMs, embedding models, and vector databases
- **Command-Line Interface**: Simple CLI for document ingestion and querying

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd agentic-rag-engine
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Usage

1. **Ingest documents** into the vector database:
```bash
python src/cli.py ingest data/project_docs --recursive
```

2. **Query** the system:
```bash
python src/cli.py query "What are the main features of the project?"
```

3. **Check system status**:
```bash
python src/cli.py status
```

## Architecture

### Multi-Agent Workflow

1. **Query Analyzer Agent**: Analyzes user queries to extract key entities and break down complex questions
2. **Retrieval Agent**: Performs intelligent document retrieval based on the analysis
3. **Synthesis Agent**: Generates comprehensive answers with proper source citations

### Supported Document Types

- PDF files (`.pdf`)
- Markdown files (`.md`) 
- Text files (`.txt`)

## Configuration

The system uses `config.yaml` for configuration. Key settings include:

- **LLM Model**: Default is `gpt-4o`
- **Embedding Model**: Default is `text-embedding-3-large`
- **Vector Database**: ChromaDB with persistent storage
- **Chunking Strategy**: Semantic chunking with configurable sizes

## Project Structure

```
/agentic-rag-engine
├── data/                    # Source documents for indexing
├── notebooks/              # Jupyter notebooks for experimentation
├── src/                    # Main source code
│   ├── core/              # Core application logic
│   ├── components/        # Pluggable components (LLMs, vector stores)
│   └── processing/        # Document processing and chunking
└── tests/                 # Unit tests
```

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Jupyter Notebooks

Explore the system with the included notebooks:

- `notebooks/01_data_ingestion_test.ipynb` - Test document ingestion
- `notebooks/02_retrieval_test.ipynb` - Test retrieval functionality  
- `notebooks/03_full_workflow_demo.ipynb` - Complete workflow demonstration

## License

MIT License - see LICENSE file for details.