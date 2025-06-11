# Installation Guide

This guide will help you install and set up the Agentic RAG Workflow Engine on your system.

## Prerequisites

Before installing the Agentic RAG Workflow Engine, ensure you have:

- **Python 3.8 or higher** installed on your system
- **Git** for cloning the repository
- **OpenAI API account** with API key access
- At least **2GB of free disk space** for the vector database

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/agentic-rag-engine.git
cd agentic-rag-engine
```

### 2. Install Dependencies

The project uses modern Python package management. Install all required dependencies:

```bash
pip install -r requirements.txt
```

**Required packages include:**
- `click` - Command-line interface framework
- `chromadb` - Vector database for document storage
- `openai` - OpenAI API client
- `pdfplumber` - PDF text extraction
- `pydantic` - Data validation and settings
- `pyyaml` - YAML configuration parsing
- `rich` - Enhanced terminal output
- `numpy` - Numerical computations
- `markdown` - Markdown processing

### 3. Configure OpenAI API

The system requires an OpenAI API key for:
- Query analysis using GPT-4o
- Document embedding generation
- Answer synthesis

#### Get Your API Key
1. Visit [OpenAI Platform](https://platform.openai.com)
2. Create an account or sign in
3. Navigate to API Keys section
4. Create a new API key

#### Set Environment Variable
Set your API key as an environment variable:

**Linux/macOS:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Windows:**
```cmd
set OPENAI_API_KEY=your-api-key-here
```

**Permanent Setup (Linux/macOS):**
Add to your `~/.bashrc` or `~/.zshrc`:
```bash
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### 4. Verify Installation

Test that everything is working correctly:

```bash
cd src
python cli.py status
```

You should see output showing:
- Vector Database: Connected
- OpenAI API Key: Configured
- Current configuration details

## Configuration

The system uses a `config.yaml` file for configuration. The default configuration should work for most users, but you can customize:

```yaml
llm:
  model: gpt-4o
  temperature: 0.1
  max_tokens: 2000

embeddings:
  model: text-embedding-3-large

vector_store:
  type: chromadb
  path: ./vector_db
  collection_name: documents

chunking:
  chunk_size: 1000
  chunk_overlap: 200
  semantic_chunking: true

retrieval:
  top_k: 5
  similarity_threshold: 0.7
```

## Directory Structure

After installation, your directory should look like:

```
agentic-rag-engine/
├── data/                    # Place your documents here
│   ├── project_docs/
│   └── research_papers/
├── src/                     # Main source code
├── notebooks/              # Jupyter notebooks for testing
├── tests/                  # Unit tests
├── config.yaml            # Configuration file
└── README.md
```

## Next Steps

After successful installation:

1. **Add Documents**: Place PDF, Markdown, or text files in the `data/` directory
2. **Ingest Documents**: Run `python src/cli.py ingest data/ --recursive`
3. **Start Querying**: Use `python src/cli.py query "your question here"`

## Troubleshooting

### Common Issues

**ImportError: No module named 'chromadb'**
- Solution: Run `pip install chromadb` or reinstall all dependencies

**OpenAI API Error: Invalid API Key**
- Solution: Verify your API key is correctly set and has sufficient credits

**Permission Denied on vector_db directory**
- Solution: Ensure write permissions in the project directory

**PDF Processing Errors**
- Solution: Install additional PDF dependencies: `pip install pdfplumber pypdf2`

### Getting Help

If you encounter issues:
1. Check the [Troubleshooting](Troubleshooting) page
2. Review error messages carefully
3. Ensure all dependencies are properly installed
4. Verify your OpenAI API key is valid and has credits

## System Requirements

### Minimum Requirements
- 4GB RAM
- 2GB free disk space
- Python 3.8+
- Internet connection for API calls

### Recommended Requirements
- 8GB RAM
- 10GB free disk space
- Python 3.11+
- SSD storage for better performance

## Development Installation

For development and contributing:

```bash
# Clone with development branch
git clone -b develop https://github.com/your-org/agentic-rag-engine.git

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest black flake8 jupyter
```

See the [Contributing Guide](Contributing) for more development setup details.