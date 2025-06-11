# CLI Reference

Complete command-line interface documentation for the Agentic RAG Workflow Engine.

## Basic Usage

```bash
python src/cli.py [OPTIONS] COMMAND [ARGS]...
```

## Global Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Configuration file path | `config.yaml` |
| `--help` | `-h` | Show help message | - |

## Commands

### status

Check system status and configuration.

```bash
python src/cli.py status
```

**Output includes:**
- Vector database connection status
- Number of indexed documents
- Current LLM and embedding models
- OpenAI API key status

**Example Output:**
```
Vector Database: Connected
Documents indexed: 42
LLM Model: gpt-4o
Embedding Model: text-embedding-3-large
OpenAI API Key: Configured
```

### ingest

Add documents to the vector database for querying.

```bash
python src/cli.py ingest DOCUMENTS_PATH [OPTIONS]
```

**Arguments:**
- `DOCUMENTS_PATH` - Path to file or directory containing documents

**Options:**
| Option | Short | Description |
|--------|-------|-------------|
| `--recursive` | `-r` | Process subdirectories recursively |

**Examples:**

```bash
# Ingest single file
python src/cli.py ingest data/report.pdf

# Ingest directory
python src/cli.py ingest data/project_docs

# Ingest recursively
python src/cli.py ingest data/ --recursive
```

**Supported File Types:**
- `.pdf` - PDF documents
- `.md` - Markdown files
- `.txt` - Plain text files

**Process:**
1. Discovers supported files in specified path
2. Extracts text content preserving structure
3. Applies semantic chunking for optimal retrieval
4. Generates embeddings using OpenAI
5. Stores in ChromaDB vector database

**Output:**
```
Found 3 documents to process
Processing document1.pdf...
Processing document2.md...
Processing document3.txt...
Successfully processed 3/3 documents
```

### query

Ask questions and get intelligent answers with source citations.

```bash
python src/cli.py query "YOUR QUESTION" [OPTIONS]
```

**Arguments:**
- `QUERY` - Your question as a string (use quotes for multi-word questions)

**Options:**
| Option | Short | Description |
|--------|-------|-------------|
| `--verbose` | `-v` | Show detailed workflow steps |

**Examples:**

```bash
# Basic query
python src/cli.py query "What are the main features?"

# Complex query
python src/cli.py query "Compare the different approaches discussed in the research papers"

# Verbose output
python src/cli.py query "How does the system work?" --verbose
```

**Response Format:**

**Standard Output:**
```
Generated Answer:
[Comprehensive answer with source citations using [Source X] format]

Sources:
1. data/document.pdf
   Page: 5
   Preview: First 100 characters of relevant text...

Processing time: 3.45s
```

**Verbose Output (with -v):**
```
Query Analyzer: Analyzed query structure and extracted key entities
  Query type: complex, Complexity: 4/5, Key entities: features, system

Retriever: Retrieved 5 relevant document chunks
  Sources: document1.pdf, document2.md

Synthesizer: Generated comprehensive answer with source citations
  Confidence: 0.85, Sources cited: 2

[Standard response format follows]
```

## Configuration

### Using Custom Config

Specify a different configuration file:

```bash
python src/cli.py --config /path/to/custom-config.yaml status
```

### Environment Variables

Set environment variables for sensitive configuration:

```bash
export OPENAI_API_KEY="your-api-key"
python src/cli.py status
```

## Workflow Examples

### Initial Setup

```bash
# 1. Check system status
python src/cli.py status

# 2. Add documents
python src/cli.py ingest data/ --recursive

# 3. Verify ingestion
python src/cli.py status

# 4. Start querying
python src/cli.py query "What topics are covered?"
```

### Research Workflow

```bash
# Add research papers
python src/cli.py ingest research-papers/ -r

# General overview
python src/cli.py query "What are the main research themes?"

# Specific analysis
python src/cli.py query "What methodologies are used in the neural network papers?"

# Comparative analysis
python src/cli.py query "How do the different approaches compare?" --verbose
```

### Documentation Search

```bash
# Index documentation
python src/cli.py ingest project-docs/ -r

# Find configuration info
python src/cli.py query "How do I configure authentication?"

# Troubleshooting
python src/cli.py query "What are common installation issues?"
```

## Error Handling

### Common Error Messages

**No documents found:**
```
No supported documents found.
```
*Solution: Ensure files have supported extensions (.pdf, .md, .txt)*

**OpenAI API Error:**
```
Error: OpenAI API key not found in environment variable: OPENAI_API_KEY
```
*Solution: Set your OpenAI API key as environment variable*

**Processing Error:**
```
Error processing document.pdf: [specific error]
```
*Solution: Check file permissions and format validity*

**No relevant documents:**
```
I couldn't find any relevant documents in the knowledge base to answer your query.
```
*Solution: Rephrase query or add more relevant documents*

## Performance Tips

### Optimal Usage

1. **Batch Ingestion**: Ingest all documents at once rather than individually
2. **Specific Queries**: Use specific, well-formed questions
3. **Iterative Refinement**: Use initial results to refine follow-up questions

### Query Optimization

**Good Queries:**
- "What are the technical requirements for the authentication system?"
- "How does the neural network architecture compare to traditional approaches?"
- "What are the main benefits of using semantic chunking?"

**Less Optimal:**
- "What is this?"
- "Tell me about stuff"
- "Everything about AI"

## Advanced Usage

### Programmatic Integration

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path('src').absolute()))

from core.orchestrator import WorkflowOrchestrator
from utils.config_manager import ConfigManager

# Initialize
config_manager = ConfigManager('config.yaml')
orchestrator = WorkflowOrchestrator(config_manager.config)

# Execute query
result = orchestrator.execute_workflow("Your question here")
print(result['answer'])
```

### Batch Processing

```bash
# Process multiple queries
echo "What are the main features?" | python src/cli.py query
echo "How does authentication work?" | python src/cli.py query
echo "What are the performance metrics?" | python src/cli.py query
```

### Custom Configuration

Create specialized configurations for different use cases:

```yaml
# research-config.yaml
retrieval:
  top_k: 10
  similarity_threshold: 0.6

chunking:
  chunk_size: 1500
  semantic_chunking: true
```

```bash
python src/cli.py --config research-config.yaml query "Research question"
```

## Debugging

### Verbose Mode Details

Use `--verbose` to see:
- Query analysis breakdown
- Entity extraction results
- Retrieval strategy decisions
- Source ranking process
- Confidence calculations

### Log Files

System logs are available for debugging:
- Check console output for real-time information
- Enable file logging in configuration for persistent logs

### Validation

```bash
# Test configuration
python src/cli.py status

# Test with sample query
python src/cli.py query "test query" --verbose

# Verify document count
python src/cli.py status | grep "Documents indexed"
```