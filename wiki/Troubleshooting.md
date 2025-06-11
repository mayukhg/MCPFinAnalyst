# Troubleshooting

Common issues and solutions for the Agentic RAG Workflow Engine.

## Installation Issues

### Python Version Compatibility

**Error:** `SyntaxError: invalid syntax`
```bash
File "src/cli.py", line 15
    def function() -> str:
                   ^
SyntaxError: invalid syntax
```

**Solution:** Ensure Python 3.8 or higher is installed
```bash
python --version  # Should show 3.8+
python3 --version # Try python3 if python shows older version
```

### Missing Dependencies

**Error:** `ModuleNotFoundError: No module named 'chromadb'`

**Solution:** Install required packages
```bash
pip install -r requirements.txt
# Or install individual packages
pip install chromadb openai click pydantic pyyaml rich
```

### Package Conflicts

**Error:** `VersionConflict: package-name 1.0 is installed but package-name>=2.0 is required`

**Solution:** Create clean virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration Issues

### Missing Configuration File

**Error:** `FileNotFoundError: config.yaml not found`

**Solution:** Create default configuration
```bash
python src/cli.py status  # Auto-creates config.yaml
```

### Invalid YAML Syntax

**Error:** `yaml.scanner.ScannerError: mapping values are not allowed here`

**Solution:** Check YAML formatting
```yaml
# ❌ Incorrect
llm:
model: gpt-4o
  temperature: 0.1

# ✅ Correct
llm:
  model: gpt-4o
  temperature: 0.1
```

### Configuration Validation Errors

**Error:** `Configuration validation failed: Invalid model name`

**Solution:** Use supported model names
```yaml
llm:
  model: gpt-4o  # Not gpt-4-turbo or custom names
```

## API Key Issues

### Missing API Key

**Error:** `OpenAI API key not found in environment variable: OPENAI_API_KEY`

**Solution:** Set environment variable
```bash
# Linux/macOS
export OPENAI_API_KEY="your-key-here"

# Windows
set OPENAI_API_KEY=your-key-here

# Make permanent by adding to ~/.bashrc or ~/.zshrc
echo 'export OPENAI_API_KEY="your-key-here"' >> ~/.bashrc
```

### Invalid API Key

**Error:** `AuthenticationError: Incorrect API key provided`

**Solution:** Verify API key
```bash
# Check key format (starts with sk-)
echo $OPENAI_API_KEY

# Test key validity
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Insufficient Credits

**Error:** `RateLimitError: You exceeded your current quota`

**Solution:** Check OpenAI account
1. Visit [OpenAI Platform](https://platform.openai.com)
2. Check billing and usage
3. Add payment method if needed

## Document Processing Issues

### No Documents Found

**Error:** `No supported documents found`

**Solution:** Check file locations and extensions
```bash
# Verify files exist
ls -la data/
ls -la data/project_docs/

# Check file extensions
find data/ -name "*.pdf" -o -name "*.md" -o -name "*.txt"
```

### PDF Processing Errors

**Error:** `ImportError: PDF processing requires 'pdfplumber'`

**Solution:** Install PDF processing libraries
```bash
pip install pdfplumber pypdf2
```

**Error:** `PdfReadError: EOF marker not found`

**Solution:** Verify PDF file integrity
```bash
# Try different PDF processing approach
# Edit config.yaml to use fallback method
```

### File Permission Errors

**Error:** `PermissionError: [Errno 13] Permission denied`

**Solution:** Check file permissions
```bash
# Make files readable
chmod 644 data/project_docs/*

# Make directories accessible
chmod 755 data/project_docs/
```

### Large File Issues

**Error:** `MemoryError: Unable to allocate array`

**Solution:** Reduce chunk size or process files individually
```yaml
chunking:
  chunk_size: 500  # Reduce from default 1000
  chunk_overlap: 100
```

## Vector Database Issues

### ChromaDB Connection Errors

**Error:** `ConnectionError: Unable to connect to ChromaDB`

**Solution:** Check directory permissions and disk space
```bash
# Check available disk space
df -h

# Verify vector_db directory permissions
ls -la vector_db/
mkdir -p vector_db
chmod 755 vector_db
```

### Database Corruption

**Error:** `sqlite3.DatabaseError: database disk image is malformed`

**Solution:** Reset vector database
```bash
# Backup if needed
mv vector_db vector_db_backup

# Reingest documents
python src/cli.py ingest data/ --recursive
```

### Embedding Generation Failures

**Error:** `OpenAIError: The model 'text-embedding-invalid' does not exist`

**Solution:** Use valid embedding model
```yaml
embeddings:
  model: text-embedding-3-large  # Valid model name
```

## Query Processing Issues

### No Relevant Documents Found

**Error:** `I couldn't find any relevant documents in the knowledge base`

**Solutions:**
1. **Rephrase query:** Use different keywords
2. **Check similarity threshold:** Lower threshold in config
3. **Verify document ingestion:** Ensure documents were processed

```yaml
retrieval:
  similarity_threshold: 0.5  # Lower from 0.7
```

### Query Timeout

**Error:** `TimeoutError: Request timed out`

**Solutions:**
1. **Check internet connection**
2. **Reduce max_tokens**
3. **Simplify query**

```yaml
llm:
  max_tokens: 1000  # Reduce from 2000
```

### API Rate Limiting

**Error:** `RateLimitError: Rate limit exceeded`

**Solution:** Implement delays or upgrade OpenAI plan
```bash
# Wait before retrying
sleep 60
python src/cli.py query "your question"
```

## Performance Issues

### Slow Response Times

**Symptoms:** Queries taking >30 seconds

**Solutions:**
1. **Reduce retrieval scope**
```yaml
retrieval:
  top_k: 3  # Reduce from 5
```

2. **Use faster model**
```yaml
llm:
  model: gpt-3.5-turbo  # Faster than gpt-4o
```

3. **Simplify chunking**
```yaml
chunking:
  semantic_chunking: false
  chunk_size: 500
```

### High Memory Usage

**Symptoms:** System running out of memory

**Solutions:**
1. **Process fewer documents at once**
2. **Reduce chunk overlap**
3. **Use smaller embedding model**

```yaml
embeddings:
  model: text-embedding-3-small  # Smaller than large
```

### Large Vector Database

**Symptoms:** Slow startup or search

**Solutions:**
1. **Regular maintenance**
```bash
# Remove old/unused documents
# Reindex periodically
```

2. **Optimize chunk size**
```yaml
chunking:
  chunk_size: 1200  # Larger chunks = fewer vectors
```

## CLI Issues

### Command Not Found

**Error:** `python: command not found` or `cli.py: No such file or directory`

**Solutions:**
```bash
# Use correct Python command
python3 src/cli.py status

# Verify working directory
pwd  # Should be in project root
ls src/cli.py  # Should exist
```

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:** Run from correct directory
```bash
# Run from project root, not src directory
cd /path/to/agentic-rag-engine
python src/cli.py status
```

### Unicode Errors

**Error:** `UnicodeDecodeError: 'utf-8' codec can't decode`

**Solution:** Check document encoding
```bash
# Check file encoding
file -i data/document.txt

# Convert if needed
iconv -f latin1 -t utf-8 document.txt > document_utf8.txt
```

## Development Issues

### LSP/IDE Errors

**Symptoms:** Type hints and imports showing errors

**Solution:** Configure Python path in IDE
```json
// VS Code settings.json
{
    "python.analysis.extraPaths": ["./src"]
}
```

### Import Resolution

**Error:** Relative import errors during development

**Solution:** Add src to Python path
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

## Debug Mode

### Enable Verbose Logging

```bash
# Get detailed workflow information
python src/cli.py query "test question" --verbose
```

### Check System Status

```bash
# Comprehensive system check
python src/cli.py status
```

### Test Configuration

```python
# Python debug script
from utils.config_manager import ConfigManager
config = ConfigManager('config.yaml')
print("Configuration loaded:", config.config.model_dump())
```

## Getting Help

### Information to Include

When reporting issues, provide:
1. **Error message:** Complete error text
2. **System info:** OS, Python version
3. **Configuration:** Relevant config sections
4. **Steps to reproduce:** What you were doing
5. **Log output:** Verbose mode output if available

### Diagnostic Commands

```bash
# System information
python --version
pip list | grep -E "(openai|chromadb|click)"

# Configuration check
python src/cli.py status

# Test query with verbose output
python src/cli.py query "test" --verbose
```

### Emergency Reset

If system is completely broken:
```bash
# Full reset
rm -rf vector_db/
rm config.yaml
python src/cli.py status  # Recreates defaults
python src/cli.py ingest data/ --recursive
```