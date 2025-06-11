# Contributing

Welcome to the Agentic RAG Workflow Engine! We appreciate your interest in contributing to this project.

## Getting Started

### Development Environment Setup

1. **Fork and Clone**
```bash
git clone https://github.com/your-username/agentic-rag-engine.git
cd agentic-rag-engine
```

2. **Create Development Environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .  # Install in development mode
```

3. **Install Development Dependencies**
```bash
pip install pytest black flake8 mypy jupyter pre-commit
```

4. **Set Up Pre-commit Hooks**
```bash
pre-commit install
```

## Development Workflow

### Branching Strategy

- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/feature-name` - Individual feature development
- `bugfix/issue-description` - Bug fixes
- `hotfix/critical-fix` - Critical production fixes

### Making Changes

1. **Create Feature Branch**
```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

2. **Make Your Changes**
- Follow coding standards (see below)
- Add tests for new functionality
- Update documentation as needed

3. **Test Your Changes**
```bash
# Run tests
pytest tests/

# Check code formatting
black --check src/
flake8 src/

# Type checking
mypy src/
```

4. **Commit Changes**
```bash
git add .
git commit -m "feat: add new feature description"
```

5. **Push and Create Pull Request**
```bash
git push origin feature/your-feature-name
# Create PR on GitHub
```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line Length**: 88 characters (Black default)
- **Imports**: Use absolute imports, group by standard/third-party/local
- **Type Hints**: Required for all public functions
- **Docstrings**: Google style for all classes and functions

### Code Formatting

**Use Black for automatic formatting:**
```bash
black src/ tests/
```

**Example of good code style:**
```python
"""
Module docstring describing the purpose.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path

import click
from rich.console import Console

from utils.logger import get_logger


class DocumentProcessor:
    """Processes documents for the RAG system.
    
    This class handles document loading, text extraction,
    and chunking for optimal retrieval performance.
    """
    
    def __init__(self, config: Config) -> None:
        """Initialize the document processor.
        
        Args:
            config: System configuration object.
        """
        self.config = config
        self.logger = get_logger(__name__)
    
    def process_document(
        self, 
        file_path: Path,
        chunk_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Process a document into chunks.
        
        Args:
            file_path: Path to the document file.
            chunk_size: Optional custom chunk size.
            
        Returns:
            List of document chunks with metadata.
            
        Raises:
            DocumentProcessingError: If processing fails.
        """
        try:
            # Implementation here
            pass
        except Exception as e:
            self.logger.error(f"Failed to process {file_path}: {e}")
            raise DocumentProcessingError(f"Processing failed: {e}")
```

### Testing Standards

**Write comprehensive tests:**
```python
"""
Test module for document processing functionality.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from processing.loader import DocumentLoader
from utils.config_manager import ConfigManager


class TestDocumentLoader:
    """Test cases for DocumentLoader class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ConfigManager('test-config.yaml').config
    
    @pytest.fixture
    def loader(self, config):
        """Create DocumentLoader instance."""
        return DocumentLoader(config)
    
    def test_load_text_document(self, loader, tmp_path):
        """Test loading a simple text document."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content for processing.")
        
        # Process document
        chunks = loader.load_document(test_file)
        
        # Assertions
        assert len(chunks) > 0
        assert chunks[0]['text'] == "Test content for processing."
        assert chunks[0]['metadata']['filename'] == "test.txt"
    
    def test_load_nonexistent_file(self, loader):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError):
            loader.load_document(Path("nonexistent.txt"))
    
    @patch('processing.loader.pdfplumber')
    def test_load_pdf_document(self, mock_pdfplumber, loader, tmp_path):
        """Test PDF document processing."""
        # Mock PDF processing
        mock_pdf = Mock()
        mock_page = Mock()
        mock_page.extract_text.return_value = "PDF content"
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        # Create mock PDF file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"mock pdf content")
        
        # Process document
        chunks = loader.load_document(test_file)
        
        # Assertions
        assert len(chunks) > 0
        assert "PDF content" in chunks[0]['text']
```

## Project Structure

### Adding New Components

**For new LLM providers:**
```python
# src/components/llms.py
from .llms import BaseLLM

class CustomLLM(BaseLLM):
    """Custom LLM implementation."""
    
    def generate_completion(self, messages, **kwargs):
        # Implementation
        pass

# Update LLMManager._initialize_llm() to include new provider
```

**For new vector stores:**
```python
# src/components/vector_stores.py
class CustomVectorStore:
    """Custom vector store implementation."""
    
    def similarity_search(self, query, k):
        # Implementation
        pass

# Update VectorStoreManager to support new store type
```

**For new document formats:**
```python
# src/processing/loader.py
def _extract_custom_format(self, file_path: Path) -> str:
    """Extract text from custom format."""
    # Implementation
    pass

# Add to DocumentLoader.load_document() format detection
```

### Documentation Updates

**Update relevant wiki pages:**
- Architecture overview for new components
- Configuration guide for new options
- API documentation for new classes/methods

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_document_processor.py

# Run with verbose output
pytest -v

# Run only failed tests
pytest --lf
```

### Test Categories

**Unit Tests**: Test individual components in isolation
```bash
pytest tests/unit/
```

**Integration Tests**: Test component interactions
```bash
pytest tests/integration/
```

**End-to-End Tests**: Test complete workflows
```bash
pytest tests/e2e/
```

### Writing Tests

**Test Structure:**
```python
def test_feature_behavior():
    """Test that feature behaves correctly under normal conditions."""
    # Arrange
    setup_test_data()
    
    # Act
    result = call_function_under_test()
    
    # Assert
    assert result.is_expected()

def test_feature_error_handling():
    """Test that feature handles errors gracefully."""
    with pytest.raises(ExpectedError):
        call_function_with_invalid_input()
```

**Use fixtures for common setup:**
```python
@pytest.fixture
def sample_document():
    """Create sample document for testing."""
    return {
        'text': 'Sample document content',
        'metadata': {'filename': 'test.txt'}
    }
```

## Documentation

### Code Documentation

**Docstring Format (Google Style):**
```python
def complex_function(
    param1: str,
    param2: Optional[int] = None,
    param3: List[str] = None
) -> Dict[str, Any]:
    """Brief description of what the function does.
    
    Longer description providing more context about the function's
    purpose, behavior, and any important implementation details.
    
    Args:
        param1: Description of first parameter.
        param2: Description of optional parameter.
        param3: Description of list parameter.
        
    Returns:
        Description of return value structure.
        
    Raises:
        ValueError: When param1 is invalid.
        ProcessingError: When processing fails.
        
    Example:
        >>> result = complex_function("test", param2=5)
        >>> print(result['status'])
        'success'
    """
```

### Wiki Updates

**For new features, update:**
1. Architecture Overview - Component descriptions
2. API Documentation - New classes and methods
3. Configuration - New configuration options
4. Quick Start - Usage examples
5. CLI Reference - New commands or options

## Pull Request Guidelines

### PR Title Format

Use conventional commit format:
- `feat: add new feature`
- `fix: resolve bug in component`
- `docs: update API documentation`
- `refactor: improve code structure`
- `test: add missing test cases`

### PR Description Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring
- [ ] Performance improvement

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass locally
- [ ] No breaking changes (or marked as such)

## Related Issues
Closes #123
References #456
```

### Review Process

**All PRs require:**
1. Passing CI checks
2. Code review from maintainer
3. Updated tests and documentation
4. No merge conflicts

**Review Criteria:**
- Code quality and style compliance
- Test coverage and quality
- Documentation completeness
- Performance impact assessment
- Breaking change evaluation

## Issue Reporting

### Bug Reports

**Include:**
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version)
- Error messages and stack traces
- Configuration details (if relevant)

**Template:**
```markdown
## Bug Description
Clear description of the bug.

## Steps to Reproduce
1. Step one
2. Step two
3. See error

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens.

## Environment
- OS: Ubuntu 20.04
- Python: 3.11.0
- Package Version: 1.0.0

## Additional Context
Any other relevant information.
```

### Feature Requests

**Include:**
- Clear description of proposed feature
- Use case and motivation
- Proposed implementation approach
- Potential impact on existing features

## Release Process

### Version Numbering

We follow Semantic Versioning (SemVer):
- `MAJOR.MINOR.PATCH`
- Major: Breaking changes
- Minor: New features (backward compatible)
- Patch: Bug fixes (backward compatible)

### Release Checklist

1. Update version numbers
2. Update CHANGELOG.md
3. Create release branch
4. Run full test suite
5. Update documentation
6. Create GitHub release
7. Deploy to package registry

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Maintain professional communication

### Getting Help

**For development questions:**
- Check existing documentation
- Search existing issues
- Ask in discussions section
- Join development chat (if available)

**For bug reports:**
- Use GitHub Issues
- Provide detailed information
- Follow issue templates

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to the Agentic RAG Workflow Engine!