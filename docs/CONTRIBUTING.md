# Contributing to bkvy

We love your input! We want to make contributing to the bkvy as easy and transparent as possible.

## How to Contribute

### Reporting Bugs

- **Use GitHub Issues** to report bugs
- **Check existing issues** first to avoid duplicates
- **Include detailed information**:
  - Your configuration (without API keys)
  - Steps to reproduce
  - Expected vs actual behavior
  - Error messages and logs
  - System information (OS, Python version)

### Suggesting Features

- **Open a GitHub Discussion** for feature requests
- **Describe the use case** and why it would be valuable
- **Consider implementation complexity** and backwards compatibility
- **Check the roadmap** to see if it's already planned

### Contributing Code

#### Getting Started

1. **Fork the repository**
2. **Create a feature branch** from `main`
3. **Make your changes**
4. **Test thoroughly**
5. **Submit a pull request**

#### Development Setup

```bash
# Clone your fork
git clone https://github.com/NitroxHead/bkvy.git
cd bkvy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy example configurations
cp config/providers.example.json config/providers.json
cp config/routing.example.json config/routing.json

# Add test API keys (optional - for integration testing)
nano config/providers.json
```

#### Running Tests

```bash
# Basic functionality tests
python test_refactored.py

# Configuration validation
python -c "
import json
with open('config/providers.json') as f: 
    json.load(f)
print('Config is valid')
"

# API integration test
python -c "
import requests
response = requests.get('http://localhost:10006/health')
print('API is responding:', response.status_code == 200)
"
```

#### Code Style

- **Follow Python PEP 8** style guidelines
- **Use type hints** for function parameters and return values
- **Write clear docstrings** for all functions and classes
- **Keep functions focused** on a single responsibility
- **Use meaningful variable names**

**Example:**

```python
async def calculate_queue_wait_time(
    provider: str, 
    model: str, 
    api_key_id: str,
    avg_response_time_ms: int
) -> float:
    """
    Calculate estimated queue wait time for a specific combination.

    Args:
        provider: Provider name (openai, anthropic, gemini)
        model: Model name 
        api_key_id: API key identifier
        avg_response_time_ms: Average response time in milliseconds

    Returns:
        Estimated wait time in seconds
    """
    # Implementation here...
```

#### Pull Request Guidelines

- **Clear title and description** explaining the change
- **Reference related issues** using `#issue_number`
- **Include test results** showing your changes work
- **Update documentation** if needed
- **Keep changes focused** - one feature per PR
- **Ensure backwards compatibility** unless it's a breaking change

#### Commit Messages

Use clear, descriptive commit messages:

```bash
# Good
feat: Add support for OpenAI o1 reasoning models
fix: Handle rate limit errors gracefully in Anthropic client  
docs: Update API documentation with new endpoints

# Bad  
Update stuff
Fix bug
Changes
```

## Development Guidelines

### Adding New Providers

1. **Update data classes** in `bkvy/models/data_classes.py`
2. **Add provider client** logic in `bkvy/core/llm_client.py`
3. **Update configuration** schema and examples
4. **Add tests** for the new provider
5. **Update documentation**

### Adding New Features

1. **Discuss in GitHub Issues** first for major features
2. **Maintain backwards compatibility** when possible
3. **Add comprehensive tests**
4. **Update all relevant documentation**
5. **Consider configuration impact**

### Code Review Process

1. **Automated checks** run on all PRs
2. **Maintainer review** required before merge
3. **Address feedback** promptly and thoroughly
4. **Squash commits** before merge if needed

## Documentation

### Writing Documentation

- **Use clear, concise language**
- **Include practical examples**
- **Keep it up-to-date** with code changes
- **Follow existing formatting**

### Documentation Structure

- `README.md` - Project overview and quick start
- `docs/SETUP.md` - Detailed setup instructions
- `docs/API.md` - Complete API reference
- `docs/CONFIGURATION.md` - Configuration guide
- `docs/EXAMPLES.md` - Usage examples

## Testing

### Test Categories

1. **Unit Tests** - Individual component functionality
2. **Integration Tests** - End-to-end workflows
3. **Configuration Tests** - Config validation and loading
4. **API Tests** - HTTP endpoint functionality

### Test Naming

```python
def test_intelligence_routing_selects_cheapest_model():
    """Test that low intelligence requests route to cheapest available model"""

def test_rate_limit_manager_handles_rpm_exceeded():
    """Test rate limit manager properly handles RPM limits"""

def test_gemini_client_disables_thinking_for_simple_requests():
    """Test that Gemini client disables thinking for low intelligence"""
```

### Mock External APIs

When testing, mock external API calls:

```python
import asyncio
from unittest.mock import AsyncMock, patch

@patch('aiohttp.ClientSession.post')
async def test_openai_api_call(mock_post):
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Test response"}}]
    }
    mock_post.return_value.__aenter__.return_value = mock_response

    # Your test here
```

## Community

### Code of Conduct

- **Be respectful** and inclusive
- **Help newcomers** learn and contribute
- **Focus on constructive feedback**
- **Assume good intentions**

### Getting Help

- **GitHub Discussions** for questions and ideas
- **GitHub Issues** for bugs and feature requests  
- **Documentation** for setup and usage help

### Recognition

Contributors are recognized through:

- **GitHub contributor listings**
- **Release notes** for significant contributions
- **Special mentions** for exceptional help

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):

- `MAJOR.MINOR.PATCH`
- Major: Breaking changes
- Minor: New features, backwards compatible
- Patch: Bug fixes

### Release Notes

Each release includes:

- **New features** and improvements
- **Bug fixes** and performance improvements
- **Breaking changes** (if any)
- **Contributor acknowledgments**

## Security

### Reporting Security Issues

- **Email security concerns** to maintainers privately
- **Don't open public issues** for security vulnerabilities
- **Include detailed information** about the issue
- **Allow time for fixes** before public disclosure

### Security Best Practices

- **Never commit API keys** or sensitive data
- **Use environment variables** for configuration
- **Validate all inputs** thoroughly
- **Handle errors securely** without leaking information

## Questions?

Don't hesitate to ask questions:

- **Open a Discussion** for general questions
- **Create an Issue** for bugs or feature requests
- **Review existing documentation** first

Thank you for contributing to the bkvy! 🚀
