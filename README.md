# Amazon Q Developer Python Client

A standalone Python client for interacting with Amazon Q Developer's streaming chat API. This client supports both Builder ID (personal) and AWS SigV4 (enterprise) authentication methods.

## Features

- ✅ **Dual Authentication**: Builder ID (Bearer token) and AWS SigV4 support
- ✅ **Streaming Responses**: Real-time streaming of AI responses
- ✅ **Multi-turn Conversations**: Maintain conversation history
- ✅ **Image Support**: Send images along with text messages
- ✅ **Model Selection**: Choose specific AI models
- ✅ **Error Handling**: Comprehensive error handling and logging
- ✅ **Async/Await**: Modern Python async programming

## Quick Start

### 1. Install Dependencies

```bash
uv sync --native-tls
```

### 2. Extract Bearer Token (for Builder ID auth)

```bash
# First, make sure you're logged into the Q CLI
q login

# Extract your bearer token (reads from auth_kv table in CLI database)
uv run python extract_token.py
```

### 3. Run Tests

```bash
# Run basic functionality tests
uv run python test_runner.py

# Run full test suite
uv run pytest

# Run integration tests with real API (requires token)
uv run python simple_test.py
```

## Usage Examples

### Simple Chat

```python
import asyncio
from amazon_q_client import AmazonQClient, AuthType

async def main():
    # Load token from file
    with open('bearer_token.txt', 'r') as f:
        bearer_token = f.read().strip()
    
    client = AmazonQClient(
        auth_type=AuthType.BUILDER_ID,
        bearer_token=bearer_token
    )
    
    # Simple message
    response = await client.send_simple_message("Hello, Amazon Q!")
    print(response)

asyncio.run(main())
```

### Streaming Chat

```python
async def streaming_example():
    client = AmazonQClient(
        auth_type=AuthType.BUILDER_ID,
        bearer_token=bearer_token
    )
    
    print("Q: Write a Python function")
    print("A: ", end="", flush=True)
    
    async for event in client.send_message("Write a Python function"):
        if 'assistantResponseEvent' in event:
            content = event['assistantResponseEvent'].get('content', '')
            print(content, end="", flush=True)
    
    print()  # New line
```

### Enterprise Authentication (SigV4)

```python
# Uses AWS credentials from environment/profile
client = AmazonQClient(
    auth_type=AuthType.SIGV4,
    region="us-east-1"
)

response = await client.send_simple_message("Hello from enterprise!")
```

## Project Structure

```
amazon-q-python-client/
├── amazon_q_client.py      # Main client implementation
├── extract_token.py        # Token extraction utility
├── test_runner.py          # Basic test runner
├── simple_test.py          # Integration tests
├── tests/                  # Unit tests
│   └── test_amazon_q_client.py
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## Authentication

### Builder ID (Personal Use)

1. **Login to Q CLI**: `q login`
2. **Extract token**: `uv run python extract_token.py`
3. **Use in code**:
   ```python
   with open('bearer_token.txt', 'r') as f:
       bearer_token = f.read().strip()
   
   client = AmazonQClient(
       auth_type=AuthType.BUILDER_ID,
       bearer_token=bearer_token
   )
   ```

### AWS SigV4 (Enterprise)

1. **Set up AWS credentials** (environment variables, profile, or IAM role)
2. **Use in code**:
   ```python
   client = AmazonQClient(
       auth_type=AuthType.SIGV4,
       region="us-east-1"
   )
   ```

## Testing

The project includes comprehensive tests:

- **Unit Tests**: Test individual components and functions
- **Integration Tests**: Test API interactions with mocked responses
- **Real API Tests**: Test against actual Amazon Q API (requires authentication)

### Run Tests

```bash
# Basic functionality tests
uv run python test_runner.py

# Full unit test suite
uv run pytest -v

# Integration tests with real API
uv run python simple_test.py
```

## Development

### Setup Development Environment

```bash
# Install with dev dependencies
uv sync --native-tls --group dev

# Run linting
uv run black amazon_q_client.py
uv run ruff check amazon_q_client.py

# Run type checking
uv run mypy amazon_q_client.py
```

### Project Configuration

The project uses modern Python tooling:

- **uv**: Fast Python package manager
- **pytest**: Testing framework with async support
- **black**: Code formatting
- **ruff**: Fast Python linter
- **mypy**: Static type checking

## Architecture

### How It Works

1. **Authentication Layer**:
   - Builder ID: Uses OAuth 2.0 Bearer tokens
   - SigV4: Uses AWS request signing with access keys

2. **API Endpoints**:
   - Builder ID: `https://q.{region}.amazonaws.com/` (CodeWhisperer Streaming API)
   - SigV4: `https://q.{region}.amazonaws.com/sendMessage` (Q Developer API)

3. **Request Format**:
   - Content-Type: `application/x-amz-json-1.0`
   - Streaming responses with Server-Sent Events format

4. **Event Types**:
   - `assistantResponseEvent`: Text responses
   - `codeEvent`: Code snippets
   - `citationEvent`: Source citations
   - `metadataEvent`: Response metadata

## Troubleshooting

### Common Issues

1. **"Bearer token required" error**:
   - Run `uv run python extract_token.py` to get your token
   - Make sure you're logged in: `q login`

2. **"AWS credentials required" error**:
   - Set up AWS credentials (see Authentication section)
   - Verify with: `aws sts get-caller-identity`

3. **"API request failed with status 401"**:
   - Token expired - re-run `q login` and extract new token
   - Check if you have access to Amazon Q Developer

4. **Certificate errors with uv**:
   - Use `--native-tls` flag: `uv sync --native-tls`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite: `uv run pytest`
5. Submit a pull request

## License

This project follows the same license as the main Amazon Q CLI repository (MIT OR Apache-2.0).
