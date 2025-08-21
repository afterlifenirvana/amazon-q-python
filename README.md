# Amazon Q Developer Python Client - Enhanced Edition

A comprehensive Python client for interacting with Amazon Q Developer's streaming chat API. This enhanced version includes all the advanced capabilities found in the official Q CLI, including tool calling, structured outputs, context management, and more.

## 🚀 Enhanced Features

- ✅ **Dual Authentication**: Builder ID (Bearer token) and AWS SigV4 support
- ✅ **Streaming Responses**: Real-time streaming of AI responses
- ✅ **Multi-turn Conversations**: Maintain conversation history
- ✅ **Tool Calling & Function Execution**: Define and execute custom tools
- ✅ **Structured Outputs**: Support for JSON and structured data
- ✅ **Image Support**: Send images along with text messages
- ✅ **Model Selection**: Choose specific AI models with parameters
- ✅ **Context Management**: Rich context including editor, shell, git, and environment state
- ✅ **User Intent**: Specify user intentions for better responses
- ✅ **Caching**: Configure response caching for performance
- ✅ **Token Usage Tracking**: Monitor token consumption
- ✅ **Agent Tasks**: Support for different agent task types
- ✅ **Error Handling**: Comprehensive error handling and logging
- ✅ **Async/Await**: Modern Python async programming

## 🆕 New Capabilities

### Tool Calling and Function Execution
```python
# Define tools
calculator_tool = client.create_tool_specification(
    name="calculator",
    description="Perform mathematical calculations",
    parameters={
        "expression": {"type": "string", "description": "Math expression"}
    },
    required=["expression"]
)

# Automatic tool execution
async for event in client.send_message_with_tools(
    "What's 15 * 23?",
    tools=[calculator_tool],
    tool_handler=my_tool_handler
):
    # Handle streaming response with automatic tool execution
    pass
```

### Rich Context Management
```python
# Create comprehensive context
context = client.create_context(
    editor_state=EditorState(
        cursor_state=CursorState(position=Position(line=42, character=15)),
        relevant_text_documents=[text_document]
    ),
    shell_state=ShellState(
        shell_name="zsh",
        history=[ShellHistoryEntry(command="git status", exit_code=0)]
    ),
    git_state=GitState(status="On branch main, 2 files modified"),
    env_state=EnvState(operating_system="macOS")
)

response = await client.send_simple_message(
    "Help me debug this code",
    context=context,
    user_intent=UserIntent.IMPROVE_CODE
)
```

### Advanced Model Configuration
```python
client = AmazonQClient(
    auth_type=AuthType.BUILDER_ID,
    bearer_token=bearer_token,
    max_tokens=4000,
    temperature=0.7,
    top_p=0.9,
    customization_arn="arn:aws:codewhisperer:us-east-1:123456789012:customization/my-model"
)

# List available models
models = await client.list_available_models()
print(f"Available: {[m.model_id for m in models.models]}")
```

### Image Analysis
```python
# Load and send images
with open("diagram.png", "rb") as f:
    image_data = f.read()

image_block = client.create_image_block(
    image_data=image_data,
    image_format=ImageFormat.PNG
)

response = await client.send_simple_message(
    "Analyze this architecture diagram",
    images=[image_block]
)
```

### Performance Caching
```python
cache_point = CachePoint(
    type=CachePointType.PERSISTENT,
    ttl_seconds=3600  # 1 hour
)

response = await client.send_simple_message(
    "Analyze this large codebase",
    cache_point=cache_point
)
```

## 📦 Installation

```bash
# Install dependencies
uv sync --native-tls

# Or with pip
pip install aiohttp boto3
```

## 🔧 Setup

### 1. Extract Bearer Token (for Builder ID auth)

```bash
# First, make sure you're logged into the Q CLI
q login

# Extract your bearer token
uv run python extract_token.py
```

### 2. Basic Usage

```python
import asyncio
from amazon_q_client import AmazonQClient, AuthType, UserIntent

async def main():
    # Load token
    with open('bearer_token.txt', 'r') as f:
        bearer_token = f.read().strip()
    
    client = AmazonQClient(
        auth_type=AuthType.BUILDER_ID,
        bearer_token=bearer_token,
        max_tokens=4000,
        temperature=0.7
    )
    
    # Simple message
    response = await client.send_simple_message(
        "Write a Python function to calculate fibonacci numbers",
        user_intent=UserIntent.CODE_GENERATION
    )
    print(response)

asyncio.run(main())
```

## 🛠 Advanced Examples

### Tool Calling with Structured Outputs

```python
async def advanced_tool_example():
    # Define a complex tool
    file_analyzer = client.create_tool_specification(
        name="analyze_file",
        description="Analyze code files for issues",
        parameters={
            "file_path": {"type": "string"},
            "analysis_type": {
                "type": "string",
                "enum": ["syntax", "performance", "security"]
            },
            "include_suggestions": {"type": "boolean", "default": True}
        },
        required=["file_path", "analysis_type"]
    )
    
    async def tool_handler(tool_name: str, tool_input: dict):
        if tool_name == "analyze_file":
            # Return structured data
            return {
                "file": tool_input["file_path"],
                "issues_found": 3,
                "severity": "medium",
                "suggestions": [
                    "Add type hints",
                    "Optimize loop performance",
                    "Add error handling"
                ]
            }
    
    async for event in client.send_message_with_tools(
        "Analyze my Python file for performance issues",
        tools=[file_analyzer],
        tool_handler=tool_handler
    ):
        # Process streaming response
        pass
```

### Multi-Modal Conversation

```python
async def multimodal_example():
    # Load image
    with open("code_screenshot.png", "rb") as f:
        image_data = f.read()
    
    image_block = client.create_image_block(image_data, ImageFormat.PNG)
    
    # Create rich context
    context = client.create_context(
        editor_state=EditorState(
            cursor_state=CursorState(position=Position(line=25, character=10))
        ),
        shell_state=ShellState(
            shell_name="bash",
            history=[ShellHistoryEntry(command="python main.py", exit_code=1)]
        )
    )
    
    response = await client.send_simple_message(
        "I'm getting an error at this line. Can you help debug it?",
        images=[image_block],
        context=context,
        user_intent=UserIntent.EXPLAIN_CODE_SELECTION,
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0"
    )
    
    print(response)
```

### Enterprise Authentication

```python
# Enterprise with SigV4
client = AmazonQClient(
    auth_type=AuthType.SIGV4,
    region="us-east-1",
    customization_arn="arn:aws:codewhisperer:us-east-1:123456789012:customization/enterprise-model"
)

response = await client.send_simple_message(
    "Generate enterprise-compliant code",
    max_tokens=2000,
    temperature=0.3
)
```

## 📊 Event Types

The enhanced client supports all event types from the streaming API:

- `assistantResponseEvent`: Text responses from the assistant
- `toolUseEvent`: Tool execution requests
- `codeEvent`: Code blocks with syntax highlighting
- `citationEvent`: Source citations and references
- `metadataEvent`: Conversation metadata (ID, etc.)
- `tokenUsageEvent`: Token consumption information
- `followupPromptEvent`: Suggested follow-up questions
- `invalidStateEvent`: Error states and recovery

## 🧪 Testing

```bash
# Run basic functionality tests
uv run python test_runner.py

# Run enhanced examples
uv run python enhanced_examples.py

# Run full test suite
uv run pytest -v

# Run integration tests with real API
uv run python simple_test.py
```

## 📁 Project Structure

```
amazon-q-python-client/
├── amazon_q_client.py          # Enhanced client implementation
├── enhanced_examples.py        # Comprehensive examples
├── extract_token.py           # Token extraction utility
├── test_runner.py             # Basic test runner
├── simple_test.py             # Integration tests
├── tests/                     # Unit tests
│   └── test_amazon_q_client.py
├── pyproject.toml             # Project configuration
└── README.md                  # This file
```

## 🔐 Authentication Methods

### Builder ID (Personal Use)
1. **Login**: `q login`
2. **Extract token**: `uv run python extract_token.py`
3. **Use in code**: Load from `bearer_token.txt`

### AWS SigV4 (Enterprise)
1. **Set up AWS credentials** (environment, profile, or IAM role)
2. **Use in code**: `AuthType.SIGV4` with region

## 🎯 User Intents

Specify user intentions for better responses:

- `CODE_GENERATION`: Generate new code
- `EXPLAIN_CODE_SELECTION`: Explain selected code
- `IMPROVE_CODE`: Optimize or refactor code
- `GENERATE_UNIT_TESTS`: Create test cases
- `APPLY_COMMON_BEST_PRACTICES`: Apply coding standards
- `CITE_SOURCES`: Include source citations
- `SHOW_EXAMPLES`: Provide usage examples

## 🚨 Error Handling

The client includes comprehensive error handling:

```python
try:
    response = await client.send_simple_message("Hello")
except Exception as e:
    if "401" in str(e):
        print("Authentication failed - check your token")
    elif "429" in str(e):
        print("Rate limit exceeded - please wait")
    elif "context window" in str(e).lower():
        print("Message too long - try shorter input")
    else:
        print(f"Unexpected error: {e}")
```

## 🔧 Configuration Options

```python
client = AmazonQClient(
    auth_type=AuthType.BUILDER_ID,
    bearer_token=token,
    region="us-east-1",
    endpoint_url="https://custom-endpoint.com",  # Custom endpoint
    max_tokens=4000,                             # Response length limit
    temperature=0.7,                             # Creativity (0.0-1.0)
    top_p=0.9,                                  # Nucleus sampling
    customization_arn="arn:aws:...",            # Custom model
)
```

## 📈 Performance Tips

1. **Use caching** for expensive operations
2. **Specify model IDs** for consistent performance
3. **Provide rich context** for better responses
4. **Use appropriate user intents** for task-specific optimization
5. **Monitor token usage** to optimize costs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite: `uv run pytest`
5. Submit a pull request

## 📄 License

This project follows the same license as the main Amazon Q CLI repository (MIT OR Apache-2.0).

## 🆘 Troubleshooting

### Common Issues

1. **"Bearer token required" error**:
   ```bash
   uv run python extract_token.py
   ```

2. **"AWS credentials required" error**:
   ```bash
   aws configure
   # or set environment variables
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   ```

3. **"API request failed with status 401"**:
   - Token expired: `q login` and re-extract
   - Check Q Developer access permissions

4. **Certificate errors**:
   ```bash
   uv sync --native-tls
   ```

5. **Tool execution errors**:
   - Ensure tool handler functions are async
   - Validate tool input schemas
   - Handle exceptions in tool handlers

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔗 Related Resources

- [Amazon Q Developer Documentation](https://docs.aws.amazon.com/amazonq/)
- [Q CLI Repository](https://github.com/aws/amazon-q-developer-cli)
- [AWS SDK for Python](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
