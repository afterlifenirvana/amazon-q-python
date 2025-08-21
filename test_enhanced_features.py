#!/usr/bin/env python3
"""
Test Enhanced Features

This file tests all the new enhanced capabilities of the Amazon Q Python client.
"""

import asyncio
import pytest
from amazon_q_client import (
    AmazonQClient, AuthType, UserIntent, ImageFormat, CachePointType,
    AgentTaskType, ToolResultStatus, EnvState, GitState, ShellState, 
    EditorState, CachePoint, EnvironmentVariable, ShellHistoryEntry,
    Position, Range, TextDocument, RelevantTextDocument, CursorState,
    Tool, ToolSpecification, ToolInputSchema, ToolResult, ToolResultContentBlock
)


def test_data_structures():
    """Test that all data structures can be created correctly"""
    
    # Test TokenUsage
    from amazon_q_client import TokenUsage
    token_usage = TokenUsage(
        uncached_input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        cache_read_input_tokens=10,
        cache_write_input_tokens=5
    )
    assert token_usage.total_tokens == 150
    
    # Test ImageBlock
    from amazon_q_client import ImageBlock, ImageSource
    image_block = ImageBlock(
        format=ImageFormat.PNG,
        source=ImageSource(bytes="base64encodeddata")
    )
    assert image_block.format == ImageFormat.PNG
    
    # Test CachePoint
    cache_point = CachePoint(
        type=CachePointType.PERSISTENT,
        ttl_seconds=3600
    )
    assert cache_point.type == CachePointType.PERSISTENT
    
    # Test EnvState
    env_state = EnvState(
        operating_system="macOS",
        environment_variables=[
            EnvironmentVariable(key="PYTHON_VERSION", value="3.11.5")
        ]
    )
    assert env_state.operating_system == "macOS"
    assert len(env_state.environment_variables) == 1
    
    # Test GitState
    git_state = GitState(status="On branch main")
    assert git_state.status == "On branch main"
    
    # Test ShellState
    shell_state = ShellState(
        shell_name="zsh",
        history=[ShellHistoryEntry(command="ls -la", exit_code=0)]
    )
    assert shell_state.shell_name == "zsh"
    assert len(shell_state.history) == 1
    
    # Test EditorState
    cursor_state = CursorState(position=Position(line=10, character=5))
    text_doc = RelevantTextDocument(
        text_document=TextDocument(uri="file:///test.py", language_id="python")
    )
    editor_state = EditorState(
        cursor_state=cursor_state,
        relevant_text_documents=[text_doc]
    )
    assert editor_state.cursor_state.position.line == 10
    assert len(editor_state.relevant_text_documents) == 1


def test_client_initialization():
    """Test client initialization with enhanced parameters"""
    
    # Test Builder ID client with enhanced parameters
    client = AmazonQClient(
        auth_type=AuthType.BUILDER_ID,
        bearer_token="test_token",
        max_tokens=4000,
        temperature=0.7,
        top_p=0.9,
        customization_arn="arn:aws:codewhisperer:us-east-1:123456789012:customization/test"
    )
    
    assert client.auth_type == AuthType.BUILDER_ID
    assert client.max_tokens == 4000
    assert client.temperature == 0.7
    assert client.top_p == 0.9
    assert client.customization_arn == "arn:aws:codewhisperer:us-east-1:123456789012:customization/test"
    
    # Test SigV4 client
    try:
        client_sigv4 = AmazonQClient(
            auth_type=AuthType.SIGV4,
            region="us-west-2"
        )
        assert client_sigv4.auth_type == AuthType.SIGV4
        assert client_sigv4.region == "us-west-2"
    except ValueError:
        # Expected if no AWS credentials are available
        pass


def test_tool_creation():
    """Test tool specification creation"""
    
    client = AmazonQClient(
        auth_type=AuthType.BUILDER_ID,
        bearer_token="test_token"
    )
    
    # Create a tool specification
    tool = client.create_tool_specification(
        name="calculator",
        description="Perform mathematical calculations",
        parameters={
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate"
            },
            "precision": {
                "type": "integer",
                "description": "Number of decimal places",
                "default": 2
            }
        },
        required=["expression"]
    )
    
    assert tool.tool_specification.name == "calculator"
    assert tool.tool_specification.description == "Perform mathematical calculations"
    assert "expression" in tool.tool_specification.input_schema.json["properties"]
    assert tool.tool_specification.input_schema.json["required"] == ["expression"]


def test_tool_result_creation():
    """Test tool result creation"""
    
    client = AmazonQClient(
        auth_type=AuthType.BUILDER_ID,
        bearer_token="test_token"
    )
    
    # Create tool result with text
    text_result = client.create_tool_result(
        tool_use_id="test_id_1",
        result="The calculation result is 42",
        status=ToolResultStatus.SUCCESS
    )
    
    assert text_result.tool_use_id == "test_id_1"
    assert text_result.status == ToolResultStatus.SUCCESS
    assert len(text_result.content) == 1
    assert text_result.content[0].text == "The calculation result is 42"
    
    # Create tool result with JSON
    json_result = client.create_tool_result(
        tool_use_id="test_id_2",
        result={"result": 42, "operation": "addition"},
        status=ToolResultStatus.SUCCESS
    )
    
    assert json_result.tool_use_id == "test_id_2"
    assert json_result.content[0].json == {"result": 42, "operation": "addition"}


def test_context_creation():
    """Test context creation"""
    
    client = AmazonQClient(
        auth_type=AuthType.BUILDER_ID,
        bearer_token="test_token"
    )
    
    # Create context with all state types
    env_state = EnvState(operating_system="Linux")
    git_state = GitState(status="Clean working directory")
    shell_state = ShellState(shell_name="bash")
    editor_state = EditorState()
    
    context = client.create_context(
        editor_state=editor_state,
        shell_state=shell_state,
        git_state=git_state,
        env_state=env_state
    )
    
    assert context.env_state.operating_system == "Linux"
    assert context.git_state.status == "Clean working directory"
    assert context.shell_state.shell_name == "bash"
    assert context.editor_state is not None


def test_image_block_creation():
    """Test image block creation"""
    
    client = AmazonQClient(
        auth_type=AuthType.BUILDER_ID,
        bearer_token="test_token"
    )
    
    # Create test image data (simple PNG header)
    test_image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
    
    image_block = client.create_image_block(
        image_data=test_image_data,
        image_format=ImageFormat.PNG
    )
    
    assert image_block.format == ImageFormat.PNG
    assert image_block.source.bytes is not None
    # Should be base64 encoded
    import base64
    decoded = base64.b64decode(image_block.source.bytes)
    assert decoded == test_image_data


@pytest.mark.asyncio
async def test_model_listing():
    """Test model listing functionality"""
    
    client = AmazonQClient(
        auth_type=AuthType.BUILDER_ID,
        bearer_token="test_token"
    )
    
    models = await client.list_available_models()
    
    assert len(models.models) > 0
    assert models.default_model is not None
    assert models.default_model.model_id is not None
    
    # Check that we have expected models
    model_ids = [m.model_id for m in models.models]
    assert "anthropic.claude-3-5-sonnet-20241022-v2:0" in model_ids


def test_json_serialization():
    """Test JSON serialization of enhanced types"""
    
    client = AmazonQClient(
        auth_type=AuthType.BUILDER_ID,
        bearer_token="test_token"
    )
    
    # Test enum serialization
    assert client._json_serializer(UserIntent.CODE_GENERATION) == "CODE_GENERATION"
    assert client._json_serializer(ImageFormat.PNG) == "png"
    assert client._json_serializer(CachePointType.PERSISTENT) == "PERSISTENT"
    
    # Test string serialization
    assert client._json_serializer("test_string") == "test_string"
    
    # Test number serialization
    assert client._json_serializer(42) == "42"


def test_camel_case_conversion():
    """Test snake_case to camelCase conversion"""
    
    client = AmazonQClient(
        auth_type=AuthType.BUILDER_ID,
        bearer_token="test_token"
    )
    
    # Test simple conversion
    assert client._to_camel_case("snake_case") == "snakeCase"
    assert client._to_camel_case("user_input_message") == "userInputMessage"
    assert client._to_camel_case("single") == "single"
    
    # Test dict conversion
    test_dict = {
        "user_input_message": {
            "content": "test",
            "model_id": "test_model",
            "user_intent": "CODE_GENERATION"
        },
        "conversation_id": "test_id"
    }
    
    converted = client._convert_dict_to_camel_case(test_dict)
    
    assert "userInputMessage" in converted
    assert "conversationId" in converted
    assert converted["userInputMessage"]["modelId"] == "test_model"
    assert converted["userInputMessage"]["userIntent"] == "CODE_GENERATION"


def test_dict_cleaning():
    """Test None value removal from dictionaries"""
    
    client = AmazonQClient(
        auth_type=AuthType.BUILDER_ID,
        bearer_token="test_token"
    )
    
    test_dict = {
        "valid_key": "valid_value",
        "none_key": None,
        "nested_dict": {
            "valid_nested": "value",
            "none_nested": None
        },
        "list_with_nones": [1, None, 2, None, 3]
    }
    
    cleaned = client._clean_dict(test_dict)
    
    assert "valid_key" in cleaned
    assert "none_key" not in cleaned
    assert "valid_nested" in cleaned["nested_dict"]
    assert "none_nested" not in cleaned["nested_dict"]
    assert cleaned["list_with_nones"] == [1, 2, 3]


def test_user_intents():
    """Test all user intent values"""
    
    intents = [
        UserIntent.APPLY_COMMON_BEST_PRACTICES,
        UserIntent.CITE_SOURCES,
        UserIntent.CODE_GENERATION,
        UserIntent.EXPLAIN_CODE_SELECTION,
        UserIntent.EXPLAIN_LINE_BY_LINE,
        UserIntent.GENERATE_CLOUDFORMATION_TEMPLATE,
        UserIntent.GENERATE_UNIT_TESTS,
        UserIntent.IMPROVE_CODE,
        UserIntent.SHOW_EXAMPLES,
        UserIntent.SUGGEST_ALTERNATE_IMPLEMENTATION
    ]
    
    # Ensure all intents have proper values
    for intent in intents:
        assert intent.value is not None
        assert isinstance(intent.value, str)
        assert len(intent.value) > 0


if __name__ == "__main__":
    # Run tests
    test_data_structures()
    test_client_initialization()
    test_tool_creation()
    test_tool_result_creation()
    test_context_creation()
    test_image_block_creation()
    test_json_serialization()
    test_camel_case_conversion()
    test_dict_cleaning()
    test_user_intents()
    
    # Run async tests
    asyncio.run(test_model_listing())
    
    print("✅ All enhanced feature tests passed!")
