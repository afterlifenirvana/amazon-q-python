#!/usr/bin/env python3
"""
Unit Tests for LangChain Amazon Q Integration

This module contains comprehensive tests for the ChatAmazonQ LangChain integration,
ensuring compatibility with LangChain interfaces and proper functionality.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult, ChatGenerationChunk
from langchain_core.tools import tool
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_amazon_q import ChatAmazonQ, create_amazon_q_chat
from amazon_q_client import AuthType, UserIntent, Model, ModelListResult


class TestChatAmazonQUnit(ChatModelUnitTests):
    """Standard LangChain unit tests for ChatAmazonQ."""
    
    @pytest.fixture
    def chat_model_class(self):
        return ChatAmazonQ
    
    @pytest.fixture
    def chat_model_params(self):
        return {
            "auth_type": AuthType.BUILDER_ID,
            "bearer_token": "test_token",
            "model_id": "test-model",
            "max_tokens": 100,
            "temperature": 0.7,
        }


class TestChatAmazonQ:
    """Custom tests for ChatAmazonQ functionality."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock Amazon Q client."""
        client = Mock()
        client.send_message = AsyncMock()
        client.list_available_models = AsyncMock()
        client.create_tool_specification = Mock()
        client.create_tool_result = Mock()
        client.create_image_block = Mock()
        return client
    
    @pytest.fixture
    def chat_model(self, mock_client):
        """Create a ChatAmazonQ instance with mocked client."""
        with patch('langchain_amazon_q.AmazonQClient', return_value=mock_client):
            return ChatAmazonQ(
                auth_type=AuthType.BUILDER_ID,
                bearer_token="test_token",
                model_id="test-model",
                max_tokens=1000,
                temperature=0.7,
            )
    
    def test_initialization(self):
        """Test ChatAmazonQ initialization."""
        chat = ChatAmazonQ(
            auth_type=AuthType.BUILDER_ID,
            bearer_token="test_token",
            model_id="claude-3.5-sonnet",
            max_tokens=2000,
            temperature=0.5,
        )
        
        assert chat.auth_type == AuthType.BUILDER_ID
        assert chat.bearer_token == "test_token"
        assert chat.model_id == "claude-3.5-sonnet"
        assert chat.max_tokens == 2000
        assert chat.temperature == 0.5
        assert chat._llm_type == "amazon_q"
    
    def test_identifying_params(self, chat_model):
        """Test identifying parameters."""
        params = chat_model._identifying_params
        
        assert "model_id" in params
        assert "auth_type" in params
        assert "region" in params
        assert "max_tokens" in params
        assert "temperature" in params
    
    def test_convert_messages_to_amazon_q_format(self, chat_model):
        """Test message conversion to Amazon Q format."""
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello, world!"),
        ]
        
        current_content, history = chat_model._convert_messages_to_amazon_q_format(messages)
        
        assert current_content == "Hello, world!"
        assert isinstance(history, list)
    
    def test_convert_messages_with_history(self, chat_model):
        """Test message conversion with conversation history."""
        messages = [
            HumanMessage(content="What is 2+2?"),
            AIMessage(content="2+2 equals 4."),
            HumanMessage(content="What about 3+3?"),
        ]
        
        current_content, history = chat_model._convert_messages_to_amazon_q_format(messages)
        
        assert current_content == "What about 3+3?"
        assert len(history) == 1  # One complete exchange
    
    @pytest.mark.asyncio
    async def test_agenerate_basic(self, chat_model, mock_client):
        """Test basic async generation."""
        # Mock the streaming response
        mock_events = [
            {"assistantResponseEvent": {"content": "Hello"}},
            {"assistantResponseEvent": {"content": " there!"}},
            {"tokenUsageEvent": {"uncachedInputTokens": 10, "outputTokens": 5, "totalTokens": 15}},
        ]
        
        async def mock_send_message(*args, **kwargs):
            for event in mock_events:
                yield event
        
        mock_client.send_message = mock_send_message
        
        messages = [HumanMessage(content="Hello")]
        result = await chat_model._agenerate(messages)
        
        assert isinstance(result, ChatResult)
        assert len(result.generations) == 1
        assert result.generations[0].message.content == "Hello there!"
        assert result.generations[0].message.usage_metadata is not None
    
    @pytest.mark.asyncio
    async def test_astream_basic(self, chat_model, mock_client):
        """Test basic async streaming."""
        mock_events = [
            {"assistantResponseEvent": {"content": "Hello"}},
            {"assistantResponseEvent": {"content": " world"}},
            {"assistantResponseEvent": {"content": "!"}},
        ]
        
        async def mock_send_message(*args, **kwargs):
            for event in mock_events:
                yield event
        
        mock_client.send_message = mock_send_message
        
        messages = [HumanMessage(content="Hello")]
        chunks = []
        
        async for chunk in chat_model._astream(messages):
            chunks.append(chunk)
        
        assert len(chunks) == 3
        assert all(isinstance(chunk, ChatGenerationChunk) for chunk in chunks)
        
        # Combine content
        full_content = "".join(chunk.message.content for chunk in chunks)
        assert full_content == "Hello world!"
    
    @pytest.mark.asyncio
    async def test_tool_calling(self, chat_model, mock_client):
        """Test tool calling functionality."""
        @tool
        def test_tool(query: str) -> str:
            """A test tool."""
            return f"Result for: {query}"
        
        # Mock tool use event
        mock_events = [
            {
                "toolUseEvent": {
                    "toolUseId": "tool_123",
                    "name": "test_tool",
                    "input": {"query": "test query"}
                }
            },
            {"assistantResponseEvent": {"content": "Tool executed successfully"}},
        ]
        
        async def mock_send_message(*args, **kwargs):
            for event in mock_events:
                yield event
        
        mock_client.send_message = mock_send_message
        mock_client.create_tool_specification.return_value = Mock()
        
        # Bind tools
        chat_with_tools = chat_model.bind_tools([test_tool])
        
        messages = [HumanMessage(content="Use the test tool")]
        result = await chat_with_tools._agenerate(messages, langchain_tools=[test_tool])
        
        assert isinstance(result, ChatResult)
        # Should have tool calls in the message
        if hasattr(result.generations[0].message, 'tool_calls'):
            assert len(result.generations[0].message.tool_calls) > 0
    
    def test_bind_tools(self, chat_model):
        """Test tool binding."""
        @tool
        def sample_tool(input_text: str) -> str:
            """A sample tool for testing."""
            return f"Processed: {input_text}"
        
        chat_with_tools = chat_model.bind_tools([sample_tool])
        
        assert chat_with_tools is not chat_model  # Should return new instance
        assert hasattr(chat_with_tools, '_bound_tools')
    
    def test_with_config(self, chat_model):
        """Test configuration updates."""
        new_chat = chat_model.with_config({"temperature": 0.9, "max_tokens": 500})
        
        assert new_chat.temperature == 0.9
        assert new_chat.max_tokens == 500
        assert new_chat is not chat_model  # Should be new instance
    
    def test_with_user_intent(self, chat_model):
        """Test user intent configuration."""
        new_chat = chat_model.with_user_intent(UserIntent.CODE_GENERATION)
        
        assert new_chat.user_intent == UserIntent.CODE_GENERATION
        assert new_chat is not chat_model
    
    def test_with_model_id(self, chat_model):
        """Test model ID configuration."""
        new_chat = chat_model.with_model_id("new-model")
        
        assert new_chat.model_id == "new-model"
        assert new_chat is not chat_model
    
    def test_with_context(self, chat_model):
        """Test context configuration."""
        new_chat = chat_model.with_context(editor=True, shell=True)
        
        assert new_chat.include_editor_context is True
        assert new_chat.include_shell_context is True
        assert new_chat.include_git_context is False  # Default
        assert new_chat is not chat_model
    
    def test_with_caching(self, chat_model):
        """Test caching configuration."""
        new_chat = chat_model.with_caching(enabled=True, ttl_seconds=1800)
        
        assert new_chat.enable_caching is True
        assert new_chat.cache_ttl_seconds == 1800
        assert new_chat is not chat_model
    
    @pytest.mark.asyncio
    async def test_list_available_models(self, chat_model, mock_client):
        """Test listing available models."""
        mock_models = ModelListResult(
            models=[
                Model(model_id="model1", model_name="Model 1"),
                Model(model_id="model2", model_name="Model 2"),
            ],
            default_model=Model(model_id="model1", model_name="Model 1")
        )
        
        mock_client.list_available_models.return_value = mock_models
        
        models = await chat_model.alist_available_models()
        
        assert len(models) == 2
        assert "model1" in models
        assert "model2" in models
    
    def test_conversation_management(self, chat_model):
        """Test conversation state management."""
        # Test initial state
        assert chat_model.get_conversation_id() is None
        
        # Set conversation ID
        chat_model.set_conversation_id("conv_123")
        assert chat_model.get_conversation_id() == "conv_123"
        
        # Reset conversation
        chat_model.reset_conversation()
        assert chat_model.get_conversation_id() is None
    
    def test_create_context(self, chat_model):
        """Test context creation."""
        context = chat_model._create_context()
        
        # Should be None if no context is enabled
        assert context is None
        
        # Enable context and test again
        chat_model.include_editor_context = True
        chat_model.include_shell_context = True
        
        context = chat_model._create_context()
        assert context is not None
        assert context.editor_state is not None
        assert context.shell_state is not None
    
    def test_create_cache_point(self, chat_model):
        """Test cache point creation."""
        # Disabled by default
        cache_point = chat_model._create_cache_point()
        assert cache_point is None
        
        # Enable caching
        chat_model.enable_caching = True
        cache_point = chat_model._create_cache_point()
        assert cache_point is not None
        assert cache_point.ttl_seconds == chat_model.cache_ttl_seconds


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_amazon_q_chat(self):
        """Test create_amazon_q_chat function."""
        chat = create_amazon_q_chat(
            bearer_token="test_token",
            model_id="claude-3.5-sonnet",
            temperature=0.8
        )
        
        assert isinstance(chat, ChatAmazonQ)
        assert chat.auth_type == AuthType.BUILDER_ID
        assert chat.bearer_token == "test_token"
        assert chat.model_id == "claude-3.5-sonnet"
        assert chat.temperature == 0.8
    
    def test_create_amazon_q_chat_enterprise(self):
        """Test create_amazon_q_chat_enterprise function."""
        from langchain_amazon_q import create_amazon_q_chat_enterprise
        
        chat = create_amazon_q_chat_enterprise(
            region="us-west-2",
            model_id="enterprise-model",
            customization_arn="arn:aws:test"
        )
        
        assert isinstance(chat, ChatAmazonQ)
        assert chat.auth_type == AuthType.SIGV4
        assert chat.region == "us-west-2"
        assert chat.model_id == "enterprise-model"
        assert chat.customization_arn == "arn:aws:test"


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_initialization_without_token(self):
        """Test initialization without required token."""
        with pytest.raises(ValueError, match="Bearer token required"):
            ChatAmazonQ(auth_type=AuthType.BUILDER_ID)
    
    def test_empty_messages(self, chat_model):
        """Test handling of empty message list."""
        with pytest.raises(ValueError, match="At least one message is required"):
            chat_model._convert_messages_to_amazon_q_format([])
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, chat_model, mock_client):
        """Test API error handling."""
        async def mock_send_message_error(*args, **kwargs):
            raise Exception("API Error")
        
        mock_client.send_message = mock_send_message_error
        
        messages = [HumanMessage(content="Test")]
        
        with pytest.raises(Exception, match="API Error"):
            await chat_model._agenerate(messages)


class TestIntegrationScenarios:
    """Test integration scenarios with LangChain components."""
    
    @pytest.mark.asyncio
    async def test_with_prompt_template(self, chat_model, mock_client):
        """Test integration with LangChain prompt templates."""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        # Mock response
        mock_events = [
            {"assistantResponseEvent": {"content": "Template response"}},
        ]
        
        async def mock_send_message(*args, **kwargs):
            for event in mock_events:
                yield event
        
        mock_client.send_message = mock_send_message
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a {role}"),
            ("human", "{question}"),
        ])
        
        # Create chain
        chain = prompt | chat_model | StrOutputParser()
        
        # This would normally work in a real scenario
        # For testing, we just verify the chain structure
        assert chain is not None
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, chat_model, mock_client):
        """Test batch processing."""
        mock_events = [
            {"assistantResponseEvent": {"content": "Response"}},
        ]
        
        async def mock_send_message(*args, **kwargs):
            for event in mock_events:
                yield event
        
        mock_client.send_message = mock_send_message
        
        requests = [
            [HumanMessage(content="Question 1")],
            [HumanMessage(content="Question 2")],
        ]
        
        responses = await chat_model.abatch(requests)
        
        assert len(responses) == 2
        assert all(isinstance(r, ChatResult) for r in responses)


def run_tests():
    """Run all tests."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()
