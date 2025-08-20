#!/usr/bin/env python3
"""
Unit tests for Amazon Q Client
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Add parent directory to path to import our module
sys.path.insert(0, str(Path(__file__).parent.parent))

from amazon_q_client import (
    AmazonQClient, 
    AuthType, 
    UserInputMessage, 
    ChatMessage, 
    ConversationState,
    ImageBlock,
    AWSSignatureV4
)


class TestAmazonQClient:
    """Test cases for AmazonQClient"""
    
    def test_init_builder_id_success(self):
        """Test successful initialization with Builder ID auth"""
        client = AmazonQClient(
            auth_type=AuthType.BUILDER_ID,
            bearer_token="test_token"
        )
        
        assert client.auth_type == AuthType.BUILDER_ID
        assert client.bearer_token == "test_token"
        assert client.region == "us-east-1"
        assert client.endpoint_url == "https://q.us-east-1.amazonaws.com"
        assert client.signer is None
    
    def test_init_builder_id_no_token(self):
        """Test initialization fails without bearer token for Builder ID"""
        with pytest.raises(ValueError, match="Bearer token required"):
            AmazonQClient(auth_type=AuthType.BUILDER_ID)
    
    @patch('boto3.Session')
    def test_init_sigv4_with_boto3(self, mock_session):
        """Test SigV4 initialization using boto3 credentials"""
        mock_credentials = Mock()
        mock_credentials.access_key = "test_access_key"
        mock_credentials.secret_key = "test_secret_key"
        mock_credentials.token = "test_session_token"
        
        mock_session_instance = Mock()
        mock_session_instance.get_credentials.return_value = mock_credentials
        mock_session.return_value = mock_session_instance
        
        client = AmazonQClient(auth_type=AuthType.SIGV4)
        
        assert client.auth_type == AuthType.SIGV4
        assert client.signer is not None
        assert client.signer.access_key == "test_access_key"
        assert client.signer.secret_key == "test_secret_key"
        assert client.signer.session_token == "test_session_token"
    
    def test_init_sigv4_with_explicit_credentials(self):
        """Test SigV4 initialization with explicit credentials"""
        client = AmazonQClient(
            auth_type=AuthType.SIGV4,
            aws_access_key_id="explicit_access_key",
            aws_secret_access_key="explicit_secret_key",
            aws_session_token="explicit_session_token"
        )
        
        assert client.signer.access_key == "explicit_access_key"
        assert client.signer.secret_key == "explicit_secret_key"
        assert client.signer.session_token == "explicit_session_token"
    
    def test_get_headers_builder_id(self):
        """Test header generation for Builder ID auth"""
        client = AmazonQClient(
            auth_type=AuthType.BUILDER_ID,
            bearer_token="test_token"
        )
        
        headers = client._get_headers()
        
        assert headers['Content-Type'] == 'application/x-amz-json-1.0'
        assert headers['Authorization'] == 'Bearer test_token'
        assert 'User-Agent' in headers
    
    def test_get_headers_sigv4(self):
        """Test header generation for SigV4 auth"""
        client = AmazonQClient(
            auth_type=AuthType.SIGV4,
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret"
        )
        
        headers = client._get_headers()
        
        assert headers['Content-Type'] == 'application/x-amz-json-1.0'
        assert 'Authorization' not in headers  # Added during signing
        assert 'User-Agent' in headers
    
    def test_prepare_request_builder_id(self):
        """Test request preparation for Builder ID"""
        client = AmazonQClient(
            auth_type=AuthType.BUILDER_ID,
            bearer_token="test_token"
        )
        
        payload = {"test": "data"}
        url, headers, payload_str = client._prepare_request("GenerateAssistantResponse", payload)
        
        assert url == "https://q.us-east-1.amazonaws.com/"
        assert headers['X-Amz-Target'] == 'CodeWhispererStreaming.GenerateAssistantResponse'
        assert headers['Authorization'] == 'Bearer test_token'
        assert json.loads(payload_str) == payload
    
    def test_prepare_request_sigv4(self):
        """Test request preparation for SigV4"""
        client = AmazonQClient(
            auth_type=AuthType.SIGV4,
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret"
        )
        
        payload = {"test": "data"}
        url, headers, payload_str = client._prepare_request("SendMessage", payload)
        
        assert url == "https://q.us-east-1.amazonaws.com/sendMessage"
        assert 'Authorization' in headers  # Added by signing
        assert 'x-amz-date' in headers
        assert json.loads(payload_str) == payload
    
    def test_clean_dict(self):
        """Test dictionary cleaning (removing None values)"""
        client = AmazonQClient(
            auth_type=AuthType.BUILDER_ID,
            bearer_token="test_token"
        )
        
        dirty_dict = {
            "key1": "value1",
            "key2": None,
            "key3": {
                "nested1": "value2",
                "nested2": None,
                "nested3": ["item1", None, "item2"]
            },
            "key4": [None, "item3", None]
        }
        
        clean_dict = client._clean_dict(dirty_dict)
        
        expected = {
            "key1": "value1",
            "key3": {
                "nested1": "value2",
                "nested3": ["item1", "item2"]
            },
            "key4": ["item3"]
        }
        
        assert clean_dict == expected


class TestDataClasses:
    """Test data classes"""
    
    def test_user_input_message(self):
        """Test UserInputMessage creation"""
        message = UserInputMessage(
            content="Hello, world!",
            model_id="test-model",
            origin="CLI"
        )
        
        assert message.content == "Hello, world!"
        assert message.model_id == "test-model"
        assert message.origin == "CLI"
        assert message.images is None
    
    def test_image_block(self):
        """Test ImageBlock creation"""
        image = ImageBlock(
            format="png",
            source={"bytes": "base64_encoded_data"}
        )
        
        assert image.format == "png"
        assert image.source["bytes"] == "base64_encoded_data"
    
    def test_conversation_state(self):
        """Test ConversationState creation"""
        user_message = UserInputMessage(content="Test message")
        chat_message = ChatMessage(user_input_message=user_message)
        
        conversation = ConversationState(
            current_message=chat_message,
            conversation_id="test-conv-id"
        )
        
        assert conversation.current_message == chat_message
        assert conversation.conversation_id == "test-conv-id"
        assert conversation.history is None


class TestAWSSignatureV4:
    """Test AWS Signature V4 implementation"""
    
    def test_init(self):
        """Test AWSSignatureV4 initialization"""
        signer = AWSSignatureV4("access_key", "secret_key", "session_token")
        
        assert signer.access_key == "access_key"
        assert signer.secret_key == "secret_key"
        assert signer.session_token == "session_token"
    
    def test_sign_request(self):
        """Test request signing"""
        signer = AWSSignatureV4("AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
        
        headers = {
            'Content-Type': 'application/x-amz-json-1.0'
        }
        
        signed_headers = signer.sign_request(
            'POST',
            'https://q.us-east-1.amazonaws.com/sendMessage',
            headers,
            '{"test": "payload"}',
            'q',
            'us-east-1'
        )
        
        # Check that required headers are added
        assert 'Authorization' in signed_headers
        assert 'x-amz-date' in signed_headers
        assert 'host' in signed_headers
        assert signed_headers['host'] == 'q.us-east-1.amazonaws.com'
        
        # Check authorization header format
        auth_header = signed_headers['Authorization']
        assert auth_header.startswith('AWS4-HMAC-SHA256')
        assert 'Credential=' in auth_header
        assert 'SignedHeaders=' in auth_header
        assert 'Signature=' in auth_header


class TestIntegration:
    """Integration tests (mocked)"""
    
    @pytest.mark.asyncio
    async def test_send_message_mock_response(self):
        """Test send_message with mocked HTTP response"""
        client = AmazonQClient(
            auth_type=AuthType.BUILDER_ID,
            bearer_token="test_token"
        )
        
        # Mock response data
        mock_events = [
            b'data: {"assistantResponseEvent": {"content": "Hello"}}\n',
            b'data: {"assistantResponseEvent": {"content": " world!"}}\n',
            b'data: [DONE]\n'
        ]
        
        # Create a proper async iterator for the content
        class MockContent:
            def __init__(self, events):
                self.events = iter(events)
            
            def __aiter__(self):
                return self
            
            async def __anext__(self):
                try:
                    return next(self.events)
                except StopIteration:
                    raise StopAsyncIteration
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            # Create mock objects
            mock_session = Mock()
            mock_response = Mock()
            mock_response.status = 200
            mock_response.content = MockContent(mock_events)
            
            # Create async context managers
            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cm.__aexit__ = AsyncMock(return_value=None)
            
            mock_response_cm = AsyncMock()
            mock_response_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response_cm.__aexit__ = AsyncMock(return_value=None)
            
            # Wire up the mocks
            mock_session_class.return_value = mock_session_cm
            mock_session.post.return_value = mock_response_cm
            
            events = []
            async for event in client.send_message("Test message"):
                events.append(event)
            
            assert len(events) == 2
            assert events[0]["assistantResponseEvent"]["content"] == "Hello"
            assert events[1]["assistantResponseEvent"]["content"] == " world!"
    
    @pytest.mark.asyncio
    async def test_send_simple_message_mock_response(self):
        """Test send_simple_message with mocked response"""
        client = AmazonQClient(
            auth_type=AuthType.BUILDER_ID,
            bearer_token="test_token"
        )
        
        # Mock the send_message method directly
        async def mock_send_message(*args, **kwargs):
            yield {"assistantResponseEvent": {"content": "Hello"}}
            yield {"assistantResponseEvent": {"content": " world!"}}
        
        client.send_message = mock_send_message
        
        response = await client.send_simple_message("Test message")
        assert response == "Hello world!"
    
    @pytest.mark.asyncio
    async def test_send_message_http_error(self):
        """Test send_message handles HTTP errors"""
        client = AmazonQClient(
            auth_type=AuthType.BUILDER_ID,
            bearer_token="test_token"
        )
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            # Create mock objects
            mock_session = Mock()
            mock_response = Mock()
            mock_response.status = 401
            mock_response.text = AsyncMock(return_value="Unauthorized")
            
            # Create async context managers
            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cm.__aexit__ = AsyncMock(return_value=None)
            
            mock_response_cm = AsyncMock()
            mock_response_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response_cm.__aexit__ = AsyncMock(return_value=None)
            
            # Wire up the mocks
            mock_session_class.return_value = mock_session_cm
            mock_session.post.return_value = mock_response_cm
            
            with pytest.raises(Exception, match="API request failed with status 401"):
                async for event in client.send_message("Test message"):
                    pass
