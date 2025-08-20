#!/usr/bin/env python3
"""
Amazon Q Developer Standalone Python Client

This client provides both Builder ID (Bearer token) and Enterprise (SigV4) authentication
methods to interact with Amazon Q Developer's streaming chat API.
"""

import json
import time
import uuid
import asyncio
import aiohttp
import boto3
from typing import Dict, List, Optional, AsyncGenerator, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from urllib.parse import urljoin
import base64
import hashlib
import hmac
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuthType(Enum):
    BUILDER_ID = "builder_id"
    SIGV4 = "sigv4"


class ChatTriggerType(Enum):
    MANUAL = "MANUAL"
    AUTO = "AUTO"
    
    def __str__(self):
        return self.value


@dataclass
class ImageBlock:
    """Represents an image in the chat message"""

    format: str  # "png", "jpeg", etc.
    source: Dict[str, str]  # {"bytes": base64_encoded_image}


@dataclass
class UserInputMessage:
    """User input message structure"""

    content: str
    model_id: Optional[str] = None
    images: Optional[List[ImageBlock]] = None
    user_input_message_context: Optional[Dict] = None
    user_intent: Optional[Dict] = None
    origin: Optional[str] = "CLI"
    cache_point: Optional[Dict] = None
    client_cache_config: Optional[Dict] = None


@dataclass
class ChatMessage:
    """Generic chat message wrapper"""

    user_input_message: Optional[UserInputMessage] = None
    assistant_response_message: Optional[Dict] = None


@dataclass
class ConversationState:
    """Conversation state for API requests"""

    current_message: ChatMessage
    chat_trigger_type: ChatTriggerType = ChatTriggerType.MANUAL
    conversation_id: Optional[str] = None
    workspace_id: Optional[str] = None
    history: Optional[List[ChatMessage]] = None
    customization_arn: Optional[str] = None
    agent_continuation_id: Optional[str] = None
    agent_task_type: Optional[str] = None


class AWSSignatureV4:
    """AWS Signature Version 4 signing utility"""

    def __init__(self, access_key: str, secret_key: str, session_token: Optional[str] = None):
        self.access_key = access_key
        self.secret_key = secret_key
        self.session_token = session_token

    def sign_request(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        payload: str,
        service: str,
        region: str,
    ) -> Dict[str, str]:
        """Sign an AWS request using SigV4"""
        from urllib.parse import urlparse

        parsed_url = urlparse(url)
        host = parsed_url.netloc
        path = parsed_url.path or "/"

        # Create canonical request
        canonical_headers = []
        signed_headers = []

        # Add required headers
        headers["host"] = host
        headers["x-amz-date"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        if self.session_token:
            headers["x-amz-security-token"] = self.session_token

        # Sort headers
        for key in sorted(headers.keys()):
            canonical_headers.append(f"{key.lower()}:{headers[key]}")
            signed_headers.append(key.lower())

        canonical_headers_str = "\n".join(canonical_headers)
        signed_headers_str = ";".join(signed_headers)

        # Create payload hash
        payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        # Create canonical request
        canonical_request = (
            f"{method}\n{path}\n\n{canonical_headers_str}\n\n{signed_headers_str}\n{payload_hash}"
        )

        # Create string to sign
        algorithm = "AWS4-HMAC-SHA256"
        credential_scope = f"{headers['x-amz-date'][:8]}/{region}/{service}/aws4_request"
        string_to_sign = f"{algorithm}\n{headers['x-amz-date']}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"

        # Calculate signature
        def sign(key: bytes, msg: str) -> bytes:
            return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

        k_date = sign(f"AWS4{self.secret_key}".encode("utf-8"), headers["x-amz-date"][:8])
        k_region = sign(k_date, region)
        k_service = sign(k_region, service)
        k_signing = sign(k_service, "aws4_request")

        signature = hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

        # Create authorization header
        authorization = f"{algorithm} Credential={self.access_key}/{credential_scope}, SignedHeaders={signed_headers_str}, Signature={signature}"
        headers["Authorization"] = authorization

        return headers


class AmazonQClient:
    """Amazon Q Developer Python Client"""

    def __init__(
        self,
        auth_type: AuthType = AuthType.BUILDER_ID,
        bearer_token: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region: str = "us-east-1",
        endpoint_url: Optional[str] = None,
    ):
        """
        Initialize Amazon Q Client

        Args:
            auth_type: Authentication type (BUILDER_ID or SIGV4)
            bearer_token: Bearer token for Builder ID auth
            aws_access_key_id: AWS access key for SigV4 auth
            aws_secret_access_key: AWS secret key for SigV4 auth
            aws_session_token: AWS session token for SigV4 auth (optional)
            region: AWS region
            endpoint_url: Custom endpoint URL (optional)
        """
        self.auth_type = auth_type
        self.bearer_token = bearer_token
        self.region = region

        # Set endpoint URL
        if endpoint_url:
            self.endpoint_url = endpoint_url
        else:
            self.endpoint_url = f"https://q.{region}.amazonaws.com"

        # Initialize SigV4 signer if using AWS credentials
        if auth_type == AuthType.SIGV4:
            if not aws_access_key_id or not aws_secret_access_key:
                # Try to get credentials from boto3
                session = boto3.Session()
                credentials = session.get_credentials()
                if credentials:
                    aws_access_key_id = credentials.access_key
                    aws_secret_access_key = credentials.secret_key
                    aws_session_token = credentials.token
                else:
                    raise ValueError("AWS credentials required for SigV4 authentication")

            self.signer = AWSSignatureV4(
                aws_access_key_id, aws_secret_access_key, aws_session_token
            )
        else:
            self.signer = None
            if not bearer_token:
                raise ValueError("Bearer token required for Builder ID authentication")

    def _get_headers(self) -> Dict[str, str]:
        """Get base headers for requests"""
        headers = {
            "Content-Type": "application/x-amz-json-1.0",
            "User-Agent": "Amazon-Q-Python-Client/1.0.0",
            "Accept": "application/json",
        }

        if self.auth_type == AuthType.BUILDER_ID:
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        return headers

    def _prepare_request(self, operation: str, payload: Dict) -> tuple[str, Dict[str, str], str]:
        """Prepare request URL, headers, and payload"""
        if self.auth_type == AuthType.BUILDER_ID:
            # CodeWhisperer Streaming API
            url = urljoin(self.endpoint_url, f"/")
            headers = self._get_headers()
            headers["X-Amz-Target"] = f"AmazonCodeWhispererStreamingService.{operation}"
        else:
            # Q Developer Streaming API
            url = urljoin(self.endpoint_url, f"/")
            headers = self._get_headers()
            headers["X-Amz-Target"] = f"AmazonQDeveloperStreamingService.{operation}"

        payload_str = json.dumps(payload, default=self._json_serializer)

        # Sign request if using SigV4
        if self.auth_type == AuthType.SIGV4:
            headers = self.signer.sign_request("POST", url, headers, payload_str, "q", self.region)

        return url, headers, payload_str
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for special types"""
        if isinstance(obj, Enum):
            return obj.value
        return str(obj)

    async def send_message(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        model_id: Optional[str] = None,
        images: Optional[List[ImageBlock]] = None,
        history: Optional[List[ChatMessage]] = None,
    ) -> AsyncGenerator[Dict, None]:
        """
        Send a message to Amazon Q and stream the response

        Args:
            message: The user message
            conversation_id: Optional conversation ID to continue a conversation
            model_id: Optional model ID to use
            images: Optional list of images
            history: Optional conversation history

        Yields:
            Dict: Streaming response events
        """
        # Create user input message
        user_message = UserInputMessage(
            content=message, model_id=model_id, images=images, origin="CLI"
        )

        # Create conversation state
        conversation_state = ConversationState(
            current_message=ChatMessage(user_input_message=user_message),
            conversation_id=conversation_id,
            history=history,
        )

        # Convert to dict and clean None values
        conversation_state_dict = self._clean_dict(asdict(conversation_state))
        
        # Convert to camelCase for API compatibility
        conversation_state_dict = self._convert_dict_to_camel_case(conversation_state_dict)
        
        # Wrap in the correct input structure for the API
        if self.auth_type == AuthType.BUILDER_ID:
            # For CodeWhisperer, wrap in GenerateAssistantResponseInput
            payload = {
                "conversationState": conversation_state_dict
            }
        else:
            # For Q Developer, use the conversation state directly
            payload = conversation_state_dict

        # Determine operation based on auth type
        operation = (
            "GenerateAssistantResponse" if self.auth_type == AuthType.BUILDER_ID else "SendMessage"
        )

        # Prepare request
        url, headers, payload_str = self._prepare_request(operation, payload)

        logger.info(f"Sending request to {url}")
        logger.info(f"Operation: {operation}")
        logger.info(f"Headers: {headers}")
        logger.info(f"Payload: {payload_str}")

        # Send streaming request
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=payload_str) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"API request failed with status {response.status}: {error_text}"
                    )

                # Process streaming response
                async for chunk in response.content.iter_chunked(1024):
                    if chunk:
                        try:
                            # Try to find JSON events in the binary stream
                            chunk_str = chunk.decode("utf-8", errors="ignore")
                            
                            # Look for JSON objects in the chunk
                            import re
                            json_pattern = r'\{"[^"]+":"[^"]*"\}'
                            matches = re.findall(json_pattern, chunk_str)
                            
                            for match in matches:
                                try:
                                    event_data = json.loads(match)
                                    if "content" in event_data:
                                        # Create a properly formatted event
                                        yield {
                                            "assistantResponseEvent": {
                                                "content": event_data["content"]
                                            }
                                        }
                                    elif "conversationId" in event_data:
                                        # Initial response event
                                        yield {
                                            "metadataEvent": {
                                                "conversationId": event_data["conversationId"]
                                            }
                                        }
                                except json.JSONDecodeError:
                                    continue
                                    
                        except Exception as e:
                            logger.debug(f"Error processing chunk: {e}")
                            continue
                        except Exception as e:
                            logger.error(f"Error processing line: {e}")
                            continue

    async def send_simple_message(self, message: str, **kwargs) -> str:
        """
        Send a message and return the complete response as a string

        Args:
            message: The user message
            **kwargs: Additional arguments for send_message

        Returns:
            str: Complete response text
        """
        response_parts = []

        async for event in self.send_message(message, **kwargs):
            # Extract text content from various event types
            if "assistantResponseEvent" in event:
                content = event["assistantResponseEvent"].get("content", "")
                response_parts.append(content)
            elif "codeEvent" in event:
                content = event["codeEvent"].get("content", "")
                response_parts.append(content)
            elif "textEvent" in event:
                content = event["textEvent"].get("content", "")
                response_parts.append(content)

        return "".join(response_parts)

    def _to_camel_case(self, snake_str: str) -> str:
        """Convert snake_case to camelCase"""
        components = snake_str.split('_')
        return components[0] + ''.join(word.capitalize() for word in components[1:])
    
    def _convert_dict_to_camel_case(self, data):
        """Recursively convert dictionary keys from snake_case to camelCase"""
        if isinstance(data, dict):
            return {
                self._to_camel_case(key): self._convert_dict_to_camel_case(value)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [self._convert_dict_to_camel_case(item) for item in data]
        else:
            return data

    def _clean_dict(self, d: Dict) -> Dict:
        """Remove None values from dictionary recursively"""
        if isinstance(d, dict):
            return {k: self._clean_dict(v) for k, v in d.items() if v is not None}
        elif isinstance(d, list):
            return [self._clean_dict(item) for item in d if item is not None]
        else:
            return d
