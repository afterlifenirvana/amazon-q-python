#!/usr/bin/env python3
"""
Amazon Q Developer Standalone Python Client

This client provides both Builder ID (Bearer token) and Enterprise (SigV4) authentication
methods to interact with Amazon Q Developer's streaming chat API.

Enhanced Features:
- Model selection and management
- Token usage tracking
- Context management (editor, shell, git, environment)
- Tool calling and structured outputs
- User intent specification
- Cache configuration
- Customization ARN support
- Agent task management
- Comprehensive error handling
"""

import json
import time
import uuid
import asyncio
import aiohttp
import boto3
from typing import Dict, List, Optional, AsyncGenerator, Union, Any
from dataclasses import dataclass, asdict, field
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


class UserIntent(Enum):
    """User intent for the conversation"""
    APPLY_COMMON_BEST_PRACTICES = "APPLY_COMMON_BEST_PRACTICES"
    CITE_SOURCES = "CITE_SOURCES"
    CODE_GENERATION = "CODE_GENERATION"
    EXPLAIN_CODE_SELECTION = "EXPLAIN_CODE_SELECTION"
    EXPLAIN_LINE_BY_LINE = "EXPLAIN_LINE_BY_LINE"
    GENERATE_CLOUDFORMATION_TEMPLATE = "GENERATE_CLOUDFORMATION_TEMPLATE"
    GENERATE_UNIT_TESTS = "GENERATE_UNIT_TESTS"
    IMPROVE_CODE = "IMPROVE_CODE"
    SHOW_EXAMPLES = "SHOW_EXAMPLES"
    SUGGEST_ALTERNATE_IMPLEMENTATION = "SUGGEST_ALTERNATE_IMPLEMENTATION"


class ImageFormat(Enum):
    """Supported image formats"""
    PNG = "png"
    JPEG = "jpeg"
    GIF = "gif"
    WEBP = "webp"


class CachePointType(Enum):
    """Cache point types"""
    EPHEMERAL = "EPHEMERAL"
    PERSISTENT = "PERSISTENT"


class AgentTaskType(Enum):
    """Agent task types"""
    ANSWER_GENERATION = "ANSWER_GENERATION"
    CODE_GENERATION = "CODE_GENERATION"
    UNIT_TEST_GENERATION = "UNIT_TEST_GENERATION"


class ToolResultStatus(Enum):
    """Tool result status"""
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"


@dataclass
class TokenUsage:
    """Token usage information"""
    uncached_input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_read_input_tokens: Optional[int] = None
    cache_write_input_tokens: Optional[int] = None


@dataclass
class ImageSource:
    """Image source data"""
    bytes: str  # base64 encoded image data


@dataclass
class ImageBlock:
    """Represents an image in the chat message"""
    format: ImageFormat
    source: ImageSource


@dataclass
class CachePoint:
    """Cache point configuration"""
    type: CachePointType
    ttl_seconds: Optional[int] = None


@dataclass
class ClientCacheConfig:
    """Client cache configuration"""
    max_cache_size: Optional[int] = None
    ttl_seconds: Optional[int] = None


@dataclass
class EnvironmentVariable:
    """Environment variable"""
    key: str
    value: str


@dataclass
class EnvState:
    """Environment state context"""
    operating_system: Optional[str] = None
    environment_variables: Optional[List[EnvironmentVariable]] = None


@dataclass
class GitState:
    """Git state context"""
    status: Optional[str] = None


@dataclass
class ShellHistoryEntry:
    """Shell history entry"""
    command: str
    exit_code: Optional[int] = None


@dataclass
class ShellState:
    """Shell state context"""
    shell_name: Optional[str] = None
    history: Optional[List[ShellHistoryEntry]] = None


@dataclass
class Position:
    """Position in a document"""
    line: int
    character: int


@dataclass
class Range:
    """Range in a document"""
    start: Position
    end: Position


@dataclass
class TextDocument:
    """Text document information"""
    uri: str
    language_id: Optional[str] = None
    version: Optional[int] = None
    text: Optional[str] = None


@dataclass
class RelevantTextDocument:
    """Relevant text document with selection"""
    text_document: TextDocument
    selection_range: Optional[Range] = None


@dataclass
class CursorState:
    """Cursor state in editor"""
    position: Position
    selection: Optional[Range] = None


@dataclass
class EditorState:
    """Editor state context"""
    cursor_state: Optional[CursorState] = None
    relevant_text_documents: Optional[List[RelevantTextDocument]] = None


@dataclass
class ToolInputSchema:
    """Tool input schema definition"""
    json: Dict[str, Any]


@dataclass
class ToolSpecification:
    """Tool specification"""
    name: str
    description: str
    input_schema: ToolInputSchema


@dataclass
class Tool:
    """Tool definition"""
    tool_specification: ToolSpecification


@dataclass
class ToolUse:
    """Tool use request"""
    tool_use_id: str
    name: str
    input: Dict[str, Any]


@dataclass
class ToolResultContentBlock:
    """Tool result content block"""
    text: Optional[str] = None
    json: Optional[Dict[str, Any]] = None


@dataclass
class ToolResult:
    """Tool execution result"""
    tool_use_id: str
    content: List[ToolResultContentBlock]
    status: Optional[ToolResultStatus] = None


@dataclass
class UserInputMessageContext:
    """Additional context for user input messages"""
    editor_state: Optional[EditorState] = None
    shell_state: Optional[ShellState] = None
    git_state: Optional[GitState] = None
    env_state: Optional[EnvState] = None
    tools: Optional[List[Tool]] = None
    tool_results: Optional[List[ToolResult]] = None
    additional_context: Optional[List[Dict]] = None


@dataclass
class UserInputMessage:
    """User input message structure"""
    content: str
    model_id: Optional[str] = None
    images: Optional[List[ImageBlock]] = None
    user_input_message_context: Optional[UserInputMessageContext] = None
    user_intent: Optional[UserIntent] = None
    origin: Optional[str] = "CLI"
    cache_point: Optional[CachePoint] = None
    client_cache_config: Optional[ClientCacheConfig] = None


@dataclass
class AssistantResponseMessage:
    """Assistant response message structure"""
    content: str
    model_id: Optional[str] = None
    token_usage: Optional[TokenUsage] = None


@dataclass
class ChatMessage:
    """Generic chat message wrapper"""
    user_input_message: Optional[UserInputMessage] = None
    assistant_response_message: Optional[AssistantResponseMessage] = None


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
    agent_task_type: Optional[AgentTaskType] = None


@dataclass
class Model:
    """Model information"""
    model_id: str
    model_name: Optional[str] = None
    description: Optional[str] = None
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None


@dataclass
class ModelListResult:
    """Result from listing available models"""
    models: List[Model]
    default_model: Model


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


@dataclass
class ImageBlock:
    """Represents an image in the chat message"""

    format: str  # "png", "jpeg", etc.
    source: Dict[str, str]  # {"bytes": base64_encoded_image}


class AmazonQClient:
    """Amazon Q Developer Python Client with Enhanced Capabilities"""

    def __init__(
        self,
        auth_type: AuthType = AuthType.BUILDER_ID,
        bearer_token: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region: str = "us-east-1",
        endpoint_url: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        customization_arn: Optional[str] = None,
    ):
        """
        Initialize Amazon Q Client with enhanced capabilities

        Args:
            auth_type: Authentication type (BUILDER_ID or SIGV4)
            bearer_token: Bearer token for Builder ID auth
            aws_access_key_id: AWS access key for SigV4 auth
            aws_secret_access_key: AWS secret key for SigV4 auth
            aws_session_token: AWS session token for SigV4 auth (optional)
            region: AWS region
            endpoint_url: Custom endpoint URL (optional)
            max_tokens: Maximum tokens for response generation
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Top-p sampling parameter
            customization_arn: ARN for model customization
        """
        self.auth_type = auth_type
        self.bearer_token = bearer_token
        self.region = region
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.customization_arn = customization_arn

        # Initialize model cache (like Rust client)
        self._model_cache = None
        self._cache_timestamp = None
        self._cache_ttl = 300  # 5 minutes cache TTL

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
            "User-Agent": "Amazon-Q-Python-Client/2.0.0",
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

    def create_tool_specification(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        required: Optional[List[str]] = None
    ) -> Tool:
        """
        Create a tool specification for function calling
        
        Args:
            name: Tool name
            description: Tool description
            parameters: JSON schema for tool parameters
            required: List of required parameter names
            
        Returns:
            Tool specification
        """
        schema = {
            "type": "object",
            "properties": parameters,
        }
        if required:
            schema["required"] = required
            
        return Tool(
            tool_specification=ToolSpecification(
                name=name,
                description=description,
                input_schema=ToolInputSchema(json=schema)
            )
        )

    def create_tool_result(
        self,
        tool_use_id: str,
        result: Union[str, Dict[str, Any]],
        status: ToolResultStatus = ToolResultStatus.SUCCESS
    ) -> ToolResult:
        """
        Create a tool result from tool execution
        
        Args:
            tool_use_id: ID of the tool use request
            result: Tool execution result (text or JSON)
            status: Execution status
            
        Returns:
            Tool result
        """
        if isinstance(result, str):
            content = [ToolResultContentBlock(text=result)]
        else:
            content = [ToolResultContentBlock(json=result)]
            
        return ToolResult(
            tool_use_id=tool_use_id,
            content=content,
            status=status
        )

    async def list_available_models(self) -> ModelListResult:
        """
        List available models using real API call (matching Rust client implementation)
        
        Returns:
            Available models and default model from API or fallback
        """
        return await self.list_available_models_cached()
    
    async def list_available_models_cached(self) -> ModelListResult:
        """
        Get available models with caching (exactly like Rust client)
        """
        # Check cache
        if (hasattr(self, '_model_cache') and self._model_cache is not None and 
            hasattr(self, '_cache_timestamp') and self._cache_timestamp is not None and 
            time.time() - self._cache_timestamp < getattr(self, '_cache_ttl', 300)):
            return self._model_cache
        
        # Cache miss, fetch from API
        try:
            result = await self._list_available_models_api()
            self._model_cache = result
            self._cache_timestamp = time.time()
            return result
        except Exception as e:
            # Fallback to hardcoded models like Rust client does
            logger.warning(f"Failed to fetch models from API: {e}, using fallback list")
            return self._get_fallback_models()
    
    async def _list_available_models_api(self) -> ModelListResult:
        """
        Call the real list_available_models API exactly like Rust client
        """
        # Build query parameters and JSON body
        query_params = "origin=CLI"
        body_data = {"origin": "CLI"}
        
        # Prepare headers
        headers = {
            "Content-Type": "application/x-amz-json-1.0",
            "User-Agent": "Amazon-Q-Python-Client/2.0.0",
            "Accept": "application/json",
            "X-Amz-Target": "AmazonCodeWhispererService.ListAvailableModels"
        }
        
        if self.auth_type == AuthType.BUILDER_ID:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        
        url = f"{self.endpoint_url}/?{query_params}"
        body = json.dumps(body_data)
        
        # Make the API call
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=body) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"list_available_models API failed with status {response.status}: {error_text}")
                
                response_text = await response.text()
                response_data = json.loads(response_text)
                
                # Parse models
                models = []
                for model_data in response_data.get("models", []):
                    max_input_tokens = 200000  # Default
                    max_output_tokens = 8192   # Default
                    
                    if "tokenLimits" in model_data:
                        max_input_tokens = model_data["tokenLimits"].get("maxInputTokens", 200000)
                        max_output_tokens = model_data["tokenLimits"].get("maxOutputTokens", 8192)
                    
                    model = Model(
                        model_id=model_data["modelId"],
                        model_name=model_data.get("modelName", model_data["modelId"]),
                        description=model_data.get("description"),
                        max_input_tokens=max_input_tokens,
                        max_output_tokens=max_output_tokens
                    )
                    models.append(model)
                
                # Parse default model
                default_model_data = response_data["defaultModel"]
                default_max_input = 200000
                default_max_output = 8192
                
                if "tokenLimits" in default_model_data:
                    default_max_input = default_model_data["tokenLimits"].get("maxInputTokens", 200000)
                    default_max_output = default_model_data["tokenLimits"].get("maxOutputTokens", 8192)
                
                default_model = Model(
                    model_id=default_model_data["modelId"],
                    model_name=default_model_data.get("modelName", default_model_data["modelId"]),
                    description=default_model_data.get("description"),
                    max_input_tokens=default_max_input,
                    max_output_tokens=default_max_output
                )
                
                return ModelListResult(models=models, default_model=default_model)
    
    def _get_fallback_models(self) -> ModelListResult:
        """
        Fallback models when API fails (matching Rust client + real API models)
        """
        fallback_models = [
            # Real API models (primary)
            Model(
                model_id="claude-sonnet-4",
                model_name="claude-sonnet-4",
                description="Anthropic Claude Sonnet 4 - May 2025 release",
                max_input_tokens=200000,
                max_output_tokens=8192
            ),
            Model(
                model_id="claude-3.7-sonnet",
                model_name="claude-3.7-sonnet", 
                description="Anthropic Claude 3.7 Sonnet - February 2025 release",
                max_input_tokens=200000,
                max_output_tokens=8192
            ),
            Model(
                model_id="claude-3.5-sonnet",
                model_name="claude-3.5-sonnet",
                description="Anthropic Claude 3.5v2 Sonnet - October 2024 release", 
                max_input_tokens=200000,
                max_output_tokens=8192
            ),
            # Extended models (that work but not in API)
            Model(
                model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
                model_name="Claude 3.5 Sonnet v2 (Full ARN)",
                description="Claude 3.5 Sonnet with full ARN identifier",
                max_input_tokens=200000,
                max_output_tokens=8192
            ),
            Model(
                model_id="amazon.nova-pro-v1:0",
                model_name="Amazon Nova Pro",
                description="Amazon's flagship model",
                max_input_tokens=300000,
                max_output_tokens=5000
            )
        ]
        
        default_model = fallback_models[0]  # claude-sonnet-4 as default (matches API)
        
        return ModelListResult(models=fallback_models, default_model=default_model)
    
    def invalidate_model_cache(self):
        """Invalidate the model cache (like Rust client)"""
        if hasattr(self, '_model_cache'):
            self._model_cache = None
        if hasattr(self, '_cache_timestamp'):
            self._cache_timestamp = None
        logger.info("Model cache invalidated")

    async def send_message(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        model_id: Optional[str] = None,
        images: Optional[List[ImageBlock]] = None,
        history: Optional[List[ChatMessage]] = None,
        tools: Optional[List[Tool]] = None,
        tool_results: Optional[List[ToolResult]] = None,
        user_intent: Optional[UserIntent] = None,
        context: Optional[UserInputMessageContext] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        cache_point: Optional[CachePoint] = None,
        workspace_id: Optional[str] = None,
        agent_task_type: Optional[AgentTaskType] = None,
    ) -> AsyncGenerator[Dict, None]:
        """
        Send a message to Amazon Q and stream the response with enhanced capabilities

        Args:
            message: The user message
            conversation_id: Optional conversation ID to continue a conversation
            model_id: Optional model ID to use
            images: Optional list of images
            history: Optional conversation history
            tools: Optional list of available tools for function calling
            tool_results: Optional list of tool execution results
            user_intent: Optional user intent specification
            context: Optional additional context (editor, shell, git, env state)
            max_tokens: Maximum tokens for response (overrides client default)
            temperature: Sampling temperature (overrides client default)
            top_p: Top-p sampling (overrides client default)
            cache_point: Cache configuration
            workspace_id: Workspace identifier
            agent_task_type: Type of agent task

        Yields:
            Dict: Streaming response events
        """
        # Use provided parameters or fall back to client defaults
        effective_max_tokens = max_tokens or self.max_tokens
        effective_temperature = temperature or self.temperature
        effective_top_p = top_p or self.top_p

        # Create enhanced user input message context
        if context is None:
            context = UserInputMessageContext()
        
        # Add tools and tool results to context
        if tools:
            context.tools = tools
        if tool_results:
            context.tool_results = tool_results

        # Create user input message with all enhanced features
        user_message = UserInputMessage(
            content=message,
            model_id=model_id,
            images=images,
            user_input_message_context=context,
            user_intent=user_intent,
            origin="CLI",
            cache_point=cache_point,
            client_cache_config=ClientCacheConfig() if cache_point else None
        )

        # Create conversation state
        conversation_state = ConversationState(
            current_message=ChatMessage(user_input_message=user_message),
            conversation_id=conversation_id,
            workspace_id=workspace_id,
            history=history,
            customization_arn=self.customization_arn,
            agent_task_type=agent_task_type,
        )

        # Convert to dict and clean None values
        conversation_state_dict = self._clean_dict(asdict(conversation_state))
        
        # Convert to camelCase for API compatibility
        conversation_state_dict = self._convert_dict_to_camel_case(conversation_state_dict)
        
        # Add inference parameters if specified
        if effective_max_tokens or effective_temperature or effective_top_p:
            inference_config = {}
            if effective_max_tokens:
                inference_config["maxTokens"] = effective_max_tokens
            if effective_temperature:
                inference_config["temperature"] = effective_temperature
            if effective_top_p:
                inference_config["topP"] = effective_top_p
            conversation_state_dict["inferenceConfiguration"] = inference_config
        
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
        logger.debug(f"Headers: {headers}")
        logger.debug(f"Payload: {payload_str}")

        # Send streaming request
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=payload_str) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"API request failed with status {response.status}: {error_text}"
                    )

                # Process streaming response (AWS EventStream format)
                buffer = b""
                async for chunk in response.content.iter_chunked(1024):
                    if chunk:
                        buffer += chunk
                        
                        # Process complete messages from buffer
                        while True:
                            events, remaining_buffer = self._parse_eventstream_buffer(buffer)
                            if not events:
                                break
                            
                            buffer = remaining_buffer
                            for event in events:
                                yield event

    def _parse_eventstream_buffer(self, buffer: bytes) -> tuple[List[Dict], bytes]:
        """
        Parse AWS EventStream buffer and return events + remaining buffer.
        
        Simplified approach: look for JSON patterns in the binary data.
        """
        events = []
        
        try:
            # Convert buffer to string, ignoring binary parts
            buffer_str = buffer.decode('utf-8', errors='ignore')
            
            # Look for JSON patterns
            import re
            
            # Find conversation ID
            conv_match = re.search(r'\{"conversationId":"([^"]*)"[^}]*\}', buffer_str)
            if conv_match:
                conv_id = conv_match.group(1)
                events.append({
                    'metadataEvent': {
                        'conversationId': conv_id
                    }
                })
            
            # Find assistant response content
            content_matches = re.findall(r'\{"content":"([^"]*)"[^}]*\}', buffer_str)
            for content in content_matches:
                if content:  # Skip empty content
                    events.append({
                        'assistantResponseEvent': {
                            'content': content
                        }
                    })
            
            # Find tool use events
            tool_matches = re.findall(r'\{"toolUseId":"([^"]*)"[^}]*"name":"([^"]*)"[^}]*"input":(\{[^}]*\})[^}]*\}', buffer_str)
            for tool_use_id, name, input_str in tool_matches:
                try:
                    input_json = json.loads(input_str)
                    events.append({
                        'toolUseEvent': {
                            'toolUseId': tool_use_id,
                            'name': name,
                            'input': input_json
                        }
                    })
                except json.JSONDecodeError:
                    pass
            
            # Find token usage events
            token_matches = re.findall(r'\{"uncachedInputTokens":(\d+),"outputTokens":(\d+),"totalTokens":(\d+)[^}]*\}', buffer_str)
            for uncached, output, total in token_matches:
                events.append({
                    'tokenUsageEvent': {
                        'uncachedInputTokens': int(uncached),
                        'outputTokens': int(output),
                        'totalTokens': int(total)
                    }
                })
            
        except Exception as e:
            logger.debug(f"Error parsing EventStream buffer: {e}")
        
        # Return empty buffer since we processed everything
        return events, b""

    async def send_message_with_tools(
        self,
        message: str,
        tools: List[Tool],
        tool_handler: callable,
        **kwargs
    ) -> AsyncGenerator[Dict, None]:
        """
        Send a message with tool calling capability and automatic tool execution
        
        Args:
            message: The user message
            tools: List of available tools
            tool_handler: Function to handle tool execution
            **kwargs: Additional arguments for send_message
            
        Yields:
            Dict: Streaming response events
        """
        conversation_id = kwargs.get('conversation_id')
        history = kwargs.get('history', [])
        
        while True:
            tool_results = []
            tool_uses = []
            
            # Send message with tools
            async for event in self.send_message(message, tools=tools, **kwargs):
                yield event
                
                # Collect tool use requests
                if "toolUseEvent" in event:
                    tool_use_data = event["toolUseEvent"]
                    tool_uses.append(ToolUse(
                        tool_use_id=tool_use_data["toolUseId"],
                        name=tool_use_data["name"],
                        input=tool_use_data["input"]
                    ))
                
                # Update conversation ID if provided
                if "metadataEvent" in event and "conversationId" in event["metadataEvent"]:
                    conversation_id = event["metadataEvent"]["conversationId"]
            
            # If no tools were used, we're done
            if not tool_uses:
                break
                
            # Execute tools and collect results
            for tool_use in tool_uses:
                try:
                    result = await tool_handler(tool_use.name, tool_use.input)
                    tool_results.append(self.create_tool_result(
                        tool_use.tool_use_id,
                        result,
                        ToolResultStatus.SUCCESS
                    ))
                except Exception as e:
                    tool_results.append(self.create_tool_result(
                        tool_use.tool_use_id,
                        f"Error: {str(e)}",
                        ToolResultStatus.ERROR
                    ))
            
            # Continue conversation with tool results
            message = ""  # Empty message when providing tool results
            kwargs.update({
                'conversation_id': conversation_id,
                'tool_results': tool_results,
                'tools': tools  # Keep tools available for potential follow-up
            })

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

    def create_context(
        self,
        editor_state: Optional[EditorState] = None,
        shell_state: Optional[ShellState] = None,
        git_state: Optional[GitState] = None,
        env_state: Optional[EnvState] = None,
    ) -> UserInputMessageContext:
        """
        Create a user input message context with various state information
        
        Args:
            editor_state: Editor state information
            shell_state: Shell state information  
            git_state: Git state information
            env_state: Environment state information
            
        Returns:
            UserInputMessageContext with the provided state
        """
        return UserInputMessageContext(
            editor_state=editor_state,
            shell_state=shell_state,
            git_state=git_state,
            env_state=env_state
        )

    def create_image_block(
        self,
        image_data: bytes,
        image_format: ImageFormat = ImageFormat.PNG
    ) -> ImageBlock:
        """
        Create an image block from image data
        
        Args:
            image_data: Raw image bytes
            image_format: Image format
            
        Returns:
            ImageBlock for use in messages
        """
        encoded_data = base64.b64encode(image_data).decode('utf-8')
        return ImageBlock(
            format=image_format,
            source=ImageSource(bytes=encoded_data)
        )
