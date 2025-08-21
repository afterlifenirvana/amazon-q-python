#!/usr/bin/env python3
"""
Test different approaches for model selection
"""

import asyncio
import json
import aiohttp
import re
from dataclasses import asdict
from amazon_q_client import (
    AmazonQClient, AuthType, UserInputMessage, ChatMessage, 
    ConversationState, ChatTriggerType, UserIntent
)

async def test_model_selection():
    """Test different approaches for model selection"""
    
    print("=== Model Selection Test ===")
    
    try:
        # Load bearer token
        with open('bearer_token.txt', 'r') as f:
            bearer_token = f.read().strip()
        
        def to_camel_case(snake_str):
            components = snake_str.split('_')
            return components[0] + ''.join(word.capitalize() for word in components[1:])
        
        def convert_to_camel_case(data):
            if isinstance(data, dict):
                return {
                    to_camel_case(key): convert_to_camel_case(value)
                    for key, value in data.items()
                    if value is not None
                }
            elif isinstance(data, list):
                return [convert_to_camel_case(item) for item in data]
            else:
                if hasattr(data, 'value'):
                    return data.value
                return data
        
        # Approach 1: Try model ID at conversation state level
        print("1. Testing model ID at conversation state level:")
        
        user_message = UserInputMessage(
            content="What is the capital of India?",
            origin="CLI"
        )
        
        conversation_state = ConversationState(
            current_message=ChatMessage(user_input_message=user_message),
            chat_trigger_type=ChatTriggerType.MANUAL
        )
        
        conversation_dict = asdict(conversation_state)
        conversation_camel = convert_to_camel_case(conversation_dict)
        
        # Add model ID at conversation state level
        conversation_camel["modelId"] = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        
        payload = {"conversationState": conversation_camel}
        
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        headers = {
            "Content-Type": "application/x-amz-json-1.0",
            "User-Agent": "Amazon-Q-Python-Client/2.0.0",
            "Accept": "application/json",
            "Authorization": f"Bearer {bearer_token}",
            "X-Amz-Target": "AmazonCodeWhispererStreamingService.GenerateAssistantResponse"
        }
        
        url = "https://q.us-east-1.amazonaws.com/"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=json.dumps(payload)) as response:
                print(f"Status: {response.status}")
                if response.status == 200:
                    print("✅ Success with conversation-level model ID!")
                    
                    # Process response
                    full_response = ""
                    async for chunk in response.content.iter_chunked(1024):
                        if chunk:
                            chunk_str = chunk.decode("utf-8", errors="replace")
                            json_patterns = re.findall(r'\{"[^"]+":"[^"]*"[^}]*\}', chunk_str)
                            
                            for json_str in json_patterns:
                                try:
                                    event_data = json.loads(json_str)
                                    if "content" in event_data:
                                        content = event_data["content"]
                                        full_response += content
                                        print(content, end="", flush=True)
                                except json.JSONDecodeError:
                                    continue
                    
                    print(f"\n✅ Response: {full_response}")
                    return  # Success, exit
                else:
                    error_text = await response.text()
                    print(f"❌ Error: {error_text}")
        
        print()
        
        # Approach 2: Try model ID as a separate parameter in the payload
        print("2. Testing model ID as separate payload parameter:")
        
        conversation_camel_clean = convert_to_camel_case(asdict(conversation_state))
        payload2 = {
            "conversationState": conversation_camel_clean,
            "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0"
        }
        
        print(f"Payload: {json.dumps(payload2, indent=2)}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=json.dumps(payload2)) as response:
                print(f"Status: {response.status}")
                if response.status == 200:
                    print("✅ Success with separate model ID parameter!")
                    
                    # Process response
                    full_response = ""
                    async for chunk in response.content.iter_chunked(1024):
                        if chunk:
                            chunk_str = chunk.decode("utf-8", errors="replace")
                            json_patterns = re.findall(r'\{"[^"]+":"[^"]*"[^}]*\}', chunk_str)
                            
                            for json_str in json_patterns:
                                try:
                                    event_data = json.loads(json_str)
                                    if "content" in event_data:
                                        content = event_data["content"]
                                        full_response += content
                                        print(content, end="", flush=True)
                                except json.JSONDecodeError:
                                    continue
                    
                    print(f"\n✅ Response: {full_response}")
                    return  # Success, exit
                else:
                    error_text = await response.text()
                    print(f"❌ Error: {error_text}")
        
        print()
        
        # Approach 3: Check if model selection is not supported in CodeWhisperer API
        print("3. Model selection might not be supported in CodeWhisperer API")
        print("   The CodeWhisperer API might use a default model and not accept model selection.")
        print("   Let's test with the default model and see what we get:")
        
        # Test with default model (no model ID specified)
        conversation_camel_default = convert_to_camel_case(asdict(conversation_state))
        payload_default = {"conversationState": conversation_camel_default}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=json.dumps(payload_default)) as response:
                print(f"Default model status: {response.status}")
                if response.status == 200:
                    print("✅ Default model works. Processing response...")
                    
                    # Process response
                    full_response = ""
                    async for chunk in response.content.iter_chunked(1024):
                        if chunk:
                            chunk_str = chunk.decode("utf-8", errors="replace")
                            json_patterns = re.findall(r'\{"[^"]+":"[^"]*"[^}]*\}', chunk_str)
                            
                            for json_str in json_patterns:
                                try:
                                    event_data = json.loads(json_str)
                                    if "content" in event_data:
                                        content = event_data["content"]
                                        full_response += content
                                        print(content, end="", flush=True)
                                except json.JSONDecodeError:
                                    continue
                    
                    print(f"\n📝 Default model response: {full_response}")
                    print("💡 Note: The CodeWhisperer API appears to use a default model.")
                    print("   For specific model selection, you might need to use the Q Developer API with SigV4 auth.")
                else:
                    error_text = await response.text()
                    print(f"❌ Default model error: {error_text}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_model_selection())
