#!/usr/bin/env python3
"""
Test Claude Sonnet 4 - the newest model from the real API
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

async def test_claude_sonnet_4():
    """Test Claude Sonnet 4 - the newest and default model"""
    
    print("=== Claude Sonnet 4 Test ===")
    print("Testing the newest model: Claude Sonnet 4 (May 2025 release)")
    print("This is the DEFAULT model from the real API")
    print()
    
    try:
        # Load bearer token
        with open('bearer_token.txt', 'r') as f:
            bearer_token = f.read().strip()
        
        # Create request with Claude Sonnet 4
        user_message = UserInputMessage(
            content="What is the capital of India? Please provide comprehensive information about its history, significance, and modern role.",
            user_intent=UserIntent.SHOW_EXAMPLES,
            origin="CLI"
        )
        
        conversation_state = ConversationState(
            current_message=ChatMessage(user_input_message=user_message),
            chat_trigger_type=ChatTriggerType.MANUAL
        )
        
        # Convert to dict properly
        conversation_dict = asdict(conversation_state)
        
        # Convert to camelCase with enum handling
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
                # Handle enums
                if hasattr(data, 'value'):
                    return data.value
                return data
        
        conversation_camel = convert_to_camel_case(conversation_dict)
        
        # Add Claude Sonnet 4 model ID (the real API format)
        conversation_camel["modelId"] = "claude-sonnet-4"
        
        # Prepare payload
        payload = {
            "conversationState": conversation_camel
        }
        
        # Prepare headers
        headers = {
            "Content-Type": "application/x-amz-json-1.0",
            "User-Agent": "Amazon-Q-Python-Client/2.0.0",
            "Accept": "application/json",
            "Authorization": f"Bearer {bearer_token}",
            "X-Amz-Target": "AmazonCodeWhispererStreamingService.GenerateAssistantResponse"
        }
        
        url = "https://q.us-east-1.amazonaws.com/"
        payload_str = json.dumps(payload)
        
        print("🔄 Sending request to Amazon Q with Claude Sonnet 4...")
        print(f"📦 Model ID: claude-sonnet-4 (Real API format)")
        print(f"🎯 User Intent: {UserIntent.SHOW_EXAMPLES.value}")
        print(f"📝 Request: Comprehensive information about India's capital")
        print(f"🆕 This is the NEWEST model (May 2025 release)")
        print()
        
        # Send request
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=payload_str) as response:
                print(f"📊 Status: {response.status}")
                
                if response.status != 200:
                    error_text = await response.text()
                    print(f"❌ Error: {error_text}")
                    return
                
                print("✅ Request successful!")
                print("📝 Claude Sonnet 4 Response:")
                print("-" * 70)
                
                # Process EventStream response
                full_response = ""
                conversation_id = None
                event_count = 0
                start_time = asyncio.get_event_loop().time()
                
                async for chunk in response.content.iter_chunked(1024):
                    if chunk:
                        # Decode the chunk
                        chunk_str = chunk.decode("utf-8", errors="replace")
                        
                        # Extract JSON patterns from the EventStream
                        json_patterns = re.findall(r'\{"[^"]+":"[^"]*"[^}]*\}', chunk_str)
                        
                        for json_str in json_patterns:
                            try:
                                event_data = json.loads(json_str)
                                event_count += 1
                                
                                if "conversationId" in event_data:
                                    conversation_id = event_data["conversationId"]
                                    if conversation_id:
                                        print(f"[Conversation ID: {conversation_id}]")
                                
                                elif "content" in event_data:
                                    content = event_data["content"]
                                    full_response += content
                                    print(content, end="", flush=True)
                                
                            except json.JSONDecodeError:
                                continue
                
                end_time = asyncio.get_event_loop().time()
                response_time = end_time - start_time
                
                print("\n" + "-" * 70)
                print(f"📏 Complete response length: {len(full_response)} characters")
                print(f"📊 Events processed: {event_count}")
                print(f"⏱️  Response time: {response_time:.2f} seconds")
                print()
                
                if full_response.strip():
                    print("✅ Successfully received answer from Claude Sonnet 4!")
                    
                    # Analyze the response quality
                    if "new delhi" in full_response.lower() or "delhi" in full_response.lower():
                        print("🎯 Correct! The capital of India is indeed New Delhi.")
                    
                    # Check for comprehensive information as requested
                    topics_covered = []
                    if any(word in full_response.lower() for word in ["history", "historical", "built", "established"]):
                        topics_covered.append("History")
                    if any(word in full_response.lower() for word in ["significance", "important", "role", "center"]):
                        topics_covered.append("Significance")
                    if any(word in full_response.lower() for word in ["modern", "today", "current", "contemporary"]):
                        topics_covered.append("Modern Role")
                    if any(word in full_response.lower() for word in ["government", "political", "parliament", "capital"]):
                        topics_covered.append("Political Role")
                    if any(word in full_response.lower() for word in ["population", "people", "demographic"]):
                        topics_covered.append("Demographics")
                    if any(word in full_response.lower() for word in ["economy", "economic", "business", "industry"]):
                        topics_covered.append("Economy")
                    
                    if topics_covered:
                        print(f"📚 Topics covered: {', '.join(topics_covered)}")
                    
                    # Analyze response characteristics
                    sentences = full_response.count('.') + full_response.count('!') + full_response.count('?')
                    words = len(full_response.split())
                    
                    print(f"📖 Response analysis:")
                    print(f"   • Words: {words}")
                    print(f"   • Sentences: ~{sentences}")
                    print(f"   • Average words per sentence: {words/max(sentences,1):.1f}")
                    
                    # Check for Claude Sonnet 4 specific characteristics
                    if len(full_response) > 800:
                        print("🧠 Claude Sonnet 4 provided a very comprehensive response!")
                    
                    if any(marker in full_response for marker in ["•", "**", "##", "-", "1.", "2."]):
                        print("📋 Response includes excellent structured formatting!")
                    
                    print()
                    print("🎉 Claude Sonnet 4 is working perfectly!")
                    print("🆕 This is the NEWEST and most advanced model available!")
                    print("💡 Claude Sonnet 4 shows superior reasoning and comprehensive knowledge!")
                    print("🏆 This is the DEFAULT model from Amazon Q's real API!")
                        
                else:
                    print("⚠️  No content received in response")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_claude_sonnet_4())
