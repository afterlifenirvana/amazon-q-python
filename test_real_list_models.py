#!/usr/bin/env python3
"""
Test real list_available_models API call with correct format
"""

import asyncio
import json
import aiohttp
from amazon_q_client import AmazonQClient, AuthType

async def test_real_list_models_api():
    """Test the real list_available_models API call with correct format"""
    
    print("=== Real List Models API Test ===")
    print("Testing the actual list_available_models API call")
    print()
    
    try:
        # Load bearer token
        with open('bearer_token.txt', 'r') as f:
            bearer_token = f.read().strip()
        
        # Prepare the request exactly like the Rust client
        # The API expects both query parameters AND JSON body
        
        # Query parameters
        query_params = "origin=CLI"
        
        # JSON body (same parameters)
        body_data = {
            "origin": "CLI"
        }
        
        # Headers
        headers = {
            "Content-Type": "application/x-amz-json-1.0",
            "User-Agent": "Amazon-Q-Python-Client/2.0.0",
            "Accept": "application/json",
            "Authorization": f"Bearer {bearer_token}",
            "X-Amz-Target": "AmazonCodeWhispererService.ListAvailableModels"
        }
        
        url = f"https://q.us-east-1.amazonaws.com/?{query_params}"
        body = json.dumps(body_data)
        
        print(f"🌐 URL: {url}")
        print(f"📋 Headers: {headers}")
        print(f"📦 Body: {body}")
        print()
        
        print("🔄 Making API call...")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=body) as response:
                print(f"📊 Status: {response.status}")
                print(f"📋 Response Headers: {dict(response.headers)}")
                
                if response.status == 200:
                    response_data = await response.json()
                    print("✅ Success! API response:")
                    print(json.dumps(response_data, indent=2))
                    
                    # Parse the models
                    models = response_data.get("models", [])
                    default_model = response_data.get("defaultModel", {})
                    
                    print(f"\n📊 Found {len(models)} models")
                    print(f"🎯 Default model: {default_model.get('modelName', default_model.get('modelId'))}")
                    
                    print("\n📋 Available models:")
                    for i, model in enumerate(models, 1):
                        model_name = model.get("modelName", model.get("modelId"))
                        model_id = model.get("modelId")
                        description = model.get("description", "")
                        
                        print(f"  {i:2d}. {model_name}")
                        print(f"      ID: {model_id}")
                        if description:
                            print(f"      Description: {description}")
                        
                        # Token limits
                        if "tokenLimits" in model:
                            limits = model["tokenLimits"]
                            max_input = limits.get("maxInputTokens", "N/A")
                            max_output = limits.get("maxOutputTokens", "N/A")
                            print(f"      Tokens: Input={max_input}, Output={max_output}")
                        print()
                    
                    print("🎉 Real API call successful!")
                    print("💡 Now we can implement this in the enhanced client")
                    
                else:
                    error_text = await response.text()
                    print(f"❌ Error: {error_text}")
                    
                    # Try different approaches
                    print("\n🔄 Trying alternative approaches...")
                    
                    # Try 1: Only query parameters, no body
                    print("1. Trying with only query parameters...")
                    async with session.post(f"https://q.us-east-1.amazonaws.com/?{query_params}", 
                                          headers=headers, data="") as resp:
                        print(f"   Status: {resp.status}")
                        if resp.status != 200:
                            error = await resp.text()
                            print(f"   Error: {error[:100]}...")
                    
                    # Try 2: Only body, no query parameters
                    print("2. Trying with only JSON body...")
                    async with session.post("https://q.us-east-1.amazonaws.com/", 
                                          headers=headers, data=body) as resp:
                        print(f"   Status: {resp.status}")
                        if resp.status != 200:
                            error = await resp.text()
                            print(f"   Error: {error[:100]}...")
                    
                    # Try 3: Different target header
                    print("3. Trying with different X-Amz-Target...")
                    alt_headers = headers.copy()
                    alt_headers["X-Amz-Target"] = "AmazonCodeWhispererStreamingService.ListAvailableModels"
                    async with session.post(url, headers=alt_headers, data=body) as resp:
                        print(f"   Status: {resp.status}")
                        if resp.status != 200:
                            error = await resp.text()
                            print(f"   Error: {error[:100]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_real_list_models_api())
