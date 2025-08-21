#!/usr/bin/env python3
"""
Debug Test for LangChain Amazon Q Integration
"""

import asyncio
import logging
import sys
from pathlib import Path

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.messages import HumanMessage
from langchain_amazon_q import create_amazon_q_chat

async def test_basic():
    """Test basic functionality with debug output."""
    print("🧪 Testing Basic LangChain Integration with Debug Mode")
    
    try:
        # Load bearer token
        with open('bearer_token.txt', 'r') as f:
            bearer_token = f.read().strip()
        print(f"✅ Bearer token loaded: {bearer_token[:20]}...")
    except FileNotFoundError:
        print("❌ Bearer token not found. Run 'python extract_token.py' first.")
        return False
    
    try:
        # Create chat model
        print("Creating ChatAmazonQ instance...")
        chat = create_amazon_q_chat(
            bearer_token=bearer_token,
            model_id="claude-3.5-sonnet",
            temperature=0.7,
            max_tokens=200
        )
        print(f"✅ Chat model created: {chat}")
        print(f"   - Model ID: {chat.model_id}")
        print(f"   - Auth type: {chat.auth_type}")
        print(f"   - Temperature: {chat.temperature}")
        
        # Test basic invoke
        print("\nTesting basic invoke...")
        message = HumanMessage(content="Say 'Hello from LangChain integration!' and nothing else.")
        print(f"Sending message: {message.content}")
        
        response = await chat.ainvoke([message])
        print(f"✅ Response received: {response}")
        print(f"   - Content: {response.content}")
        print(f"   - Type: {type(response)}")
        
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            print(f"   - Token usage: {response.usage_metadata}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_streaming():
    """Test streaming with debug output."""
    print("\n🧪 Testing Streaming")
    
    try:
        with open('bearer_token.txt', 'r') as f:
            bearer_token = f.read().strip()
        
        chat = create_amazon_q_chat(
            bearer_token=bearer_token,
            model_id="claude-3.5-sonnet"
        )
        
        print("Testing streaming...")
        message = HumanMessage(content="Count from 1 to 3, one number at a time.")
        print(f"Streaming message: {message.content}")
        
        print("Stream output: ", end="")
        chunk_count = 0
        
        async for chunk in chat.astream([message]):
            chunk_count += 1
            print(f"[Chunk {chunk_count}: {chunk.content}]", end="", flush=True)
        
        print(f"\n✅ Streaming completed with {chunk_count} chunks")
        return True
        
    except Exception as e:
        print(f"\n❌ Streaming test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run debug tests."""
    print("🚀 LangChain Amazon Q Debug Test Suite\n")
    
    # Test basic functionality
    basic_success = await test_basic()
    
    if basic_success:
        # Test streaming if basic works
        streaming_success = await test_streaming()
    else:
        streaming_success = False
    
    print("\n" + "="*50)
    print("📊 Debug Test Results")
    print("="*50)
    print(f"Basic functionality: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"Streaming: {'✅ PASS' if streaming_success else '❌ FAIL'}")
    
    if basic_success and streaming_success:
        print("\n🎉 All debug tests passed!")
        return True
    else:
        print("\n⚠️ Some tests failed. Check debug output above.")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Tests interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
