#!/usr/bin/env python3
"""
Detailed Debug Test for LangChain Amazon Q Integration
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
from amazon_q_client import AmazonQClient, AuthType

async def test_raw_client():
    """Test the raw Amazon Q client to see what responses we get."""
    print("🧪 Testing Raw Amazon Q Client")
    
    try:
        # Load bearer token
        with open('bearer_token.txt', 'r') as f:
            bearer_token = f.read().strip()
        print(f"✅ Bearer token loaded: {bearer_token[:20]}...")
    except FileNotFoundError:
        print("❌ Bearer token not found. Run 'python extract_token.py' first.")
        return False
    
    try:
        # Create raw client
        client = AmazonQClient(
            auth_type=AuthType.BUILDER_ID,
            bearer_token=bearer_token,
            max_tokens=200,
            temperature=0.7
        )
        
        print("Testing raw client send_message...")
        
        events = []
        async for event in client.send_message("Say 'Hello from raw client!' and nothing else."):
            print(f"Raw event: {event}")
            events.append(event)
        
        print(f"Total events received: {len(events)}")
        
        # Try send_simple_message
        print("\nTesting send_simple_message...")
        simple_response = await client.send_simple_message("Say 'Hello from simple!' and nothing else.")
        print(f"Simple response: '{simple_response}'")
        
        return len(events) > 0 or simple_response
        
    except Exception as e:
        print(f"❌ Raw client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_langchain_with_debug():
    """Test LangChain integration with detailed debug."""
    print("\n🧪 Testing LangChain Integration with Detailed Debug")
    
    try:
        from langchain_amazon_q import create_amazon_q_chat
        
        with open('bearer_token.txt', 'r') as f:
            bearer_token = f.read().strip()
        
        chat = create_amazon_q_chat(
            bearer_token=bearer_token,
            model_id="claude-3.5-sonnet",
            temperature=0.7,
            max_tokens=200
        )
        
        print("Testing LangChain _agenerate method...")
        
        # Test the internal _agenerate method directly
        messages = [HumanMessage(content="Say 'Hello from LangChain!' and nothing else.")]
        
        print("Calling _agenerate...")
        result = await chat._agenerate(messages)
        
        print(f"_agenerate result: {result}")
        print(f"Result type: {type(result)}")
        print(f"Generations: {result.generations}")
        if result.generations:
            print(f"First generation: {result.generations[0]}")
            print(f"Message content: '{result.generations[0].message.content}'")
        
        return True
        
    except Exception as e:
        print(f"❌ LangChain detailed test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run detailed debug tests."""
    print("🚀 Detailed Debug Test Suite\n")
    
    # Test raw client first
    raw_success = await test_raw_client()
    
    if raw_success:
        # Test LangChain integration
        langchain_success = await test_langchain_with_debug()
    else:
        langchain_success = False
    
    print("\n" + "="*50)
    print("📊 Detailed Debug Results")
    print("="*50)
    print(f"Raw client: {'✅ PASS' if raw_success else '❌ FAIL'}")
    print(f"LangChain integration: {'✅ PASS' if langchain_success else '❌ FAIL'}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️ Tests interrupted")
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
