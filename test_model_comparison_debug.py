#!/usr/bin/env python3
"""
Model Comparison Test with Debug Logging
"""

import asyncio
import time
import sys
import logging
from pathlib import Path

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.messages import HumanMessage
from langchain_amazon_q import create_amazon_q_chat


async def test_single_model_debug():
    """Test a single model with detailed debug output."""
    print("🧪 Testing Single Model with Debug Output")
    
    try:
        # Load bearer token
        with open('bearer_token.txt', 'r') as f:
            bearer_token = f.read().strip()
        print(f"✅ Bearer token loaded: {bearer_token[:20]}...")
    except FileNotFoundError:
        print("❌ Bearer token not found.")
        return
    
    # Create chat model
    print("Creating chat model...")
    chat = create_amazon_q_chat(
        bearer_token=bearer_token,
        model_id="claude-3.5-sonnet",
        temperature=0.7,
        max_tokens=200
    )
    print(f"✅ Chat model created: {chat}")
    
    # Test prompt
    prompt = "What is 2+2? Give a short answer."
    print(f"📝 Testing prompt: {prompt}")
    
    # Test with debug
    print("🔍 Calling ainvoke...")
    start_time = time.time()
    
    response = await chat.ainvoke([HumanMessage(content=prompt)])
    
    end_time = time.time()
    
    print(f"✅ Response received in {end_time - start_time:.2f}s")
    print(f"📄 Response type: {type(response)}")
    print(f"📄 Response content: '{response.content}'")
    print(f"📄 Response length: {len(response.content)} characters")
    
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        print(f"🔢 Token usage: {response.usage_metadata}")
    
    if hasattr(response, 'response_metadata') and response.response_metadata:
        print(f"📊 Response metadata: {response.response_metadata}")
    
    # Test streaming too
    print("\n🔍 Testing streaming...")
    chunks = []
    async for chunk in chat.astream([HumanMessage(content="Count from 1 to 3")]):
        print(f"📦 Chunk: {chunk}")
        if chunk.content:
            chunks.append(chunk.content)
            print(f"   Content: '{chunk.content}'")
    
    print(f"✅ Streaming complete. Total chunks: {len(chunks)}")
    print(f"📄 Combined content: '{''.join(chunks)}'")


async def test_two_models_comparison():
    """Test two models side by side with debug output."""
    print("\n" + "="*60)
    print("🧪 Testing Two Models Comparison")
    print("="*60)
    
    try:
        with open('bearer_token.txt', 'r') as f:
            bearer_token = f.read().strip()
    except FileNotFoundError:
        print("❌ Bearer token not found.")
        return
    
    models = ["claude-3.5-sonnet", "claude-3.7-sonnet"]
    prompt = "Explain quantum computing in exactly 2 sentences."
    
    print(f"📝 Prompt: {prompt}")
    print(f"🤖 Models: {', '.join(models)}")
    
    for model_id in models:
        print(f"\n🔍 Testing {model_id}...")
        
        try:
            # Create model
            chat = create_amazon_q_chat(
                bearer_token=bearer_token,
                model_id=model_id,
                temperature=0.7,
                max_tokens=150
            )
            
            # Get response
            start_time = time.time()
            response = await chat.ainvoke([HumanMessage(content=prompt)])
            end_time = time.time()
            
            print(f"✅ {model_id} Response ({end_time - start_time:.2f}s):")
            print("-" * 40)
            print(response.content)
            print("-" * 40)
            
            if response.usage_metadata:
                print(f"🔢 Tokens: {response.usage_metadata}")
            
        except Exception as e:
            print(f"❌ {model_id} failed: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """Run debug tests."""
    print("🚀 Amazon Q Model Comparison - Debug Mode")
    
    # Test 1: Single model with detailed debug
    await test_single_model_debug()
    
    # Test 2: Two models comparison
    await test_two_models_comparison()
    
    print("\n✅ Debug tests complete!")


if __name__ == "__main__":
    asyncio.run(main())
