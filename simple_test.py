#!/usr/bin/env python3
"""
Simple integration test with real Amazon Q API
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from amazon_q_client import AmazonQClient, AuthType


async def test_simple_message():
    """Test sending a simple message to Amazon Q"""
    print("🤖 Testing simple message with Amazon Q...")

    # Check for token file
    token_file = Path("bearer_token.txt")
    if not token_file.exists():
        print("❌ No bearer_token.txt found!")
        print("Please run: python extract_token.py")
        return False

    try:
        # Load token
        with open(token_file, "r") as f:
            bearer_token = f.read().strip()

        if not bearer_token:
            print("❌ Empty token file!")
            return False

        print(f"✓ Loaded token (length: {len(bearer_token)})")

        # Create client
        client = AmazonQClient(auth_type=AuthType.BUILDER_ID, bearer_token=bearer_token)

        print("✓ Created Amazon Q client")

        # Send a simple test message
        print("📤 Sending test message: 'Hello, can you help me?'")

        response = await client.send_simple_message("Hello, can you help me?")

        print("📥 Response received:")
        print("-" * 40)
        print(response)
        print("-" * 40)

        if response and len(response) > 0:
            print("✅ Simple message test: PASS")
            return True
        else:
            print("❌ Empty response received")
            return False

    except Exception as e:
        print(f"❌ Simple message test: FAIL - {e}")
        return False


async def test_streaming_message():
    """Test streaming message response"""
    print("\n🌊 Testing streaming message...")

    token_file = Path("bearer_token.txt")
    if not token_file.exists():
        print("❌ No bearer_token.txt found!")
        return False

    try:
        with open(token_file, "r") as f:
            bearer_token = f.read().strip()

        client = AmazonQClient(auth_type=AuthType.BUILDER_ID, bearer_token=bearer_token)

        print("📤 Sending streaming test: 'Write a simple Python function'")
        print("📥 Streaming response:")
        print("-" * 40)

        event_count = 0
        async for event in client.send_message("Write a simple Python function"):
            event_count += 1

            # Print different event types
            if "assistantResponseEvent" in event:
                content = event["assistantResponseEvent"].get("content", "")
                print(content, end="", flush=True)
            elif "codeEvent" in event:
                content = event["codeEvent"].get("content", "")
                print(f"\n[CODE]\n{content}\n[/CODE]\n", end="", flush=True)
            elif "metadataEvent" in event:
                print(f"\n[METADATA: {event['metadataEvent']}]\n", end="", flush=True)

            # Limit events for testing
            if event_count > 50:
                print("\n[... truncated for testing ...]")
                break

        print("\n" + "-" * 40)

        if event_count > 0:
            print(f"✅ Streaming test: PASS ({event_count} events received)")
            return True
        else:
            print("❌ No events received")
            return False

    except Exception as e:
        print(f"❌ Streaming test: FAIL - {e}")
        return False


async def test_conversation():
    """Test multi-turn conversation"""
    print("\n💬 Testing multi-turn conversation...")

    token_file = Path("bearer_token.txt")
    if not token_file.exists():
        print("❌ No bearer_token.txt found!")
        return False

    try:
        with open(token_file, "r") as f:
            bearer_token = f.read().strip()

        client = AmazonQClient(auth_type=AuthType.BUILDER_ID, bearer_token=bearer_token)

        # First message
        print("📤 Message 1: 'What is Python?'")
        response1 = await client.send_simple_message("What is Python?")
        print(f"📥 Response 1: {response1[:100]}...")

        # Follow-up message (note: conversation history not implemented in this simple test)
        print("\n📤 Message 2: 'Can you give me an example?'")
        response2 = await client.send_simple_message("Can you give me an example?")
        print(f"📥 Response 2: {response2[:100]}...")

        if response1 and response2:
            print("✅ Conversation test: PASS")
            return True
        else:
            print("❌ One or more responses were empty")
            return False

    except Exception as e:
        print(f"❌ Conversation test: FAIL - {e}")
        return False


async def main():
    """Run integration tests"""
    print("🧪 Amazon Q Python Client Integration Tests")
    print("=" * 60)

    # Check if we're logged into Q CLI
    print("📋 Prerequisites:")
    print("  1. Make sure you're logged in: q login")
    print("  2. Extract your token: python extract_token.py")
    print("  3. Verify CLI works: q chat 'hello'")
    print()

    all_passed = True

    # Run tests
    if not await test_simple_message():
        all_passed = False

    if not await test_streaming_message():
        all_passed = False

    if not await test_conversation():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All integration tests PASSED!")
        print("\n✨ Your Python client is working correctly!")
        print("\n📚 Next steps:")
        print("  - Check out examples.py for more advanced usage")
        print("  - Integrate the client into your own Python applications")
        print("  - Explore different models and conversation features")
    else:
        print("❌ Some integration tests FAILED!")
        print("\n🔧 Troubleshooting:")
        print("  - Verify you're logged in: q login")
        print("  - Check your token: python extract_token.py")
        print("  - Test CLI directly: q chat 'hello'")
        print("  - Check network connectivity")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
