#!/usr/bin/env python3
"""
Simple test to ask Amazon Q about the capital of India
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from amazon_q_client import AmazonQClient, AuthType


async def ask_capital_of_india():
    """Ask Amazon Q about the capital of India"""
    print("🤖 Testing Amazon Q Client - Asking about the capital of India")
    print("=" * 60)

    # Check for token file
    token_file = Path("bearer_token.txt")
    if not token_file.exists():
        print("❌ No bearer_token.txt found!")
        print("Please run: uv run python extract_token.py")
        return False

    try:
        # Load token
        with open(token_file, "r") as f:
            bearer_token = f.read().strip()

        if not bearer_token:
            print("❌ Empty token file!")
            return False

        print(f"✓ Loaded token (length: {len(bearer_token)})")

        # Create client with us-east-1 region where the endpoint exists
        client = AmazonQClient(
            auth_type=AuthType.BUILDER_ID, 
            bearer_token=bearer_token,
            region="us-east-1"
        )
        print("✓ Created Amazon Q client")

        # Ask the question
        question = "What is the capital of India?"
        print(f"📤 Question: {question}")
        print("📥 Amazon Q Response:")
        print("-" * 40)

        response = await client.send_simple_message(question)
        print(response)
        print("-" * 40)

        if response and len(response) > 0:
            print("✅ Successfully received response from Amazon Q!")
            return True
        else:
            print("❌ Empty response received")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main function"""
    success = await ask_capital_of_india()
    
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        print("\n🔧 Troubleshooting:")
        print("  - Make sure you're logged in: q login")
        print("  - Extract your token: uv run python extract_token.py")
        print("  - Test CLI directly: q chat 'hello'")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
