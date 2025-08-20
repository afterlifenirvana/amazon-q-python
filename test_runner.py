#!/usr/bin/env python3
"""
Simple test runner for Amazon Q Client
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from amazon_q_client import AmazonQClient, AuthType


async def test_basic_functionality():
    """Test basic client functionality without real API calls"""
    print("🧪 Testing basic client functionality...")

    # Test 1: Client initialization
    print("  ✓ Testing client initialization...")
    try:
        client = AmazonQClient(auth_type=AuthType.BUILDER_ID, bearer_token="test_token")
        assert client.auth_type == AuthType.BUILDER_ID
        assert client.bearer_token == "test_token"
        print("    ✅ Builder ID client initialization: PASS")
    except Exception as e:
        print(f"    ❌ Builder ID client initialization: FAIL - {e}")
        return False

    # Test 2: Headers generation
    print("  ✓ Testing headers generation...")
    try:
        headers = client._get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test_token"
        assert headers["Content-Type"] == "application/x-amz-json-1.0"
        print("    ✅ Headers generation: PASS")
    except Exception as e:
        print(f"    ❌ Headers generation: FAIL - {e}")
        return False

    # Test 3: Request preparation
    print("  ✓ Testing request preparation...")
    try:
        payload = {"test": "data"}
        url, headers, payload_str = client._prepare_request("GenerateAssistantResponse", payload)
        assert url == "https://q.us-east-1.amazonaws.com/"
        assert "X-Amz-Target" in headers
        print("    ✅ Request preparation: PASS")
    except Exception as e:
        print(f"    ❌ Request preparation: FAIL - {e}")
        return False

    # Test 4: Dictionary cleaning
    print("  ✓ Testing dictionary cleaning...")
    try:
        dirty_dict = {"key1": "value1", "key2": None, "key3": {"nested": None}}
        clean_dict = client._clean_dict(dirty_dict)
        assert "key2" not in clean_dict
        assert "nested" not in clean_dict["key3"]
        print("    ✅ Dictionary cleaning: PASS")
    except Exception as e:
        print(f"    ❌ Dictionary cleaning: FAIL - {e}")
        return False

    return True


async def test_with_real_token():
    """Test with real token if available"""
    print("\n🔑 Testing with real token (if available)...")

    token_file = Path("bearer_token.txt")
    if not token_file.exists():
        print("  ⚠️  No bearer_token.txt found - skipping real token tests")
        print("  💡 Run 'python extract_token.py' to extract your token")
        return True

    try:
        with open(token_file, "r") as f:
            bearer_token = f.read().strip()

        if not bearer_token:
            print("  ⚠️  Empty token file - skipping real token tests")
            return True

        print(f"  ✓ Found token (length: {len(bearer_token)})")

        # Create client with real token
        client = AmazonQClient(auth_type=AuthType.BUILDER_ID, bearer_token=bearer_token)

        print("  ✅ Real token client creation: PASS")
        print("  💡 Token appears valid - ready for real API calls")

        return True

    except Exception as e:
        print(f"  ❌ Real token test: FAIL - {e}")
        return False


async def test_sigv4_setup():
    """Test SigV4 setup"""
    print("\n🔐 Testing SigV4 authentication setup...")

    try:
        # Test with explicit credentials
        client = AmazonQClient(
            auth_type=AuthType.SIGV4,
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
        )

        assert client.signer is not None
        assert client.signer.access_key == "test_key"
        print("  ✅ SigV4 client with explicit credentials: PASS")

        return True

    except Exception as e:
        print(f"  ❌ SigV4 setup: FAIL - {e}")
        return False


async def main():
    """Run all tests"""
    print("🚀 Amazon Q Python Client Test Suite")
    print("=" * 50)

    all_passed = True

    # Run basic functionality tests
    if not await test_basic_functionality():
        all_passed = False

    # Test with real token if available
    if not await test_with_real_token():
        all_passed = False

    # Test SigV4 setup
    if not await test_sigv4_setup():
        all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests PASSED!")
        print("\n📋 Next steps:")
        print("  1. Extract your token: python extract_token.py")
        print("  2. Run real API test: python simple_test.py")
        print("  3. Run full test suite: uv run pytest")
    else:
        print("❌ Some tests FAILED!")
        print("Please check the errors above and fix them.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
