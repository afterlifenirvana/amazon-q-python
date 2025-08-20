#!/usr/bin/env python3
"""
Helper script to extract Bearer token from Amazon Q CLI keychain
"""

import json
import sqlite3
import os
import sys
from pathlib import Path
import keyring
from typing import Optional


def get_q_cli_data_dir() -> Path:
    """Get the Amazon Q CLI data directory"""
    if sys.platform == "darwin":  # macOS
        return Path.home() / "Library" / "Application Support" / "amazon-q"
    elif sys.platform == "linux":
        return Path.home() / ".local" / "share" / "Amazon Q"
    elif sys.platform == "win32":
        return Path(os.environ.get("APPDATA", "")) / "Amazon Q"
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")


def extract_token_from_keychain() -> Optional[str]:
    """Extract Bearer token from system keychain"""
    try:
        # Try to get the token from keychain using the same key as the CLI
        token_data = keyring.get_password("Amazon Q", "codewhisperer:odic:token")
        if token_data:
            token_info = json.loads(token_data)
            return token_info.get("access_token")
    except Exception as e:
        print(f"Failed to extract from keychain: {e}")

    return None


def extract_token_from_database() -> Optional[str]:
    """Extract Bearer token from CLI database"""
    try:
        data_dir = get_q_cli_data_dir()
        db_path = data_dir / "data.sqlite3"

        if not db_path.exists():
            print(f"Database not found at {db_path}")
            return None

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Query the auth_kv table (corrected from 'secrets' table)
        cursor.execute("SELECT value FROM auth_kv WHERE key = ?", ("codewhisperer:odic:token",))
        result = cursor.fetchone()

        if result:
            token_data = json.loads(result[0])
            access_token = token_data.get("access_token")
            if access_token:
                return access_token

        conn.close()

    except Exception as e:
        print(f"Failed to extract from database: {e}")

    return None


def main():
    """Main function to extract and display token"""
    print("Amazon Q CLI Token Extractor")
    print("=" * 40)

    # Try keychain first
    print("Trying to extract token from keychain...")
    token = extract_token_from_keychain()

    if not token:
        print("Keychain extraction failed, trying database...")
        token = extract_token_from_database()

    if token:
        print(f"✅ Token extracted successfully!")
        print(f"Token (first 50 chars): {token[:50]}...")

        # Save to file for easy use
        with open("bearer_token.txt", "w") as f:
            f.write(token)
        print("💾 Token saved to 'bearer_token.txt'")

    else:
        print("❌ No token found")
        print("\nTroubleshooting:")
        print("1. Make sure you're logged in: `q login`")
        print("2. Check if the CLI is working: `q chat 'hello'`")


if __name__ == "__main__":
    main()
