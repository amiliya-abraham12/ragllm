"""
Start a public ngrok tunnel for the LlamaRAG Assist API.

FIRST-TIME SETUP:
  1. Sign up for free at https://dashboard.ngrok.com/signup
  2. Copy your auth token from https://dashboard.ngrok.com/get-started/your-authtoken
  3. Run:  python start_ngrok.py YOUR_AUTH_TOKEN
  
AFTER SETUP (token is saved):
  python start_ngrok.py

This creates a public URL like https://xxxx-xx-xx.ngrok-free.app
that anyone in the world can use to access your API.
"""

import sys
from pyngrok import ngrok, conf  # type: ignore

PORT = 8000

def main():
    # If auth token provided as argument, save it
    if len(sys.argv) > 1:
        token = sys.argv[1]
        ngrok.set_auth_token(token)
        print(f"✅ Auth token saved!\n")

    # Start tunnel
    print(f"🚀 Starting ngrok tunnel to localhost:{PORT}...")
    try:
        tunnel = ngrok.connect(PORT, "http")
        public_url = tunnel.public_url

        print("\n" + "=" * 60)
        print("🌍 YOUR PUBLIC API URL:")
        print(f"\n   {public_url}")
        print(f"\n   Docs: {public_url}/docs")
        print(f"   Health: {public_url}/api/health")
        print(f"   Ask: POST {public_url}/api/ask")
        print("=" * 60)
        print("\n📋 Share this URL with anyone — they can access your API!")
        print("   Press Ctrl+C to stop the tunnel.\n")

        # Keep running
        ngrok_process = ngrok.get_ngrok_process()
        ngrok_process.proc.wait()

    except Exception as e:
        if "ERR_NGROK_105" in str(e) or "auth" in str(e).lower():
            print("\n❌ Auth token required!")
            print("   1. Sign up free: https://dashboard.ngrok.com/signup")
            print("   2. Run: python start_ngrok.py YOUR_AUTH_TOKEN\n")
        else:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
