"""
Make LlamaRAG Assist API publicly accessible using localhost.run.
No sign-up or installation needed — uses SSH tunneling.

Usage:
    python start_public.py

Requirements:
    - API server must be running: python -m api.main
    - SSH must be available (comes with Windows 10+)
"""

import subprocess
import sys
import re


def main():
    port = 8000

    print("=" * 58)
    print("  🌍 LlamaRAG Assist — Public Access Tunnel")
    print("=" * 58)
    print(f"\n📡 Creating public tunnel to localhost:{port}...")
    print("   Using: localhost.run (free, no sign-up)\n")

    try:
        # Start SSH tunnel
        process = subprocess.Popen(
            ["ssh", "-R", f"80:localhost:{port}", "-o", "StrictHostKeyChecking=no", "nokey@localhost.run"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Read output and find the public URL
        url_found = False
        stdout = process.stdout
        if stdout is None:
            print("❌ Failed to capture output")
            return
        for line in iter(stdout.readline, ""):
            line = line.strip()
            if not line:
                continue

            # Look for the URL in the output
            url_match = re.search(r"(https?://[a-zA-Z0-9._-]+\.lhr\.life)", line)
            if url_match and not url_found:
                public_url = url_match.group(1)
                url_found = True
                print("=" * 58)
                print("  ✅ YOUR PUBLIC API URL:")
                print(f"\n  🔗 {public_url}")
                print(f"\n  📖 Docs:    {public_url}/docs")
                print(f"  💚 Health:  {public_url}/api/health")
                print(f"  🤖 Ask:     POST {public_url}/api/ask")
                print(f"\n  📋 Share this URL — anyone can access your API!")
                print("=" * 58)
                print("\n  Press Ctrl+C to stop the tunnel.\n")

            # Also print raw output for debugging
            if not url_found:
                print(f"   {line}")

        process.wait()

    except FileNotFoundError:
        print("❌ SSH not found! Install OpenSSH:")
        print("   Settings → Apps → Optional Features → Add OpenSSH Client")
    except KeyboardInterrupt:
        print("\n\n🛑 Tunnel closed.")
        process.terminate()
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
