# run.py
# ============================================================
# ONE FILE TO RULE THEM ALL
# Run this file to get the full app working from scratch.
# Works on Windows, Mac, and Linux.
# ============================================================

import os
import sys
import subprocess
import platform

VENV_DIR = "venv"
IS_WINDOWS = platform.system() == "Windows"
PYTHON = sys.executable


def print_step(msg: str):
    print(f"\n{'='*55}")
    print(f"  {msg}")
    print(f"{'='*55}")


def run(cmd: list, check: bool = True):
    result = subprocess.run(cmd, check=check)
    return result.returncode == 0


def check_env_file():
    """Make sure .env exists with a Groq API key."""
    if not os.path.exists(".env"):
        print("\n⚠️  No .env file found!")
        print("Get a FREE Groq API key at: https://console.groq.com")
        key = input("Paste your Groq API key here: ").strip()
        if not key.startswith("gsk_"):
            print("❌ Key doesn't look right. Make sure it starts with 'gsk_'")
            sys.exit(1)
        with open(".env", "w") as f:
            f.write(f"GROQ_API_KEY={key}\n")
        print("✅ .env file created!")
    else:
        with open(".env") as f:
            content = f.read()
        if "your_groq_api_key_here" in content or "GROQ_API_KEY=" not in content:
            print("\n⚠️  Your .env file doesn't have a real API key!")
            print("Get a FREE Groq API key at: https://console.groq.com")
            key = input("Paste your Groq API key here: ").strip()
            with open(".env", "w") as f:
                f.write(f"GROQ_API_KEY={key}\n")
            print("✅ API key saved!")
        else:
            print("✅ .env file found!")


def install_dependencies():
    """Install all required packages."""
    print_step("📦 Installing dependencies...")
    run([PYTHON, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run([PYTHON, "-m", "spacy", "download", "en_core_web_sm", "-q"])
    print("✅ All dependencies installed!")


def check_embeddings():
    """Check if any domain index exists, offer to set one up."""
    domains = ["science", "medical", "general"]
    available = [d for d in domains if os.path.exists(f"embeddings/{d}_index.bin")]

    if available:
        print(f"✅ Found indexes for: {', '.join(available)}")
        return

    print("\n⚠️  No domain indexes found. Let's set one up.")
    print("Recommended: start with 'science' (fastest, ~30 seconds)")
    print("Options: science / medical / general / all")
    choice = input("Which domain? [science]: ").strip().lower() or "science"
    run([PYTHON, "setup.py"], check=False)


def launch_app():
    """Launch the Streamlit dashboard."""
    print_step("🚀 Launching Dashboard...")
    print("Opening at: http://localhost:8501")
    print("Press Ctrl+C to stop.\n")
    subprocess.run([PYTHON, "-m", "streamlit", "run", "app/dashboard.py",
                    "--server.port=8501"])


def main():
    print("""
╔══════════════════════════════════════════════╗
║   🧠 LLM Reliability RAG + KG + NLI System  ║
║         Starting up...                       ║
╚══════════════════════════════════════════════╝
    """)

    # Step 1: Check API key
    print_step("🔑 Step 1: Checking API Key")
    check_env_file()

    # Step 2: Install dependencies
    print_step("📦 Step 2: Checking Dependencies")
    try:
        import streamlit
        import faiss
        import groq
        import spacy
        print("✅ All dependencies already installed!")
    except ImportError:
        install_dependencies()

    # Step 3: Check domain indexes
    print_step("🗂️  Step 3: Checking Knowledge Base")
    check_embeddings()

    # Step 4: Launch
    print_step("🌐 Step 4: Launching App")
    launch_app()


if __name__ == "__main__":
    main()