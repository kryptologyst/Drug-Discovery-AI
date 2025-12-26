"""
Simple script to run the Streamlit demo.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit demo."""
    demo_path = Path(__file__).parent / "demo" / "app.py"
    
    if not demo_path.exists():
        print(f"Demo file not found: {demo_path}")
        sys.exit(1)
    
    print("Starting Drug Discovery AI Demo...")
    print("Open your browser to http://localhost:8501")
    print("Press Ctrl+C to stop the demo")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(demo_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\nDemo stopped.")

if __name__ == "__main__":
    main()
