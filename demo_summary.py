"""
Project summary and demonstration script.
"""

import os
import sys
from pathlib import Path
import subprocess

def print_project_summary():
    """Print project summary."""
    print("🧬 Drug Discovery AI - Project Summary")
    print("=" * 60)
    print()
    
    print("📁 Project Structure:")
    print("├── src/                    # Source code")
    print("│   ├── models/            # Neural network models")
    print("│   ├── data/              # Data processing utilities")
    print("│   ├── losses/            # Loss functions")
    print("│   ├── metrics/           # Evaluation metrics")
    print("│   ├── train/             # Training utilities")
    print("│   ├── eval/              # Evaluation utilities")
    print("│   └── utils/             # Core utilities")
    print("├── configs/               # Configuration files")
    print("├── demo/                  # Streamlit demo application")
    print("├── tests/                 # Unit tests")
    print("├── assets/                # Generated plots and models")
    print("├── data/                  # Data directory")
    print("├── requirements.txt        # Python dependencies")
    print("├── pyproject.toml         # Project configuration")
    print("├── README.md              # Project documentation")
    print("└── DISCLAIMER.md          # Medical disclaimer")
    print()
    
    print("🔬 Model Types Implemented:")
    print("• Fingerprint-based Neural Networks (Morgan, MACCS, RDKit)")
    print("• Graph Neural Networks (GCN, GAT)")
    print("• Random Forest Baselines")
    print("• Ensemble Models")
    print("• Uncertainty Quantification")
    print()
    
    print("📊 Features:")
    print("• Molecular fingerprinting and graph construction")
    print("• Comprehensive evaluation metrics (RMSE, MAE, R², calibration)")
    print("• Interactive Streamlit demo")
    print("• Type hints and documentation")
    print("• Unit tests and validation")
    print("• Configuration management")
    print("• Device fallback (CUDA → MPS → CPU)")
    print("• Deterministic seeding for reproducibility")
    print()
    
    print("⚠️ IMPORTANT DISCLAIMER:")
    print("This software is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.")
    print("NOT for clinical use or medical decision-making.")
    print("Always consult qualified healthcare professionals.")
    print()


def run_demo():
    """Run the Streamlit demo."""
    print("🚀 Starting Drug Discovery AI Demo...")
    print("=" * 40)
    
    demo_path = Path(__file__).parent / "demo" / "app.py"
    
    if not demo_path.exists():
        print(f"❌ Demo file not found: {demo_path}")
        return False
    
    print("Opening Streamlit demo...")
    print("The demo will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the demo")
    print()
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(demo_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        return True
    except KeyboardInterrupt:
        print("\nDemo stopped.")
        return True
    except Exception as e:
        print(f"Error running demo: {e}")
        return False


def run_training_example():
    """Run a simple training example."""
    print("🏋️ Running Training Example...")
    print("=" * 40)
    
    train_script = Path(__file__).parent / "simple_train.py"
    
    if not train_script.exists():
        print(f"❌ Training script not found: {train_script}")
        return False
    
    try:
        result = subprocess.run([sys.executable, str(train_script)], 
                              capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running training example: {e}")
        return False


def run_tests():
    """Run the test suite."""
    print("🧪 Running Tests...")
    print("=" * 40)
    
    test_script = Path(__file__).parent / "run_tests.py"
    
    if not test_script.exists():
        print(f"❌ Test script not found: {test_script}")
        return False
    
    try:
        result = subprocess.run([sys.executable, str(test_script)], 
                              capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


def main():
    """Main function."""
    print_project_summary()
    
    print("Available Commands:")
    print("1. Run Demo (Streamlit)")
    print("2. Run Training Example")
    print("3. Run Tests")
    print("4. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == "1":
                run_demo()
                break
            elif choice == "2":
                run_training_example()
                break
            elif choice == "3":
                run_tests()
                break
            elif choice == "4":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
