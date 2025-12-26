"""
Simple test runner for drug discovery AI project.
"""

import sys
from pathlib import Path
import subprocess

def run_tests():
    """Run the test suite."""
    print("Running Drug Discovery AI Tests...")
    print("=" * 40)
    
    # Add src to path
    sys.path.append(str(Path(__file__).parent))
    
    try:
        # Run pytest
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", 
            "-v", 
            "--tb=short"
        ], capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


def run_simple_test():
    """Run a simple functionality test."""
    print("Running Simple Functionality Test...")
    print("-" * 30)
    
    try:
        # Test imports
        from src.utils.core import set_seed, get_device, validate_smiles
        from src.data.molecular import smiles_to_morgan_fingerprint, generate_synthetic_dataset
        from src.models import create_model
        
        print("✅ All imports successful")
        
        # Test basic functionality
        set_seed(42)
        device = get_device()
        print(f"✅ Device: {device}")
        
        # Test SMILES validation
        assert validate_smiles("CCO") == True
        assert validate_smiles("invalid") == False
        print("✅ SMILES validation works")
        
        # Test fingerprint generation
        fp = smiles_to_morgan_fingerprint("CCO")
        assert fp is not None
        assert len(fp) == 1024
        print("✅ Morgan fingerprint generation works")
        
        # Test synthetic data generation
        smiles_list, targets = generate_synthetic_dataset(n_samples=10, seed=42)
        assert len(smiles_list) == 10
        assert len(targets) == 10
        print("✅ Synthetic data generation works")
        
        # Test model creation
        model = create_model(
            model_type="fingerprint",
            input_dim=1024,
            hidden_dims=[64],
            dropout=0.2
        )
        assert model is not None
        print("✅ Model creation works")
        
        print("✅ All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Main test runner."""
    print("Drug Discovery AI - Test Suite")
    print("=" * 50)
    
    # Run simple test first
    simple_success = run_simple_test()
    
    print("\n" + "=" * 50)
    
    # Run full test suite if simple test passes
    if simple_success:
        full_success = run_tests()
        
        if full_success:
            print("\n🎉 All tests completed successfully!")
        else:
            print("\n⚠️ Some tests failed, but basic functionality works")
    else:
        print("\n❌ Basic functionality test failed - check installation")
    
    print("\nNote: This is a research demonstration - not for clinical use.")


if __name__ == "__main__":
    main()
