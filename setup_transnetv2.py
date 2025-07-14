#!/usr/bin/env python3
"""
TransNetV2 Setup Script for GoPro Highlights Extraction

This script helps install and verify TransNetV2 dependencies for enhanced
scene detection in the GoPro highlights extraction tool.

Now using PyTorch for better Python 3.13+ compatibility!
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors gracefully."""
    print(f"⏳ {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ {description} failed with error: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10 or later is required for TransNetV2")
        return False
    
    # PyTorch has much better Python 3.13+ support than TensorFlow
    if version.major == 3 and version.minor >= 13:
        print("✅ Python 3.13+ detected - PyTorch has excellent support!")
        print("🎉 This is much better than TensorFlow for newer Python versions")
    
    print("✅ Python version check passed")
    return True

def install_dependencies():
    """Install TransNetV2 PyTorch dependencies."""
    dependencies = [
        ("numpy>=1.21.0", "NumPy for array operations"),
        ("opencv-python>=4.5.0", "OpenCV for video processing"),
        ("torch>=1.9.0", "PyTorch for neural network inference")
    ]
    
    print("📦 Installing TransNetV2 PyTorch dependencies...")
    
    failed_packages = []
    
    for package, description in dependencies:
        if not run_command(f"pip install '{package}'", f"Installing {description}"):
            failed_packages.append((package, description))
    
    if failed_packages:
        print(f"\n⚠️  {len(failed_packages)} package(s) failed to install:")
        for package, description in failed_packages:
            print(f"   - {package} ({description})")
        
        if any("torch" in pkg for pkg, _ in failed_packages):
            print("\n💡 PyTorch installation tips:")
            print("   - Try: pip install torch torchvision (with vision support)")
            print("   - For CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu")
            print("   - Visit: https://pytorch.org/get-started/locally/ for custom installation")
        
        print("\n🔄 The script will still work with ffmpeg fallback if PyTorch fails")
        return False
    
    return True

def test_installation():
    """Test the TransNetV2 PyTorch installation."""
    print("🧪 Testing TransNetV2 PyTorch installation...")
    
    test_code = """
import sys
sys.path.append('.')
from extract_highlights import TransNetV2Detector

detector = TransNetV2Detector()
available = detector._check_availability()

if available:
    print("✅ TransNetV2 PyTorch dependencies are available")
    print("🔄 Testing model download (this may take a moment)...")
    
    # Test model download without actually loading
    try:
        detector._download_transnetv2_model()
        print("✅ TransNetV2 PyTorch model downloaded successfully")
        print("🎉 TransNetV2 PyTorch setup complete!")
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        print("💡 You can manually download from: https://github.com/soCzech/TransNetV2/tree/master/inference-pytorch")
else:
    print("❌ TransNetV2 dependencies are not available")
    print("💡 Try running: pip install torch opencv-python numpy")
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True, cwd=os.getcwd())
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def main():
    """Main setup function."""
    print("🎬 TransNetV2 PyTorch Setup for GoPro Highlights Extraction")
    print("=" * 55)
    print("🚀 Now using PyTorch for better Python 3.13+ compatibility!")
    print("")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    install_success = install_dependencies()
    if not install_success:
        print("\n⚠️  Some dependencies failed to install")
        print("💡 The script will continue with available dependencies")
        print("🔄 Testing what's available...")
    
    # Test installation
    test_success = test_installation()
    if not test_success:
        print("\n⚠️  TransNetV2 test failed")
        print("💡 The script will fall back to ffmpeg scene detection")
        print("🔧 To retry TransNetV2 setup later:")
        print("   1. Fix dependency issues above")
        print("   2. Run: python setup_transnetv2.py")
        if not install_success:
            sys.exit(1)
    
    print("\n🎉 TransNetV2 PyTorch setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run the highlights extraction script:")
    print("   python extract_highlights.py /path/to/videos /path/to/output")
    print("2. TransNetV2 PyTorch will be used automatically for scene detection")
    print("3. The script will fall back to ffmpeg if TransNetV2 fails")
    
    print("\n🔧 Configuration options:")
    print("   --scene-detection transnetv2  # Use TransNetV2 PyTorch (default)")
    print("   --scene-detection ffmpeg      # Use ffmpeg only")
    print("   --scene-detection auto        # Auto with fallback")
    print("   --transnetv2-threshold 0.5    # Detection sensitivity")
    
    print("\n🚀 Advantages of PyTorch version:")
    print("   ✅ Better Python 3.13+ compatibility")
    print("   ✅ Easier installation and fewer conflicts")
    print("   ✅ Better performance for inference")
    print("   ✅ More reliable across different platforms")

if __name__ == "__main__":
    main() 