import tkinter as tk
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from emotion_detector.ui.app import EmotionDetectorUI



def check_dependencies():
    """Check if all required dependencies are installed"""
    required_libraries = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 
        'sklearn', 'neattext', 'nltk', 'joblib'
    ]
    
    missing_libraries = []
    for lib in required_libraries:
        try:
            __import__(lib)
        except ImportError:
            missing_libraries.append(lib)
    
    return missing_libraries


def main():
    """Main application entry point"""
    # Check for dependencies
    missing_libraries = check_dependencies()
    if missing_libraries:
        print(f"Missing required libraries: {', '.join(missing_libraries)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_libraries)}")
        return 1
    
    # Create and run the application
    root = tk.Tk()
    app = EmotionDetectorUI(root)
    root.mainloop()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())