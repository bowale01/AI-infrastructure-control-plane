#!/usr/bin/env python3
"""
Simple runner script for AI Agents application.
"""
import sys
import os
import asyncio

# Add the current directory to Python path so we can import from src
sys.path.insert(0, os.path.dirname(__file__))

# Import the main function from src
if __name__ == "__main__":
    try:
        from src.main import main
        asyncio.run(main())
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error running application: {e}")
        sys.exit(1)
