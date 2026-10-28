"""Run with .venv/Scripts/python.exe run.py, then visit http://127.0.0.1:5000."""
import argparse

from web.app import create_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Gabor + tensor voting workbench")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    create_app().run(host="127.0.0.1", port=args.port, threaded=True, debug=False)
