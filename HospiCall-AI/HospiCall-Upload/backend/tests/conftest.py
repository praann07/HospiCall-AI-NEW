import os
import sys
import tempfile

# Must run before any `app.*` import: config.Settings reads DB_PATH at import.
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="hospicall_test_"), "test.db")
