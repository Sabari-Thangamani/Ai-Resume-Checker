import subprocess
import sys
import time

try:
    # Run the app
    process = subprocess.Popen([sys.executable, 'app.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(5)  # Wait for startup
    stdout, stderr = process.communicate(timeout=10)
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
    print("Return code:", process.returncode)
except subprocess.TimeoutExpired:
    print("App is running in background")
    process.terminate()
except Exception as e:
    print("Error:", e)
