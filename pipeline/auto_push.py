import os
import subprocess
from datetime import datetime
import os

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


REPO_PATH = "C:\\Users\\mani\\OneDrive\\Desktop\\infosys"
  # CHANGE THIS
GITHUB_USERNAME = "manisudhan"
GITHUB_REPO = "smart_traffic_violation_detection"

# 1. Move to repo folder
os.chdir(REPO_PATH)

# 2. Add your GitHub credentials automatically
remote_url = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git"
subprocess.call(["git", "remote", "set-url", "origin", remote_url])

# 3. Stage all updated parquet outputs
subprocess.call(["git", "add", "output/"])

# 4. Commit with timestamp
msg = f"Auto update parquet - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
subprocess.call(["git", "commit", "-m", msg])

# 5. Push to GitHub automatically
subprocess.call(["git", "push", "origin", "main"])

print("GitHub auto-updated successfully!")
