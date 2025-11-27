import subprocess
import os

SCRIPTS = [
    "traffic_ingestion.py",
    "traffic_analysis.py",
    "plotdata.py",
    "advanced_analysis.py",
    "auto_push.py"   # Final step
]

def run_script(script):
    print(f"\n Running: {script}")
    result = subprocess.run(["python", script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("ERROR:\n", result.stderr)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # go to /pipeline/

    print("=================================")
    print("   TRAFFIC ANALYTICS PIPELINE")
    print("=================================\n")

    for script in SCRIPTS:
        run_script(script)

    print("\nPipeline finished successfully!")
