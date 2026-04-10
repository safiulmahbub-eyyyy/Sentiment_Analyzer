"""
Automated Pipeline Orchestrator
Runs existing scripts in sequence: Import → Analyze → Report
"""

import sys
import subprocess
from datetime import datetime

def run_script(script_path, description):
    """Run a Python script and return success status"""
    print("\n" + "=" * 70)
    print(f"[RUNNING] {description}")
    print("=" * 70)
    
    try:
        # Run the script as a subprocess
        result = subprocess.run(
            [sys.executable, script_path],
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Failed to run {description}: {e}")
        return False

def main():
    """Run the complete pipeline"""
    
    print("\n" + "=" * 70)
    print("AUTOMATED SENTIMENT ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Step 1: Import data
    success = run_script(
        'scripts/import_from_github.py',
        'Step 1: Import JSON data to database'
    )
    
    if not success:
        print("\n[FAILED] Pipeline stopped at import stage")
        return False
    
    # Step 2: Analyze sentiment
    success = run_script(
        'analyzer/process_posts.py',
        'Step 2: Analyze sentiment with VADER'
    )
    
    if not success:
        print("\n[FAILED] Pipeline stopped at analysis stage")
        return False
    
    # Step 3: Show results (optional, won't fail pipeline if it errors)
    print("\n" + "=" * 70)
    print("[RUNNING] Step 3: Display results summary")
    print("=" * 70)
    
    try:
        subprocess.run([sys.executable, 'analyzer/show_results.py'])
    except:
        print("[INFO] Results display skipped (script may not exist yet)")
    
    # Success!
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n[SUCCESS] All data imported and analyzed!")
    print("[DATABASE] database/tech_sentiment.db updated")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[STOPPED] Pipeline cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)