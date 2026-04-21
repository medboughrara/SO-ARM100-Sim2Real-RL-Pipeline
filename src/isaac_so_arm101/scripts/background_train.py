import os
import argparse
import subprocess
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Run Training in the background (headless) with logging.")
    parser.add_argument("--task", type=str, default="Isaac-SO-ARM100-Reach-v0", help="Task ID to train.")
    parser.add_argument("--num_envs", type=int, default=512, help="Number of environments.")
    parser.add_argument("--max_iterations", type=int, default=1000, help="Max iterations.")
    parser.add_argument("--name", type=str, default=None, help="Name for the run log.")
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.name if args.name else f"bg_train_{args.task}_{timestamp}"
    log_file = f"logs/background_logs/{run_name}.log"
    
    os.makedirs("logs/background_logs", exist_ok=True)
    
    # Construct the command
    cmd = [
        "uv", "run", "train",
        "--task", args.task,
        "--num_envs", str(args.num_envs),
        "--max_iterations", str(args.max_iterations),
        "--headless"
    ]
    
    print(f"[INFO] Launching background training for task: {args.task}")
    print(f"[INFO] Log file: {os.path.abspath(log_file)}")
    print(f"[INFO] Command: {' '.join(cmd)}")
    
    # On Windows, we use CREATE_NEW_CONSOLE or similar to detach if needed, 
    # but the simplest way to "provide a file responsible for that" is a script that pipes to log.
    with open(log_file, "w") as f:
        # Start the process
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, shell=True)
        
    print(f"[INFO] Training started with PID: {process.pid}")
    print(f"[INFO] You can close this terminal. Training will continue in the background.")
    print(f"[INFO] Use 'Get-Content {log_file} -Wait' in PowerShell to monitor.")

if __name__ == "__main__":
    main()
