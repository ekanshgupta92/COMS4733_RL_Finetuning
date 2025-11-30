import json
import numpy as np
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    args = parser.parse_args()

    print(f"Computing stats for dataset at {args.dataset}")
    
    # Find all action files
    action_files = list(args.dataset.glob("**/actions.npy"))
    if not action_files:
        print("No episodes found!")
        return

    # Load all actions
    all_actions = []
    for af in action_files:
        actions = np.load(af)
        all_actions.append(actions)
    
    # Concatenate
    all_actions = np.concatenate(all_actions, axis=0)
    
    # Compute stats
    stats = {
        "min": all_actions.min(axis=0).tolist(),
        "max": all_actions.max(axis=0).tolist(),
        "mean": all_actions.mean(axis=0).tolist(),
        "std": all_actions.std(axis=0).tolist(),
    }
    
    # Prevent divide by zero in std (replace 0 with 1.0)
    stats["std"] = [s if s > 1e-6 else 1.0 for s in stats["std"]]

    # Save
    out_path = args.dataset / "action_stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Stats saved to {out_path}")
    print("Gripper Mean:", stats["mean"][7])

if __name__ == "__main__":
    main()