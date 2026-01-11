import os
import re
import glob
import matplotlib.pyplot as plt

def parse_logs(log_dir):
    results = []
    # Get all log files
    files = glob.glob(os.path.join(log_dir, "*.log"))
    
    # Sort files numerically
    def extract_number(filename):
        # Extract the number part from filename (e.g., basic_0.log -> 0)
        basename = os.path.basename(filename)
        match = re.search(r'(\d+)\.log', basename)
        return int(match.group(1)) if match else -1
    
    files.sort(key=extract_number)
    
    print(f"Parsing {len(files)} files in {log_dir}...")
    
    for filepath in files:
        if not os.path.isfile(filepath):
            continue
            
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # Look for the final result dictionary
            # Pattern: 'AGENT_B_WIN': 102
            match = re.search(r"'AGENT_B_WIN':\s*(\d+)", content)
            if match:
                win_count = int(match.group(1))
                # Total games is 120
                win_rate = win_count / 120.0
                results.append(win_rate)
            else:
                # Try double quotes just in case
                match_dq = re.search(r'"AGENT_B_WIN":\s*(\d+)', content)
                if match_dq:
                    win_count = int(match_dq.group(1))
                    win_rate = win_count / 120.0
                    results.append(win_rate)
                else:
                    print(f"Warning: Could not parse result from {filepath}")
    return results

def main():
    # Basic Logs
    basic_dir = os.path.join("logs", "basic")
    basic_results = parse_logs(basic_dir)
    print(f"Parsed {len(basic_results)} results from Basic logs.")

    # Pro Logs
    pro_dir = os.path.join("logs", "pro")
    pro1_dir = os.path.join("logs", "pro_1")
    
    pro_results = parse_logs(pro_dir)
    pro1_results = parse_logs(pro1_dir)
    all_pro_results = pro_results + pro1_results
    print(f"Parsed {len(all_pro_results)} results from Pro logs ({len(pro_results)} + {len(pro1_results)}).")

    # Plotting Basic
    if basic_results:
        plt.figure(figsize=(10, 6))
        x_basic = range(1, len(basic_results) + 1)
        plt.scatter(x_basic, basic_results, color='blue', alpha=0.7, label='Experiment Win Rate')
        
        avg_basic = sum(basic_results)/len(basic_results)
        plt.axhline(y=avg_basic, color='red', linestyle='--', label=f'Average: {avg_basic:.1%}')
        
        plt.title("NewAgent vs BasicAgent Win Rates (10 Experiments)")
        plt.xlabel("Experiment Index")
        plt.ylabel("Win Rate")
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        output_basic = "win_rate_basic.png"
        plt.savefig(output_basic)
        print(f"Saved {output_basic}")
        plt.close()

    # Plotting Pro
    if all_pro_results:
        plt.figure(figsize=(12, 6))
        x_pro = range(1, len(all_pro_results) + 1)
        plt.scatter(x_pro, all_pro_results, color='green', alpha=0.6, label='Experiment Win Rate')
        
        avg_pro = sum(all_pro_results)/len(all_pro_results)
        plt.axhline(y=avg_pro, color='red', linestyle='--', label=f'Average: {avg_pro:.1%}')
        
        # Add a separator line between pro and pro_1 if exactly 80 results and 40 each
        if len(pro_results) == 40 and len(pro1_results) == 40:
            plt.axvline(x=40.5, color='gray', linestyle=':', label='Batch Separation')
            plt.text(20, 0.1, 'Batch 1 (pro)', ha='center')
            plt.text(60, 0.1, 'Batch 2 (pro_1)', ha='center')

        plt.title("NewAgent vs BasicAgentPro Win Rates (80 Experiments)")
        plt.xlabel("Experiment Index")
        plt.ylabel("Win Rate")
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        output_pro = "win_rate_pro.png"
        plt.savefig(output_pro)
        print(f"Saved {output_pro}")
        plt.close()

if __name__ == "__main__":
    main()
