import re
import sys

def parse_log(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    step_data = {}
    current_rank = None
    current_step = None
    current_tensor = []
    current_param_name = None

    for line in lines:
        match = re.match(r'=== DEBUG: Parameter Gradients at Step (\d+) \[Rank (\d+)\] ===', line)
        if match:
            current_step = int(match.group(1))
            current_rank = int(match.group(2))
            if current_step not in step_data:
                step_data[current_step] = {0: {}, 1: {}}
            continue
        
        if current_rank is None:
            continue
            
        if line.startswith("Param Size:") or line.startswith("Param:"):
            if current_param_name and current_tensor:
                step_data[current_step][current_rank][current_param_name] = "".join(current_tensor)
            current_param_name = line.strip()
            current_tensor = []
        elif line.startswith("Tensor(") or line.startswith("[") or line.startswith(" [") or line.startswith("]"):
            current_tensor.append(line)
        elif line.startswith("================"):
            if current_param_name and current_tensor:
                step_data[current_step][current_rank][current_param_name] = "".join(current_tensor)
            current_param_name = None
            current_tensor = []

    return step_data

def compare(step_data):
    steps = sorted(step_data.keys())
    if not steps:
        print("No debug data found.")
        return
        
    last_step = steps[-1]
    print(f"Comparing Step {last_step}")
    data0 = step_data[last_step][0]
    data1 = step_data[last_step][1]
    
    keys0 = list(data0.keys())
    keys1 = list(data1.keys())
    
    if len(keys0) != len(keys1):
        print(f"Different number of params: Rank0={len(keys0)}, Rank1={len(keys1)}")
        
    for k in keys0:
        if k not in keys1:
            print(f"Missing {k} in Rank 1")
            continue
        
        val0 = data0[k]
        val1 = data1[k]
        if val0 == val1:
            print(f" MATCH : {k}")
        else:
            print(f" *MISMATCH* : {k}")
            print(f"Rank 0: {val0[:100]}...")
            print(f"Rank 1: {val1[:100]}...")
            print("-" * 50)

if __name__ == '__main__':
    data = parse_log('test_grad_output.log')
    compare(data)
