
import json

input_path = "Qwen--Qwen2.5-Coder-14B-Instruct--main--bigcodebench-complete--refinement_acc-0-1-sanitized_calibrated.jsonl"
output_path = "Qwen--Qwen2.5-Coder-14B-Instruct--main--bigcodebench-complete--refinement_acc-0-1-sanitized_calibrated.jsonl"

seen = set()
with open(input_path, "r") as fin, open(output_path, "w") as fout:
    for line in fin:
        item = json.loads(line)
        task_id = item.get("task_id")
        if task_id not in seen:
            seen.add(task_id)
            fout.write(json.dumps(item) + "\n")
