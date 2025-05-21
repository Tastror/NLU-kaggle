import os
import csv
import json
import torch


choose = "qwen"

if choose in ["mistral", "deepseek"]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "mistralai/Mistral-7B-v0.1"; result_suffix = "mistral"
    model_name = "deepseek-ai/deepseek-math-7b-rl"; result_suffix = "deepseek"
elif choose in ["qwen"]:
    from modelscope import AutoModelForCausalLM, AutoTokenizer
    model_name = "Qwen/Qwen-7B-Chat"; result_suffix = "qwen"
else:
    print("Invalid model choice.")
    exit()


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


from huggingface_hub import login
with open("token", "r") as f:
    token = f.read().strip()
login(token=token)

try:
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    print("Model and tokenizer loaded successfully.")

except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    print("Please ensure you have a working internet connection and enough disk space.")
    print("If you have memory issues, consider using a smaller model or quantization.")
    exit()


infile = open("test.csv", "r")

result_csv = f"result_{result_suffix}.csv"
start_pos = 0
if not os.path.exists(result_csv):
    print(f"{result_csv} does not exist, creating it...")
    outfile = open(result_csv, "w")
    outfile.write("Id,label\n")
else:
    if os.path.getsize(result_csv) == 0:
        print(f"{result_csv} is empty, writing header...")
        outfile = open(result_csv, "w")
        outfile.write("Id,label\n")
    else:
        print(f"{result_csv} exists and is not empty, appending to it...")
        with open(result_csv, "r") as f:
            reader = csv.reader(f)
            last_row = list(reader)[-1]
            start_pos = int(last_row[0])
            print(f"Starting from Id: {start_pos}")
        outfile = open(result_csv, "a")

reader = csv.DictReader(infile)
processed_data = []
for row in reader:
    try:
        row['Choices'] = json.loads(row['Choices'])
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for Choices in row {row['Id']}: {e}")
        row['Choices'] = []
    try:
        row['Id'] = int(row['Id'])
    except ValueError:
        print(f"Warning: Could not convert Id '{row['Id']}' to integer.")
    processed_data.append({
        "data": f"question: {row['Question']}\nchoices: A: {row['Choices'][0]}, B: {row['Choices'][1]}, C: {row['Choices'][2]}, D: {row['Choices'][3]}",
        "id": row['Id'],
    })

infile.close()

# problem_data = "question: Ali is the principal of a private school where he teaches one class. John is also the principal of a public school. John has two classes in his school. The capacity of each class is 1/8 of the capacity of Ali's class, which has 120 students. What is the total capacity of the two schools?\nchoices: A. 947 B. 899 C. 150 D. 803"

def create_prompt_cot_answer_index(problem_data):
    global choose
    if choose in ["deepseek"]:
        return f"Please give your thought process and find the answer (format example: \\boxed{{result: A}}).\n{problem_data}\nPlease reason step by step, and put your final answer within \\boxed{{}}\nThought process & answer: To "
    else:
        return f"Please give your thought process and find the answer (format example: `result: A`).\n{problem_data}\nThought process & answer: To "

def parse_cot_index_output(model_output_str):
    import re
    match = re.search(r"(?:Answer|answer|Result|result|Option|option|答案).*?([ABCD])", model_output_str)
    if match:
        data = match.group(1)
        if data in ["A","B","C","D"]:
            res = ord(data) - ord("A")
            print(f"\033[32mParsed answer index: {res}\033[0m")
            return res
    print(f"\033[31mError: Could not parse the model output\033[0m")
    return None

retry_map = {}
total_retry = 0

for problem in processed_data[start_pos:]:

    end = False
    predicted_index = 0

    print(f"\nProblem ID: {problem['id']}")
    prompt_for_model_cot = create_prompt_cot_answer_index(problem["data"])
    print(prompt_for_model_cot)
    inputs_cot = tokenizer(prompt_for_model_cot, return_tensors="pt").to(device)

    while not end:

        outputs_cot = model.generate(**inputs_cot, max_new_tokens=99999)
        response_ids_cot = outputs_cot[0][inputs_cot.input_ids.shape[1]:]
        model_response_raw_cot = tokenizer.decode(response_ids_cot, skip_special_tokens=True).strip()

        print(f"\nRaw Model Response: {model_response_raw_cot}")
        predicted_index = parse_cot_index_output(model_response_raw_cot)

        if predicted_index is not None:
            end = True
        else:
            retry_map[problem['id']] = retry_map.get(problem['id'], 0) + 1
            total_retry += 1
            end = False

    print("\033[33m")
    print(f"Retry count for problem ID {problem['id']}: {retry_map.get(problem['id'], 0)}")
    print(f"total retry count: {total_retry}")
    print("\033[0m")

    outfile.write(f"{problem['id']},{predicted_index}\n")
    outfile.flush()

outfile.close()

print(f"Total retries: {total_retry}")
with open("retry_map.json", "w") as f:
    json.dump(retry_map, f)
print("Retry map saved to retry_map.json")
