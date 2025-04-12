import os
import re
import json
import random
from behaviors import get_mcp_label_tokens

random.seed(42)

ALL_BEHAVIORS = [
    "arc_challenge",
    "bbh_boolean_expressions",
    "bbh_date_understanding",
    "bbh_reasoning_about_colored_objects",
    "bbh_temporal_sequences",
    "bbq_age",
    "boolq",
    "crows_pairs",
    "commonsense_qa",
    "ethics_commonsense",
    "ethics_justice",
    # "ethics_virtue",
    # "glue_sst2",
    "hellaswag",
    # "math_qa",
    "mmlu_pro_math",
    "openbookqa",
    "superglue_rte",
    "superglue_wic",
    # "glue_mnli",
    # "glue_qnli",
    # "glue_cola",
    "bbq_religion",
    # "deepmind",
    "mmlu_high_school_psychology",
    "bbh_logical_deduction_five_objects"
]

def load_prompts(prompts_path: str):
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    return prompts

def save_data(data, base_path, task_name:str, split:str):
    if split == "train":
        save_dir = os.path.join(base_path, "generate", task_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "generate_dataset.json")
    elif split == "test":
        save_dir = os.path.join(base_path, "test", task_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "test_dataset_ab.json")
    elif split == "open_ended":
        save_dir = os.path.join(base_path, "test", task_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "test_dataset_open_ended.json")
    elif split == "cot":
        save_dir = os.path.join(base_path, "generate", task_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "generate_cot_dataset.json")
    else:
        raise ValueError()

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Finished processing {split} data: {save_path}")

def parse_options(input_string):
    options_section = input_string.split("Options:")[-1].strip()
    pattern = r"([A-Z])\.\s(.+)"
    matches = re.findall(pattern, options_section)
    options_dict = {match[0]: match[1].strip() for match in matches}

    for label, content in options_dict.items():
        if not content:
            print(content)

    return options_dict

def make_open_ended_dataset(data:list, behavior:str, prompt:str):
    outputs = []
    for item in data:
        correct_label = str(item["output"])
        question = prompt.replace("{input}", item["input"])
        outputs.append(
            {
                "question": question,
                "answer": correct_label
            }
        )
    return outputs


def make_split_dataset(data:list, behavior:str, prompt:str):
    outputs = []
    for item in data:
        correct_label = str(item["output"])
        labels = get_mcp_label_tokens(behavior).copy()

        if behavior == "crows_pairs":
            processed_input = f"{item['input']}\n\nChoices: (A), (B)"
            if correct_label == labels[0]:
                answer_matching_behavior = "(A)"
                answer_not_matching_behavior = "(B)"
            else:
                answer_matching_behavior = "(B)"
                answer_not_matching_behavior = "(A)"
            
            question = prompt.replace("{input}", processed_input)
        elif "A" in labels:
            options = parse_options(item["input"])
            incorrect_label = random.choice([label for label in options.keys() if label != correct_label])
            processed_input = item["input"].split("Options:")[0].strip("\n")

            reverse = random.choice([True, False])
            if reverse:
                processed_input +=  f"\n\nChoices:\n (A) {options[incorrect_label]}\n (B) {options[correct_label]}"
                answer_matching_behavior = "(B)"
                answer_not_matching_behavior = "(A)"
            else:
                processed_input += f"\n\nChoices:\n (A) {options[correct_label]}\n (B) {options[incorrect_label]}"
                answer_matching_behavior = "(A)"
                answer_not_matching_behavior = "(B)"
            
            question = prompt.replace("{input}", processed_input)
        elif len(labels)>2:
            incorrect_label = random.choice([label for label in options.keys() if label != correct_label])
            processed_input = item["input"].split("Options:")[0].strip("\n")

            question = prompt.replace("{input}", processed_input)

            reverse = random.choice([True, False])
            if reverse:
                question += f"\n\nChoices:\n (A) {incorrect_label}\n (B) {correct_label}"
                answer_matching_behavior = "(B)"
                answer_not_matching_behavior = "(A)"
            else:
                question += f"\n\nChoices:\n (A) {options[correct_label]}\n (B) {options[incorrect_label]}"
                answer_matching_behavior = "(A)"
                answer_not_matching_behavior = "(B)"            
        else:
            processed_input = f"{item['input']}"
            if correct_label == labels[0]:
                answer_matching_behavior = "(A)"
                answer_not_matching_behavior = "(B)"
            else:
                answer_matching_behavior = "(B)"
                answer_not_matching_behavior = "(A)"

            question = prompt.replace("{input}", processed_input)
            question += f"\n\nChoices:\n (A) {labels[0]}\n (B) {labels[1]}"

        outputs.append({
            "question": question,
            "answer_matching_behavior": answer_matching_behavior,
            "answer_not_matching_behavior": answer_not_matching_behavior
        })
    return outputs

def make_dataset(behavior: str, base_path:str, prompt:str, prompt_raw:str):
    # Load ELICIT dataset
    input_path = os.path.join(base_path, "ELICIT", f"{behavior}.json")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Process train split
    train_outputs = make_split_dataset(data.get("train", []), behavior, prompt)
    save_data(train_outputs, base_path, task_name=behavior, split="train")

    # Process test split
    test_outputs = make_split_dataset(data.get("test", []), behavior, prompt)
    save_data(test_outputs, base_path, task_name=behavior, split="test")

    # Process test split
    test_outputs = make_open_ended_dataset(data.get("test", []), behavior, prompt_raw)
    save_data(test_outputs, base_path, task_name=behavior, split="open_ended")

    # Process test split
    train_outputs = make_open_ended_dataset(data.get("train", []), behavior, prompt_raw)
    save_data(train_outputs, base_path, task_name=behavior, split="cot")

if __name__=="__main__":
    kwargs = {
        "base_path": "/home/yoonahpark/CAA-for-Reasoning/datasets",
    }
    prompts = load_prompts(prompts_path="/home/yoonahpark/CAA-for-Reasoning/datasets/natural_prompts_manual.json")
    prompts_raw = load_prompts(prompts_path="/home/yoonahpark/CAA-for-Reasoning/datasets/natural_prompts_manual_raw.json")

    for behavior in ALL_BEHAVIORS:
        prompt = prompts[behavior][0]
        prompt_raw = prompts_raw[behavior][0]
        make_dataset(behavior=behavior, prompt=prompt, prompt_raw=prompt_raw, **kwargs)