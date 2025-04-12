import os
import json
import random
from multiwoz.prompt_generator import PromptGenerator

def get_error_type(error_item):
    if error_item["jga"] == 1:
        return "Correct"
    
    if error_item["parsed_pred"] == error_item["parsed_gold"]:
        return "Error Prop"
    
    if error_item["completion"].startswith("**Dialogue state change after Latest Conversation Between System and User:**"):
        return "Header, Incorrect"
    
    completion = error_item["completion"].replace("**Dialogue state change after Latest Conversation Between System and User:**", "")
    completion = completion.strip().strip("\n").strip()
    
    return "Incorrect"

def load_json(load_path: str):
    with open(load_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def make_error_data(item_ids, prompt_refs, error_data_path:str, filter_correct_sv:bool = False, filter_correct_tsv:bool = False, get_error_type:bool = False):
    if error_data_path:
        error_data = load_json(load_path=error_data_path)
        error_data = {item["dialog_id"]: item for item in error_data}
    else:
        error_data = None

    outputs = []
    for test_id, (prompt, ref) in zip(item_ids, prompt_refs):
        if error_data is not None:
            error_item = error_data[test_id]
            
            if filter_correct_sv and error_item["jga"] == 1:
                continue
            if filter_correct_tsv and error_item["parsed_pred"] == error_item["parsed_gold"]:
                continue
            
            outputs.append({
                "question": prompt,
                "answer_matching_behavior": "**Dialogue state change after Latest Conversation Between System and User:** \n    "+ref,
                "answer_not_matching_behavior": "**Dialogue state change after Latest Conversation Between System and User:** \n    "+str(error_item["parsed_pred"]),
            })
            if get_error_type:
                outputs[-1]["error_type"] = get_error_type(error_item=error_data[test_id])
        else:
            outputs.append({
                "question": prompt,
                "answer_matching_behavior": "**Dialogue state change after Latest Conversation Between System and User:** \n    "+ref,
                "answer_not_matching_behavior": "**Dialogue state change after Latest Conversation Between System and User:** \n    "
            })
        
    return outputs

def make_choice_data(item_ids, prompt_refs, shuffled_refs):
    outputs = []
    for test_id, (prompt, ref), wrong_ref in zip(item_ids, prompt_refs, shuffled_refs):   
        reverse = random.choice([True, False])     
        if not reverse:
            question = prompt + "**Dialogue state change after Latest Conversation Between System and User:**" + f"\n\nChoices:\n (A) {ref}\n (B) {wrong_ref}"
            answer_matching_behavior = "(A)"
            answer_not_matching_behavior = "(B)"
        else:
            question = prompt + "**Dialogue state change after Latest Conversation Between System and User:**" + f"\n\nChoices:\n (A) {wrong_ref}\n (B) {ref}"
            answer_matching_behavior = "(B)"
            answer_not_matching_behavior = "(A)"

        outputs.append({
            "question": question,
            "answer_matching_behavior": answer_matching_behavior,
            "answer_not_matching_behavior": answer_not_matching_behavior,
        })    
    return outputs

def save_data(data, base_path, task_name:str, split:str):
    if split == "train":
        save_dir = os.path.join(base_path, "generate", task_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "generate_dataset.json")
    elif split == "test":
        save_dir = os.path.join(base_path, "test", task_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "test_dataset_ab.json")
    else:
        raise ValueError()

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def deranged_shuffle(original):
    n = len(original)
    if n <= 1:
        raise ValueError("List must have at least 2 elements.")

    available = original[:]
    shuffled = []

    for i in range(n):
        candidates = [x for x in available if x != original[i]]
        if not candidates:
            raise ValueError("Derangement not possible due to duplicate constraints.")
        
        chosen = random.choice(candidates)
        shuffled.append(chosen)

    return shuffled

def main(base_path:str, split:str, load_path:str, configs:dict, error_data_path:str = None, error_configs:dict = {}):
    prompt_gen = PromptGenerator()
    data = load_json(load_path=load_path)
    item_ids = [item["ID"] for item in data]

    # reformat
    data = [(item, []) for item in data]

    # make prompts
    prompt_refs = prompt_gen.make_prompt(data, **configs)

    # generate error data
    outputs = make_error_data(item_ids=item_ids, prompt_refs=prompt_refs, error_data_path=error_data_path, **error_configs)

    task_name = "multiwozdst_error"
    save_data(data=outputs, base_path=base_path, task_name=task_name, split=split)

    # generate shuffle data
    refs = [ref for _, ref in prompt_refs]
    random.shuffle(refs)
    outputs_shuffled = outputs.copy()
    for i, ref in enumerate(refs):
        outputs_shuffled[i]["answer_not_matching_behavior"] = "**Dialogue state change after Latest Conversation Between System and User:** \n    "+ref

    task_name = "multiwozdst_shuffle"
    save_data(data=outputs_shuffled, base_path=base_path, task_name=task_name, split=split)

    # generate choice data
    refs = [ref for _, ref in prompt_refs]
    shuffled_refs = deranged_shuffle(refs[:])
    outputs = make_choice_data(item_ids=item_ids, prompt_refs=prompt_refs, shuffled_refs=shuffled_refs)
    shuffled_refs = deranged_shuffle(refs)

    task_name = "multiwozdst_choice"
    save_data(data=outputs, base_path=base_path, task_name=task_name, split=split)

    # generate none data
    outputs_none = outputs.copy()
    for i in range(len(outputs_none)):
        outputs_none[i]["answer_not_matching_behavior"] = "**Dialogue state change after Latest Conversation Between System and User:** \n    "
    
    task_name = "multiwozdst_none"
    save_data(data=outputs_shuffled, base_path=base_path, task_name=task_name, split=split)

if __name__=="__main__":
    configs = {
        "use_xy_concat": False,
        "natural_language": False,
        "turn_slot_values": True,
        "add_system_prompt": True,
        "add_instruction": False,
        "add_heading": True,
        "add_guidelines": False,
        "add_special_tokens": False
    }

    # main(
    #     base_path="/home/yoonahpark/CAA-for-Reasoning/datasets",
    #     split="test",
    #     load_path="/home/yoonahpark/CAA/datasets/multiwozdst/mw21_5p_train.json", 
    #     configs=configs
    # )

    main(
        base_path="/home/yoonahpark/CAA-for-Reasoning/datasets",
        split="train", # "train", "test"
        load_path="/home/yoonahpark/CAA-for-Reasoning/datasets/multiwozdst/mw24_10p_test.json", 
        configs=configs, 
        error_data_path="/home/yoonahpark/CAA-for-Reasoning/datasets/multiwozdst/error_results_14.json"
    )

    main(
        base_path="/home/yoonahpark/CAA-for-Reasoning/datasets",
        split="test",
        load_path="/home/yoonahpark/CAA/datasets/multiwozdst/mw21_5p_train.json", 
        configs=configs
    )

    


