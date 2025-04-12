"""
Use CAA to steer the model

Usage:
python prompting_with_steering.py --behaviors sycophancy --layers 10 --multipliers 0.1 0.5 1 2 5 10 --type ab --use_base_model --model_size 7b
"""

import json
from llama_wrapper import LlamaWrapper
import os
from dotenv import load_dotenv
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm
from utils.tokenize import E_INST
from steering_settings import SteeringSettings
from behaviors import (
    get_open_ended_test_data,
    get_steering_vector,
    get_system_prompt,
    ALL_BEHAVIORS,
    get_results_dir,
    get_results_all_dir,
)

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

COT_PROMPT = "Answer thinking step by step:"

def process_item_open_ended(
    item: Dict[str, str],
    model: LlamaWrapper,
    prompt: Optional[str],
) -> Dict[str, str]:
    question = item["question"] + " " + prompt
    model_output = model.generate_text(
        user_input=question, system_prompt=None, max_new_tokens=512
    )
    return {
        "question": question,
        "model_output": model_output.split(E_INST)[-1].strip(),
        "raw_model_output": model_output,
        "answer": item.get("answer")
    }

def test_steering(
    layers: List[int], multipliers: List[int], settings: SteeringSettings, overwrite=False
):
    """
    layers: List of layers to test steering on.
    multipliers: List of multipliers to test steering with.
    settings: SteeringSettings object.
    """
    save_results_dir = get_results_dir(settings.behavior)
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)
    
    model = LlamaWrapper(
        HUGGINGFACE_TOKEN,
        size=settings.model_size,
        use_chat=not settings.use_base_model,
        override_model_weights_path=settings.override_model_weights_path,
    )
    model.set_save_internal_decodings(False)
    test_data = get_open_ended_test_data(settings.behavior.replace("_cot", ""))
    for layer in layers:
        name_path = model.model_name_path
        if settings.override_vector_model is not None:
            name_path = settings.override_vector_model
        if settings.override_vector is not None:
            vector = get_steering_vector(settings.behavior, settings.override_vector, name_path, normalized=True)
        else:
            vector = get_steering_vector(settings.behavior, layer, name_path, normalized=True)
        if settings.model_size != "7b":
            vector = vector.half()
        vector = vector.to(model.device)
        for multiplier in multipliers:
            result_save_suffix = settings.make_result_save_suffix(
                layer=layer, multiplier=multiplier
            )
            save_filename = os.path.join(
                save_results_dir,
                f"results_{result_save_suffix}.json",
            )
            if os.path.exists(save_filename) and not overwrite:
                print("Found existing", save_filename, "- skipping")
                continue
            results = []
            for item in tqdm(test_data, desc=f"Layer {layer}, multiplier {multiplier}"):
                model.reset_all()
                model.set_add_activations(
                    layer, multiplier * vector
                )
                result = process_item_open_ended(
                    item=item,
                    model=model,
                    prompt="",
                )
                print(result)
                results.append(result)
                with open(
                    save_filename,
                    "w",
                ) as f:
                    json.dump(results, f, indent=4)
        # CoT
        result_save_suffix = "CoT"
        save_filename = os.path.join(
            save_results_dir,
            f"results_{result_save_suffix}.json",
        )
        if os.path.exists(save_filename) and not overwrite:
            print("Found existing", save_filename, "- skipping")
            continue
        results = []
        for item in tqdm(test_data, desc=f"Layer {layer}, multiplier {multiplier}"):
            model.reset_all()
            model.set_add_activations(
                layer, 0
            )
            result = process_item_open_ended(
                item=item,
                model=model,
                prompt=COT_PROMPT,
            )
            print(result)
            results.append(result)
            with open(
                save_filename,
                "w",
            ) as f:
                json.dump(results, f, indent=4)


def test_steering_all(
    layers: List[int], multipliers: List[int], settings: SteeringSettings, overwrite=False
):
    """
    layers: List of layers to test steering on.
    multipliers: List of multipliers to test steering with.
    settings: SteeringSettings object.
    """
    save_results_dir = get_results_all_dir(settings.behavior)
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)
    
    model = LlamaWrapper(
        HUGGINGFACE_TOKEN,
        size=settings.model_size,
        use_chat=not settings.use_base_model,
        override_model_weights_path=settings.override_model_weights_path,
    )
    model.set_save_internal_decodings(False)
    test_data = get_open_ended_test_data(settings.behavior)
    
    for multiplier in multipliers:
        result_save_suffix = settings.make_result_save_suffix(
            layer="all", multiplier=multiplier
        )
        save_filename = os.path.join(
            save_results_dir,
            f"results_{result_save_suffix}.json",
        )
        if os.path.exists(save_filename) and not overwrite:
            print("Found existing", save_filename, "- skipping")
            continue
        results = []
        for item in tqdm(test_data, desc=f"multiplier {multiplier}"):
            model.reset_all()
            name_path = model.model_name_path
            for layer in layers:
                if settings.override_vector_model is not None:
                    name_path = settings.override_vector_model
                if settings.override_vector is not None:
                    vector = get_steering_vector(settings.behavior, settings.override_vector, name_path, normalized=True)
                else:
                    vector = get_steering_vector(settings.behavior, layer, name_path, normalized=True)
                if settings.model_size != "7b":
                    vector = vector.half()
                vector = vector.to(model.device)
                model.set_add_activations(
                    layer, multiplier * vector
                )

            result = process_item_open_ended(
                item=item,
                model=model,
                prompt="" if multiplier != 0 else COT_PROMPT
            )
            results.append(result)
            with open(
                save_filename,
                "w",
            ) as f:
                json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument(
        "--behaviors",
        type=str,
        nargs="+",
        default=ALL_BEHAVIORS
    )
    parser.add_argument("--system_prompt", type=str, default=None, choices=["pos", "neg"], required=False)
    parser.add_argument("--override_vector", type=int, default=None)
    parser.add_argument("--override_vector_model", type=str, default=None)
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--model_size", type=str, choices=["7b", "13b"], default="7b")
    parser.add_argument("--override_model_weights_path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--all_layers", action="store_true", default=False)
    
    args = parser.parse_args()

    steering_settings = SteeringSettings()
    steering_settings.system_prompt = args.system_prompt
    steering_settings.override_vector = args.override_vector
    steering_settings.override_vector_model = args.override_vector_model
    steering_settings.use_base_model = args.use_base_model
    steering_settings.model_size = args.model_size
    steering_settings.override_model_weights_path = args.override_model_weights_path

    for behavior in args.behaviors:
        steering_settings.behavior = behavior
        if args.all_layers:
            test_steering_all(
                layers=args.layers,
                multipliers=args.multipliers,
                settings=steering_settings,
                overwrite=args.overwrite,
            )
        else:
            test_steering(
                layers=args.layers,
                multipliers=args.multipliers,
                settings=steering_settings,
                overwrite=args.overwrite,
            )

