"""
Use CAA to steer the model

Usage:
python prompting_with_steering.py --behaviors sycophancy --layers 10 --multipliers 0.1 0.5 1 2 5 10 --type ab --use_base_model --model_size 7b
"""

import json
from llama_wrapper import LlamaWrapper
import torch as t
import torch.nn.functional as F
import os
from dotenv import load_dotenv
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm
from utils.helpers import get_a_b_probs
from utils.tokenize import E_INST
from steering_settings import SteeringSettings
from behaviors import (
    get_open_ended_test_data,
    get_steering_vector,
    get_system_prompt,
    get_truthful_qa_data,
    get_mmlu_data,
    get_ab_test_data,
    get_mcp_test_data,
    ALL_BEHAVIORS,
    get_results_dir,
    # get_results_all_dir,
    get_results_ras_dir,
    get_mcp_label_tokens,
    get_activations_path,
    get_vector_path,
    get_ab_data_path
)
from utils.tokenize import tokenize_llama_base, tokenize_llama_chat

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

def process_item_ab(
    item: Dict[str, str],
    model: LlamaWrapper,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    question: str = item["question"]
    answer_matching_behavior = item["answer_matching_behavior"]
    answer_not_matching_behavior = item["answer_not_matching_behavior"]
    model_output = model.get_logits_from_text(
        user_input=question, model_output="(", system_prompt=system_prompt
    )
    a_prob, b_prob = get_a_b_probs(model_output, a_token_id, b_token_id)
    return {
        "question": question,
        "answer_matching_behavior": answer_matching_behavior,
        "answer_not_matching_behavior": answer_not_matching_behavior,
        "a_prob": a_prob,
        "b_prob": b_prob,
    }


def process_item_open_ended(
    item: Dict[str, str],
    model: LlamaWrapper,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    question = item["question"]
    model_output = model.generate_text(
        user_input=question, system_prompt=system_prompt, max_new_tokens=100
    )
    return {
        "question": question,
        "model_output": model_output.split(E_INST)[-1].strip(),
        "raw_model_output": model_output,
    }


def process_item_tqa_mmlu(
    item: Dict[str, str],
    model: LlamaWrapper,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    prompt = item["prompt"]
    correct = item["correct"]
    incorrect = item["incorrect"]
    category = item["category"]
    model_output = model.get_logits_from_text(
        user_input=prompt, model_output="(", system_prompt=system_prompt
    )
    a_prob, b_prob = get_a_b_probs(model_output, a_token_id, b_token_id)
    return {
        "question": prompt,
        "correct": correct,
        "incorrect": incorrect,
        "a_prob": a_prob,
        "b_prob": b_prob,
        "category": category,
    }


def test_steering(
    layers: List[int], multipliers: List[int], settings: SteeringSettings, overwrite=False, top_k=5
):
    """
    layers: List of layers to test steering on.
    multipliers: List of multipliers to test steering with.
    settings: SteeringSettings object.
    """
    with open(get_ab_data_path(behavior), "r") as f:
        train_data = json.load(f)
        
    save_results_dir = get_results_ras_dir(settings.behavior)
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)
    process_methods = {
        "ab": process_item_ab,
        # "open_ended": process_item_open_ended,
        # "truthful_qa": process_item_tqa_mmlu,
        # "mmlu": process_item_tqa_mmlu,
    }
    test_datasets = {
        "ab": get_ab_test_data(settings.behavior),
        # "open_ended": get_open_ended_test_data(settings.behavior),
        # "truthful_qa": get_truthful_qa_data(),
        # "mmlu": get_mmlu_data(),
    }
    model = LlamaWrapper(
        HUGGINGFACE_TOKEN,
        size=settings.model_size,
        use_chat=not settings.use_base_model,
        override_model_weights_path=settings.override_model_weights_path,
    )
    a_token_id = model.tokenizer.convert_tokens_to_ids("A")
    b_token_id = model.tokenizer.convert_tokens_to_ids("B")
    model.set_save_internal_decodings(False)
    test_data = test_datasets[settings.type]

    for layer in layers:
        name_path = model.model_name_path
        if settings.override_vector_model is not None:
            name_path = settings.override_vector_model
        
        activations_neg = t.load(
            get_activations_path(behavior, layer, name_path, "neg", type=None)
        )
        activations_pos = t.load(
            get_activations_path(behavior, layer, name_path, "pos", type=None)
        )
        activations_neg_norm = F.normalize(activations_neg, dim=1).to("cpu")
        vectors = activations_pos - activations_neg

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

                tokens = tokenize_llama_chat(
                    model.tokenizer,
                    user_input=item["question"],
                    model_output="(",
                )
                tokens = t.tensor(tokens).unsqueeze(0).to(model.device)

                model.get_logits(tokens)
                activations = model.get_last_activations(layer)
                activations = activations[0, -1, :].detach().cpu()

                # cosine similarity with all neg activations
                act_norm = F.normalize(activations, dim=0)  # shape [d]
                sims = t.matmul(activations_neg_norm, act_norm)

                topk_indices = sims.topk(top_k).indices  # shape: [top_k]
                topk_vectors = vectors[topk_indices]     # shape: [top_k, d]
                vector = topk_vectors.mean(dim=0)        # shape: [d]

                model.reset_all()
                model.set_add_activations(
                    layer, (multiplier * vector).to(model.device)
                )
                result = process_methods[settings.type](
                    item=item,
                    model=model,
                    system_prompt=get_system_prompt(settings.behavior, settings.system_prompt),
                    a_token_id=a_token_id,
                    b_token_id=b_token_id,
                )
                retrieved_items = [train_data[i] for i in topk_indices.tolist()]
                result["retrieved_items"] = retrieved_items
                results.append(result)
                
                with open(
                    save_filename,
                    "w",
                ) as f:
                    json.dump(results, f, indent=4)


def dynamic_steering(
    layers: List[int], multipliers: List[int], settings: SteeringSettings, overwrite=False, top_k=5
):
    """
    layers: List of layers to test steering on.
    multipliers: List of multipliers to test steering with.
    settings: SteeringSettings object.
    """
    with open(get_ab_data_path(behavior), "r") as f:
        train_data = json.load(f)
        
    save_results_dir = get_results_ras_dir(settings.behavior)
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)
    process_methods = {
        "ab": process_item_ab,
        # "open_ended": process_item_open_ended,
        # "truthful_qa": process_item_tqa_mmlu,
        # "mmlu": process_item_tqa_mmlu,
    }
    test_datasets = {
        "ab": get_ab_test_data(settings.behavior),
        # "open_ended": get_open_ended_test_data(settings.behavior),
        # "truthful_qa": get_truthful_qa_data(),
        # "mmlu": get_mmlu_data(),
    }
    model = LlamaWrapper(
        HUGGINGFACE_TOKEN,
        size=settings.model_size,
        use_chat=not settings.use_base_model,
        override_model_weights_path=settings.override_model_weights_path,
    )
    a_token_id = model.tokenizer.convert_tokens_to_ids("A")
    b_token_id = model.tokenizer.convert_tokens_to_ids("B")
    model.set_save_internal_decodings(False)
    test_data = test_datasets[settings.type]

    for layer in layers:
        name_path = model.model_name_path
        if settings.override_vector_model is not None:
            name_path = settings.override_vector_model
        
        activations_neg = t.load(
            get_activations_path(behavior, layer, name_path, "neg", type=None)
        )
        activations_pos = t.load(
            get_activations_path(behavior, layer, name_path, "pos", type=None)
        )
        activations_all = t.cat([activations_neg, activations_pos], dim=0)

        for multiplier in multipliers:
            result_save_suffix = settings.make_result_save_suffix(
                layer=layer, multiplier=multiplier
            )
            save_filename = os.path.join(
                save_results_dir,
                f"results_{result_save_suffix}_topk.json",
            )
            if os.path.exists(save_filename) and not overwrite:
                print("Found existing", save_filename, "- skipping")
                continue
            results = []

            for item in tqdm(test_data, desc=f"Layer {layer}, multiplier {multiplier}"):

                tokens = tokenize_llama_chat(
                    model.tokenizer,
                    user_input=item["question"],
                    model_output="",
                )
                tokens = t.tensor(tokens).unsqueeze(0).to(model.device)

                model.get_logits(tokens)
                activations = model.get_last_activations(layer)
                activations = activations[0, -1, :].detach().cpu()

                # distance with all activations
                dist = ((activations_all - activations.unsqueeze(0))**2).sum(dim=1).sqrt()
                _, topk_indices = dist.topk(top_k, largest=False)  # shape: [top_k]

                neg_len = activations_neg.shape[0]
                neg_mask = topk_indices < neg_len

                topk_neg_indices = topk_indices[neg_mask]

                model.reset_all()

                if topk_neg_indices.shape[0] > 0:
                    topk_vectors = (activations_pos - activations_neg)[topk_neg_indices] 
                    vector = topk_vectors.sum(0) / top_k

                    model.set_add_activations(
                        layer, (multiplier * vector).to(model.device)
                    )
                    retrieved_items = [train_data[i] for i in topk_neg_indices.tolist()]
                
                else:
                    retrieved_items = []

                result = process_methods[settings.type](
                    item=item,
                    model=model,
                    system_prompt=get_system_prompt(settings.behavior, settings.system_prompt),
                    a_token_id=a_token_id,
                    b_token_id=b_token_id,
                )
                result["retrieved_items"] = retrieved_items
                results.append(result)
                
                with open(
                    save_filename,
                    "w",
                ) as f:
                    json.dump(results, f, indent=4)


def zeroshot_steering(
    layers: List[int], multipliers: List[int], settings: SteeringSettings, overwrite=False, top_k=5
):
    """
    layers: List of layers to test steering on.
    multipliers: List of multipliers to test steering with.
    settings: SteeringSettings object.
    """
    with open(get_ab_data_path(behavior), "r") as f:
        train_data = json.load(f)
        
    save_results_dir = get_results_ras_dir(settings.behavior)
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)
    process_methods = {
        "ab": process_item_ab,
        # "open_ended": process_item_open_ended,
        # "truthful_qa": process_item_tqa_mmlu,
        # "mmlu": process_item_tqa_mmlu,
    }
    test_datasets = {
        "ab": get_ab_test_data(settings.behavior),
        # "open_ended": get_open_ended_test_data(settings.behavior),
        # "truthful_qa": get_truthful_qa_data(),
        # "mmlu": get_mmlu_data(),
    }
    model = LlamaWrapper(
        HUGGINGFACE_TOKEN,
        size=settings.model_size,
        use_chat=not settings.use_base_model,
        override_model_weights_path=settings.override_model_weights_path,
    )
    a_token_id = model.tokenizer.convert_tokens_to_ids("A")
    b_token_id = model.tokenizer.convert_tokens_to_ids("B")
    model.set_save_internal_decodings(False)
    test_data = test_datasets[settings.type]

    for layer in layers:
        name_path = model.model_name_path
        if settings.override_vector_model is not None:
            name_path = settings.override_vector_model
        
        activations_neg = t.load(
            get_activations_path(behavior, layer, name_path, "neg", type=None)
        )
        activations_pos = t.load(
            get_activations_path(behavior, layer, name_path, "pos", type=None)
        )
        activations_all = t.cat([activations_neg, activations_pos], dim=0)

        zs_suffix = settings.make_result_save_suffix(
            layer=layer, multiplier=0.0
        )
        zs_filename = os.path.join(
            save_results_dir,
            f"results_{zs_suffix}.json",
        )
        with open(zs_filename, "r") as f:
            zs_data = json.load(f)

        for multiplier in multipliers:
            result_save_suffix = settings.make_result_save_suffix(
                layer=layer, multiplier=multiplier
            )
            save_filename = os.path.join(
                save_results_dir,
                f"results_{result_save_suffix}_zeroshot.json",
            )
            if os.path.exists(save_filename) and not overwrite:
                print("Found existing", save_filename, "- skipping")
                continue
            results = []

            i=0

            for item in tqdm(test_data, desc=f"Layer {layer}, multiplier {multiplier}"):

                zs_item = zs_data[i]
                zs_label = "(A)" if zs_item["a_prob"] > zs_item["b_prob"] else "(B)"

                tokens = tokenize_llama_chat(
                    model.tokenizer,
                    user_input=item["question"],
                    model_output=zs_label,
                )
                tokens = t.tensor(tokens).unsqueeze(0).to(model.device)

                model.get_logits(tokens)
                activations = model.get_last_activations(layer)
                activations = activations[0, -2, :].detach().cpu()

                # distance with all activations
                dist = ((activations_all - activations.unsqueeze(0))**2).sum(dim=1).sqrt()
                _, topk_indices = dist.topk(top_k, largest=False)  # shape: [top_k]

                neg_len = activations_neg.shape[0]
                neg_mask = topk_indices < neg_len

                topk_neg_indices = topk_indices[neg_mask]

                model.reset_all()

                if topk_neg_indices.shape[0] > 0:
                    topk_vectors = (activations_pos - activations_neg)[topk_neg_indices] 
                    vector = topk_vectors.sum(0) / top_k
                    model.set_add_activations(
                        layer, (multiplier * vector).to(model.device)
                    )
                    retrieved_items = [train_data[i] for i in topk_neg_indices.tolist()]
                else:
                    retrieved_items = []

                result = process_methods[settings.type](
                    item=item,
                    model=model,
                    system_prompt=get_system_prompt(settings.behavior, settings.system_prompt),
                    a_token_id=a_token_id,
                    b_token_id=b_token_id,
                )
                result["retrieved_items"] = retrieved_items
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
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["ab", "open_ended", "truthful_qa", "mmlu"],
    )
    parser.add_argument("--system_prompt", type=str, default=None, choices=["pos", "neg"], required=False)
    parser.add_argument("--override_vector", type=int, default=None)
    parser.add_argument("--override_vector_model", type=str, default=None)
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--model_size", type=str, choices=["7b", "13b"], default="7b")
    parser.add_argument("--override_model_weights_path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--top_k", type=int, default=5)
    
    args = parser.parse_args()

    steering_settings = SteeringSettings()
    steering_settings.type = args.type
    steering_settings.system_prompt = args.system_prompt
    steering_settings.override_vector = args.override_vector
    steering_settings.override_vector_model = args.override_vector_model
    steering_settings.use_base_model = args.use_base_model
    steering_settings.model_size = args.model_size
    steering_settings.override_model_weights_path = args.override_model_weights_path

    for behavior in args.behaviors:
        steering_settings.behavior = behavior        
        zeroshot_steering(
            layers=args.layers,
            multipliers=args.multipliers,
            settings=steering_settings,
            overwrite=args.overwrite,
            top_k=args.top_k
        )

