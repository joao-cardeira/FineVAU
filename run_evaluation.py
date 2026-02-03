#!/usr/bin/env python3
"""Evaluate Qwen2.5-VL on the FineW3 dataset."""

import argparse
import json
import os
from typing import List, Dict, Any
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from google.genai import types
from tqdm import tqdm

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError("qwen_vl_utils is required: pip install qwen-vl-utils")

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Prompt for the LVLM to generate video descriptions
PROMPT_TEMPLATE = """Please provide a comprehensive description of the video.
In your description, detail:
- the setting, including the location, the time_of_day, the lighting conditions, the crowd density and the main salient feature of the location.
- any objects (e.g. people, vehicles, animals, or other objects in general) and physical attributes:
  - for people: their clothing, approximate age, gender and any prominent salient features
  - for other objects: category, colors
- the sequence of events in the video, with a particular emphasis on abnormal, criminal, dangerous or illegal events.
Ensure your description is as comprehensive and precise as possible."""

# Prompt for Gemini to evaluate the model's answer against ground truth
EVAL_PROMPT_TEMPLATE = """You are an impartial AI evaluator for a video anomaly understanding task. Your role is to compare a model's textual description of a video (`MODEL_ANSWER`) with a structured JSON representation of the ground truth (`GROUND_TRUTH_JSON`).

Your task is to meticulously evaluate the `MODEL_ANSWER` against the `GROUND_TRUTH_JSON` and produce a single JSON object as your output. Do not include any explanatory text before or after the JSON object.

Follow these evaluation criteria precisely:

1. Location Evaluation: For each key in the `Location` object of the ground truth (e.g., "environment", "time_of_day"), determine if the `MODEL_ANSWER` correctly identifies its value.
   - Score `1` if the information is present and correct.
   - Score `0` if the information is missing or incorrect.

2. Entity Evaluation: For each entity listed in the `Entity` array of the ground truth:
   - `presence_score`: Score `1` if the entity (identified by its "id") is mentioned in the `MODEL_ANSWER`. Score `0` if it is not mentioned.
   - `attributes`: For each attribute of the entity (e.g., "color", "clothing"), score `1` if the `MODEL_ANSWER` correctly states the attribute. Score `0` if it's missing or incorrect.

3. Event Evaluation: For each event in the `Event` array of the ground truth:
   - `score`: Compare the ground truth `event_description` with the `MODEL_ANSWER`.
     - Score `2` (Perfect Match): The `MODEL_ANSWER` accurately and completely describes the event.
     - Score `1` (Partial Match): The `MODEL_ANSWER` describes some aspects of the event correctly but misses key details or contains minor inaccuracies.
     - Score `0` (Mismatch/Miss): The `MODEL_ANSWER` fails to describe the event or provides a completely incorrect description.

Your output MUST be a single, valid JSON object structured exactly as follows:
{{
  "location": {{
    "environment": <0 or 1>,
    "time_of_day": <0 or 1>,
    "lighting_conditions": <0 or 1>,
    "crowd_density": <0 or 1>,
    "salient_feature": <0 or 1>
  }},
  "entity": [
    {{
      "id": "<ground_truth_entity_id>",
      "presence_score": <0 or 1>,
      "attributes": {{
        "<attribute_name>": <0 or 1>,
        ...
      }}
    }},
    ...
  ],
  "event": [
    {{
      "event_description": "<ground_truth_event_description>",
      "score": <0, 1, or 2>
    }},
    ...
  ]
}}

<GROUND_TRUTH_JSON>
{ground_truth}
</GROUND_TRUTH_JSON>

<MODEL_ANSWER>
{model_answer}
</MODEL_ANSWER>"""


def load_finew3_dataset():
    """Load the FineW3 dataset from HuggingFace."""
    return load_dataset("jaca-pereira/FineW3", split="test")


def create_model(
    model_name: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    mm_processor_cache_gb: float,
):
    """Initialize the vLLM model and processor."""
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        #mm_processor_cache_gb=mm_processor_cache_gb,
        #dtype="bfloat16",
        #trust_remote_code=True,
        #max_num_seqs=1,
    )

    return llm, processor


def prepare_inference_input(processor: AutoProcessor, video_path: str, prompt: str) -> Dict[str, Any]:
    """Prepare input for a single video (without running inference)."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": 1.0},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    prompt_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True
    )

    mm_data = {}
    if image_inputs:
        mm_data["image"] = image_inputs
    if video_inputs:
        mm_data["video"] = video_inputs

    return {
        "prompt": prompt_text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs
    }


def run_batch_inference(llm: LLM, processor: AutoProcessor, inputs: List[Dict[str, Any]]) -> List[str]:
    """Run inference on a batch of prepared inputs."""
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
        stop=[processor.tokenizer.eos_token],
    )

    outputs = llm.generate(inputs, sampling_params)

    return [output.outputs[0].text for output in outputs]


def evaluate_response(prediction: str, ground_truth: dict, gemini_client=None, gemini_model_name: str = None) -> dict:
    """Evaluate model response using Gemini as judge."""
    if gemini_client is None:
        # Return placeholder scores if Gemini is not configured
        return {"raw_scores": None, "error": "Gemini not configured"}

    # Build ground truth JSON in expected format
    gt_json = {
        "Location": ground_truth.get("scene", {}),
        "Entity": ground_truth.get("entities", []),
        "Event": ground_truth.get("events", {}).get("refined_events", []) if ground_truth.get("events") else [],
    }

    eval_prompt = EVAL_PROMPT_TEMPLATE.format(
        ground_truth=json.dumps(gt_json, indent=2),
        model_answer=prediction,
    )

    try:
        response = gemini_client.models.generate_content(
            model=gemini_model_name,
            contents=eval_prompt,
            config=types.GenerateContentConfig(
                temperature=0.0
            )
        )
        response_text = response.text.strip()

        # Extract JSON from response (handle markdown code blocks)
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        scores = json.loads(response_text)
        return {"raw_scores": scores}

    except Exception as e:
        return {"raw_scores": None, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL on FineW3")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use (0.0-1.0)",
    )
    parser.add_argument(
        "--mm-processor-cache-gb",
        type=float,
        default=4.0,
        help="GB of memory for multimodal processor cache",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        required=True,
        help="Directory containing video files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=-1,
        help="Batch size for vLLM inference. -1 means all inputs at once (default: -1)",
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Gemini API key for FV-Score evaluation (optional)",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-3-flash-preview",
        help="Gemini model to use for evaluation",
    )
    args = parser.parse_args()

    # Initialize Gemini for evaluation if API key provided
    gemini_client = None
    if args.gemini_api_key:
        if not GEMINI_AVAILABLE:
            print("Warning: google-genai not installed. Skipping evaluation.")
        else:
            gemini_client = genai.Client(api_key=args.gemini_api_key)
            print(f"Gemini evaluation enabled with model: {args.gemini_model}")
    else:
        print("No Gemini API key provided. Predictions will be saved without evaluation.")

    print(f"Loading model: {args.model}")
    llm, processor = create_model(
        args.model,
        args.tensor_parallel_size,
        args.gpu_memory_utilization,
        args.mm_processor_cache_gb,
    )

    print("Loading FineW3 dataset...")
    dataset = load_finew3_dataset()

    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    # Prepare all inputs first
    print("Preparing video inputs...")
    samples_list = list(dataset)
    prepared_inputs = []
    valid_indices = []  # Track which samples were successfully prepared

    for i, sample in enumerate(tqdm(samples_list, desc="Preparing inputs")):
        if sample.get("source") == "ECVA":
            continue

        video_id = sample["video_id"]
        video_path = os.path.join(args.video_dir, f"{video_id}.mp4")

        try:
            input_data = prepare_inference_input(processor, video_path, PROMPT_TEMPLATE)
            prepared_inputs.append(input_data)
            valid_indices.append(i)
        except Exception as e:
            print(f"  Error preparing {video_id}: {e}")

    # Determine batch size
    batch_size = len(prepared_inputs) if args.batch_size == -1 else args.batch_size

    # Run batch inference
    print(f"Running inference with batch size: {batch_size if args.batch_size != -1 else 'all'} ({len(prepared_inputs)} total inputs)")
    all_predictions = []

    for batch_start in tqdm(range(0, len(prepared_inputs), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(prepared_inputs))
        batch_inputs = prepared_inputs[batch_start:batch_end]

        batch_predictions = run_batch_inference(llm, processor, batch_inputs)
        all_predictions.extend(batch_predictions)

    # Build results with evaluation
    results = []
    prediction_idx = 0

    for i, sample in enumerate(samples_list):
        if sample.get("source") == "ECVA":
            continue

        video_id = sample["video_id"]

        if i in valid_indices:
            prediction = all_predictions[prediction_idx]
            prediction_idx += 1

            try:
                ground_truth = {
                    "events": json.loads(sample["events"]) if sample["events"] else None,
                    "entities": json.loads(sample["entities"]) if sample["entities"] else None,
                    "scene": json.loads(sample["scene"]) if sample["scene"] else None,
                }

                scores = evaluate_response(prediction, ground_truth, gemini_client, args.gemini_model)

                results.append({
                    "video_id": video_id,
                    "prediction": prediction,
                    "scores": scores,
                })
            except Exception as e:
                print(f"  Error evaluating {video_id}: {e}")
                results.append({
                    "video_id": video_id,
                    "prediction": prediction,
                    "error": str(e),
                })
        else:
            results.append({
                "video_id": video_id,
                "error": "Failed to prepare input",
            })

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()