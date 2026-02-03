# FineVAU Evaluation

Evaluate LVLMs on the [FineW3 dataset](https://huggingface.co/datasets/jaca-pereira/FineW3) using vLLM.

## Installation

```bash
pip install vllm datasets transformers qwen-vl-utils google-generativeai
```

## Usage

```bash
python run_evaluation.py --video-dir /path/to/videos
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-VL-7B-Instruct` | Model name or HuggingFace path |
| `--tensor-parallel-size` | `1` | Number of GPUs for tensor parallelism. Use `2` or `4` for larger models (e.g., 72B) |
| `--gpu-memory-utilization` | `0.9` | Fraction of GPU memory to use. Lower if OOM (e.g., `0.8`) |
| `--mm-processor-cache-gb` | `4.0` | Memory (GB) for caching video frames. Increase for faster processing if memory allows |
| `--video-dir` | Required | Directory containing video files named `{video_id}.mp4` |
| `--output` | `results.json` | Output file path |
| `--limit` | None | Process only first N samples (for testing) |
| `--gemini-api-key` | None | Gemini API key for FV-Score evaluation |
| `--gemini-model` | `gemini-1.5-flash` | Gemini model for evaluation |

### Examples

**Single GPU (7B model):**
```bash
python run_evaluation.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --video-dir ./videos \
    --gpu-memory-utilization 0.9
```

**Multi-GPU (72B model):**
```bash
python run_evaluation.py \
    --model Qwen/Qwen2.5-VL-72B-Instruct \
    --video-dir ./videos \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85
```

**Quick test (5 samples):**
```bash
python run_evaluation.py \
    --video-dir ./videos \
    --limit 5
```

**With FV-Score evaluation (requires Gemini API key):**
```bash
python run_evaluation.py \
    --video-dir ./videos \
    --gemini-api-key YOUR_GEMINI_API_KEY
```

## Output

Results are saved as JSON with predictions and FV-Score breakdown:

```json
[
  {
    "video_id": "Abuse023_x264",
    "prediction": "The video shows...",
    "scores": {
      "raw_scores": {
        "location": {
          "environment": 1,
          "time_of_day": 1,
          "lighting_conditions": 0,
          "crowd_density": 1,
          "salient_feature": 0
        },
        "entity": [...],
        "event": [...]
      }
    }
  }
]
```

## Citation

```bibtex
@inproceedings{pereira2026finevau,
  author    = {Pereira, Jo\~{a}o and Lopes, Vasco and Neves, Jo\~{a}o and Semedo, David},
  title     = {FineVAU: A Novel Human-Aligned Benchmark for Fine-Grained Video Anomaly Understanding},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026},
}
```