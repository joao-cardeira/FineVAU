MODEL_PROMPT = """Please provide a comprehensive description of the video.
In your description, detail:
- the setting, including the location, the time_of_day, the lighting conditions, the crowd density and the main salient feature of the location.
- any objects (e.g. people, vehicles, animals, or other objects in general) and physical attributes:
  - for people: their clothing, approximate age, gender and any prominent salient features
  - for other objects: category, colors
- the sequence of events in the video, with a particular emphasis on abnormal, criminal, dangerous or illegal events.
Ensure your description is as comprehensive and precise as possible."""

JUDGE_PROMPT = """You are an impartial AI evaluator for a video anomaly understanding task. Your role is to compare a model's textual description of a video (`MODEL_ANSWER`) with a structured JSON representation of the ground truth (`GROUND_TRUTH_JSON`).

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
