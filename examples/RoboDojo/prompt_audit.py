"""Inspect the tokenized prompt actually forwarded to the VLAct backbone."""
import json
from pathlib import Path

import numpy as np


class PromptAudit:
    def __init__(self, tokenizer, path):
        self.tokenizer = tokenizer
        self.path = Path(path)
        self.pending = None
        self.batches = 0
        self.records = []

    def arm(self, examples, **metadata):
        self.pending = (examples, metadata)

    def __call__(self, module, args, kwargs):
        if self.pending is None:
            return
        examples, metadata = self.pending
        prompts = self.tokenizer.batch_decode(kwargs["input_ids"].detach().cpu(), skip_special_tokens=False)
        for example, prompt in zip(examples, prompts, strict=True):
            if example["lang"] not in prompt:
                raise ValueError("Dataset instruction is missing from the actual backbone input")
            subtask = example.get("current_subtask")
            present = bool(subtask and f"Current subtask: {subtask}" in prompt)
            if subtask and not present:
                raise ValueError("Current subtask is missing from the actual backbone input")
            record = {**metadata, "source_episode_index": int(example["source_episode_index"]),
                      "original_episode_index": int(example.get("original_episode_index", example["source_episode_index"])),
                      "task_index": int(example["task_index"]), "frame_index": int(example["frame_index"]),
                      "timestamp": float(example["timestamp"]), "subtask_start_timestamp": example.get("subtask_start_timestamp"),
                      "current_subtask": subtask, "current_subtask_in_model_prompt": present,
                      "instruction": example["lang"], "model_prompt": prompt}
            if "action_valid_mask" in example:
                mask = np.asarray(example["action_valid_mask"], dtype=bool)
                record.update(valid_action_steps=int(mask.sum()), action_horizon=len(mask),
                              action_valid_mask=mask.tolist())
            with self.path.open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.records.append(record)
        self.batches += 1
        self.pending = None
