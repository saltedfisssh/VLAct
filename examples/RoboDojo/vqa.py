"""Frame/camera-aligned VQA from LeRobot language_events; answer-only text supervision."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from examples.RoboDojo.data import CAMERAS, RoboDojoDataset


def extract_vqa_pairs(events):
    by_camera = defaultdict(lambda: defaultdict(list))
    for event in events:
        if event.get("style") == "vqa":
            by_camera[event.get("camera")][event["role"]].append(event.get("content"))
    pairs, excluded = [], Counter()
    for camera, roles in by_camera.items():
        if camera not in {f"observation.images.{c}" for c in CAMERAS}:
            raise ValueError(f"VQA has no supported camera: {camera}")
        if len(roles["user"]) != 1 or len(roles["assistant"]) != 1:
            raise ValueError(f"Ambiguous VQA pairing for {camera}: expected one question and one answer")
        question, answer = roles["user"][0], roles["assistant"][0]
        if not question or not answer:
            raise ValueError("Empty VQA question or answer")
        value = json.loads(answer)
        kind = ("bbox" if "detections" in value else "keypoint" if "point" in value else
                "count" if "count" in value else "attribute" if "attribute" in value else
                "spatial" if "relation" in value else "unknown")
        if kind not in {"count", "attribute", "spatial"}:
            excluded[kind] += 1
            continue
        pairs.append({"camera": camera, "question": question, "answer": answer, "answer_type": kind})
    return pairs, dict(excluded)


class AnnotationVQADataset(Dataset):
    def __init__(self, root, split="train", source_override=None):
        self.data = RoboDojoDataset(root, split=split, source_override=source_override)
        self.episodes = {r["source_episode_index"]: r for r in self.data.episodes}
        self.rows = [json.loads(line) for line in (Path(root) / "vqa.jsonl").read_text().splitlines()
                     if json.loads(line)["split"] == split]
        for row in self.rows:
            ep = self.episodes[row["source_episode_index"]]
            if row["absolute_index"] != ep["offset"] + row["frame_index"]:
                raise ValueError("VQA frame does not match its episode")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        ep = self.episodes[row["source_episode_index"]]
        image = self.data.read_image(ep, row["absolute_index"], row["camera"].removeprefix("observation.images."))
        return {**row, "image": [image]}


def build_vqa_inputs(processor, examples):
    users = [[{"role": "user", "content": [
        *[{"type": "image", "image": image} for image in example["image"]],
        {"type": "text", "text": example["question"] + "\nReturn the answer as JSON."}]}] for example in examples]
    full = [user + [{"role": "assistant", "content": [{"type": "text", "text": example["answer"]}]}]
            for user, example in zip(users, examples, strict=True)]
    kwargs = dict(tokenize=True, padding=True, return_dict=True, return_tensors="pt")
    prompts = processor.apply_chat_template(users, add_generation_prompt=True, **kwargs)
    inputs = processor.apply_chat_template(full, add_generation_prompt=False, **kwargs)
    labels = torch.full_like(inputs["input_ids"], -100)
    audit = []
    for i, example in enumerate(examples):
        prompt_ids = prompts["input_ids"][i][prompts["attention_mask"][i].bool()]
        valid = inputs["attention_mask"][i].bool().nonzero().flatten()
        full_ids = inputs["input_ids"][i, valid]
        length = len(prompt_ids)
        if not torch.equal(full_ids[:length], prompt_ids) or length >= len(full_ids):
            raise ValueError("VQA assistant boundary could not be aligned with the tokenized prompt")
        labels[i, valid[length:]] = full_ids[length:]
        supervised = processor.tokenizer.decode(full_ids[length:], skip_special_tokens=False)
        if example["answer"] not in supervised:
            raise ValueError("VQA answer is missing from supervised tokens")
        audit.append({k: v for k, v in example.items() if k != "image"} | {
            "question_prompt": processor.tokenizer.decode(prompt_ids, skip_special_tokens=False),
            "supervised_text": supervised, "supervised_tokens": len(full_ids) - length,
            "prompt_tokens": length, "prompt_tokens_masked": bool((labels[i, valid[:length]] == -100).all()),
            "padding_masked": bool((labels[i, ~inputs["attention_mask"][i].bool()] == -100).all())})
    inputs["labels"] = labels
    return inputs, audit
