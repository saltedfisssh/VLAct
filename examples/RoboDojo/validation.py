"""Deterministic task-balanced validation samples shared by action and VQA branches."""
from collections import Counter, defaultdict

import numpy as np
from torch.utils.data import Subset


def balanced_validation_subset(dataset, limit, seed):
    if limit < 1:
        raise ValueError("Validation sample limit must be positive")
    groups = defaultdict(list)
    if hasattr(dataset, "rows"):
        for index, row in enumerate(dataset.rows):
            groups[int(row["task_index"])].append(index)
    else:
        offset = 0
        for episode in dataset.episodes:
            groups[int(episode["task_index"])].extend(range(offset, offset + episode["length"]))
            offset += episode["length"]
    rng = np.random.default_rng(seed)
    candidates = {task: rng.permutation(indices).tolist() for task, indices in groups.items()}
    tasks = rng.permutation(sorted(candidates)).tolist()
    indices, counts = [], Counter()
    while tasks and len(indices) < limit:
        remaining = []
        for task in tasks:
            if candidates[task] and len(indices) < limit:
                indices.append(candidates[task].pop())
                counts[task] += 1
            if candidates[task]:
                remaining.append(task)
        tasks = remaining
    return Subset(dataset, indices), {
        "method": "fixed_task_balanced_without_replacement", "seed": int(seed),
        "requested_samples": int(limit), "samples": len(indices), "dataset_indices": indices,
        "samples_by_task": dict(sorted(counts.items())),
        "complete_validation_set": len(indices) == len(dataset),
    }
