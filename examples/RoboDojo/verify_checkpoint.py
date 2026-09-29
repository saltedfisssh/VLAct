"""Strict checkpoint reload and real three-camera action inference (not simulator evaluation)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# Resolve the dataset helper even when launched from the RoboDojo environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.RoboDojo.data import RoboDojoDataset, write_json
from examples.RoboDojo.prompt_audit import PromptAudit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--policy-source', type=Path, help='Optional RoboDojo policy/starVLA/source_starvla checkout')
    parser.add_argument('--base-vlm', help='Local base VLM path override for another server')
    parser.add_argument('--source-override')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.policy_source:
        sys.path.insert(0, str(args.policy_source.resolve()))
    if args.policy_source:
        from starVLA.model.framework.base_framework import build_framework
    else:
        from starVLA.model.framework import build_framework
    import starVLA
    run = args.checkpoint.resolve().parents[1]
    cfg = OmegaConf.load(run / 'config.yaml')
    cfg.trainer.enable_gradient_checkpointing = False
    if args.base_vlm:
        cfg.framework.qwenvl.base_vlm = args.base_vlm
    model = build_framework(cfg)
    weights = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=True)
    result = model.load_state_dict(weights, strict=True)
    del weights
    model.eval().cuda()
    dataset = RoboDojoDataset(args.cache, split='val', source_override=args.source_override)
    if cfg.datasets.vla_data.get('language_mode', 'task') != dataset.manifest['spec']['language_mode']:
        raise ValueError('Checkpoint and verification cache use different language conditioning')
    sample = dataset[0]
    # Deliberately omit actions and labels for future steps from inference inputs.
    examples = [{'image': sample['image'], 'lang': sample['lang']}]
    audit_path = run / ('inference_inputs_robodojo.jsonl' if args.policy_source else 'inference_inputs.jsonl')
    audit = PromptAudit(model.qwen_vl_interface.processor.tokenizer, audit_path)
    model.qwen_vl_interface.register_forward_pre_hook(audit, with_kwargs=True)
    audit.arm([sample], phase='inference')
    with torch.inference_mode():
        predicted = model.predict_action(examples=examples)['normalized_actions']
    predicted = np.asarray(predicted, dtype=np.float32)
    assert predicted.shape == (1, 50, 14) and np.isfinite(predicted).all()
    stats = json.loads((run / 'dataset_statistics.json').read_text())['arx_x5']['action']
    low, high = np.array(stats['q01']), np.array(stats['q99'])
    absolute = .5 * (predicted + 1) * (high - low) + low
    assert np.isfinite(absolute).all()
    report = {'checkpoint': str(args.checkpoint.resolve()), 'framework_source': list(starVLA.__path__),
              'strict_load': not result.missing_keys and not result.unexpected_keys,
              'normalized_action_shape': list(predicted.shape), 'finite_actions': True,
              'source_episode_index': sample['source_episode_index'], 'frame_index': sample['frame_index'],
              'cameras': 3, 'include_state': bool(cfg.datasets.vla_data.include_state),
              'language_mode': dataset.manifest['spec']['language_mode'],
              'current_subtask': sample['current_subtask'],
              'current_subtask_in_model_prompt': audit.records[0]['current_subtask_in_model_prompt'],
              'inference_first_action_abs_qpos': absolute[0, 0].tolist(), 'simulator_evaluation': False}
    if cfg.datasets.get('vqa_data', {}).get('enabled', False):
        from examples.RoboDojo.vqa import AnnotationVQADataset, build_vqa_inputs
        vqa_sample = AnnotationVQADataset(args.cache, split='val', source_override=args.source_override)[0]
        inputs, vqa_audit = build_vqa_inputs(model.qwen_vl_interface.processor, [vqa_sample])
        with torch.inference_mode():
            vqa_loss = model.qwen_vl_interface(**inputs.to(model.device), return_dict=True, use_cache=False).loss
        if not torch.isfinite(vqa_loss):
            raise FloatingPointError('Non-finite VQA loss after checkpoint reload')
        report['vqa_reload'] = {**vqa_audit[0], 'answer_cross_entropy': float(vqa_loss),
                                'answer_only_supervision': True, 'free_generation_evaluated': False}
    target = args.output or run / ('verify_robodojo.json' if args.policy_source else 'verify_checkpoint.json')
    write_json(target, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
