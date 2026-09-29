"""VQA pairing and answer supervision must survive writer order and left padding."""
import unittest
from pathlib import Path

import torch
from PIL import Image

from examples.RoboDojo.vqa import build_vqa_inputs, extract_vqa_pairs


class VQATest(unittest.TestCase):
    def test_pair_by_camera_not_adjacent_row_order(self):
        events = [
            {"role": "assistant", "style": "vqa", "camera": "observation.images.cam_high", "content": '{"count": 2}'},
            {"role": "assistant", "style": "vqa", "camera": "observation.images.cam_left_wrist", "content": '{"count": 1}'},
            {"role": "user", "style": "vqa", "camera": "observation.images.cam_left_wrist", "content": 'How many left?'},
            {"role": "user", "style": "vqa", "camera": "observation.images.cam_high", "content": 'How many high?'},
        ]
        pairs, excluded = extract_vqa_pairs(events)
        self.assertEqual(excluded, {})
        self.assertEqual({x["question"]: x["answer"] for x in pairs},
                         {'How many high?': '{"count": 2}', 'How many left?': '{"count": 1}'})
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            extract_vqa_pairs(events[:-1])

    def test_coordinate_labels_remain_excluded_until_calibrated(self):
        events = [{"role": role, "style": "vqa", "camera": "observation.images.cam_high", "content": text}
                  for role, text in [('user', 'Where is it?'), ('assistant', '{"point": [499, 588]}')]]
        pairs, excluded = extract_vqa_pairs(events)
        self.assertEqual(pairs, [])
        self.assertEqual(excluded, {"keypoint": 1})

    def test_real_qwen_template_masks_prompt_and_padding(self):
        from transformers import AutoProcessor
        base = Path('/mnt/xiaoyu_ssd_intern/pretrained_checkpoints/Qwen3-VL-4B-Instruct-Action')
        if not base.exists():
            self.skipTest('Local Qwen3-VL processor required')
        processor = AutoProcessor.from_pretrained(base, local_files_only=True)
        processor.tokenizer.padding_side = 'left'
        image = Image.new('RGB', (64, 64), color=(128, 0, 0))
        examples = [{'image': [image], 'question': 'How many objects?', 'answer': '{"count": 2}'},
                    {'image': [image], 'question': 'What is the color of the object near the robot hand?',
                     'answer': '{"attribute": "color", "value": "red"}'}]
        inputs, records = build_vqa_inputs(processor, examples)
        for i, record in enumerate(records):
            self.assertTrue(record['prompt_tokens_masked'])
            self.assertTrue(record['padding_masked'])
            self.assertGreater(record['supervised_tokens'], 0)
            self.assertNotIn(examples[i]['answer'], record['question_prompt'])
            self.assertIn(examples[i]['answer'], record['supervised_text'])
            self.assertTrue(torch.any(inputs['labels'][i] != -100))
        self.assertTrue(torch.any(inputs['attention_mask'] == 0))


if __name__ == '__main__':
    unittest.main()
