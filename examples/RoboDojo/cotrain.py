"""QwenOFT action + annotation-VQA objective; checkpoint keys remain plain QwenOFT."""
import json

from examples.RoboDojo.vqa import build_vqa_inputs
from starVLA.model.framework.QwenOFT import Qwenvl_OFT


class VQACoTrainingOFT(Qwenvl_OFT):
    def __init__(self, config):
        super().__init__(config)
        self.vqa_weight = float(config.datasets.vqa_data.loss_weight)
        self.vqa_audit_path = None
        self.vqa_audit_batches = 0
        self.vqa_audited = 0

    def forward(self, examples=None, vqa_examples=None, **kwargs):
        result = super().forward(examples, **kwargs) if examples is not None else {}
        if vqa_examples is not None:
            inputs, audit = build_vqa_inputs(self.qwen_vl_interface.processor, vqa_examples)
            inputs = inputs.to(self.qwen_vl_interface.model.device)
            output = self.qwen_vl_interface(**inputs, output_hidden_states=False, return_dict=True, use_cache=False)
            result["vqa_loss"] = output.loss
            # A single backward/DeepSpeed step combines both objectives.
            result["loss"] = result.get("action_loss", 0.) + self.vqa_weight * output.loss
            if self.vqa_audit_path is not None and self.vqa_audited < self.vqa_audit_batches:
                with self.vqa_audit_path.open("a") as handle:
                    for record in audit:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                self.vqa_audited += 1
        return result
