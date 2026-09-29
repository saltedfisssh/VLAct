"""Continuation preserves committed metrics and rejects partial rank states."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from examples.RoboDojo.continue_training import latest_checkpoint, archive_uncommitted_metrics


class RecoveryTest(unittest.TestCase):
    def state(self, run, step):
        path = run / "training_states" / f"step_{step}"
        (path / "pytorch_model").mkdir(parents=True)
        (path / "progress.json").write_text(json.dumps({"step": step}))
        (path / "latest").write_text("pytorch_model")
        files = [path / "custom_checkpoint_0.pkl", path / "pytorch_model/mp_rank_00_model_states.pt"]
        for rank in range(8):
            files += [path / f"random_states_{rank}.pkl",
                      path / f"pytorch_model/bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"]
        for file in files:
            with zipfile.ZipFile(file, "w") as archive:
                archive.writestr("state.pkl", b"test state")
        return path

    def test_latest_requires_all_eight_ranks_and_commit_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            good = self.state(run, 1000)
            partial = self.state(run, 1500)
            (partial / "random_states_7.pkl").unlink()
            self.assertEqual(latest_checkpoint(run), (1000, good))
            (partial / "progress.json").unlink()
            self.assertEqual(latest_checkpoint(run), (1000, good))
            (good / "pytorch_model/bf16_zero_pp_rank_6_mp_rank_00_optim_states.pt").write_bytes(b"broken")
            with self.assertRaisesRegex(RuntimeError, "No complete"):
                latest_checkpoint(run)

    def test_rollback_keeps_complete_audit_of_uncommitted_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            archive = run / "audit"
            archive.mkdir()
            text = "".join(json.dumps({"step": step, "loss": 0.1}) + "\n" for step in [999, 1000, 1001, 1479])
            (run / "metrics.jsonl").write_text(text)
            archive_uncommitted_metrics(run, 1000, archive)
            self.assertEqual((archive / "metrics_before_resume.jsonl").read_text(), text)
            self.assertEqual([json.loads(line)["step"] for line in (run / "metrics.jsonl").read_text().splitlines()],
                             [999, 1000])


if __name__ == "__main__":
    unittest.main()
