"""CPU smoke and regression tests: python -m unittest discover -s tests -v."""

import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataload.dataset import FusionDataset, load_image
from engine import load_checkpoint, validate
from loss import FusionLoss
from model import FusionNet
from train import check_disjoint, main as train_main
from validate import main as validate_main


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def write_split(self, name, channels=8, count=3):
        root = self.root / name
        rng = np.random.default_rng(7)
        for directory in ("MS", "PAN", "UPMS"):
            (root / directory).mkdir(parents=True)
        for index in range(count):
            for directory, suffix, shape, axes in (
                ("MS", "ms", (channels, 3, 3), "CYX"),
                ("PAN", "pan", (9, 11), "YX"),
                ("UPMS", "mul", (channels, 9, 11), "CYX"),
            ):
                data = rng.random(shape).astype(np.float32)
                tifffile.imwrite(root / directory / f"scene_a_{index}_{suffix}.tif",
                                 data, photometric="minisblack", metadata={"axes": axes})
        return root

    def test_tiff_layouts_and_constant_normalization(self):
        image = np.arange(5 * 7 * 3, dtype=np.float32).reshape(5, 7, 3)
        hwc = self.root / "hwc.tif"
        chw = self.root / "chw.tif"
        tifffile.imwrite(hwc, image, photometric="rgb")
        tifffile.imwrite(chw, image.transpose(2, 0, 1), photometric="minisblack", metadata={"axes": "CYX"})
        torch.testing.assert_close(load_image(hwc), load_image(chw))
        constant = self.root / "constant.tif"
        tifffile.imwrite(constant, np.ones((5, 7), dtype=np.float32))
        self.assertEqual(load_image(constant).shape, (1, 5, 7))
        self.assertTrue(torch.equal(load_image(constant), torch.zeros(1, 5, 7)))

    def test_dataset_pairing_channels_and_errors(self):
        root = self.write_split("data")
        dataset = FusionDataset(root, num_channels=8)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[0][2].shape, (8, 9, 11))
        with self.assertRaisesRegex(ValueError, "expected 4 MS channels"):
            FusionDataset(root, num_channels=4)[0]
        with self.assertRaisesRegex(ValueError, "separate input files"):
            check_disjoint(dataset, dataset)
        (root / "PAN" / "scene_a_0_pan.tif").unlink()
        with self.assertRaisesRegex(ValueError, "match exactly"):
            FusionDataset(root, num_channels=8)

    def test_model_channel_counts_and_spatial_padding(self):
        for channels in (1, 3, 4, 8):
            model = FusionNet(channels)
            fused = model(torch.rand(1, 1, 9, 11), torch.rand(1, channels, 9, 11))
            self.assertEqual(fused.shape, (1, channels, 9, 11))
            self.assertTrue(torch.isfinite(fused).all())

    def test_every_band_contributes_to_loss(self):
        pan, upms = torch.rand(2, 1, 8, 8), torch.rand(2, 8, 8, 8)
        fused = torch.rand_like(upms, requires_grad=True)
        losses = FusionLoss()(pan, upms, fused)
        self.assertEqual(set(losses), {"loss", "spectral", "spatial"})
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(fused.grad).all())
        self.assertTrue((fused.grad.abs().sum((0, 2, 3)) > 0).all())

    def test_validation_is_read_only_and_batch_invariant(self):
        model = FusionNet(8)
        dataset = TensorDataset(torch.rand(3, 8, 2, 2), torch.rand(3, 1, 8, 8), torch.rand(3, 8, 8, 8))
        before = {key: value.clone() for key, value in model.state_dict().items()}
        modes = []
        handle = model.register_forward_pre_hook(lambda module, _: modes.append((module.training, torch.is_grad_enabled())))
        self.addCleanup(handle.remove)
        first = validate(model, DataLoader(dataset, batch_size=1), FusionLoss(), torch.device("cpu"))
        second = validate(model, DataLoader(dataset, batch_size=2), FusionLoss(), torch.device("cpu"))
        self.assertTrue(model.training)
        self.assertTrue(all(mode == (False, False) for mode in modes))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        for name in first:
            self.assertAlmostEqual(first[name], second[name], places=5)

    def test_empty_validation_is_rejected(self):
        dataset = TensorDataset(torch.empty(0, 4, 2, 2), torch.empty(0, 1, 8, 8), torch.empty(0, 4, 8, 8))
        with self.assertRaisesRegex(ValueError, "empty dataset"):
            validate(FusionNet(), DataLoader(dataset), FusionLoss(), torch.device("cpu"))

    def test_training_checkpoint_validation_and_resume(self):
        train_dir, val_dir = self.write_split("train"), self.write_split("val")
        output = self.root / "run"
        args = ["--train-dir", str(train_dir), "--val-dir", str(val_dir), "--output-dir", str(output),
                "--num-channels", "8", "--batch-size", "2", "--device", "cpu", "--save-every", "1"]
        with contextlib.redirect_stdout(io.StringIO()):
            history = train_main(args + ["--epochs", "2"])
            best = load_checkpoint(output / "best.pth", "cpu")
            self.assertEqual(best["val_metrics"]["loss"], min(row["val_loss"] for row in history))
            standalone = validate_main(["--checkpoint", str(output / "best.pth"), "--val-dir", str(val_dir),
                                        "--batch-size", "1", "--output", str(output / "validation.json")])
            self.assertAlmostEqual(standalone["loss"], best["val_metrics"]["loss"], places=5)
            history = train_main(args + ["--epochs", "3", "--resume", str(output / "last.pth")])
        last = load_checkpoint(output / "last.pth", "cpu")
        self.assertEqual(set(last["val_metrics"]), {"loss", "spectral", "spatial"})
        self.assertEqual(set(last["train_metrics"]), {"loss", "spectral", "spatial"})
        self.assertEqual(last["epoch"], 3)
        self.assertEqual([row["epoch"] for row in history], [1, 2, 3])
        self.assertTrue((output / "epoch_0003.pth").exists())
        self.assertTrue((output / "validation.json").exists())
        with (output / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 3)
        earlier = load_checkpoint(output / "epoch_0001.pth", "cpu")
        self.assertTrue(any(not torch.equal(value, earlier["model_state_dict"][key])
                            for key, value in last["model_state_dict"].items()))

if __name__ == "__main__":
    unittest.main()
