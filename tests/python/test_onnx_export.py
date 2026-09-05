"""Unit tests for ONNX model export, quantization, metadata schema, and numerical parity."""

import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from training.data.scalers import RollingZScoreScaler
from training.models.export_onnx import (
    ONNXExporter,
    export_model_package,
)
from training.models.forecaster import ClowForecaster


@pytest.fixture
def small_forecaster():
    """Create a small deterministic ClowForecaster model for export testing."""
    torch.manual_seed(42)
    model = ClowForecaster(
        input_dim=8,
        d_model=32,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        quantiles=(0.10, 0.50, 0.90),
    )
    model.eval()
    return model


def test_export_forecaster_onnx_valid(small_forecaster):
    """Test exporting forecaster to ONNX and validating model structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "forecaster.onnx"
        res_path = ONNXExporter.export_forecaster(
            model=small_forecaster,
            output_path=out_path,
            input_dim=8,
            context_length=32,
            batch_size=1,
            dynamic_axes=True,
        )
        assert res_path.exists()
        assert res_path.stat().st_size > 0

        # Load and check with onnx
        onnx_model = onnx.load(str(res_path))
        onnx.checker.check_model(onnx_model)

        input_names = [inp.name for inp in onnx_model.graph.input]
        output_names = [out.name for out in onnx_model.graph.output]
        assert "context_features" in input_names
        assert "anatomy_pred" in output_names
        assert "direction_logit" in output_names
        assert "direction_prob" in output_names
        assert "quantiles_high" in output_names
        assert "quantiles_low" in output_names


def test_export_numerical_parity_single_and_batch(small_forecaster):
    """Test numerical equivalence between PyTorch and ONNX Runtime within 1e-4 tolerance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "forecaster.onnx"
        ONNXExporter.export_forecaster(
            model=small_forecaster,
            output_path=out_path,
            input_dim=8,
            context_length=32,
            dynamic_axes=True,
        )

        # Verify batch=1
        errors_b1 = ONNXExporter.verify_numerical_parity(
            torch_model=small_forecaster,
            onnx_path=out_path,
            input_dim=8,
            context_length=32,
            batch_size=1,
            tolerance=1e-4,
        )
        for out_name, err in errors_b1.items():
            assert err < 1e-4, f"Batch 1 error {err} too high for {out_name}"

        # Verify batch=4
        errors_b4 = ONNXExporter.verify_numerical_parity(
            torch_model=small_forecaster,
            onnx_path=out_path,
            input_dim=8,
            context_length=32,
            batch_size=4,
            tolerance=1e-4,
        )
        for out_name, err in errors_b4.items():
            assert err < 1e-4, f"Batch 4 error {err} too high for {out_name}"


def test_quantization_dynamic_int8(small_forecaster):
    """Test dynamic INT8 quantization on exported ONNX model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fp32_path = Path(tmpdir) / "model.onnx"
        ONNXExporter.export_forecaster(
            model=small_forecaster,
            output_path=fp32_path,
            input_dim=8,
            context_length=32,
        )

        int8_path = ONNXExporter.quantize_model(fp32_path)
        assert int8_path.exists()

        # Run inference on quantized model
        session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
        test_input = np.random.randn(1, 32, 8).astype(np.float32)
        outputs = session.run(None, {"context_features": test_input})
        assert len(outputs) == 5
        # Verify shapes
        assert outputs[0].shape == (1, 4)  # anatomy_pred
        assert outputs[1].shape == (1, 1)  # direction_logit
        assert outputs[2].shape == (1, 1)  # direction_prob
        assert outputs[3].shape == (1, 3)  # quantiles_high
        assert outputs[4].shape == (1, 3)  # quantiles_low


def test_export_model_package_and_metadata(small_forecaster):
    """Test full model packaging with manifest and JSON schema serialization."""
    feature_names = [f"feat_{i}" for i in range(8)]
    scaler = RollingZScoreScaler(window=32)

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = export_model_package(
            model=small_forecaster,
            feature_names=feature_names,
            output_dir=tmpdir,
            model_id="test_forecaster",
            context_length=32,
            scaler=scaler,
            quantize=True,
        )

        assert Path(manifest["onnx_path"]).exists()
        assert Path(manifest["quantized_onnx_path"]).exists()
        assert Path(manifest["metadata_path"]).exists()
        assert len(manifest["onnx_sha256"]) == 64
        assert len(manifest["metadata_sha256"]) == 64

        with open(manifest["metadata_path"]) as f:
            meta = json.load(f)

        assert meta["model_id"] == "test_forecaster"
        assert meta["input_dim"] == 8
        assert meta["feature_names"] == feature_names
        assert meta["quantiles"] == [0.10, 0.50, 0.90]
        assert meta["input_schema"]["name"] == "context_features"
        assert meta["architecture"]["d_model"] == 32
