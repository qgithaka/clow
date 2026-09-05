"""ONNX Model Export and Optimization Engine for Clow.

Converts trained PyTorch foundation models into optimized .onnx binaries
with dynamic batching, dynamic quantization, graph optimization,
and metadata schema packaging for sub-5ms native C++ execution.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from training.data.scalers import RollingZScoreScaler, RollingRobustScaler, RollingMinMaxScaler
from training.models.forecaster import ClowForecaster

logger = logging.getLogger("clow.models.export_onnx")


class ONNXExporter:
    """Exports and optimizes PyTorch models for high-performance ONNX Runtime inference."""

    @staticmethod
    def export_forecaster(
        model: ClowForecaster,
        output_path: Union[str, Path],
        input_dim: int,
        context_length: int = 64,
        batch_size: int = 1,
        dynamic_axes: bool = True,
        opset_version: int = 17,
    ) -> Path:
        """Exports a ClowForecaster PyTorch model to ONNX format.
        
        Args:
            model: Instantiated ClowForecaster module.
            output_path: Destination file path for .onnx binary.
            input_dim: Number of input stationary features per bar.
            context_length: Sliding window sequence length (default 64).
            batch_size: Default export batch size (default 1).
            dynamic_axes: Whether to allow variable batch sizes at runtime.
            opset_version: ONNX operator set version (default 17).
            
        Returns:
            Path to exported ONNX model.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.eval()

        dummy_input = torch.randn(batch_size, context_length, input_dim, dtype=torch.float32)

        input_names = ["context_features"]
        output_names = [
            "anatomy_pred",
            "direction_logit",
            "direction_prob",
            "quantiles_high",
            "quantiles_low",
        ]

        dynamic_axes_cfg = None
        if dynamic_axes:
            dynamic_axes_cfg = {
                "context_features": {0: "batch_size"},
                "anatomy_pred": {0: "batch_size"},
                "direction_logit": {0: "batch_size"},
                "direction_prob": {0: "batch_size"},
                "quantiles_high": {0: "batch_size"},
                "quantiles_low": {0: "batch_size"},
            }

        # Export torch model to ONNX using legacy TorchScript engine for stable cross-platform serialization
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes_cfg,
                dynamo=False,
            )
        except TypeError:
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes_cfg,
            )

        # Check and validate exported ONNX model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info(f"Successfully exported and validated ONNX model to {output_path}")

        return output_path

    @staticmethod
    def quantize_model(
        input_onnx_path: Union[str, Path],
        output_onnx_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Applies dynamic INT8 quantization to an ONNX model for CPU acceleration.
        
        Args:
            input_onnx_path: Path to FP32 ONNX model.
            output_onnx_path: Destination path for INT8 quantized model.
            
        Returns:
            Path to quantized ONNX model.
        """
        from onnxruntime.quantization import QuantType, quantize_dynamic

        input_onnx_path = Path(input_onnx_path)
        if output_onnx_path is None:
            output_onnx_path = input_onnx_path.with_name(
                input_onnx_path.stem + "_quantized.onnx"
            )
        else:
            output_onnx_path = Path(output_onnx_path)

        quantize_dynamic(
            model_input=str(input_onnx_path),
            model_output=str(output_onnx_path),
            weight_type=QuantType.QInt8,
        )
        logger.info(f"Quantized model saved to {output_onnx_path}")
        return output_onnx_path

    @staticmethod
    def verify_numerical_parity(
        torch_model: ClowForecaster,
        onnx_path: Union[str, Path],
        input_dim: int,
        context_length: int = 64,
        batch_size: int = 1,
        tolerance: float = 1e-4,
    ) -> Dict[str, float]:
        """Verifies output numerical equivalence between PyTorch and ONNX Runtime.
        
        Args:
            torch_model: Source PyTorch model.
            onnx_path: Path to exported ONNX model.
            input_dim: Feature dimension.
            context_length: Sequence length.
            batch_size: Test batch size.
            tolerance: Maximum permissible absolute difference.
            
        Returns:
            Dictionary mapping output tensor names to max absolute errors.
        """
        torch_model.eval()
        np.random.seed(42)
        sample_np = np.random.randn(batch_size, context_length, input_dim).astype(np.float32)
        sample_torch = torch.from_numpy(sample_np)

        # 1. PyTorch inference
        with torch.no_grad():
            torch_outputs = torch_model(sample_torch)

        # 2. ONNX Runtime inference
        ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_inputs = {"context_features": sample_np}
        ort_outputs = ort_session.run(None, ort_inputs)
        output_names = [o.name for o in ort_session.get_outputs()]

        max_errors = {}
        for name, ort_val in zip(output_names, ort_outputs):
            torch_val = torch_outputs[name].detach().cpu().numpy()
            diff = np.max(np.abs(torch_val - ort_val))
            max_errors[name] = float(diff)
            if diff > tolerance:
                raise AssertionError(
                    f"Numerical disparity exceeded tolerance for output '{name}': "
                    f"max error {diff:.6f} > tolerance {tolerance:.6f}"
                )

        logger.info(f"Numerical verification passed. Max errors: {max_errors}")
        return max_errors


def create_model_metadata(
    model_id: str,
    feature_names: List[str],
    context_length: int,
    quantiles: Sequence[float],
    scaler: Optional[Union[RollingZScoreScaler, RollingRobustScaler, RollingMinMaxScaler]] = None,
    architecture_params: Optional[Dict[str, Any]] = None,
    description: str = "Clow Next-Candle Transformer Forecaster",
) -> Dict[str, Any]:
    """Builds complete serialization metadata schema for C++ deployment.
    
    Args:
        model_id: Unique model identifier (e.g. 'clow_forecaster_v1').
        feature_names: Exact list of feature names matching model input tensor columns.
        context_length: Number of time-series bars in sliding window context.
        quantiles: Quantile probability levels predicted (e.g. [0.10, 0.50, 0.90]).
        scaler: Scaler instance with fitted parameters for streaming normalizer.
        architecture_params: Optional dict of neural architecture hyperparameters.
        description: Description of model function.
        
    Returns:
        Structured metadata dictionary ready for JSON serialization.
    """
    scaler_info = {}
    if scaler is not None:
        win = getattr(scaler, "window", getattr(scaler, "window_size", 64))
        scaler_info = {
            "scaler_type": type(scaler).__name__,
            "window_size": win,
            "clip_val": getattr(scaler, "clip_val", None),
            "eps": getattr(scaler, "eps", 1e-8),
            "feature_names": getattr(scaler, "feature_names", feature_names),
        }
        if hasattr(scaler, "feature_means") and scaler.feature_means is not None:
            scaler_info["feature_means"] = {k: float(v) for k, v in scaler.feature_means.items()}
        if hasattr(scaler, "feature_stds") and scaler.feature_stds is not None:
            scaler_info["feature_stds"] = {k: float(v) for k, v in scaler.feature_stds.items()}
        if hasattr(scaler, "feature_mins") and scaler.feature_mins is not None:
            scaler_info["feature_mins"] = {k: float(v) for k, v in scaler.feature_mins.items()}
        if hasattr(scaler, "feature_maxs") and scaler.feature_maxs is not None:
            scaler_info["feature_maxs"] = {k: float(v) for k, v in scaler.feature_maxs.items()}
        if hasattr(scaler, "feature_medians") and scaler.feature_medians is not None:
            scaler_info["feature_medians"] = {k: float(v) for k, v in scaler.feature_medians.items()}
        if hasattr(scaler, "feature_iqrs") and scaler.feature_iqrs is not None:
            scaler_info["feature_iqrs"] = {k: float(v) for k, v in scaler.feature_iqrs.items()}

    metadata = {
        "model_id": model_id,
        "description": description,
        "format": "ONNX",
        "context_length": context_length,
        "input_dim": len(feature_names),
        "feature_names": feature_names,
        "quantiles": list(quantiles),
        "num_anatomy_targets": 4,
        "anatomy_target_names": ["body_ratio", "upper_wick", "lower_wick", "range_to_atr"],
        "input_schema": {
            "name": "context_features",
            "shape": ["batch_size", context_length, len(feature_names)],
            "dtype": "float32",
        },
        "output_schema": {
            "anatomy_pred": {"shape": ["batch_size", 4], "dtype": "float32"},
            "direction_logit": {"shape": ["batch_size", 1], "dtype": "float32"},
            "direction_prob": {"shape": ["batch_size", 1], "dtype": "float32"},
            "quantiles_high": {"shape": ["batch_size", len(quantiles)], "dtype": "float32"},
            "quantiles_low": {"shape": ["batch_size", len(quantiles)], "dtype": "float32"},
        },
        "scaler": scaler_info,
        "architecture": architecture_params or {},
    }
    return metadata


def export_model_package(
    model: ClowForecaster,
    feature_names: List[str],
    output_dir: Union[str, Path],
    model_id: str = "clow_forecaster_v1",
    context_length: int = 64,
    scaler: Optional[Any] = None,
    quantize: bool = True,
) -> Dict[str, Any]:
    """Packages model into complete production container with ONNX binary and metadata.
    
    Args:
        model: Trained ClowForecaster instance.
        feature_names: List of input feature names in order.
        output_dir: Target deployment directory.
        model_id: Unique model identifier.
        context_length: Sequence length.
        scaler: Optional fitted scaler for normalizer parameters.
        quantize: Whether to also create an INT8 quantized ONNX binary.
        
    Returns:
        Manifest dictionary with paths and SHA-256 checksums.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export FP32 ONNX
    onnx_path = output_dir / f"{model_id}.onnx"
    ONNXExporter.export_forecaster(
        model=model,
        output_path=onnx_path,
        input_dim=len(feature_names),
        context_length=context_length,
    )

    # 2. Checksum of FP32 ONNX
    with open(onnx_path, "rb") as f:
        onnx_sha256 = hashlib.sha256(f.read()).hexdigest()

    quantized_path = None
    quantized_sha256 = None
    if quantize:
        quantized_path = ONNXExporter.quantize_model(onnx_path)
        with open(quantized_path, "rb") as f:
            quantized_sha256 = hashlib.sha256(f.read()).hexdigest()

    # 3. Create and save metadata JSON
    arch_params = {
        "d_model": model.d_model,
        "nhead": model.nhead,
        "num_layers": model.num_layers,
        "dim_feedforward": model.dim_feedforward,
    }
    metadata = create_model_metadata(
        model_id=model_id,
        feature_names=feature_names,
        context_length=context_length,
        quantiles=model.quantiles,
        scaler=scaler,
        architecture_params=arch_params,
    )
    metadata["onnx_sha256"] = onnx_sha256
    if quantized_sha256:
        metadata["quantized_onnx_sha256"] = quantized_sha256

    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(metadata_path, "rb") as f:
        meta_sha256 = hashlib.sha256(f.read()).hexdigest()

    manifest = {
        "model_id": model_id,
        "onnx_path": str(onnx_path),
        "onnx_sha256": onnx_sha256,
        "quantized_onnx_path": str(quantized_path) if quantized_path else None,
        "quantized_onnx_sha256": quantized_sha256,
        "metadata_path": str(metadata_path),
        "metadata_sha256": meta_sha256,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Model package successfully generated in {output_dir}")
    return manifest
