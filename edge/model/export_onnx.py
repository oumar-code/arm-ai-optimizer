"""
export_onnx.py
Exports the trained PyTorch LSTM and Isolation Forest models to ONNX
format and then applies INT8 post-training quantisation via ONNX Runtime.

Also exports the LSTM as FP16 for latency comparison benchmarks.

Usage:
    python export_onnx.py [--model-dir /app/model]
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("export-onnx")

WINDOW_LEN = 60
N_FEATURES = 21
LSTM_HIDDEN = 64
LSTM_LAYERS = 2

# ── Re-define model architecture (must match train.py) ────────────────────────
import torch.nn as nn


class RULPredictor(nn.Module):
    def __init__(self, input_size: int, hidden: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)
# ─────────────────────────────────────────────────────────────────────────────


def export_lstm(model_dir: Path) -> None:
    pt_path = model_dir / "lstm_rul.pt"
    if not pt_path.exists():
        log.warning("lstm_rul.pt not found — skipping LSTM export")
        return

    model = RULPredictor(N_FEATURES, LSTM_HIDDEN, LSTM_LAYERS)
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, WINDOW_LEN, N_FEATURES)
    onnx_fp32_path = str(model_dir / "lstm_rul_fp32.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_fp32_path,
        export_params=True,
        opset_version=17,
        input_names=["features"],
        output_names=["rul_hours"],
        dynamic_axes={"features": {0: "batch_size"}},
        do_constant_folding=True,
    )
    log.info("Exported LSTM FP32 → %s", onnx_fp32_path)

    # Verify the exported model
    onnx_model = onnx.load(onnx_fp32_path)
    onnx.checker.check_model(onnx_model)
    log.info("ONNX model verified ✓")

    # INT8 dynamic quantisation (works on Arm64 without calibration data)
    onnx_int8_path = str(model_dir / "lstm_rul_int8.onnx")
    quantize_dynamic(
        onnx_fp32_path,
        onnx_int8_path,
        weight_type=QuantType.QInt8,
    )
    log.info("Quantised LSTM INT8 → %s", onnx_int8_path)

    # Quick inference sanity check
    sess = onnxruntime.InferenceSession(
        onnx_int8_path,
        providers=["CPUExecutionProvider"],
    )
    out = sess.run(None, {"features": dummy_input.numpy()})
    log.info("INT8 sanity check — predicted RUL: %.2f h", out[0][0])

    # Create a symlink so the inference service always loads "pdm_model.onnx"
    link = model_dir / "pdm_model.onnx"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to("lstm_rul_int8.onnx")
    log.info("Symlinked pdm_model.onnx → lstm_rul_int8.onnx")


def export_isolation_forest_onnx(model_dir: Path) -> None:
    """
    Export Isolation Forest to ONNX via skl2onnx.
    Falls back gracefully if skl2onnx is not installed.
    """
    if_path = model_dir / "isolation_forest.pkl"
    if not if_path.exists():
        log.warning("isolation_forest.pkl not found — skipping IF export")
        return

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        log.warning(
            "skl2onnx not installed — Isolation Forest will run via sklearn at inference time"
        )
        return

    with open(if_path, "rb") as fh:
        clf = pickle.load(fh)

    initial_type = [("float_input", FloatTensorType([None, N_FEATURES]))]
    onnx_model = convert_sklearn(clf, initial_types=initial_type, target_opset=17)
    out_path = str(model_dir / "isolation_forest.onnx")
    with open(out_path, "wb") as fh:
        fh.write(onnx_model.SerializeToString())
    log.info("Isolation Forest exported → %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/app/model", help="Directory with trained models")
    args = parser.parse_args()
    model_dir = Path(args.model_dir)

    export_lstm(model_dir)
    export_isolation_forest_onnx(model_dir)
    log.info("Export complete.")


if __name__ == "__main__":
    main()
