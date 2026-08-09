from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def create_session(model_path: str | Path, threads: int = 4):
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(threads)
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])


def validate_onnx_file(model_path: str | Path) -> None:
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)


def compare_outputs(torch_output, onnx_path: str | Path, input_array: np.ndarray, atol: float = 1e-4, rtol: float = 1e-3):
    session = create_session(onnx_path)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_array})[0]
    expected = torch_output.detach().cpu().numpy()
    close = bool(np.allclose(expected, output, atol=atol, rtol=rtol))
    maximum = float(np.max(np.abs(expected - output)))
    if not close:
        raise RuntimeError(f"PyTorch/ONNX output mismatch; max absolute error={maximum}")
    return {"allclose": close, "maximum_absolute_error": maximum, "atol": atol, "rtol": rtol}


def load_deployment_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
