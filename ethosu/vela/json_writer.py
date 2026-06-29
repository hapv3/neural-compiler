# SPDX-FileCopyrightText: Copyright 2021, 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Description:
# Functions used to write to a json format (.json) file.
import base64
import json

import numpy as np

from .high_level_command_to_npu_op import get_region
from .nn_graph import PassPlacement
from .operation import Op


def _to_list(val):
    if val is None:
        return []
    if isinstance(val, np.ndarray):
        return val.flatten().tolist()
    if isinstance(val, (list, tuple)):
        return [float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v for v in val]
    return val


def _base64_encode(data):
    if isinstance(data, np.ndarray):
        data = data.tobytes()
    return base64.b64encode(data).decode("ascii")

 
def _quantization_entry(quant_type, scale, zero_point):
    qt = (quant_type or "").lower()
    if qt in ("", "none"):
        return {
            "quant_type": "none",
            "scale": [],
            "zero_point": []
        }

    scale_arr = np.array(scale, dtype=np.float32).flatten()
    zero_arr = np.array(zero_point, dtype=np.int32).flatten()

    if qt == "per_channel":
        return {"quant_type": qt, "scale": scale_arr.tolist(), "zero_point": zero_arr.tolist()}

    if scale_arr.size == 0 or zero_arr.size == 0:
        return {
            "quant_type": "none",
            "scale": [],
            "zero_point": []
        }

    return {
        "quant_type": qt,
        "scale": [float(scale_arr[0])],
        "zero_point": [int(zero_arr[0])]
    }


def _quantization_entry_from_tensor(tensor):
    if hasattr(tensor, "quant_type"):
        quant_type = getattr(tensor, "quant_type", "none") or "none"
        return _quantization_entry(quant_type, getattr(tensor, "scale", None), getattr(tensor, "zero_point", None))

    quant = getattr(tensor, "quantization", None)
    if quant is None or not quant.is_valid():
        return _quantization_entry("none", None, None)

    quant_type = "per_channel" if quant.is_per_axis() else "per_tensor"
    return _quantization_entry(quant_type, quant.scale_f32, quant.zero_point)


def write_rawdata_output(nng, arch, filename):
    subgraphs_to_write = [sg for sg in nng.subgraphs if sg.placement == PassPlacement.Cpu]

    for sg_idx, sg in enumerate(subgraphs_to_write):
        custom_op = None
        for ps in sg.passes:
            for op in ps.ops:
                if op.type == Op.CustomNpuOp:
                    custom_op = op
                    break
            if custom_op:
                break

        if custom_op:
            ifm_shapes = []
            ifm_elem_sizes = []
            ifm_regions = []
            ifm_offsets = []
            ifm_quantization = []
            ofm_shapes = []
            ofm_elem_sizes = []
            ofm_regions = []
            ofm_offsets = []
            ofm_quantization = []
            cmd_stream_tensor, weight_tensor, scratch_tensor, scratch_fast_tensor = custom_op.inputs[:4]
            weight_region = get_region(weight_tensor.mem_type, arch)
            scratch_region = get_region(scratch_tensor.mem_type, arch)
            scratch_fast_region = get_region(scratch_fast_tensor.mem_type, arch)
            for ifm in custom_op.inputs[4:]:
                ifm_shapes.append(ifm.get_full_shape())
                ifm_regions.append(get_region(ifm.mem_type, arch))
                ifm_offsets.append(ifm.address)
                ifm_elem_sizes.append(ifm.element_size())
                ifm_quantization.append(_quantization_entry_from_tensor(ifm))
            for ofm in custom_op.outputs:
                ofm_shapes.append(ofm.get_full_shape())
                ofm_regions.append(get_region(ofm.mem_type, arch))
                ofm_offsets.append(ofm.address)
                ofm_elem_sizes.append(ofm.element_size())
                ofm_quantization.append(_quantization_entry_from_tensor(ofm))

            data = {
                "cmd_data": _base64_encode(cmd_stream_tensor.values),
                "weight_data": _base64_encode(weight_tensor.values),
                "weight_region": int(weight_region),
                "scratch_shape": _to_list(scratch_tensor.shape),
                "scratch_region": int(scratch_region),
                "scratch_size": int(scratch_tensor.shape[0]),
                "scratch_fast_shape": _to_list(scratch_fast_tensor.shape),
                "scratch_fast_region": int(scratch_fast_region),
                "scratch_fast_size": int(scratch_fast_tensor.shape[0]),
                "input_shape": [_to_list(s) for s in ifm_shapes],
                "input_elem_size": ifm_elem_sizes,
                "input_region": ifm_regions,
                "input_offset": ifm_offsets,
                "output_shape": [_to_list(s) for s in ofm_shapes],
                "output_elem_size": ofm_elem_sizes,
                "output_region": ofm_regions,
                "output_offset": ofm_offsets,
                "variable_shape": [],
                "variable_elem_size": [],
                "variable_region": [],
                "variable_offset": [],
                "input_quantization": ifm_quantization,
                "output_quantization": ofm_quantization,
                "variable_quantization": [],
            }

            filename_sg = f"{filename}_vela_sg{sg_idx}.json"
            with open(filename_sg, "w") as f:
                json.dump(data, f)


def write_rawdata_output_from_model(filename, model):
    data = {
        "cmd_data": _base64_encode(model.command_stream),
        "weight_data": _base64_encode(model.read_only.data),
        "weight_region": int(model.read_only.region),
        "scratch_shape": [int(model.scratch.size)],
        "scratch_region": int(model.scratch.region),
        "scratch_size": int(model.scratch.size),
        "scratch_fast_shape": [int(model.scratch_fast.size)],
        "scratch_fast_region": int(model.scratch_fast.region),
        "scratch_fast_size": int(model.scratch_fast.size),
        "input_shape": [list(t.shape) for t in model.inputs],
        "input_elem_size": [int(t.element_size) for t in model.inputs],
        "input_region": [int(t.region) for t in model.inputs],
        "input_offset": [int(t.address) for t in model.inputs],
        "output_shape": [list(t.shape) for t in model.outputs],
        "output_elem_size": [int(t.element_size) for t in model.outputs],
        "output_region": [int(t.region) for t in model.outputs],
        "output_offset": [int(t.address) for t in model.outputs],
        "variable_shape": [list(t.shape) for t in model.variables],
        "variable_elem_size": [int(t.element_size) for t in model.variables],
        "variable_region": [int(t.region) for t in model.variables],
        "variable_offset": [int(t.address) for t in model.variables],
        "input_quantization": [_quantization_entry_from_tensor(t) for t in model.inputs],
        "output_quantization": [_quantization_entry_from_tensor(t) for t in model.outputs],
        "variable_quantization": [_quantization_entry_from_tensor(t) for t in model.variables],
    }

    filename_sg = f"{filename}_vela.json"
    with open(filename_sg, "w") as f:
        json.dump(data, f)
