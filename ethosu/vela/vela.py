#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2020-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
"""Compile a neural network model for Arm Ethos-U NPUs."""
import argparse
import glob
import mmap
import os
import sys
import time
from configparser import ConfigParser
from typing import List
from typing import Optional

import flatbuffers

from . import architecture_features
from . import compiler_driver
from . import model_reader
from . import rawdata_writer
from . import scheduler
from . import stats_writer
from . import tflite_writer
from ._version import __version__
from .api import API_VERSION
from .debug_database import DebugDatabase
from .errors import InputFileError
from .errors import VelaError
from .hillclimb_allocation import HillClimbAllocator
from .nn_graph import NetworkType
from .nn_graph import TensorAllocator
from .tensor import MemArea
from .tensor import Tensor
from .tflite.Model import Model
from .tflite_mapping import builtin_operator_map
from .tflite_mapping import builtin_operator_name_map
from .tflite_mapping import optype_to_builtintype
from .tflite_model_semantic import TFLiteSemantic
from .tflite_supported_operators import TFLiteSupportedOperators
from .tosa_model_semantic import TosaSemantic
from .tosa_supported_operators import TosaSupportedOperators
from ethosu import regor

TFLITE_MAGIC = 0x334C4654
TOSA_MAGIC = 0x41534F54


def process(
    input_name, enable_debug_db, arch, model_reader_options, compiler_options, scheduler_options, output_format
):
    if compiler_options.timing:
        start = time.time()

    os.makedirs(compiler_options.output_dir, exist_ok=True)
    output_basename = os.path.join(compiler_options.output_dir, os.path.splitext(os.path.basename(input_name))[0])
    DebugDatabase.show_warnings = enable_debug_db

    nng, network_type = model_reader.read_model(input_name, model_reader_options)

    if not nng:
        raise InputFileError(input_name, "Input file could not be read")

    if compiler_options.verbose_operators:
        nng.print_operators()

    if compiler_options.timing:
        stop = time.time()
        print("Model reading took %f s" % (stop - start))
        start = time.time()

    compiler_driver.compiler_driver(nng, arch, compiler_options, scheduler_options, network_type, output_basename)

    summary_csv_file = "{0}_summary_{1}.csv".format(output_basename, arch.system_config)
    stats_writer.write_summary_metrics_csv(nng, summary_csv_file, arch)

    stats_writer.print_performance_metrics(
        nng,
        show_cpu_operations=compiler_options.show_cpu_operations,
        verbose_weights=compiler_options.verbose_weights,
        verbose_cycle_estimate=compiler_options.verbose_cycle_estimate,
        arch=arch,
    )

    output_tfl_filename = output_basename + "_vela.tflite"
    if output_format == "tflite":
        tflite_writer.write_tflite(nng, output_tfl_filename)
    elif output_format == "raw":
        rawdata_writer.write_rawdata_output(nng, arch, output_basename)
    else:
        assert False, f"Unsupported output_format = {output_format}"

    if enable_debug_db:
        file_offsets = calculate_operator_file_offsets(output_tfl_filename)
        for idx, offset in enumerate(sorted(file_offsets)):
            sg = find_subgraph_with_command_stream_order(nng, idx)
            if sg is not None:
                DebugDatabase.set_stream_offset(sg, offset)
        debug_filename = output_basename + "_debug.xml"
        DebugDatabase.write(debug_filename, input_name, output_tfl_filename)

    if compiler_options.timing:
        stop = time.time()
        print("Compiler driver took %f s" % (stop - start))

    return nng


def process_regor(
    arch,
    input_name,
    accelerator,
    system_config,
    options,
    enable_debug_db,
    output_dir,
    output_format,
    verbose_weights=False,
    verbose_cycle_estimate=False,
    show_cpu_operations=False,
    verbose_performance=False,
):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_name, "rb") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as network:
            fmt = get_format(network)
            compiled_model = regor.compile(accelerator, network, fmt, system_config, options=options, verbose=True)

    model_name = os.path.splitext(os.path.basename(input_name))[0]

    output_basename = os.path.join(output_dir, model_name)

    if output_format == "tflite":
        assert isinstance(compiled_model, regor.CompiledTFLiteModel)
        output_name = output_basename + "_vela.tflite"
        with open(output_name, "wb") as f:
            f.write(compiled_model.model)
    elif output_format == "raw":
        assert isinstance(compiled_model, regor.CompiledRawModel)
        rawdata_writer.write_rawdata_output_from_model(output_basename, compiled_model)
    else:
        assert False, f"Unsupported output_format = {output_format}"

    summary_csv_file = "{0}_summary_{1}.csv".format(output_basename, arch.system_config)

    if verbose_performance:
        stats_writer.write_regor_perlayer_performance_csv(
            arch, compiled_model.opt_database, compiled_model.perf_report, output_basename
        )
        stats_writer.print_regor_perlayer_performance(
            arch, compiled_model.opt_database, compiled_model.perf_report, output_basename
        )

    stats_writer.print_regor_performance_metrics(
        arch,
        compiled_model.perf_report,
        model_name,
        summary_csv_file,
        compiled_model.opt_database,
        verbose_weights,
        verbose_cycle_estimate,
        show_cpu_operations,
    )
    if enable_debug_db:
        stats_writer.write_regor_db(compiled_model.opt_database, output_basename)


def find_subgraph_with_command_stream_order(nng, idx):
    for sg in nng.subgraphs:
        if sg.generated_stream_id == idx:
            return sg
    return None


def calculate_operator_file_offsets(name: str):
    # Read the vela optimized TFLite file
    with open(name, "rb") as f:
        buf = bytearray(f.read())
    # Calculate the file offsets for each custom operator
    file_offsets = []
    model = Model.GetRootAsModel(buf, 0)
    for idx in range(model.SubgraphsLength()):  # However only one subgraph is supported as of now
        sg = model.Subgraphs(idx)
        for idx in range(sg.OperatorsLength()):
            operator = sg.Operators(idx)
            if model.OperatorCodes(operator.OpcodeIndex()).CustomCode() is not None:
                tensor_idx = operator.Inputs(0)
                tensor = sg.Tensors(tensor_idx)
                buffer = model.Buffers(tensor.Buffer())
                offset = flatbuffers.number_types.UOffsetTFlags.py_type(buffer._tab.Offset(4))
                file_offsets.append(buffer._tab.Vector(offset))
    return file_offsets


def print_subgraph_io_summary(nng):
    """Print a summary of all the input and output tensor sizes for all subgraphs.
    Also displays the total tensor size and the memory used area for sram.
    """

    print("Subgraph IO Summary")
    print("-------------------")
    print(f"NNG: {nng.name}")
    max_sg_size = 0
    for sg in reversed(nng.subgraphs):
        print(f"  NNG Subgraph: {sg.name} = {sg.placement}")
        sg_size = 0

        if hasattr(sg, "scratch_tensor") and sg.scratch_tensor is not None:
            sg_tensors = sg.input_tensors + [sg.scratch_tensor] + sg.output_tensors
        else:
            sg_tensors = sg.input_tensors + sg.output_tensors

        for tens in sg_tensors:
            if tens in sg.input_tensors:
                tens_dir = "In"
            elif tens in sg.output_tensors:
                tens_dir = "Out"
            else:
                tens_dir = "In/Out"

            size = tens.elements() * tens.element_size() / 1024.0
            sg_size = sg_size + size
            print(f"         Tensor [{tens_dir}]: {tens.name} = {size} KiB")

        print(f"      Total Size = {sg_size} KiB")
        print(f"      SRAM Memory Used = {sg.memory_used.get(MemArea.Sram, 0) / 1024.0} KiB")
        max_sg_size = max(sg_size, max_sg_size)

    print(f"   Maximum NNG Subgraph Size = {max_sg_size} KiB")


def generate_supported_ops():
    # Exclude network type from generation by adding value to exclude list.
    # To easily exclude NetworkType from generated documentation.
    exclude_generation_network_type_value = [NetworkType.TOSA.value]

    def _exclude_list_names(constraint, exclude_list):
        constraints_excluded_names = [
            optype_to_builtintype(op) for op, exclude_constraint in exclude_list if constraint in exclude_constraint
        ]
        return f" - [{', '.join(sorted(constraints_excluded_names))}]" if constraints_excluded_names else ""

    def _regor_generic_constraints(supported_ops):
        # extract generic constraints from Regor supported-ops
        # A generic constraint applies to all opTypes except a list of exceptions
        # a constraint is considered generic if it applies to more than 80% of the opTypes
        threshold = 0.20 * len(supported_ops)
        allconstraints = set()
        exceptions = dict()
        for op in supported_ops:
            for constraint in supported_ops[op]:
                allconstraints.add(constraint)
                exceptions[constraint] = list()
        for op in supported_ops:
            for constraint in [c for c in allconstraints if c not in supported_ops[op]]:
                exceptions[constraint].append(op)
        return {k: v for k, v in exceptions.items() if len(v) < threshold}

    def _regor_specific_constraints(supported_ops, generic_constraints):
        # extract specific constraints from Regor supported-ops
        # A constraint is considered op-specific if it's not in the generic constraints
        specific_constraints = dict()
        # create op-specific constraints
        for op in sorted(supported_ops):
            constraints = supported_ops[op]
            # Regor renames ARG_MAX to avoid name-collisions
            op = op.replace("ARGMAX", "ARG_MAX")
            specific_constraints[op] = sorted(filter(lambda c: c not in generic_constraints, constraints))
        return specific_constraints

    supported_ops_u85 = regor.tflite_operator_constraints("EthosU85")
    u85_generic = _regor_generic_constraints(supported_ops_u85)
    u85_specific = _regor_specific_constraints(supported_ops_u85, u85_generic)

    # Add license for supported ops
    lines = [
        "<!--",
        "SPDX-FileCopyrightText: Copyright 2020-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>",
        "",
        "SPDX-License-Identifier: Apache-2.0",
        "",
        "Licensed under the Apache License, Version 2.0 (the License); you may",
        "not use this file except in compliance with the License.",
        "You may obtain a copy of the License at",
        "",
        "www.apache.org/licenses/LICENSE-2.0",
        "",
        "Unless required by applicable law or agreed to in writing, software",
        "distributed under the License is distributed on an AS IS BASIS, WITHOUT",
        "WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
        "See the License for the specific language governing permissions and",
        "limitations under the License.",
        "-->",
        "",
    ]

    lines += [
        "# Supported Ops",
        "",
        "This file was automatically generated by Vela using the `--supported-ops-report` parameter.  ",
        f"Vela version: `{__version__}`",
        "",
        "This file complies with",
        "[**Gitiles Markdown syntax**](https://gerrit.googlesource.com/gitiles/+/HEAD/Documentation/markdown.md)",
        "",
        "Summary table of constraints for:",
    ]

    # Ethos-U55 and Ethos-U65 TFLite and TOSA summary links
    for network_type in NetworkType:
        if network_type.value in exclude_generation_network_type_value:
            continue

        lines += [
            f"- [Ethos-U55 and Ethos-U65 {network_type.name}]"
            f"(#ethos-u55-and-ethos-u65-{network_type.name.lower()}-summary-table)",
        ]

    # Ethos-U85 TFLite summary link
    lines += [
        "- [Ethos-U85 TFLite](#ethos-u85-tflite-summary-table)",
    ]

    # Ethos-U55 and Ethos-U65 TFLite and TOSA Summary Table
    for network_type in NetworkType:
        if network_type.value in exclude_generation_network_type_value:
            continue

        lines += [
            "",
            f"## Ethos-U55 and Ethos-U65 {network_type.name} Summary Table",
            "",
        ]
        if network_type == NetworkType.TFLite:
            lines += [
                "The table below contains TFLite operators that can be placed on the Ethos-U55 and Ethos-U65.  ",
                "If the constraints are not met, then that operator will be scheduled on the CPU instead.  ",
                "For any other TFLite operator not listed, will be left untouched and scheduled on the CPU.  ",
                "Please check the supported operator list for your chosen runtime for further information.",
                "",
                "| Operator | TFLite Constraints |",
                "| --- | --- |",
            ]
            semantic_checker = TFLiteSemantic()
            supported = TFLiteSupportedOperators()
        elif network_type == NetworkType.TOSA:
            lines += [
                "The table below contains TOSA operators that can be placed on the Ethos-U NPU.  ",
                "Note: There is limited support for compiling a TOSA neural network (EXPERIMENTAL).  ",
                "The related constraints have not yet been populated in the list.",
                "",
                "| Operator | TOSA Constraints |",
                "| --- | --- |",
            ]
            semantic_checker = TosaSemantic()
            supported = TosaSupportedOperators()
        else:
            raise ValueError

        op_constraint_links = []
        op_list = sorted(((op, builtin_operator_name_map[op]) for op in builtin_operator_map), key=lambda x: x[1])
        for op, name in op_list:
            internal_op = builtin_operator_map[op][0]
            if internal_op in TFLiteSupportedOperators.supported_operators:
                links = f"[Generic](#ethos-u55-and-ethos-u65-{network_type.name.lower()}-generic-constraints)"
                if (
                    internal_op in supported.specific_constraints
                    or internal_op in semantic_checker.specific_constraints
                ):
                    links += (
                        f", [Specific](#ethos-u55-and-ethos-u65-{network_type.name.lower()}-{name.lower()}-constraints)"
                    )
                    op_constraint_links.append((internal_op, name))
                lines.append(f"| {name} | {links} |")

        if network_type == NetworkType.TFLite:
            # Ethos-U85 TFLite Summary Table
            lines += [
                "",
                "## Ethos-U85 TFLite Summary Table",
                "",
            ]
            lines += [
                "The table below contains TFLite operators that can be placed on the Ethos-U85.  ",
                "If the constraints are not met, then that operator will be scheduled on the CPU instead.  ",
                "For any other TFLite operator not listed, will be left untouched and scheduled on the CPU.  ",
                "Please check the supported operator list for your chosen runtime for further information.  ",
                "",
                "| Operator | TFLite Constraints |",
                "| --- | --- |",
            ]
            for op in u85_specific:
                sconstraints = u85_specific[op]
                links = "[Generic](#ethos-u85-tflite-generic-constraints)"
                if len(sconstraints):
                    links += f", [Specific](#ethos-u85-tflite-{op.lower()}-constraints)"
                lines.append(f"| {op.upper()} | {links} |")

        # Ethos-U55 and Ethos-U65 generic constraints
        lines += [
            "",
            f"## Ethos-U55 and Ethos-U65 {network_type.name} Generic Constraints",
            "",
            "This is a list of constraints that most operators must satisfy in order to be scheduled on the NPU.  ",
            "(Operators excluded from certain constraints are listed as exceptions )\n" "",
        ]
        for constraint in semantic_checker.generic_constraints:
            # Markdown needs two spaces at the end of a line to render it as a separate line
            reason = constraint.__doc__.replace("\n", "  \n")
            exclude_list = TFLiteSemantic.get_generic_constraint_exclude_list().items()
            lines.append(f"- {reason}")
            excluded_ops = _exclude_list_names(constraint, exclude_list)
            if excluded_ops:
                lines.append(f"  - Exceptions: {excluded_ops}")
        for constraint in supported.generic_constraints:
            # Markdown needs two spaces at the end of a line to render it as a separate line
            reason = constraint.__doc__.replace("\n", "  \n")
            exclude_list = supported.generic_constraints_exceptions.items()
            lines.append(f"- {reason}")
            excluded_ops = _exclude_list_names(constraint, exclude_list)
            if excluded_ops:
                lines.append(f"  - Exceptions: {excluded_ops}")
        lines += ["", "## Ethos-U55 and Ethos-U65 Specific Operator constraints"]
        for op, name in op_constraint_links:
            lines += [
                "",
                f"### Ethos-U55 and Ethos-U65 {network_type.name} {name} Constraints",
                "",
                f"This is a list of constraints that the {name} operator must satisfy in order to be scheduled on the"
                " NPU.",
                "",
            ]
            for constraint in semantic_checker.specific_constraints[op]:
                # Markdown needs two spaces at the end of a line to render it as a separate line
                reason = constraint.__doc__.replace("\n", "  \n")
                lines.append(f"- {reason}")
            for constraint in supported.specific_constraints[op]:
                # Markdown needs two spaces at the end of a line to render it as a separate line
                reason = constraint.__doc__.replace("\n", "  \n")
                lines.append(f"- {reason}")

    # Generic TFLite constraints for Ethos-U85
    lines += [
        "",
        "## Ethos-U85 TFLite Generic Constraints",
        "",
        "This is a list of constraints that most operators must satisfy in order to be scheduled on the NPU.  ",
        "(Operators excluded from certain constraints are listed as exceptions )\n" "",
        "",
    ]
    for constraint in u85_generic:
        exceptions = u85_generic[constraint]
        lines.append(f"- {constraint}")
        if len(exceptions):
            lines.append(f"  - Exceptions: [{', '.join(exceptions)}]")

    # Op-specific TFLite constraints for Ethos-U85
    lines += [
        "",
        "## Ethos-U85 Specific Operator Constraints",
    ]

    for name in u85_specific:
        constraints = u85_specific[name]
        if not len(constraints):
            continue
        lines += [
            "",
            f"### Ethos-U85 TFLite {name.upper()} Constraints",
            "",
            f"This is a list of constraints that the {name.upper()} operator "
            "must satisfy in order to be scheduled on the"
            " NPU.",
            "",
        ]
        for constraint in constraints:
            lines.append(f"- {constraint}")

    # Note. this will generate the file in the CWD
    filepath = os.path.join(os.getcwd(), "SUPPORTED_OPS.md")
    with open(filepath, "wt") as md:
        md.writelines(line + "\n" for line in lines)
        print(f"Report file: {filepath}")


def get_compiler_config(
    enable_debug_db: bool,
    verbose_all: bool,
    verbose_high_level_command_stream: bool,
    verbose_register_command_stream: bool,
    optimize: str,
    arena_cache_size: int,
    verbose_schedule: bool,
    verbose_allocation: bool,
    verbose_graph: bool,
    verbose_quantization: bool,
    verbose_packing: bool,
    verbose_performance: bool,
    show_cpu_operations: bool,
    output_format: str,
    disable_chaining: bool,
    disable_fwd: bool,
    disable_cascading: bool,
    disable_buffering: bool,
    disable_ifm_reuse: bool,
    cop_format: str,
    separate_io_regions: bool,
    cpu_tensor_alignment: int,
    tensor_allocator: str,
) -> str:
    """Build compiler config file."""
    config = "\n[compiler]\n"
    if verbose_high_level_command_stream:
        config += "verbose_high_level_command_stream=true\n"
    if verbose_register_command_stream:
        config += "verbose_register_command_stream=true\n"
    if verbose_performance or show_cpu_operations or enable_debug_db:
        config += "enable_db=true\n"
    if output_format == "raw":
        config += "output_format=Raw\n"
    else:
        config += "output_format=TFLite\n"
    config += f"cop_format={cop_format}\n"

    config += "\n[scheduler]\n"
    config += f"optimize={optimize}\n"
    config += f"arena_size_limit={arena_cache_size}\n"
    if verbose_schedule:
        config += "verbose=true\n"
    if verbose_allocation:
        config += "verbose_allocation=true\n"
    config += "disable_feature="
    if disable_chaining:
        config += "Grouping|"
    if disable_fwd:
        config += "FWD|"
    if disable_cascading:
        config += "Cascading|"
    if disable_buffering:
        config += "WeightBuffering|"
    if disable_ifm_reuse:
        config += "ReuseIFM|"
    config = config.rstrip("|") + "\n"
    if separate_io_regions:
        config += "separate_io_regions=true\n"
    config += f"cpu_tensor_alignment={cpu_tensor_alignment}\n"
    config += f"tensor_allocator={tensor_allocator}\n"

    config += "\n[graph]\n"
    if verbose_graph:
        config += "verbose=true\n"
    if verbose_quantization:
        config += "verbose_quantization=true\n"

    return config


def get_format(in_data: mmap.mmap) -> str:
    """Infere format based on input file."""
    ret = "UNDEFINED"
    if in_data.size() < 8:
        return ret
    second_word = int.from_bytes(in_data[4:8], "little")
    if second_word == TFLITE_MAGIC:
        ret = "TFLITE"
    if second_word == TOSA_MAGIC:
        ret = "TOSA"
    return ret


def list_config_files():
    print("Available config files:")
    path_length = len(architecture_features.CONFIG_FILES_PATH + os.path.sep)
    for config in glob.glob(os.path.join(architecture_features.CONFIG_FILES_PATH, "*", "*.ini")):
        print(config[path_length:])


def list_configs(config_filename):
    vela_config_filename = config_filename
    if not os.path.isfile(vela_config_filename):
        vela_config_filename = os.path.join(architecture_features.CONFIG_FILES_PATH, config_filename)
        if not os.path.isfile(vela_config_filename):
            assert False, f"Cannot find config file {config_filename}"

    if os.path.splitext(vela_config_filename)[1] != ".ini":
        assert False, f"Specified file {config_filename} is not a Vela config file"

    print(f"Configurations defined in {vela_config_filename}:")
    vela_config = ConfigParser()
    vela_config.read(vela_config_filename)
    for section in vela_config.sections():
        print(f"   {section}")


class DeprecatedStoreAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        print(f"Warning: Option {option_string} is deprecated and will be removed in a future release.")
        setattr(namespace, self.dest, values)


class DeprecatedStoreTrueAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        print(f"Warning: Option {option_string} is deprecated and will be removed in a future release.")
        setattr(namespace, self.dest, True)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the main entry point."""
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(prog="vela", description="Neural network model compiler for Arm Ethos-U NPUs")
    parser.register("action", "deprecated_store", DeprecatedStoreAction)
    parser.register("action", "deprecated_store_true", DeprecatedStoreTrueAction)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--api-version",
        action="version",
        version="Warning: Option --api-version is deprecated and will be removed in a future release."
        f" Version: {API_VERSION}",
        help="[DEPRECATED] Displays the version of the external API.",
    )
    parser.add_argument(
        "--supported-ops-report",
        action="store_true",
        help="Generate the SUPPORTED_OPS.md file in the current working directory and exit",
    )

    parser.add_argument(
        "--list-config-files",
        action="store_true",
        help=(
            "Display all available configurations in the `config_files` folder and exit. To select config file, "
            "use the --config argument with one of the listed config files (For example: --config Arm/vela.ini )"
        ),
    )
    parser.add_argument(
        "--list-configs",
        type=str,
        help=("Display all configurations defined in the specified config file"),
    )

    # set network nargs to be optional to allow the support-ops-report CLI option to be used standalone
    parser.add_argument(
        "network",
        metavar="NETWORK",
        type=str,
        nargs="?",
        default=None,
        help="Filename of the input network",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Output directory to write files to (default: %(default)s)"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="tflite",
        choices=["tflite", "raw"],
        help="Output format (default: %(default)s).",
    )
    parser.add_argument(
        "--enable-debug-db",
        action="store_true",
        default=None,
        help="Enables the calculation and writing of a network debug database to output directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        action="append",
        help="Vela configuration file(s) in Python ConfigParser .ini file format",
    )
    parser.add_argument("--verbose-all", action="store_true", help="Enable all verbose options")
    parser.add_argument(
        "--verbose-config", action="store_true", help="Enable system configuration and memory mode debug"
    )
    parser.add_argument("--verbose-graph", action="store_true", help="Enable graph optimizer debug")
    parser.add_argument("--verbose-quantization", action="store_true", help="Enable quantization debug")
    parser.add_argument(
        "--verbose-packing", action="deprecated_store_true", help="[DEPRECATED] Enable pass packing debug"
    )
    parser.add_argument(
        "--verbose-tensor-purpose", action="deprecated_store_true", help="[DEPRECATED] Enable tensor purpose debug"
    )
    parser.add_argument("--verbose-tensor-format", action="store_true", help="Enable tensor format debug")
    parser.add_argument("--verbose-schedule", action="store_true", help="Enable schedule debug")
    parser.add_argument("--verbose-allocation", action="store_true", help="Enable tensor allocation debug")
    parser.add_argument(
        "--verbose-high-level-command-stream", action="store_true", help="Enable high level command stream debug"
    )
    parser.add_argument(
        "--verbose-register-command-stream", action="store_true", help="Enable register command stream debug"
    )
    parser.add_argument(
        "--verbose-operators", action="deprecated_store_true", help="[DEPRECATED] Enable operator list debug"
    )
    parser.add_argument("--verbose-weights", action="store_true", help="Enable weights information debug")
    parser.add_argument("--verbose-cycle-estimate", action="store_true", help="Enable cycle estimate information debug")
    parser.add_argument("--verbose-performance", action="store_true", help="Enable performance information debug")
    parser.add_argument(
        "--verbose-progress", action="deprecated_store_true", help="[DEPRECATED] Enable progress information debug"
    )
    parser.add_argument(
        "--show-cpu-operations", action="store_true", help="Show the operations that fall back to the CPU"
    )
    parser.add_argument(
        "--timing", action="deprecated_store_true", help="[DEPRECATED] Time the compiler doing operations"
    )
    parser.add_argument(
        "--force-symmetric-int-weights",
        action="store_true",
        help="Forces all zero points to 0 for signed integer weights",
    )
    parser.add_argument(
        "--accelerator-config",
        type=str,
        default="ethos-u55-256",
        choices=list(architecture_features.Accelerator.member_list()),
        help="Accelerator configuration to use (default: %(default)s)",
    )
    parser.add_argument(
        "--system-config",
        type=str,
        default=architecture_features.ArchitectureFeatures.DEFAULT_CONFIG,
        help="System configuration to select from the Vela configuration file (default: %(default)s)",
    )
    parser.add_argument(
        "--memory-mode",
        type=str,
        default=architecture_features.ArchitectureFeatures.DEFAULT_CONFIG,
        help="Memory mode to select from the Vela configuration file (default: %(default)s)",
    )
    parser.add_argument(
        "--tensor-allocator",
        default=TensorAllocator.HillClimb,
        type=lambda s: TensorAllocator[s],
        choices=list(TensorAllocator),
        help="Tensor Allocator algorithm (default: %(default)s)",
    )
    parser.add_argument(
        "--show-subgraph-io-summary",
        action="deprecated_store_true",
        help="[DEPRECATED] Shows a summary of all the subgraphs and their inputs and outputs",
    )
    parser.add_argument(
        "--max-block-dependency",
        action="deprecated_store",
        type=int,
        default=architecture_features.ArchitectureFeatures.MAX_BLOCKDEP,
        choices=range(0, architecture_features.ArchitectureFeatures.MAX_BLOCKDEP + 1),
        help=(
            "[DEPRECATED] Set the maximum value that can be used for the block dependency between npu kernel operations"
            " (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--optimise",
        type=lambda s: scheduler.OptimizationStrategy[s],
        default=scheduler.OptimizationStrategy.Performance,
        choices=list(scheduler.OptimizationStrategy),
        help=(
            "Set the optimisation strategy. The Size strategy results in minimal SRAM usage (does not use"
            " arena-cache-size). The Performance strategy results in maximal performance (uses the arena-cache-size"
            " if specified) (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--arena-cache-size",
        type=int,
        help=(
            "Set the size of the arena cache memory area, in bytes. If specified, this option overrides the memory"
            " mode attribute with the same name in a Vela configuration file"
        ),
    )
    parser.add_argument(
        "--cpu-tensor-alignment",
        type=int,
        default=Tensor.AllocationQuantum,
        help=(
            "Controls the allocation byte alignment of CPU tensors including Ethos-U Custom"
            " operator inputs and outputs (default: %(default)s Bytes)"
        ),
    )
    parser.add_argument(
        "--recursion-limit",
        action="deprecated_store",
        type=int,
        default=1000,
        help=(
            "[DEPRECATED] Set the recursion depth limit, may result in RecursionError"
            " if too low (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--hillclimb-max-iterations",
        action="deprecated_store",
        type=int,
        default=HillClimbAllocator.MAX_ITERATIONS,
        help=(
            "[DEPRECATED] Set the maximum number of iterations the Hill Climb tensor allocator"
            " will run (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--cop-format",
        choices=["COP1", "COP2"],
        default="COP1",
    )
    parser.add_argument(
        "--separate-io-regions",
        action="store_true",
        help="Use separate regions for input and output tensors (implies COP2 driver actions format)",
    )

    # debug options
    parser.add_argument(
        "--debug-force-legacy-core",
        action="deprecated_store_true",
        help="Debug: Use the deprecated legacy Python compilation core",
    )
    parser.add_argument(
        "--debug-force-regor", action="deprecated_store_true", help="[DEPRECATED] Debug: Force the use of the regor"
    )
    parser.add_argument("--disable-chaining", action="deprecated_store_true", default=False, help="[DEPRECATED]")
    parser.add_argument("--disable-fwd", action="deprecated_store_true", default=False, help="[DEPRECATED]")
    parser.add_argument("--disable-cascading", action="deprecated_store_true", default=False, help="[DEPRECATED]")
    parser.add_argument("--disable-buffering", action="deprecated_store_true", default=False, help="[DEPRECATED]")
    parser.add_argument("--disable-ifm-reuse", action="deprecated_store_true", default=False, help="[DEPRECATED]")
    args = parser.parse_args(argv)

    # Generate the supported ops report and exit
    if args.supported_ops_report:
        generate_supported_ops()
        return 0

    if args.list_config_files:
        list_config_files()
        return 0

    if args.list_configs:
        list_configs(args.list_configs)
        return 0

    if args.network is None:
        parser.error("the following argument is required: NETWORK")

    if args.cop_format == "COP1" and args.separate_io_regions:
        parser.error("Driver actions format 'COP2' is required for --separate-io-regions")

    def _parse_config(config):
        # Make sure the correct separator is used depending on OS
        config = os.path.normpath(config)

        if os.path.splitext(config)[1] != ".ini":
            raise InputFileError(config, "Configuration files must use the .ini extension")

        if (
            len(config.split(os.path.sep)) == 2
            and not config.startswith(os.path.sep)
            and not config.startswith(".")
            and not config.startswith("~")
        ):
            config_path = os.path.join(architecture_features.CONFIG_FILES_PATH, config)
        else:
            # Check if the configuration file is correctly placed inside the config_files directory
            if os.access(
                os.path.join(architecture_features.CONFIG_FILES_PATH, *config.split(os.path.sep)[-2:]),
                os.R_OK,
            ):
                rel_path = os.path.join(*config.split(os.path.sep)[-2:])
                print(
                    f"Warning: Consider accessing the configuration by --config {rel_path} since it is located "
                    "inside the config_files directory."
                )
            config_path = config

        if not os.access(config_path, os.R_OK):
            raise InputFileError(
                config_path,
                "File not found or is not readable. The configuration file is either not located in a folder "
                "directly under the `config_files` directory or its path has not been provided correctly.",
            )

        return config_path

    # check all config files exist because they will be read as a group
    config_files = [_parse_config(cfg) for cfg in args.config] if args.config else None

    if args.cpu_tensor_alignment < 16 or args.cpu_tensor_alignment & (args.cpu_tensor_alignment - 1) != 0:
        parser.error(
            "Invalid argument to --cpu-tensor-alignment = {} (must be greater than or equal to 16 and a power of 2)"
            "".format(args.cpu_tensor_alignment)
        )

    if args.debug_force_legacy_core and args.debug_force_regor:
        parser.error(
            "Cannot force the use of both the deprecated legacy Python compilation core and the Regor C++ compilation"
            " core at the same time. Please choose only one of these options."
        )

    if args.verbose_all:
        for v in vars(args):
            if v.startswith("verbose") and v != "verbose_all":
                setattr(args, v, True)

    sys.setrecursionlimit(args.recursion_limit)

    arch = architecture_features.ArchitectureFeatures(
        vela_config_files=config_files,
        system_config=args.system_config,
        memory_mode=args.memory_mode,
        accelerator_config=args.accelerator_config,
        max_blockdep=args.max_block_dependency,
        verbose_config=args.verbose_config,
        arena_cache_size=args.arena_cache_size,
    )

    model_reader_options = model_reader.ModelReaderOptions()

    # Vela's legacy Python compilation core has been deprecated.
    # However, this can be overridden to still be used for TFLite and Ethos-U55 or Ethos-U65 by using the
    # `--debug-force-legacy-core` option, until fully removed.
    # All Ethos-U85 or TOSA network compilations always use Vela's C++ compilation core (Regor).
    if arch.is_ethos_u85_system or args.network.lower().endswith(".tosa") or not args.debug_force_legacy_core:
        if args.debug_force_legacy_core:
            parser.error(
                "Forcing the use of the deprecated legacy Python compilation core is not possible for this target"
                " system or input network. \nThe legacy core only supports Ethos-U55 and Ethos-U65 systems and TFLite"
                " input networks. Please remove the --debug-force-legacy-core option and try again."
            )
        system_config = "[architecture]\n"
        system_config += f"macs={arch.num_macs_per_cycle}\n"
        system_config += f"cores={arch.ncores}\n"

        system_config += "[vela]\n"
        system_config += f"system_config_name={arch.system_config}\n"
        system_config += f"memory_mode_name={arch.memory_mode}\n"

        # TODO MLBEDSW-8190: Improve support for multiple config files
        for config_file in arch.vela_config_files:
            with open(config_file, "rb") as f:
                system_config += f.read().decode("utf-8")

        # Transform Vela accelerator config string format to Regor format
        accelerator_config = args.accelerator_config
        substrings = accelerator_config.split("-")
        substrings = [substring.capitalize() for substring in substrings[:-1]]
        accelerator = "".join(substrings)

        options = get_compiler_config(
            args.enable_debug_db,
            args.verbose_all,
            args.verbose_high_level_command_stream,
            args.verbose_register_command_stream,
            args.optimise,
            # TODO MLBEDSW-8191: Clean up arena_cache_size selection
            # arch.arena_cache_size will use value from CLI option if available, otherwise from config file
            arch.arena_cache_size,
            args.verbose_schedule,
            args.verbose_allocation,
            args.verbose_graph,
            args.verbose_quantization,
            args.verbose_packing,
            args.verbose_performance,
            args.show_cpu_operations,
            args.output_format,
            args.disable_chaining,
            args.disable_fwd,
            args.disable_cascading,
            args.disable_buffering,
            args.disable_ifm_reuse,
            args.cop_format,
            args.separate_io_regions,
            args.cpu_tensor_alignment,
            args.tensor_allocator,
        )

        process_regor(
            arch,
            args.network,
            accelerator,
            system_config,
            options,
            args.enable_debug_db,
            args.output_dir,
            args.output_format,
            args.verbose_weights,
            args.verbose_cycle_estimate,
            args.show_cpu_operations,
            args.verbose_performance,
        )

    else:
        compiler_options = compiler_driver.CompilerOptions(
            verbose_graph=args.verbose_graph,
            verbose_quantization=args.verbose_quantization,
            verbose_packing=args.verbose_packing,
            verbose_tensor_purpose=args.verbose_tensor_purpose,
            verbose_tensor_format=args.verbose_tensor_format,
            verbose_allocation=args.verbose_allocation,
            verbose_high_level_command_stream=args.verbose_high_level_command_stream,
            verbose_register_command_stream=args.verbose_register_command_stream,
            verbose_operators=args.verbose_operators,
            verbose_weights=args.verbose_weights,
            verbose_cycle_estimate=args.verbose_cycle_estimate,
            verbose_performance=args.verbose_performance,
            verbose_progress=args.verbose_progress,
            show_cpu_operations=args.show_cpu_operations,
            tensor_allocator=args.tensor_allocator,
            timing=args.timing,
            force_symmetric_int_weights=args.force_symmetric_int_weights,
            output_dir=args.output_dir,
            cpu_tensor_alignment=args.cpu_tensor_alignment,
            hillclimb_max_iterations=args.hillclimb_max_iterations,
        )

        scheduler_options = scheduler.SchedulerOptions(
            optimization_strategy=args.optimise,
            sram_target=arch.arena_cache_size,
            verbose_schedule=args.verbose_schedule,
            verbose_progress=args.verbose_progress,
        )

        try:
            nng = process(
                args.network,
                args.enable_debug_db,
                arch,
                model_reader_options,
                compiler_options,
                scheduler_options,
                args.output_format,
            )

        except VelaError as e:
            print(e.data)
            return 1

        if args.show_subgraph_io_summary:
            print_subgraph_io_summary(nng)

    return 0
