"""Audit retained compute dumps and compare independent AFU Mean + MAC output.

No simulation, recompilation, oracle replacement, or report relabeling. PASS is
restricted to the named numerical observations, not every captured TCDM byte.
"""

import argparse
import csv
import io
import json
from pathlib import Path
import struct
import sys
import zlib

import numpy as np

from benchmark import difference
from runtime import ROOT, make_interpreter


def audit_compute_dumps(entries, planned, commands, read_bytes, sha):
    """Hash/command identity checks are integrity checks, not numerical PASS."""
    expected = {item['command']: item for item in planned}
    if len(expected) != len(planned) or len(entries) != len(planned):
        raise ValueError('Duplicate or incomplete compute capture plan')
    result = {}
    for entry in entries:
        command = entry['command']
        if command in result or command not in expected or not 1 <= command <= len(commands):
            raise ValueError('Duplicate, unexpected or invalid compute capture')
        if any(entry.get(key) != value for key, value in expected[command].items()):
            raise ValueError('Compute capture differs from the pinned plan')
        if sha(commands[command - 1]) != entry['command_sha256']:
            raise ValueError('Captured command hash mismatch')
        compressed = read_bytes(entry['path'])
        if len(compressed) != entry['compressed_bytes'] or sha(compressed) != entry['compressed_sha256']:
            raise ValueError('Compressed compute capture hash/size mismatch')
        memory = zlib.decompress(compressed)
        if len(memory) != 512 * 1024 or len(memory) != entry['tcdm_bytes'] or sha(memory) != entry['tcdm_sha256']:
            raise ValueError('Decompressed compute capture hash/size mismatch')
        result[command] = memory
    return result


def c32_tensor(memory, offset, shape):
    shape = tuple(map(int, shape))
    if len(shape) not in (3, 4) or shape[0] != 1 or min(shape) <= 0:
        raise ValueError('Expected static batch-one spatial/token tensor')
    channels, spatial = shape[-1], int(np.prod(shape[1:-1]))
    groups = (channels + 31) // 32
    if offset < 0 or offset + groups * spatial * 32 > len(memory):
        raise ValueError('C32 observation exceeds TCDM')
    blocked = np.frombuffer(memory, np.int8, groups * spatial * 32, offset)
    return blocked.reshape(groups, spatial, 32).transpose(1, 0, 2).reshape(spatial, groups * 32)[:, :channels].reshape(shape)


def scratch_offset(command, position):
    region, index, offset = struct.unpack_from('<HHI', command, position)
    if region != 6 or index != 0 or offset % 32:
        raise ValueError('Expected aligned internal TCDM reference')
    return offset


def audit_command_pmu(stream, commands, expected_cycles):
    rows = csv.DictReader(stream)
    count, cycles = 0, 0
    for count, row in enumerate(rows, 1):
        if count > len(commands) or int(row['command']) != count:
            raise ValueError('PMU command sequence is incomplete or duplicated')
        kind, size, flags, layer, tile = struct.unpack_from('<HHIII', commands[count - 1])
        if tuple(int(row[key]) for key in ('type', 'size', 'flags', 'layer_id', 'tile_id')) != (kind, size, flags, layer, tile):
            raise ValueError('PMU row descriptor differs from executed command')
        if int(row['cycle']) < 0:
            raise ValueError('Negative per-command cycle count')
        cycles += int(row['cycle'])
    if count != len(commands) or cycles != expected_cycles:
        raise ValueError('PMU count/cycles do not reconcile with invocation counters')
    return dict(rows=count, columns=len(rows.fieldnames), command_cycles=cycles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--neural-ai-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite retained comparison evidence')
    sys.path.insert(0, str(args.neural_ai_root / 'sw/test/compiler_runtime'))
    from yolo_reference_replay import checked_read, load_case, runtime as rtl_runtime, sha
    case, model, *_ = load_case(args.case)
    if json.loads((ROOT / 'build/manifest.json').read_text()) != case['reference_build']:
        raise ValueError('Reference MAC build differs from the pinned case')
    commands, _ = rtl_runtime._logical_command_records(model)
    report_path = args.case.parent / 'byte_exact.json'
    report = json.loads(report_path.read_text())
    for key in ('model', 'source', 'firmware', 'input', 'compute_dump_reference'):
        if report[key] != case[key]:
            raise ValueError(f'Retained report {key} differs from case')
    if (case.get('full_graph') is not True or report.get('start_mode') != 'reset' or
            case['first_command'] != 0 or case['end_command'] != len(commands) or
            report['pmu']['command_begin'] != len(commands) or report['pmu']['command_end'] != len(commands)):
        raise ValueError('Expected complete full-reset command execution')
    manifest = json.loads(checked_read(case['compute_dump_reference']))
    for key in ('model', 'source', 'firmware', 'input', 'reference_build'):
        if manifest[key] != case[key]:
            raise ValueError(f'Compute manifest {key} differs from case')
    dump_manifest = checked_read(report['compute_dumps']['manifest'])
    entries = [json.loads(line) for line in dump_manifest.splitlines()]
    dumps = audit_compute_dumps(entries, manifest['compute_checkpoints'], commands,
                                 lambda path: Path(path).read_bytes(), sha)
    if len(dumps) != report['compute_dumps']['captured']:
        raise ValueError('Compute capture count differs from original report')
    csv_path = args.case.parent / 'command_pmu.csv'
    with csv_path.open(newline='') as stream:
        pmu_csv = audit_command_pmu(stream, commands, report['pmu']['command_active_cycles'])
    pmu_csv.update(path=str(csv_path.resolve()), sha256=sha(csv_path.read_bytes()))
    snapshots = []
    for saved in report['intermediate_snapshots']:
        metadata = json.loads(checked_read(saved['evidence']))
        if metadata['case']['sha256'] != sha(args.case.read_bytes()) or metadata['model'] != case['model']:
            raise ValueError('Snapshot provenance differs from case')
        rtl_runtime._verify_yolo320_snapshot(model, checked_read(saved['snapshot']), expected_boundary=saved['command'])
        snapshots.append(saved['command'])
    runner = make_interpreter(checked_read(case['source']), 'neural_ai_afu_mean_reference', preserve=True)
    if runner.reference_coverage()['mac_fallback_nodes']:
        raise ValueError('Unsupported MAC kernels in the independent reference')
    detail = runner.get_input_details()[0]
    runner.set_tensor(detail['index'], np.frombuffer(checked_read(case['input']), np.int8).reshape(detail['shape']))
    control, actual, source_mean = runner.invoke_comparison()
    expected = np.frombuffer(checked_read(case['final_output']['expected']), np.int8).reshape(control.shape)
    if not np.array_equal(control, expected):
        raise AssertionError('Source-Mean control changed the pinned original hybrid oracle')
    p = runner.plan
    reference_tensors = np.load(io.BytesIO(checked_read(manifest['reference_tensors'])), allow_pickle=False)
    prefix_checks = []
    for i in p.prefix_nodes:
        for tensor in p.ops[i].outputs:
            stats = difference(runner.prefix.get_tensor(int(tensor)), reference_tensors[f'tensor_{tensor}'])
            if stats['mismatches']:
                raise AssertionError('Reference prefix changed a pinned source tensor')
            prefix_checks.append(dict(source_node=i, tensor=int(tensor), **stats))
    if not np.array_equal(source_mean, reference_tensors[f'tensor_{p.output}']):
        raise AssertionError('Source Mean changed its pinned tensor')
    means = [(n, c) for n, c in enumerate(commands, 1) if struct.unpack_from('<H', c)[0] == 14]
    if len(means) != 1:
        raise ValueError('Verifier requires exactly one fused AFU Mean command')
    number, command = means[0]
    if len(command) != 64 or not struct.unpack_from('<I', command, 4)[0] & 16:
        raise ValueError('Missing fused Mean requantization flag')
    height, width, channels = struct.unpack_from('<3I', command, 32)
    if (height * width, channels) != (p.spatial, p.input_shape[-1]):
        raise ValueError('Source Mean and command dimensions differ')
    emitted = struct.unpack_from('<iIiiI', command, 44)
    if emitted != tuple(p.quant[key] for key in ('output_multiplier', 'output_shift', 'input_offset',
                                                'output_zero_point', 'double_round_shift')):
        raise AssertionError('Independently generated Mean qparams differ from compiled command')
    observed_input = c32_tensor(dumps[number - 1], scratch_offset(command, 16), p.input_shape)
    offset = scratch_offset(command, 24)
    observed_mean = np.frombuffer(dumps[number], np.int8, channels, offset).reshape(p.output_shape)
    comparisons = [dict(command=number - 1, scope='Complete pre-Mean source tensor',
                        **difference(observed_input, runner.prefix.get_tensor(p.input))),
                   dict(command=number, scope='Complete Mean output', **difference(observed_mean, runner.mean))]
    # Compare the complete materialized FC output before final boundary unpack.
    # Reject other layouts/strides instead of interpreting them as compact data.
    boundary = commands[-1]
    if (struct.unpack_from('<H', boundary)[0] != 18 or len(boundary) != 96 or
            struct.unpack_from('<4H', boundary, 32) != (2, 2, 1, 1) or
            struct.unpack_from('<HHI', boundary, 24) != (4, 0, 0)):
        raise ValueError('Verifier requires final INT8 ROW32-to-public-NHWC copy')
    dims = struct.unpack_from('<4I', boundary, 40)
    valid, source_stride, dest_stride = struct.unpack_from('<3I', boundary, 56)
    if (dims != (1, 1, 1, actual.size) or valid != actual.size or source_stride < valid or dest_stride != valid):
        raise ValueError('Unsupported final ROW32 observation shape/strides')
    for n in (len(commands) - 1, len(commands)):
        observed = np.frombuffer(dumps[n], np.int8, valid, scratch_offset(boundary, 16)).reshape(actual.shape)
        comparisons.append(dict(command=n, scope='Complete final ROW32 source', **difference(observed, actual)))
    snapshot_output = rtl_runtime._verify_yolo320_snapshot(
        model, Path(report['snapshot']).read_bytes(), expected_boundary=len(commands))[3]
    if sha(snapshot_output) != report['final_output']['actual_sha256']:
        raise ValueError('Retained final output differs from original report')
    if (args.case.parent / 'output-actual.bin').read_bytes() != snapshot_output:
        raise ValueError('Retained output file differs from final snapshot')
    final = difference(np.frombuffer(snapshot_output, np.int8).reshape(actual.shape), actual)
    failed = final['mismatches'] or any(row['mismatches'] for row in comparisons)
    result = dict(status='FAIL' if failed else 'PASS', reference=p.description(),
                  source_case=dict(path=str(args.case.resolve()), sha256=sha(args.case.read_bytes())),
                  original_report=dict(path=str(report_path.resolve()), sha256=sha(report_path.read_bytes()), status=report['status']),
                  reference_sources_sha256={name: sha((Path(__file__).parent / name).read_bytes())
                                            for name in ('afu_mean.py', 'runtime.py', 'verify_afu_mean.py')},
                  integrity=dict(compute_dumps=len(dumps), snapshots=len(snapshots),
                                 compute_manifest_sha256=sha(dump_manifest)),
                  prefix_source_tensors_unchanged=prefix_checks,
                  original_oracle_unchanged=True, source_mean_control_vs_original_oracle=difference(control, expected),
                  numerical_compute_observations=comparisons,
                  remaining_compute_dumps_numerical_status='NOT_COMPARED',
                  mean_afu_vs_original=difference(runner.mean, source_mean),
                  final_reference_vs_retained_rtl=final, reference_vs_unchanged_hybrid=difference(actual, control),
                  output_sha256=dict(original_hybrid=sha(control.tobytes()), afu_mean_reference=sha(actual.tobytes()),
                                     retained_rtl=sha(snapshot_output)),
                  top1=dict(original_hybrid=int(control.argmax()), afu_mean_reference=int(actual.argmax())),
                  pmu=report['pmu'], command_pmu_csv=pmu_csv, reference_mac_build=case['reference_build'],
                  scope='Only listed tensors and final output for this retained input/model; not every capture, dataset accuracy, or a new RTL run.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'integrity', 'numerical_compute_observations',
                                          'mean_afu_vs_original', 'final_reference_vs_retained_rtl', 'top1')}, indent=2))
    del runner
    if failed:
        raise SystemExit('Independent AFU Mean reference differs from retained RTL; see report')


if __name__ == '__main__':
    main()
