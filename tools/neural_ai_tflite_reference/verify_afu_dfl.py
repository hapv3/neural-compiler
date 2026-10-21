"""Check independent AFU DFL reference against a retained YOLO replay (no sim)."""

import argparse
import json
from pathlib import Path
import struct
import sys

import numpy as np

from benchmark import difference
from runtime import make_interpreter


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
    report = json.loads((args.case.parent / 'byte_exact.json').read_text())
    snapshots = {}
    for saved in report['intermediate_snapshots']:
        metadata = json.loads(checked_read(saved['evidence']))
        if metadata['case']['sha256'] != sha(args.case.read_bytes()) or metadata['model'] != case['model']:
            raise ValueError('Snapshot provenance differs from case')
        snapshots[saved['command']] = rtl_runtime._verify_yolo320_snapshot(
            model, checked_read(saved['snapshot']), expected_boundary=saved['command'])[1]
    runner = make_interpreter(case['source']['path'], 'neural_ai_afu_dfl_reference', preserve=True)
    detail = runner.get_input_details()[0]
    runner.set_tensor(detail['index'], np.frombuffer(checked_read(case['input']), np.int8).reshape(detail['shape']))
    control, actual, _ = runner.invoke_comparison()
    expected = np.frombuffer(checked_read(case['final_output']['expected']), np.int8).reshape(control.shape)
    if not np.array_equal(control, expected):
        raise AssertionError('Control changed the original pinned hybrid oracle')
    commands, _ = rtl_runtime._logical_command_records(model)
    constants = None
    for index in range(struct.unpack_from('<I', model, 20)[0]):
        kind, _, offset, size = struct.unpack_from('<4I', model, 64 + 32 * index)
        if kind == 2:
            constants = model[offset:offset + size]
    if constants is None:
        raise ValueError('Missing NAI constants')
    evidence = []
    covered = set()
    logits = runner.prefix.get_tensor(runner.plan.input)
    for n, command in enumerate(commands, 1):
        if struct.unpack_from('<H', command)[0] != 28:
            continue
        locations, layout, multiplier, shift = struct.unpack_from('<2IiI', command, 56)
        if layout != 1:
            raise ValueError('Snapshot verifier requires C32-split DFL input')
        matches = [g for g in runner.plan.groups if (g['locations'], g['multiplier'], g['shift']) ==
                   (locations, multiplier, shift)]
        if len(matches) != 1:
            raise ValueError('Ambiguous source-to-command DFL group')
        group = matches[0]
        first = group['first']
        if first in covered:
            raise ValueError('Repeated DFL source group')
        covered.add(first)
        source, dest = (struct.unpack_from('<I', command, o)[0] for o in (20, 28))
        before = snapshots[n - 1]
        saved_logits = np.frombuffer(before, np.int8, locations * 64, source).reshape(2, locations, 32)
        saved_logits = saved_logits.transpose(1, 0, 2).reshape(locations, 4, 16).transpose(1, 0, 2)[None]
        source_logits = logits[:, :, first:first + locations]
        observed = np.frombuffer(snapshots[n], np.int8, 4 * locations, dest).reshape(1, 4, locations)
        predicted = runner.distances[:, :, first:first + locations]
        for offset, independent in ((44, runner.plan.exp), (52, runner.plan.reciprocal)):
            table_offset = struct.unpack_from('<I', command, offset)[0]
            deployed = np.frombuffer(constants, '<u4', 256, table_offset)
            if not np.array_equal(deployed, independent):
                raise AssertionError('Independently generated LUT differs from deployed model')
        stats = dict(command=n, first=first, locations=locations,
                     input_vs_source=difference(saved_logits, source_logits),
                     output_vs_rtl=difference(predicted, observed), independent_luts_match=True)
        if stats['input_vs_source']['mismatches'] or stats['output_vs_rtl']['mismatches']:
            raise AssertionError(stats)
        evidence.append(stats)
    if covered != {g['first'] for g in runner.plan.groups}:
        raise AssertionError('Not every source DFL group was checked')
    final_snapshot = rtl_runtime._verify_yolo320_snapshot(
        model, Path(report['snapshot']).read_bytes(), expected_boundary=case['end_command'])[3]
    observed_output = np.frombuffer(final_snapshot, np.int8).reshape(actual.shape)
    final = difference(actual, observed_output)
    if final['mismatches']:
        raise AssertionError(f'AFU reference still differs from retained final RTL output: {final}')
    result = dict(status='PASS', reference=runner.plan.description(), source_case_sha256=sha(args.case.read_bytes()),
                  reference_sources_sha256={name: sha((Path(__file__).parent / name).read_bytes())
                                            for name in ('afu_dfl.py', 'runtime.py', 'verify_afu_dfl.py')},
                  original_oracle_unchanged=True, control_vs_original_oracle=difference(control, expected),
                  checkpoints=evidence, final_reference_vs_retained_rtl=final,
                  reference_vs_unchanged_hybrid=difference(actual, control),
                  scope='Only this retained input/model. Prior failing hybrid report is not rewritten; no new RTL simulation.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    del runner


if __name__ == '__main__':
    main()
