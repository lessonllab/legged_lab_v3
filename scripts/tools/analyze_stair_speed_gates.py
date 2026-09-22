"""Summarize overlapping stair promotion blockers from frozen evaluation episodes."""
import argparse
import json
import math
from pathlib import Path


def speed_reasons(row, prefix=''):
    seconds, tracking, actual, command = [row[prefix + key] for key in
        ('cruise_seconds', 'cruise_tracking', 'cruise_actual_mps', 'cruise_command_mps')]
    finite = all(math.isfinite(x) for x in (seconds, tracking, actual, command))
    return {'nonfinite': not finite, 'short_cruise': finite and seconds < .5,
            'low_tracking': finite and tracking <= .6, 'no_command': finite and command <= 0.,
            'underspeed': finite and actual < .8 * command, 'overspeed': finite and actual > 1.2 * command}


def summarize(rows):
    result = {'episodes': len(rows)}
    for key in ('failed', 'traversed', 'settled', 'terminal_geometry', 'passed', 'speed_passed'):
        result[key] = sum(bool(row[key]) for row in rows)
    for prefix in ('', 'warm_'):
        failures = [speed_reasons(row, prefix) for row in rows]
        for key in failures[0]:
            result[prefix + key] = sum(row[key] for row in failures)
        result[prefix + 'full_pass'] = sum(row['passed'] and not any(f.values()) for row, f in zip(rows, failures))
        for key in ('cruise_seconds', 'cruise_actual_mps', 'cruise_command_mps', 'cruise_tracking'):
            result[prefix + key + '_mean'] = sum(row[prefix + key] for row in rows) / len(rows)
    result['settled_then_lost_exit'] = sum(row['settled'] and not row['terminal_geometry'] for row in rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    results = []
    for model in data['results']:
        cases = {}
        for case in sorted({r['case'] for r in model['episodes'] if r['group'] in (1, 2)}):
            rows = [r for r in model['episodes'] if r['case'] == case]
            cases[case] = summarize(rows)
        results.append({'checkpoint': model['checkpoint'], 'cases': cases})
    output = {'source': str(args.input.resolve()),
              'scope': 'Completed episodes in fixed evaluation only. Reasons overlap. warm_ excludes first 0.5 s; no policy change.',
              'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
