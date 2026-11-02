"""Surface errors returned by Inspect eval instead of calling every return a success."""
import json


def save_eval_summary(logs, output, control_hz):
    runs, errors = [], []
    for log in logs:
        scenes = [{'scene_id': sample.scene_id, 'status': sample.status,
                   'error': sample.error, 'termination_reasons': list(sample.termination_reasons),
                   'scores': sample.reduced} for sample in log.samples]
        runs.append({'status': log.status, 'error': log.error,
                     'total_steps': log.stats.total_steps,
                     'simulation_seconds': log.stats.total_steps / control_hz,
                     'scenes': scenes})
        for error in [log.error, *(sample.error for sample in log.samples)]:
            if error and error not in errors:
                errors.append(error)
        if log.status == 'error' and not log.error:
            errors.append('Inspect returned error status; see per-scene logs')
    if not runs:
        errors.append('Inspect returned no evaluation logs')
    summary = {'runs': runs, 'errors': errors}
    (output / 'eval_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary
