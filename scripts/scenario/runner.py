#!/usr/bin/env python3
"""Run event-driven scenarios sequentially and preserve evidence in JSON bundles."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time

from model import Scheduler, load_recipe
from compare import compare_frame

ROOT = Path(__file__).resolve().parents[2]
PREFIXES = {'[milestone] ': 'milestone', '[scenario] ': 'scenario'}


def parse_event(line):
    for prefix, kind in PREFIXES.items():
        if line.startswith(prefix):
            try:
                event = json.loads(line[len(prefix):])
                if isinstance(event, dict):
                    return kind, event
            except json.JSONDecodeError:
                pass
    return None, None


def crash_line(line):
    return any(token in line for token in (
        'Memory fault / invalid instruction pointer', 'Illegal or unimplemented instruction',
        'Assertion failed:', 'EXC_BAD_ACCESS', '[Fault]', 'Segmentation fault', 'stack_chk_fail'))


def scene_metrics(events, end):
    """Counter deltas, with explicit coverage and no fabricated zero-FPS scenes."""
    samples = [e for e in events if e.get('type') == 'metrics']
    marks = [e for e in events if e.get('type') == 'milestone' and e.get('role') == 'scene']
    result = []
    for index, mark in enumerate(marks):
        stop = marks[index + 1]['t'] if index + 1 < len(marks) else end
        selected = [s for s in samples if mark['t'] <= s['t'] <= stop]
        row = dict(scene=mark['name'], start=mark['t'], end=stop, samples=len(selected), fps=None,
                   unsupported_packets=None)
        if len(selected) >= 2 and selected[-1]['t'] > selected[0]['t']:
            a, b = selected[0], selected[-1]
            row.update(fps=(b['frames'] - a['frames']) / (b['t'] - a['t']),
                       unsupported_packets=b['unsupported_packets'] - a['unsupported_packets'],
                       measured_seconds=b['t'] - a['t'])
        result.append(row)
    return result


def outcome(recipe, milestones, returncode, timed_out, crash_evidence, captures, errors, supervisor_killed=False):
    expected = recipe.get('expected', {})
    missing = [n for n in expected.get('milestones', []) if n not in milestones]
    forbidden = [n for n in expected.get('forbid_milestones', []) if n in milestones]
    crashed = (returncode is not None and returncode < 0 and not supervisor_killed) or bool(crash_evidence)
    failures = list(errors)
    if missing:
        failures.append('missing milestones: ' + ', '.join(missing))
    if forbidden:
        failures.append('forbidden milestones: ' + ', '.join(forbidden))
    if crashed:
        failures.append('crash/fault evidence observed')
    if timed_out:
        failures.append('scenario timed out')
    if returncode not in (None, 0) and not timed_out:
        failures.append(f'process exited with status {returncode}')
    if any(c.get('comparison', {}).get('passed') is False for c in captures):
        failures.append('golden-frame comparison failed')
    return dict(passed=not failures, crashed=bool(crashed), failures=failures, missing=missing)


def run_once(recipe, binary, output, timeout, fast_forward, discovery=False):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    with binary.open('rb') as executable:
        runner_hash = hashlib.sha256(executable.read()).hexdigest()
    scheduler = Scheduler(recipe)
    events, captures, errors, crashes, actions = [], [], [], [], []
    sent_logs, pending_captures = set(), {}
    supervisor_killed = False
    input_queue = []
    inflight_input = None
    next_input_time = 0
    started, timed_out, requested_stop, ready = time.monotonic(), False, False, False
    end_deadline = started + timeout
    # Only explicit recipe switches affect a run; inherited debug/input flags
    # would otherwise make identical recipes behave differently across shells.
    env = {key: value for key, value in os.environ.items() if not key.startswith('PS4_')}
    for key, value in recipe.get('env', {}).items():
        env[key] = str(value)
    with tempfile.TemporaryDirectory(prefix='ps4-scenario-') as temporary:
        fifo = Path(temporary) / 'control.fifo'
        os.mkfifo(fifo, 0o600)
        channel = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)
        env.update(PS4_SCENARIO=str(output / 'recipe.json'), PS4_SCENARIO_CONTROL=str(fifo),
                   PS4_SCENARIO_FAST_FORWARD='1' if fast_forward else '0', PS4_DEBUG_DIR=str(output / 'captures'),
                   PS4_NO_VSYNC='1', PS4_FLIP_STATS='1')
        if discovery:
            env['PS4_SCENARIO_DISCOVERY'] = '1'
        eboot = Path(recipe['eboot'])
        if not eboot.is_absolute():
            eboot = ROOT / eboot
        command = [str(binary), '--headless', '--budget', '0', str(eboot)]
        lines = queue.Queue()
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True, text=True, errors='replace', bufsize=1)

        def read_output():
            for line in proc.stdout:
                lines.put(line)
            lines.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()

        def send(action):
            message = {k: v for k, v in action.items() if k not in ('step', 'hold')}
            data = (json.dumps(message, separators=(',', ':')) + '\n').encode()
            try:
                if os.write(channel, data) != len(data):
                    raise OSError('partial FIFO write')
                actions.append(dict(message, sent_t=time.monotonic() - started))
            except OSError as error:
                errors.append(f'control channel: {error}')

        eof = False
        try:
            with (output / 'emulator.log').open('w', buffering=1) as log:
                while not eof:
                    now = time.monotonic()
                    due = scheduler.due(now)
                    for action in due:
                        if action['op'] == 'buttons':
                            input_queue.append(action)
                            continue
                        if action['op'] == 'capture':
                            pending_captures[action['id']] = action['step']
                        elif action['op'] == 'stop':
                            requested_stop = True
                            end_deadline = min(end_deadline, now + 8)
                        send(action)
                    if input_queue and inflight_input is None and now >= next_input_time:
                        action = input_queue.pop(0)
                        send(action)
                        inflight_input = action['id']
                        next_input_time = now + action['hold']
                    if now >= end_deadline and proc.poll() is None:
                        timed_out = not requested_stop
                        supervisor_killed = True
                        if requested_stop:
                            errors.append("runtime did not stop within 8 seconds")
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                    try:
                        line = lines.get(timeout=0.02)
                    except queue.Empty:
                        continue
                    if line is None:
                        eof = True
                        continue
                    log.write(line)
                    kind, event = parse_event(line)
                    if event is not None:
                        if kind == 'milestone':
                            event['type'] = 'milestone'
                            definition = next((m for m in recipe['milestones'] if m['name'] == event.get('name')), None)
                            if definition is None:
                                errors.append('unknown emitted milestone: ' + str(event.get('name')))
                                continue
                            event['role'] = definition.get('role', 'milestone')
                            if scheduler.milestone(event, time.monotonic()):
                                print(f"  {event['name']}: {event['t']:.3f}s, frame {event['frame']}", flush=True)
                        elif event.get('type') == 'input_sampled':
                            if event.get('id') == inflight_input:
                                inflight_input = None
                        elif event.get('type') == 'ready':
                            ready = True
                        elif event.get('type') == 'capture':
                            step = pending_captures.pop(event['id'], None)
                            capture = dict(event)
                            folder = Path(event['path'])
                            if not all((folder / name).is_file() for name in ('report.json', 'target_0000000000000000.png', 'target_0000000000000000.ppm')):
                                errors.append(f"capture {event['id']}: incomplete artifact set")
                            if step and 'reference' in step:
                                try:
                                    capture['comparison'] = compare_frame(Path(event['path']) / 'target_0000000000000000.ppm',
                                                                          ROOT / step['reference'], step.get('tolerance', 0.08))
                                except (ValueError, OSError) as error:
                                    errors.append(f"capture {event['id']}: {error}")
                            captures.append(capture)
                        elif event.get('type') == 'error':
                            errors.append(event.get('message', 'runtime scenario error'))
                        events.append(event)
                    elif '[scenario-error]' in line:
                        errors.append(line.strip())
                    if crash_line(line):
                        crashes.append(line.strip())
                    if kind is None:
                        for name in scheduler.log_matches(line):
                            if name not in sent_logs:
                                send(dict(op='log', name=name))
                                sent_logs.add(name)
                proc.wait(timeout=2)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            reader.join(timeout=2)
            proc.stdout.close()
            os.close(channel)
    duration = time.monotonic() - started
    if not ready:
        errors.append('runtime scenario service never became ready')
    if pending_captures:
        errors.append('captures not completed: ' + ', '.join(pending_captures))
    sent_inputs = {a['id'] for a in actions if a['op'] == 'buttons'}
    applied_inputs = {e['id'] for e in events if e.get('type') == 'input'}
    sampled_inputs = {e['id'] for e in events if e.get('type') == 'input_sampled'}
    if input_queue or inflight_input is not None:
        errors.append('input sequence was not fully sampled by the guest')
    if sent_inputs - sampled_inputs:
        errors.append('input states not sampled: ' + ', '.join(sorted(sent_inputs - sampled_inputs)))
    if sent_inputs - applied_inputs:
        errors.append('input commands not acknowledged: ' + ', '.join(sorted(sent_inputs - applied_inputs)))
    completed_captures = {c['id'] for c in captures}
    for step in recipe.get('steps', []):
        if step['action'] == 'capture' and step['after'] in scheduler.reached and step['id'] not in completed_captures:
            errors.append('expected capture missing: ' + step['id'])

    # A stop may be a declared action, but timeouts remain failures even when
    # earlier milestones passed. Assertions/faults can exit with status zero.
    result = outcome(recipe, scheduler.reached, proc.returncode, timed_out, crashes, captures, errors, supervisor_killed)
    metrics = [e for e in events if e.get('type') == 'metrics']
    unsupported = metrics[-1]['unsupported_packets'] if metrics else None
    maximum = recipe.get('expected', {}).get('max_unsupported_packets')
    if maximum is not None and (unsupported is None or unsupported > maximum):
        result['passed'] = False
        result['failures'].append('unsupported packet count missing or exceeds expectation')
    report = dict(result, recipe=recipe['name'], revision=revision, source_dirty=dirty, runner_sha256=runner_hash, command=command,
                  duration=duration, timeout=timeout, timed_out=timed_out, requested_stop=requested_stop,
                  returncode=proc.returncode, supervisor_killed=supervisor_killed, fast_forward=fast_forward, milestones=scheduler.reached,
                  furthest_milestone=next(reversed(scheduler.reached), None),
                  furthest_observed_scene=next((n for n in reversed(scheduler.reached) if any(m['name']==n and m.get('role')=='scene' for m in recipe['milestones'])), None), scenes=scene_metrics(events, max((e.get('t', 0) for e in events), default=0)),
                  unsupported_packets=unsupported, captures=captures, actions=actions, events=events, crash_evidence=crashes)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recipe', type=Path)
    parser.add_argument('--runner', type=Path, default=ROOT / 'build-opt/ps4-runner')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repeat', type=int, choices=(1, 2), default=1)
    parser.add_argument('--timeout', type=float, help='override recipe timeout (maximum 180 seconds)')
    parser.add_argument('--fast-forward', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--discover', action='store_true', help='log first-seen shader/texture fingerprints')
    args = parser.parse_args()
    try:
        recipe = load_recipe(args.recipe)
        timeout = args.timeout if args.timeout is not None else recipe['timeout']
        if not 0 < timeout <= 180:
            raise ValueError('timeout must be greater than zero and at most 180 seconds')
        binary = args.runner.resolve()
        if args.fast_forward and not recipe.get('fast_forward_until'):
            raise ValueError('--fast-forward requires fast_forward_until in the recipe')
        if not binary.is_file():
            raise ValueError(f'runner not found: {binary}')
        if not (ROOT / recipe['eboot']).is_file():
            raise ValueError(f"eboot not found: {recipe['eboot']}")
        output = (args.output or ROOT / 'build-opt/scenarios' / (recipe['name'] + '-' +
                  datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
        # All scenario invocations on this checkout share one advisory lock.
        # The coordinator must still avoid builds and non-scenario launches.
        (ROOT / 'build-opt').mkdir(exist_ok=True)
        with open(Path(tempfile.gettempdir()) / f'ps4-scenario-{os.getuid()}.lock', 'a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('another scenario runner is active')
            output.mkdir(parents=True, exist_ok=False)
            results = []
            for i in range(args.repeat):
                existing = subprocess.run(['pgrep', '-x', 'ps4-runner'], capture_output=True, text=True)
                if existing.returncode == 0:
                    raise ValueError('a ps4-runner process is already active; refusing another emulator')
                print(f"{recipe['name']}: run {i + 1}/{args.repeat}", flush=True)
                results.append(run_once(recipe, binary, output / f'run_{i + 1}', timeout, args.fast_forward, args.discover))
            passed = sum(r['passed'] for r in results)
            summary = dict(recipe=recipe['name'], runs=len(results), passed=passed, pass_rate=passed / len(results),
                           furthest_milestones=[r['furthest_milestone'] for r in results],
                           furthest_observed_scenes=[r['furthest_observed_scene'] for r in results])
            (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            print(f"{passed}/{len(results)} passed; reports: {output}")
            return 0 if passed == len(results) else 1
    except (ValueError, OSError, json.JSONDecodeError, re.error) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
