"""Recipe validation, event matching, and milestone-relative action scheduling."""
import heapq
import json
import math
from pathlib import Path
import re

BUTTONS = {'cross': 0x4000, 'circle': 0x2000, 'square': 0x8000, 'triangle': 0x1000,
           'escape': 8, 'right': 0x20, 'left': 0x80, 'up': 0x10, 'down': 0x40, 'options': 8,
           'touchpad': 0x100000, 'l1': 0x400, 'r1': 0x800}


def number(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f'{label} must be a finite number >= {minimum}')
    return value


def load_recipe(path):
    recipe = json.loads(Path(path).read_text())
    if not isinstance(recipe, dict) or recipe.get('version') != 1:
        raise ValueError('recipe must be an object with version 1')
    for key in ('name', 'eboot'):
        if not isinstance(recipe.get(key), str) or not recipe[key]:
            raise ValueError(f'{key} must be a nonempty string')
    if not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_-]*', recipe['name']):
        raise ValueError('name must be a safe recipe identifier')
    number(recipe.get('timeout'), 'timeout', 0.1)
    milestones = recipe.get('milestones')
    if not isinstance(milestones, list) or not milestones:
        raise ValueError('milestones must be a nonempty list')
    names = set()
    for milestone in milestones:
        if not isinstance(milestone, dict) or not isinstance(milestone.get('source'), dict):
            raise ValueError('each milestone needs a source object')
        name = milestone.get('name')
        if not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_]*', name) or name in names:
            raise ValueError(f'invalid or duplicate milestone name: {name}')
        if milestone.get('after') is not None and (not isinstance(milestone['after'], str) or milestone['after'] not in names):
            raise ValueError(f'{name}: after must reference an earlier milestone')
        if milestone.get('role') not in (None, 'scene', 'milestone'):
            raise ValueError(f'{name}: role must be scene or milestone')
        names.add(name)
        source = milestone.get('source', {})
        kind = source.get('type')
        field = {'input': 'id', 'nid': 'name', 'shader': 'hash', 'texture': 'address', 'log': 'pattern', 'start': None}.get(kind)
        if kind not in ('nid', 'shader', 'texture', 'log', 'start', 'input'):
            raise ValueError(f'{name}: unknown source type {kind}')
        if field and (not isinstance(source.get(field), str) or not source[field]):
            raise ValueError(f'{name}: source requires {field}')
        if kind in ('shader', 'texture'):
            if not re.fullmatch(r'0x[0-9a-fA-F]{1,16}', source[field]):
                raise ValueError(f'{name}: {field} must be a hexadecimal string')
        if source.get('stage') not in (None, 'VS', 'PS', 'CS'):
            raise ValueError(f'{name}: invalid shader stage')
        if kind == 'log':
            re.compile(source['pattern'])
    steps = recipe.get('steps', [])
    if not isinstance(steps, list):
        raise ValueError('steps must be a list')
    ids = set()
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError('each step must be an object')
        if not isinstance(step.get('id'), str) or not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_]*', step['id']) or step['id'] in ids:
            raise ValueError('steps need unique safe ids')
        ids.add(step['id'])
        if not isinstance(step.get('after'), str) or step['after'] not in names:
            raise ValueError(f"{step['id']}: unknown milestone")
        number(step.get('delay', 0), 'delay')
        kind = step.get('action')
        if kind == 'press':
            buttons = step.get('buttons')
            if not isinstance(buttons, list) or not buttons or any(not isinstance(b, str) or b not in BUTTONS for b in buttons):
                raise ValueError(f"{step['id']}: unknown buttons")
            number(step.get('duration', 0.1), 'duration', 0.01)
            number(step.get('gap', 0.3), 'gap', 0.01)
            count = step.get('repeat', 1)
            if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 1000:
                raise ValueError('repeat must be an integer from 1 to 1000')
        elif kind == 'capture':
            if 'reference' in step and not isinstance(step['reference'], str):
                raise ValueError('reference must be a path')
            number(step.get('tolerance', 0.08), 'tolerance')
            if step.get('tolerance', 0.08) > 1:
                raise ValueError('tolerance must be <= 1')
        elif kind != 'stop':
            raise ValueError(f"unknown action: {kind}")
    expected = recipe.get('expected', {})
    if not isinstance(expected, dict):
        raise ValueError('expected must be an object')
    environment = recipe.get('env', {})
    reserved = {'PS4_AT_FRAME', 'PS4_GPU_CAPTURE_FRAME', 'PS4_GPU_CAPTURE_XCODE', 'PS4_RUN_SECONDS', 'PS4_NO_VSYNC', 'PS4_GPU_READBACK'}
    if not isinstance(environment, dict) or any(not isinstance(k, str) or k.startswith(('PS4_SCENARIO', 'PS4_AUTO_')) or k in reserved for k in environment):
        raise ValueError('env must be an object and cannot override scenario control or automatic input')
    for key in ('milestones', 'forbid_milestones'):
        if not isinstance(expected.get(key, []), list) or any(not isinstance(n, str) or n not in names for n in expected.get(key, [])):
            raise ValueError(f'expected.{key} references unknown milestones')
    if set(expected.get('milestones', [])) & set(expected.get('forbid_milestones', [])):
        raise ValueError('a milestone cannot be both required and forbidden')
    if 'max_unsupported_packets' in expected:
        number(expected['max_unsupported_packets'], 'max_unsupported_packets')
    until = recipe.get('fast_forward_until')
    if until is not None and (not isinstance(until, str) or until not in names):
        raise ValueError('fast_forward_until references an unknown milestone')
    return recipe


class Scheduler:
    def __init__(self, recipe):
        self.recipe = recipe
        self.reached = {}
        self.pending = []
        self.serial = 0
        self.pressed = {}  # independent press tokens support overlapping chords

    def push(self, when, action):
        self.serial += 1
        heapq.heappush(self.pending, (when, self.serial, action))

    def milestone(self, event, now):
        name = event['name']
        if name in self.reached:
            return False
        self.reached[name] = event
        for step in self.recipe.get('steps', []):
            if step['after'] != name:
                continue
            when = now + step.get('delay', 0)
            if step['action'] == 'press':
                duration = step.get('duration', 0.1)
                for i in range(step.get('repeat', 1)):
                    token = f"{step['id']}_{i}"
                    mask = 0
                    for button in step['buttons']:
                        mask |= BUTTONS[button]
                    t = when + i * (duration + step.get('gap', 0.3))
                    self.push(t, dict(op='press', id=token, mask=mask, hold=duration))
                    self.push(t + duration, dict(op='release', id=token, hold=step.get('gap', 0.3)))
            else:
                self.push(when, dict(op=step['action'], id=step['id'], step=step))
        return True

    def due(self, now):
        result = []
        while self.pending and self.pending[0][0] <= now:
            _, _, action = heapq.heappop(self.pending)
            if action['op'] in ('press', 'release'):
                if action['op'] == 'press':
                    self.pressed[action['id']] = action['mask']
                else:
                    self.pressed.pop(action['id'], None)
                mask = 0
                for value in self.pressed.values():
                    mask |= value
                result.append(dict(op='buttons', id=action['id'] + '_' + action['op'], buttons=mask, hold=action['hold']))
            else:
                result.append(action)
        return result

    def log_matches(self, line):
        matches = []
        for milestone in self.recipe['milestones']:
            source = milestone['source']
            if source['type'] != 'log' or milestone['name'] in self.reached:
                continue
            if milestone.get('after') and milestone['after'] not in self.reached:
                continue
            if re.search(source['pattern'], line):
                matches.append(milestone['name'])
        return matches
