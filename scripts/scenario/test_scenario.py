"""Pure unit tests: no emulator, GPU, or game assets required."""
import json
from pathlib import Path
import tempfile
import unittest
from model import load_recipe, Scheduler, BUTTONS
from compare import compare_frame
from runner import parse_event, scene_metrics, outcome, run_once, crash_line


class Scenarios(unittest.TestCase):
    def setUp(self):
        self.recipe = dict(version=1, name='fixture', eboot='fixture.elf', timeout=10,
                           milestones=[dict(name='boot', source=dict(type='start'))],
                           steps=[], expected=dict(milestones=['boot']))

    def load(self, recipe):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'recipe.json'
            path.write_text(json.dumps(recipe))
            return load_recipe(path)

    def test_validation(self):
        self.assertEqual(self.load(self.recipe), self.recipe)
        changes = [dict(timeout=-1), dict(timeout=float('nan')), dict(version=2),
                   dict(milestones=self.recipe['milestones'] * 2),
                   dict(milestones=[dict(name='bad', after='unknown', source=dict(type='start'))]),
                   dict(milestones=[dict(name='boot', source=dict(type='shader', hash='oops'))]),
                   dict(expected=dict(milestones=['missing'])),
                   dict(expected=dict(milestones=[{}])),
                   dict(fast_forward_until=[]),
                   dict(milestones=[dict(name='bad', after=[], source=dict(type='start'))])]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load(dict(self.recipe, **change))

    def test_press_validation(self):
        step = dict(id='press', after='boot', action='press', buttons=['cross'])
        for change in [dict(buttons=['oops']), dict(duration=-1), dict(repeat=0),
                       dict(repeat=True), dict(after='unknown'), dict(id='../unsafe')]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load(dict(self.recipe, steps=[dict(step, **change)]))

    def test_repeated_input_and_release(self):
        self.recipe['steps'] = [dict(id='pages', after='boot', delay=2, action='press',
                                    buttons=['right'], repeat=28, duration=0.1, gap=0.3)]
        scheduler = Scheduler(self.load(self.recipe))
        scheduler.milestone(dict(name='boot', t=0, frame=1), 100)
        self.assertEqual(scheduler.due(101.99), [])
        self.assertEqual(scheduler.due(102)[0]['buttons'], 0x20)
        self.assertEqual(scheduler.due(102.11)[0]['buttons'], 0)
        rest = scheduler.due(200)
        self.assertEqual(len(rest), 54)
        self.assertEqual(rest[-1]['buttons'], 0)
        self.assertFalse(scheduler.milestone(dict(name='boot'), 201))
        self.assertEqual(scheduler.due(300), [])

    def test_overlapping_buttons(self):
        self.recipe['steps'] = [dict(id=b, after='boot', action='press', buttons=[b],
                                    duration=d) for b, d in [('cross', 1), ('right', 2)]]
        scheduler = Scheduler(self.recipe)
        scheduler.milestone(dict(name='boot'), 0)
        self.assertEqual(scheduler.due(0)[-1]['buttons'], 0x4020)
        self.assertEqual(scheduler.due(1)[0]['buttons'], 0x20)
        self.assertEqual(scheduler.due(2)[0]['buttons'], 0)

    def test_log_dependency_and_machine_line(self):
        self.recipe['milestones'].append(dict(name='dialog', after='boot', source=dict(type='log', pattern=r'\[MsgDialog\].*EULA')))
        scheduler = Scheduler(self.recipe)
        self.assertEqual(scheduler.log_matches('[MsgDialog] EULA'), [])
        scheduler.milestone(dict(name='boot'), 0)
        self.assertEqual(scheduler.log_matches('[MsgDialog] EULA'), ['dialog'])
        kind, event = parse_event('[milestone] {"name":"dialog","frame":20,"t":2.5}')
        self.assertEqual((kind, event['name']), ('milestone', 'dialog'))
        self.assertEqual(parse_event('[milestone] broken'), (None, None))

    def test_scene_metrics_and_failures(self):
        events = [dict(type='milestone', name='boot', t=0, role='scene'),
                  dict(type='metrics', t=1, frames=10, unsupported_packets=2),
                  dict(type='metrics', t=3, frames=110, unsupported_packets=5)]
        scene = scene_metrics(events, 4)[0]
        self.assertEqual(scene['fps'], 50)
        self.assertEqual(scene['unsupported_packets'], 3)
        self.assertTrue(outcome(self.recipe, {'boot': {}}, 0, False, [], [], [])['passed'])
        self.assertFalse(outcome(self.recipe, {}, 0, False, [], [], [])['passed'])
        self.assertTrue(outcome(self.recipe, {'boot': {}}, 0, False, ['guest fault'], [], [])['crashed'])
        timed = outcome(self.recipe, {'boot': {}}, -15, True, [], [], [], True)
        self.assertFalse(timed['passed'])
        self.assertFalse(timed['crashed'])


    def test_fake_process_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = root / 'fake-runtime'
            fake.write_text("""#!/usr/bin/env python3
import json, os, time
from pathlib import Path
fifo=os.open(os.environ['PS4_SCENARIO_CONTROL'],os.O_RDONLY)
def emit(prefix, data):
 print(prefix+' '+json.dumps(dict(t=0.1,frame=1,**data)),flush=True)
emit('[scenario]',dict(type='ready'))
emit('[milestone]',dict(name='boot',source='start'))
with os.fdopen(fifo) as stream:
 for line in stream:
  c=json.loads(line)
  if c['op']=='buttons':
   emit('[scenario]',dict(type='input',id=c['id'],buttons=c['buttons']))
   time.sleep(float(os.getenv('FAKE_SAMPLE_DELAY','0')))
   if not os.getenv('FAKE_NO_SAMPLE'): emit('[scenario]',dict(type='input_sampled',id=c['id'],buttons=c['buttons']))
  elif c['op']=='stop': break
emit('[scenario]',dict(type='metrics',frames=1,unsupported_packets=0))
""")
            fake.chmod(0o755)
            self.recipe['eboot'] = str(fake)
            self.recipe['steps'] = [dict(id='cross', after='boot', delay=0.01, action='press', buttons=['cross']),
                                    dict(id='done', after='boot', delay=0.8, action='stop')]
            result = run_once(self.recipe, fake, root / 'run', 3, False)
            self.assertTrue(result['passed'], result['failures'])
            acknowledgments = [e for e in result['events'] if e.get('type') == 'input']
            self.assertEqual([e['buttons'] for e in acknowledgments], [0x4000, 0])
            self.assertTrue((root / 'run/report.json').is_file())
            # Delayed guest polling must stretch a repeated sequence instead of
            # coalescing several page turns during a capture/compile stall.
            self.recipe['steps'][0].update(repeat=4, duration=0.01, gap=0.01)
            self.recipe['steps'][1]['delay'] = 1.5
            self.recipe['env'] = {'FAKE_SAMPLE_DELAY': '0.05'}
            delayed = run_once(self.recipe, fake, root / 'delayed', 4, False)
            self.assertTrue(delayed['passed'], delayed['failures'])
            sampled = [e['id'] for e in delayed['events'] if e.get('type') == 'input_sampled']
            self.assertEqual(sampled, [f'cross_{i}_{edge}' for i in range(4) for edge in ('press', 'release')])
            self.recipe['env'] = {'FAKE_NO_SAMPLE': '1'}
            unsampled = run_once(self.recipe, fake, root / 'unsampled', 4, False)
            self.assertFalse(unsampled['passed'])
            self.assertTrue(any('not fully sampled' in error for error in unsampled['failures']))
            self.assertFalse(unsampled['crashed'])



    def test_escape_matches_window_mapping(self):
        self.assertEqual(BUTTONS['escape'], 0x0008)
        self.assertEqual(BUTTONS['options'], BUTTONS['escape'])

    def test_existing_golden_comparator(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a.ppm', Path(directory) / 'b.ppm'
            image = b'P6\n160 90\n255\n' + bytes([20, 40, 60]) * (160 * 90)
            a.write_bytes(image)
            b.write_bytes(image)
            self.assertTrue(compare_frame(a, b, 0)['passed'])
            b.write_bytes(b'P6\n160 90\n255\n' + bytes([255, 255, 255]) * (160 * 90))
            self.assertFalse(compare_frame(a, b, 0.08)['passed'])
            b.write_bytes(b'P6\n320 180\n255\n' + bytes([0, 0, 0]) * (320 * 180))
            with self.assertRaisesRegex(ValueError, 'reference must be 160x90'):
                compare_frame(a, b, 0.08)
            with self.assertRaises(ValueError):
                compare_frame(a, Path(directory) / 'missing.ppm', 0.08)

    def test_dependency_order_and_regex(self):
        for milestones in [
            [dict(name='late', after='early', source=dict(type='start'))],
            [dict(name='bad', source=dict(type='unknown'))],
            [dict(name='unsafe/path', source=dict(type='start'))],
        ]:
            with self.assertRaises(ValueError):
                self.load(dict(self.recipe, milestones=milestones))
        with self.assertRaises(ValueError):
            self.load(dict(self.recipe, env={'PS4_AUTO_PRESS': '1'}))


    def test_actual_runtime_fault_wording(self):
        self.assertTrue(crash_line('[CPU] Execution Status: Memory fault / invalid instruction pointer'))
        self.assertTrue(crash_line('[CPU] Execution Status: Illegal or unimplemented instruction'))
        self.assertFalse(crash_line('[CPU] Execution Status: Instruction executed successfully'))
        self.assertFalse(crash_line('[CPU] Execution Status: CPU Halted (HLT)'))


if __name__ == '__main__':
    unittest.main()
