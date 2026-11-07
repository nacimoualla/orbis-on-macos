"""Fixture-only library tests; never inspect the repository's games folder."""
import contextlib
import io
import os
from pathlib import Path
import struct
import sys
import subprocess
import json
import tempfile
import time
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from game_library import Library, DEFAULT_PROFILE
from run_workflow import CommandRunner, GameRunner


def title(root):
    root.mkdir(parents=True, exist_ok=True)
    header = bytearray(64)
    header[:6] = b'\x7fELF\x02\x01'
    struct.pack_into('<H', header, 18, 62)
    (root / 'eboot.elf').write_bytes(header)
    keys = b'TITLE\0TITLE_ID\0APP_VER\0'
    values = [b'Fixture\0', b'CUSA00001\0', b'1.00\0']
    entries = b''
    offset = 0
    for key_offset, value in zip((0, 6, 15), values):
        entries += struct.pack('<HHIII', key_offset, 0x204, len(value), len(value), offset)
        offset += len(value)
    data = b'\0PSF' + struct.pack('<IIII', 0x101, 68, 68 + len(keys), 3) + entries + keys + b''.join(values)
    (root / 'sce_sys').mkdir(exist_ok=True)
    (root / 'sce_sys/param.sfo').write_bytes(data)


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        self.library = Library(self.repo)

    def tearDown(self):
        self.temp.cleanup()

    def pkg(self, path=None):
        path = path or self.repo / 'games/fixture.pkg'
        path.write_bytes(b'fixture pkg')
        os.utime(path, (time.time() - 10,) * 2)
        return path

    def extractor(self, source, stage, force=False):
        title(stage / 'nested/title')
        return stage / 'nested/title'

    def test_validated_internal_source_deleted_and_receipted(self):
        source = self.pkg()
        with patch('game_library.extract_pkg', side_effect=self.extractor):
            result = self.library.import_sources()
        self.assertFalse(source.exists())
        self.assertEqual(result['games'][0]['title'], 'Fixture')
        self.assertEqual(result['games'][0]['profile'], DEFAULT_PROFILE)
        self.assertEqual(result['jobs'][0]['status'], 'complete')
        self.assertTrue(self.library.state_path.exists())

    def test_failure_preserves_source_and_does_not_retry(self):
        source = self.pkg()
        with patch('game_library.extract_pkg', return_value=None) as extract:
            self.library.import_sources()
            self.library.import_sources()
            self.assertEqual(extract.call_count, 1)
            self.library.import_sources(retry=True)
            self.assertEqual(extract.call_count, 2)
        self.assertTrue(source.exists())

    def test_existing_title_does_not_authorize_deletion(self):
        title(self.repo / 'games/existing')
        source = self.pkg()
        with patch('game_library.extract_pkg', return_value=None):
            self.library.import_sources()
        self.assertTrue(source.exists())

    def test_external_source_and_duplicate_titles_are_retained(self):
        source = self.pkg(self.repo / 'outside.pkg')
        with patch('game_library.extract_pkg', side_effect=self.extractor):
            self.library.import_sources(source)
            internal = self.pkg()
            self.library.import_sources()
        self.assertTrue(source.exists())
        self.assertFalse(internal.exists())
        self.assertEqual(len(self.library.snapshot()['games']), 2)

    def test_source_change_during_extraction_preserved(self):
        source = self.pkg()
        def changing(src, stage, force=False):
            self.extractor(src, stage)
            src.write_bytes(b'changed')
        with patch('game_library.extract_pkg', side_effect=changing):
            result = self.library.import_sources()
        self.assertTrue(source.exists())
        self.assertEqual(result['jobs'][0]['status'], 'failed')

    def test_profile_and_missing_title_persist(self):
        root = self.repo / 'external/title'
        title(root)
        profile = self.library.profile(root, dict(budget=123, jit=False))
        self.assertEqual(profile['budget'], 123)
        (root / 'eboot.elf').unlink()
        game = Library(self.repo).snapshot()['games'][0]
        self.assertEqual(game['status'], 'missing')
        self.assertFalse(game['profile']['jit'])

    def test_recent_source_waits_for_second_scan(self):
        source = self.pkg()
        os.utime(source, None)
        with patch('game_library.extract_pkg', side_effect=self.extractor) as extract:
            self.assertEqual(self.library.import_sources()['jobs'][0]['status'], 'waiting')
            extract.assert_not_called()
            self.library.import_sources()
            extract.assert_called_once()

    def test_failed_extractor_partial_valid_output_is_not_published(self):
        source = self.pkg()
        def partial(src, stage, force=False):
            self.extractor(src, stage)
            return None
        with patch('game_library.extract_pkg', side_effect=partial):
            result = self.library.import_sources()
        self.assertTrue(source.exists())
        self.assertEqual(result['games'], [])
        self.assertEqual(result['jobs'][0]['status'], 'failed')

    def test_ordinary_elf_launch_uses_profile(self):
        root = self.repo / 'games/title'
        title(root)
        with patch.object(GameRunner, 'find_executable', return_value='/fixture/runner'), patch('run_workflow.CommandRunner.run', return_value=(True, 0, '')) as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(GameRunner(str(self.repo)).launch(str(root / 'eboot.elf')))
            self.assertIn('--window', run.call_args.args[0])

    def test_interrupt_kills_and_reaps_child(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], stdout=subprocess.PIPE, start_new_session=True)
        with patch('run_workflow.subprocess.Popen', return_value=process), patch('run_workflow.selectors.DefaultSelector.select', side_effect=KeyboardInterrupt), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                CommandRunner.run(['fixture'], 'Interrupt')
        self.assertIsNotNone(process.poll())

    def test_promoted_crash_recovers_without_reextracting(self):
        source = self.pkg()
        original = self.library.save
        def crash(state):
            if any(r.get('status') == 'complete' for r in state['sources'].values()):
                raise KeyboardInterrupt()
            original(state)
        with patch('game_library.extract_pkg', side_effect=self.extractor), patch.object(self.library, 'save', side_effect=crash):
            with self.assertRaises(KeyboardInterrupt):
                self.library.import_sources()
        self.assertTrue(source.exists())
        with patch('game_library.extract_pkg') as extract:
            result = self.library.import_sources()
            extract.assert_not_called()
        self.assertFalse(source.exists())
        self.assertEqual(result['jobs'][0]['status'], 'complete')

    def test_corrupt_state_is_preserved_without_deletion_trust(self):
        source = self.pkg()
        self.library.state_path.write_text('{broken')
        with contextlib.redirect_stderr(io.StringIO()):
            self.library.snapshot()
        self.assertTrue(source.exists())
        self.assertEqual(len(list(self.library.internal.glob('state.corrupt-*.json'))), 1)

    def test_partial_promotion_crash_rolls_back_owned_title_and_retries(self):
        source = self.pkg()
        def two(src, stage, force=False):
            title(stage / 'one')
            title(stage / 'two')
            return stage / 'one'
        rename = os.rename
        calls = []
        def interrupt_second(origin, destination):
            calls.append(destination)
            if len(calls) == 2:
                raise KeyboardInterrupt()
            rename(origin, destination)
        with patch('game_library.extract_pkg', side_effect=two), patch('game_library.os.rename', side_effect=interrupt_second):
            with self.assertRaises(KeyboardInterrupt):
                self.library.import_sources()
        self.assertTrue(source.exists())
        with patch('game_library.extract_pkg', side_effect=two):
            result = self.library.import_sources()
        self.assertFalse(source.exists())
        self.assertEqual(len(result['games']), 2)
        self.assertFalse(Path(calls[0]).exists())

    def test_import_data_sync_precedes_complete_receipt(self):
        self.pkg()
        syncs = []
        save = self.library.save
        def observe_save(state):
            if any(r.get('status') == 'complete' for r in state['sources'].values()):
                self.assertTrue(syncs)
            save(state)
        from game_library import sync_tree
        def observe_sync(root):
            sync_tree(root)
            syncs.append(root)
        with patch('game_library.extract_pkg', side_effect=self.extractor), patch('game_library.sync_tree', side_effect=observe_sync), patch.object(self.library, 'save', side_effect=observe_save):
            self.library.import_sources()

    def test_tar_gz_uses_staged_archive_and_retains_source(self):
        source = self.pkg(self.repo / 'fixture.tar.gz')
        def archive(src, destination, password):
            destination.mkdir(parents=True)
            pkg = destination / 'nested.pkg'
            pkg.write_bytes(b'fixture')
            return [pkg]
        with patch('game_library.extract_archive', side_effect=archive) as unpack, patch('game_library.extract_pkg', side_effect=self.extractor):
            result = self.library.import_sources(source)
        self.assertTrue(source.exists())
        self.assertEqual(result['jobs'][0]['status'], 'complete')
        self.assertNotEqual(unpack.call_args.args[0], source)

    def test_detached_worker_survives_request_parent_and_polls_without_lock(self):
        # Empty fixture target is harmless; a preheld state lock keeps worker live.
        import fcntl
        with (self.library.internal / 'lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            script = 'import sys,json; sys.path.insert(0,sys.argv[1]); from game_library import Library; print(json.dumps(Library(sys.argv[2]).background_import()))'
            parent = subprocess.run([sys.executable, '-c', script, str(Path(__file__).resolve().parents[1] / 'scripts'), str(self.repo)], capture_output=True, text=True, timeout=5)
            self.assertEqual(parent.returncode, 0, parent.stderr)
            result = json.loads(parent.stdout)
            self.assertTrue(result['worker']['running'])
            pid = result['worker']['pid']
            os.kill(pid, 0)
            start = time.monotonic()
            self.assertTrue(self.library.snapshot()['worker']['running'])
            self.assertLess(time.monotonic() - start, 1)
            # A repeated submission queues once and never launches a second worker.
            second = self.library.background_import()
            self.assertEqual(second['worker']['pid'], pid)
            self.assertEqual(second['worker']['queued'], 1)
        deadline = time.monotonic() + 5
        while self.library.worker_status()['running'] and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(self.library.worker_status()['running'])
        self.assertTrue(Path(result['worker']['log_path']).is_file())

    def test_successor_waits_for_finishing_worker_lock(self):
        import fcntl
        processes = []
        real_popen = subprocess.Popen
        def track_spawn(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process
        with (self.library.internal / 'worker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.library.atomic_json(self.library.internal / 'worker.json', dict(running=False, pid=os.getpid()))
            with patch('game_library.subprocess.Popen', side_effect=track_spawn):
                result = self.library.background_import()
            self.assertTrue(result['worker']['running'])
            time.sleep(0.1)
            self.assertIsNone(processes[0].poll())
            self.assertEqual(len(self.library.read_json('queue.json', [])), 1)
        processes[0].wait(timeout=5)
        self.assertEqual(self.library.read_json('queue.json', []), [])
        self.assertFalse(self.library.worker_status()['running'])

    def test_worker_keeps_waiting_request_until_third_attempt(self):
        request = dict(target=str(self.repo / 'games'), retry=False)
        self.library.atomic_json(self.library.internal / 'queue.json', [request])
        own = dict(source=str(self.repo / 'games/copying.pkg'), status='waiting', message='copying')
        outside = dict(source=str(self.repo / 'outside.pkg'), status='waiting', message='unrelated')
        results = [dict(jobs=[own, outside]), dict(jobs=[own, outside]), dict(jobs=[dict(own, status='complete'), outside])]
        with patch.object(self.library, 'import_sources', side_effect=results) as imports, patch('game_library.time.sleep') as sleep, contextlib.redirect_stdout(io.StringIO()):
            self.library.run_worker()
        self.assertEqual(imports.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(self.library.read_json('queue.json', []), [])

    def test_silent_child_timeout(self):
        start = time.monotonic()
        with contextlib.redirect_stdout(io.StringIO()):
            success, code, output = CommandRunner.run([sys.executable, '-c', 'import time; time.sleep(30)'], 'Silent child', timeout=0.2)
        self.assertFalse(success)
        self.assertLess(time.monotonic() - start, 3)


if __name__ == '__main__':
    unittest.main()
