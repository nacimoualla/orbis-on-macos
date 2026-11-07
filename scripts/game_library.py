"""Shared, crash-safe game library and conservative package import service."""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
import time
import uuid
import sys
import subprocess

from extract_games import extract_pkg, extract_game_target, extract_archive, parse_sfo

DEFAULT_PROFILE = dict(jit=True, window=True, budget=0, hle_mono=False)


def walk(path, prune_titles=False):
    for base, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d != '.library' and not (Path(base) / d).is_symlink())
        if prune_titles and any(name in files and elf(Path(base) / name) for name in ('eboot.elf', 'eboot.bin')):
            dirs[:] = []
        yield Path(base), files


def metadata(root):
    try:
        return parse_sfo((root / 'sce_sys/param.sfo').read_bytes())
    except (OSError, ValueError, struct.error):
        return {}


def elf(path):
    try:
        with path.open('rb') as stream:
            data = stream.read(64)
        return len(data) == 64 and data[:6] == b'\x7fELF\x02\x01' and struct.unpack_from('<H', data, 18)[0] == 62
    except OSError:
        return False


def discover(paths):
    found = {}
    for path in paths:
        for root, files in walk(path, prune_titles=True):
            executable = next((root / name for name in ('eboot.elf', 'eboot.bin') if name in files and elf(root / name)), None)
            if not executable:
                continue
            root = root.resolve()
            info = {key: value for key, value in metadata(root).items() if isinstance(value, str)}
            found[str(root)] = dict(root=str(root), eboot=str(executable.resolve()), title=info.get('TITLE') or root.name,
                title_id=info.get('TITLE_ID', ''), app_ver=info.get('APP_VER', ''),
                icon_path=str(root / 'sce_sys/icon0.png') if (root / 'sce_sys/icon0.png').is_file() else '', status='ready')
    return found


def identity(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sync_tree(root):
    # Publication receipts must never precede durable extracted data.
    directories = []
    for base, dirs, files in os.walk(root, followlinks=False):
        base = Path(base)
        directories.append(base)
        dirs[:] = [d for d in dirs if not (base / d).is_symlink()]
        for name in files:
            path = base / name
            if path.is_symlink():
                continue
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    for path in reversed(directories):
        sync_directory(path)
    sync_directory(Path(root).parent)


def valid_state(state):
    if not isinstance(state, dict) or state.get('version') != 1:
        return False
    if not all(isinstance(state.get(k), list) for k in ('games', 'jobs', 'roots')) or not isinstance(state.get('sources'), dict):
        return False
    if not all(isinstance(r, str) for r in state['roots']):
        return False
    for game in state['games']:
        if not isinstance(game, dict) or not all(isinstance(game.get(k), str) for k in ('root', 'title', 'eboot', 'title_id', 'app_ver', 'icon_path', 'status')):
            return False
        profile = game.get('profile')
        if not isinstance(profile, dict) or not all(type(profile.get(k)) is bool for k in ('jit', 'window', 'hle_mono')) or type(profile.get('budget')) is not int or profile['budget'] < 0:
            return False
    for key, record in state['sources'].items():
        if not isinstance(record, dict) or record.get('source') != key or not all(isinstance(record.get(k), str) for k in ('status', 'message')):
            return False
        if not isinstance(record.get('identity'), list) or len(record['identity']) != 5 or not all(type(n) is int for n in record['identity']):
            return False
        if 'destinations' in record and (not isinstance(record['destinations'], list) or not all(isinstance(d, str) for d in record['destinations'])):
            return False
    return True


class Library:
    def __init__(self, repo):
        self.games = Path(repo).resolve() / 'games'
        self.internal = self.games / '.library'
        self.internal.mkdir(parents=True, exist_ok=True)
        self.state_path = self.internal / 'state.json'

    def save(self, state):
        fd, name = tempfile.mkstemp(dir=self.internal, prefix='state-', suffix='.tmp')
        try:
            with os.fdopen(fd, 'w') as out:
                json.dump(state, out, indent=2)
                out.flush()
                os.fsync(out.fileno())
            os.replace(name, self.state_path)
            directory = os.open(self.internal, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    @contextlib.contextmanager
    def locked(self):
        with (self.internal / 'lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.state_path.exists():
                try:
                    state = json.loads(self.state_path.read_text())
                    if not valid_state(state):
                        raise ValueError('invalid library state schema')
                except (ValueError, TypeError) as exc:
                    backup = self.internal / ('state.corrupt-' + uuid.uuid4().hex + '.json')
                    os.replace(self.state_path, backup)
                    sync_directory(self.internal)
                    print(f'Library state was corrupt ({exc}); preserved at {backup}. Rebuilding library; reconfigure saved profiles.', file=sys.stderr)
                    state = dict(version=1, games=[], jobs=[], roots=[str(self.games)], sources={})
            else:
                state = dict(version=1, games=[], jobs=[], roots=[str(self.games)], sources={})
            yield state
            self.save(state)

    def refresh_games(self, state):
        existing = {g['root']: dict(g, status='missing') for g in state['games']}
        for root, game in discover(state['roots']).items():
            game['profile'] = existing.get(root, {}).get('profile', DEFAULT_PROFILE.copy())
            existing[root] = game
        state['games'] = sorted(existing.values(), key=lambda g: (g['title'].casefold(), g['root']))

    def view(self, state):
        result = {k: state[k] for k in ('version', 'games', 'jobs', 'roots')}
        result['jobs'] = [{k: r[k] for k in ('source', 'status', 'message')} for r in state['sources'].values()]
        result['worker'] = self.worker_status()
        return result

    def snapshot(self):
        if self.worker_status()['running']:
            try:
                state = json.loads(self.state_path.read_text())
                if valid_state(state):
                    return self.view(state)
            except (OSError, ValueError):
                pass
            return self.view(dict(version=1, games=[], jobs=[], roots=[str(self.games)], sources={}))
        with self.locked() as state:
            self.refresh_games(state)
            return self.view(state)

    def profile(self, root, changes):
        if self.worker_status()['running']:
            raise RuntimeError('Import worker is active; profile edits are available when imports finish')
        root = str(Path(root).resolve())
        with self.locked() as state:
            if root not in state['roots']:
                state['roots'].append(root)
            self.refresh_games(state)
            game = next((g for g in state['games'] if g['root'] == root), None)
            if game is None:
                game = dict(root=root, eboot='', title=Path(root).name, title_id='', app_ver='', icon_path='', status='missing', profile=DEFAULT_PROFILE.copy())
                state['games'].append(game)
            game['profile'].update(changes)
            return game['profile'].copy()

    def import_sources(self, target=None, retry=False):
        with self.locked() as state:
            target = Path(target).resolve() if target else self.games
            if target.is_dir() and str(target) not in state['roots']:
                state['roots'].append(str(target))
            sources = [target] if target.is_file() else [base / name for base, files in walk(target) for name in sorted(files) if name.lower().endswith('.pkg')]
            for source in sources:
                if source.is_symlink():
                    continue
                key = str(source)
                ident = identity(source)
                previous = state['sources'].get(key, {})
                same = previous.get('identity') == ident
                if same and previous.get('status') == 'promoting':
                    if self.recover_promotion(previous):
                        self.save(state)
                    else:
                        previous.update(status='waiting', message='Interrupted import rolled back; retrying')
                if same and previous.get('status') == 'complete':
                    self.delete_receipted(source, previous)
                    continue
                if same and previous.get('status') == 'failed' and not retry:
                    continue
                record = dict(source=key, identity=ident, status='waiting', message='Waiting for source to finish copying')
                state['sources'][key] = record
                if not same and time.time() - source.stat().st_mtime < 5:
                    continue
                stage = Path(tempfile.mkdtemp(dir=self.internal, prefix='import-'))
                try:
                    record.update(status='extracting', message='Extracting and validating')
                    self.save(state)
                    if source.suffix.lower() == '.pkg':
                        if not extract_pkg(source, stage, force=True):
                            raise ValueError('PKG extraction failed; source retained')
                    else:
                        archive = stage / source.name
                        shutil.copy2(source, archive)
                        if source.name.lower().endswith(('.tar.gz', '.tgz')):
                            unpacked = stage / 'archive'
                            if not extract_archive(archive, unpacked, password='SuperPSX'):
                                raise ValueError('Archive extraction failed; source retained')
                            for base, files in list(walk(unpacked)):
                                for name in files:
                                    if name.lower().endswith('.tar'):
                                        if not extract_archive(base / name, unpacked / ('tar-' + uuid.uuid4().hex), password='SuperPSX'):
                                            raise ValueError('Nested tar extraction failed; source retained')
                            packages = [b / n for b, files in walk(unpacked) for n in files if n.lower().endswith('.pkg')]
                            if not packages or any(not extract_pkg(pkg, stage / 'titles', force=True) for pkg in packages):
                                raise ValueError('Archive PKG extraction failed; source retained')
                        elif not extract_game_target(archive, stage / 'titles'):
                            raise ValueError('Archive extraction failed; source retained')
                    titles = discover([stage])
                    if not titles or any(not re.fullmatch(r'[A-Za-z0-9_-]+', g['title_id']) or not isinstance(metadata(Path(g['root'])).get('TITLE'), str) or not metadata(Path(g['root'])).get('TITLE') for g in titles.values()):
                        raise ValueError('Extraction has no validated ELF and readable title metadata')
                    if identity(source) != ident:
                        raise ValueError('Source changed during extraction; retained for retry')
                    destinations = []
                    digest = uuid.uuid4().hex
                    record['token'] = digest
                    for index, game in enumerate(titles.values()):
                        dest = self.games / (game['title_id'] + '-' + digest + '-' + str(index))
                        if dest.exists():
                            raise ValueError('Destination conflict; source retained')
                        destinations.append((Path(game['root']), dest))
                    for origin, dest in destinations:
                        (origin / '.import-owner').write_text(digest)
                        sync_tree(origin)
                    record.update(status='promoting', destinations=[str(d) for _, d in destinations])
                    self.save(state)
                    for origin, dest in destinations:
                        os.rename(origin, dest)
                        sync_directory(origin.parent)
                        sync_directory(dest.parent)
                    record.update(status='complete', message='Validated and imported', destinations=[str(d) for _, d in destinations])
                    # Durable receipt is essential: existing title matches cannot authorize deletion.
                    self.save(state)
                    self.delete_receipted(source, record)
                except Exception as exc:
                    record.update(status='promoting' if record.get('status') in ('promoting', 'complete') else 'failed', message=str(exc))
                finally:
                    shutil.rmtree(stage, ignore_errors=True)
            state['jobs'] = [{k: r[k] for k in ('source', 'status', 'message')} for r in state['sources'].values()]
            self.refresh_games(state)
            return self.view(state)

    @contextlib.contextmanager
    def queue_lock(self):
        with (self.internal / 'queue.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def atomic_json(self, path, value):
        fd, name = tempfile.mkstemp(dir=self.internal, prefix='worker-', suffix='.tmp')
        try:
            with os.fdopen(fd, 'w') as out:
                json.dump(value, out)
                out.flush()
                os.fsync(out.fileno())
            os.replace(name, path)
            sync_directory(self.internal)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def read_json(self, name, default):
        try:
            return json.loads((self.internal / name).read_text())
        except (OSError, ValueError):
            return default

    def worker_status(self):
        status = self.read_json('worker.json', {})
        pid = status.get('pid')
        running = False
        if type(pid) is int and pid > 0 and status.get('running'):
            try:
                os.kill(pid, 0)
                # The dedicated flock is the authority once startup grace expires.
                with (self.internal / 'worker.lock').open('a') as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        running = time.time() - status.get('started_at', 0) < 5
                    except BlockingIOError:
                        running = True
            except (OSError, TypeError):
                pass
        status['running'] = running
        queue = self.read_json('queue.json', [])
        status['queued'] = len(queue) if isinstance(queue, list) else 0
        status['log_tail'] = ''
        log_path = status.get('log_path')
        if isinstance(log_path, str):
            try:
                with open(log_path, 'rb') as stream:
                    stream.seek(0, os.SEEK_END)
                    stream.seek(max(0, stream.tell() - 4096))
                    status['log_tail'] = stream.read().decode('utf-8', errors='replace')
            except OSError:
                pass
        return status

    def background_import(self, target=None, retry=False):
        target = str(Path(target).resolve()) if target else str(self.games)
        with self.queue_lock():
            queue = self.read_json('queue.json', [])
            if not isinstance(queue, list):
                raise RuntimeError('Import queue is corrupt; preserve queue.json and repair it before importing')
            request = dict(target=target, retry=bool(retry))
            if request not in queue:
                queue.append(request)
            self.atomic_json(self.internal / 'queue.json', queue)
            if not self.worker_status()['running']:
                logs = self.games.parent / 'Logs'
                logs.mkdir(exist_ok=True)
                log_path = logs / ('import_' + time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8] + '.log')
                with log_path.open('ab', buffering=0) as output:
                    process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(self.games.parent)],
                        stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
                self.atomic_json(self.internal / 'worker.json', dict(running=True, pid=process.pid, log_path=str(log_path), started_at=time.time(), current=''))
        return self.snapshot()

    def run_worker(self):
        with (self.internal / 'worker.lock').open('a') as lock:
            # A successor may start while its predecessor is finishing under
            # this lock. Wait so its durable queued request cannot be stranded.
            fcntl.flock(lock, fcntl.LOCK_EX)
            while True:
                with self.queue_lock():
                    queue = self.read_json('queue.json', [])
                    status = self.read_json('worker.json', {})
                    status.update(pid=os.getpid(), running=bool(queue))
                    if not queue:
                        status['current'] = ''
                        self.atomic_json(self.internal / 'worker.json', status)
                        return
                    request = queue[0]
                    status['current'] = request['target']
                    self.atomic_json(self.internal / 'worker.json', status)
                try:
                    print('Importing:', request['target'], flush=True)
                    result = self.import_sources(request['target'], request.get('retry', False))
                    # Keep the durable request until its own copying sources
                    # stabilize, even after the submitting GUI exits.
                    target = Path(request['target'])
                    directory = target.is_dir()
                    def waiting_for_request(job):
                        source = Path(job['source'])
                        belongs = source.is_relative_to(target) if directory else source == target
                        return belongs and job['status'] == 'waiting'
                    while any(waiting_for_request(j) for j in result['jobs']):
                        time.sleep(5)
                        result = self.import_sources(request['target'], request.get('retry', False))
                    print(json.dumps(result), flush=True)
                except Exception as exc:
                    print('Import failed:', str(exc), flush=True)
                with self.queue_lock():
                    queue = self.read_json('queue.json', [])
                    if queue and queue[0] == request:
                        queue.pop(0)
                    self.atomic_json(self.internal / 'queue.json', queue)

    def owned_destinations(self, record):
        token = record.get('token')
        destinations = record.get('destinations', [])
        if not isinstance(token, str) or not re.fullmatch(r'[0-9a-f]{32}', token) or not destinations:
            return False
        for name in destinations:
            path = Path(name)
            if path.parent != self.games or path.is_symlink() or not path.is_dir():
                return False
            try:
                if (path / '.import-owner').is_symlink() or (path / '.import-owner').read_text() != token:
                    return False
            except OSError:
                return False
        return True

    def recover_promotion(self, record):
        destinations = record.get('destinations', [])
        titles = discover(destinations) if self.owned_destinations(record) else {}
        if len(titles) == len(destinations) and titles and all(isinstance(metadata(Path(g['root'])).get('TITLE'), str) and re.fullmatch(r'[A-Za-z0-9_-]+', g['title_id']) for g in titles.values()):
            for name in destinations:
                sync_tree(Path(name))
            record.update(status='complete', message='Recovered validated import')
            return True
        # Roll back only directories carrying this transaction's durable marker.
        for name in destinations:
            one = dict(record, destinations=[name])
            if self.owned_destinations(one):
                shutil.rmtree(name)
                sync_directory(self.games)
        return False

    def delete_receipted(self, source, record):
        if source.suffix.lower() != '.pkg' or not source.is_relative_to(self.games) or source.is_symlink():
            return
        if not self.owned_destinations(record):
            return
        titles = discover(record.get('destinations', []))
        if len(titles) != len(record.get('destinations', [])) or not titles:
            return
        if all(isinstance(metadata(Path(g['root'])).get('TITLE'), str) and re.fullmatch(r'[A-Za-z0-9_-]+', g['title_id']) for g in titles.values()) and identity(source) == record['identity']:
            for destination in record['destinations']:
                sync_tree(Path(destination))
            source.unlink()
            sync_directory(source.parent)
            record['message'] = 'Validated and imported; source PKG removed'


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        Library(sys.argv[2]).run_worker()
    else:
        raise SystemExit('Internal usage: game_library.py --worker REPO')
