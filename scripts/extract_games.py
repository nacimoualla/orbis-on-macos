#!/usr/bin/env python3
"""
PS4 Game and Package Extractor

Automates extraction of PS4 games from PKG files and archives (.zip, .7z, .rar).
Handles password-protected archives (default: 'SuperPSX') and converts
extracted SELF eboot.bin files into runnable eboot.elf binaries using unpack_self.
"""

import os
import sys
import time
import shutil
import struct
import tarfile
import zipfile
import threading
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Add scripts directory to path to import unpack_self
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from unpack_self import unpack_self
except ImportError:
    unpack_self = None

PKG_TOOL_RELEASES = {
    "darwin": "https://github.com/xXJSONDeruloXx/ps4-pkg-tools/releases/download/v20260801-153821-45baedaa/ps4-pkg-tools-macOS.tar.gz",
    "linux": "https://github.com/xXJSONDeruloXx/ps4-pkg-tools/releases/download/v20260801-153821-45baedaa/ps4-pkg-tools-Linux.tar.gz",
}


def log(msg: str, prefix: str = "ℹ", color: str = "\033[96m"):
    """Print formatted message with ANSI colors if supported."""
    reset = "\033[0m" if sys.stdout.isatty() else ""
    clr = color if sys.stdout.isatty() else ""
    print(f"{clr}{prefix} {msg}{reset}")


def log_success(msg: str):
    log(msg, prefix="✓", color="\033[92m")


def log_warn(msg: str):
    log(msg, prefix="⚠", color="\033[93m")


def log_fail(msg: str):
    log(msg, prefix="✗", color="\033[91m")


class ExtractionProgressTracker:
    """
    Real-time progress tracker for game and archive extractions.
    Monitors output directory size, write speed (MB/s), ETA, and tool stdout.
    Supports animated terminal progress bar and periodic non-interactive log updates.
    """

    def __init__(self, target_dir: Path, expected_bytes: int, label: str = "Extracting"):
        self.target_dir = target_dir
        self.expected_bytes = max(1, expected_bytes)
        self.label = label
        self.start_time = time.time()
        self.last_log_time = 0.0
        self.last_pct_milestone = -1
        self.is_tty = sys.stdout.isatty()
        self.stop_event = threading.Event()
        self.current_status = ""
        self.file_info = ""
        self.error_lines: List[str] = []
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the background monitor thread."""
        self.stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def set_status(self, status: str):
        """Update current active file or status text."""
        self.current_status = status.strip()

    def set_file_info(self, file_info: str):
        """Update file counter info, e.g. [5/26]."""
        self.file_info = file_info.strip()

    def _get_dir_size(self) -> int:
        """Efficient non-recursive directory size calculation using stack."""
        if not self.target_dir.exists():
            return 0
        total = 0
        stack = [str(self.target_dir)]
        while stack:
            cur = stack.pop()
            try:
                with os.scandir(cur) as it:
                    for entry in it:
                        try:
                            if entry.is_file(follow_symlinks=False):
                                total += entry.stat().st_size
                            elif entry.is_dir(follow_symlinks=False):
                                stack.append(entry.path)
                        except OSError:
                            pass
            except OSError:
                pass
        return total

    def _render(self, written: int, force_log: bool = False):
        """Render animated progress bar or output log milestone."""
        now = time.time()
        elapsed = max(0.001, now - self.start_time)
        speed = written / elapsed
        pct = (written / self.expected_bytes) * 100.0 if self.expected_bytes > 0 else 0.0
        if pct > 99.5 and not self.stop_event.is_set():
            pct = 99.5

        cols = shutil.get_terminal_size((80, 24)).columns
        bar_len = 15 if cols < 95 else 20
        filled = int(round(bar_len * (min(100.0, pct) / 100.0)))
        bar = "█" * filled + "░" * (bar_len - filled)

        cur_gb = written / (1024**3)
        tot_gb = self.expected_bytes / (1024**3)
        speed_mb = speed / (1024 * 1024)
        rem_bytes = max(0, self.expected_bytes - written)
        eta_sec = (rem_bytes / speed) if speed > 1024 else 0
        eta_str = f"{int(eta_sec)}s" if eta_sec < 60 else f"{int(eta_sec//60)}m{int(eta_sec%60):02d}s"

        status_part = f"{self.file_info} {self.current_status}".strip()

        if self.is_tty:
            prefix = f"\r\033[96m[{bar}]\033[0m {pct:5.1f}% | {cur_gb:.2f}/{tot_gb:.2f} GB | {speed_mb:5.1f} MB/s | ETA: {eta_str}"
            rem_cols = max(0, cols - len(prefix) - 4)
            if status_part and rem_cols > 8:
                if len(status_part) > rem_cols:
                    status_part = "..." + status_part[-(rem_cols - 3):]
                line = f"{prefix} | {status_part}"
            else:
                line = prefix
            sys.stdout.write(line.ljust(cols - 1) + "\r")
            sys.stdout.flush()
        else:
            milestone = int(pct // 10) * 10
            if force_log or (now - self.last_log_time >= 5.0) or (milestone > self.last_pct_milestone):
                tag = f"[{status_part}]" if status_part else ""
                print(f"ℹ Progress: {pct:5.1f}% ({cur_gb:.2f}/{tot_gb:.2f} GB) | {speed_mb:5.1f} MB/s | ETA: {eta_str} {tag}")
                self.last_log_time = now
                self.last_pct_milestone = milestone

    def _monitor_loop(self):
        while not self.stop_event.is_set():
            written = self._get_dir_size()
            self._render(written)
            time.sleep(0.25)

    def finish(self, success: bool = True):
        """Stop tracking and print final execution summary."""
        self.stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        final_written = self._get_dir_size()
        elapsed = max(0.001, time.time() - self.start_time)
        speed_mb = (final_written / elapsed) / (1024 * 1024)
        tot_gb = final_written / (1024**3)
        time_str = f"{int(elapsed)}s" if elapsed < 60 else f"{int(elapsed//60)}m{int(elapsed%60):02d}s"

        if self.is_tty:
            sys.stdout.write("\r" + " " * (shutil.get_terminal_size((80, 24)).columns - 1) + "\r")
            sys.stdout.flush()

        if success:
            log_success(f"{self.label} completed: {tot_gb:.2f} GB in {time_str} ({speed_mb:.1f} MB/s)")
        else:
            log_fail(f"{self.label} failed after {time_str}")


def find_pkg_tool() -> Optional[Path]:
    """Find ps4-pkg-tool in PATH or scripts/tools directory."""
    # 1. Check scripts/tools
    local_tool = SCRIPT_DIR / "tools" / "ps4-pkg-tool"
    if local_tool.is_file() and os.access(local_tool, os.X_OK):
        return local_tool

    # 2. Check PATH
    which_tool = shutil.which("ps4-pkg-tool")
    if which_tool:
        return Path(which_tool)

    return None


def ensure_pkg_tool() -> Path:
    """Ensure ps4-pkg-tool is present, downloading it if necessary."""
    tool = find_pkg_tool()
    if tool:
        return tool

    tools_dir = SCRIPT_DIR / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    target_tool = tools_dir / "ps4-pkg-tool"

    platform = "darwin" if sys.platform == "darwin" else ("linux" if sys.platform.startswith("linux") else None)
    if not platform or platform not in PKG_TOOL_RELEASES:
        raise RuntimeError(f"ps4-pkg-tool binary not found and automatic download is not supported for platform: {sys.platform}")

    url = PKG_TOOL_RELEASES[platform]
    tar_path = tools_dir / "ps4-pkg-tools.tar.gz"
    log(f"Downloading ps4-pkg-tool from {url}...")

    import urllib.request
    urllib.request.urlretrieve(url, tar_path)

    log(f"Extracting ps4-pkg-tool to {tools_dir}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith("ps4-pkg-tool"):
                member.name = os.path.basename(member.name)
                tar.extract(member, path=tools_dir)

    tar_path.unlink(missing_ok=True)
    if target_tool.is_file():
        target_tool.chmod(0o755)
        log_success(f"ps4-pkg-tool installed to {target_tool}")
        return target_tool

    raise RuntimeError(f"Failed to install ps4-pkg-tool to {target_tool}")


def parse_sfo(data: bytes) -> Dict[str, any]:
    """Parse PS4 param.sfo binary format."""
    if not data or data[:4] != b"\x00PSF":
        return {}
    key_off, data_off, count = struct.unpack("<III", data[8:20])
    entries = {}
    for i in range(count):
        off = 20 + i * 16
        k_off, fmt, d_len, max_len, v_off = struct.unpack("<HHIII", data[off : off + 16])
        k = data[key_off + k_off :].split(b"\x00", 1)[0].decode("utf-8", errors="ignore")
        val = data[data_off + v_off : data_off + v_off + d_len]
        if fmt == 0x0204:
            entries[k] = val.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")
        elif fmt == 0x0404 and len(val) >= 4:
            entries[k] = struct.unpack("<I", val[:4])[0]
        else:
            entries[k] = val
    return entries


def inspect_pkg(pkg_path: Path) -> Dict[str, any]:
    """Read metadata (Content ID, Title ID, Title, App Ver) directly from PKG header."""
    info = {
        "path": pkg_path,
        "content_id": "",
        "title_id": "",
        "title": "",
        "app_ver": "",
        "is_pkg": False,
    }
    if not pkg_path.is_file():
        return info

    with open(pkg_path, "rb") as f:
        magic = f.read(4)
        if magic != b"\x7fCNT":
            return info

        info["is_pkg"] = True
        f.seek(0x10)
        entry_count = struct.unpack(">I", f.read(4))[0]
        f.seek(0x18)
        table_offset = struct.unpack(">I", f.read(4))[0]

        f.seek(0x40)
        info["content_id"] = f.read(36).decode("ascii", errors="ignore").rstrip("\x00")

        # Parse entry table to find param.sfo (entry_id == 0x1000)
        f.seek(table_offset)
        sfo_offset = None
        sfo_size = None
        for _ in range(entry_count):
            eid, fn_off, f1, f2, offset, size, _ = struct.unpack(">IIIIII8s", f.read(32))
            if eid == 0x1000:
                sfo_offset = offset
                sfo_size = size
                break

        if sfo_offset and sfo_size:
            f.seek(sfo_offset)
            sfo_data = f.read(sfo_size)
            sfo = parse_sfo(sfo_data)
            info["title"] = sfo.get("TITLE", "")
            info["title_id"] = sfo.get("TITLE_ID", "")
            info["app_ver"] = sfo.get("APP_VER", "")

    if not info["title_id"] and info["content_id"]:
        parts = info["content_id"].split("-")
        if len(parts) >= 2:
            info["title_id"] = parts[1].split("_")[0]

    return info


def make_game_slug(info: Dict[str, any], filename: str) -> str:
    """Derive an idiomatic directory slug for the game."""
    content_id = info.get("content_id", "")
    title_id = info.get("title_id", "")
    title = info.get("title", "")
    lower_fn = filename.lower()
    lower_title = title.lower()

    if "npxx51362" in content_id.lower() or "npxx51374" in content_id.lower() or title_id in {"NPXX51362", "NPXX51374"} or "flappy" in lower_fn or "flappy" in lower_title:
        return "flappy_bird"
    if "cusa00909" in content_id.lower() or title_id == "CUSA00909" or "thomas" in lower_fn or "thomas" in lower_title:
        return "thomas_was_alone"
    if "cusa09415" in content_id.lower() or title_id == "CUSA09415" or "undertale" in lower_fn or "undertale" in lower_title:
        return "undertale"
    if "cusa07010" in content_id.lower() or title_id == "CUSA07010" or "sonic" in lower_fn or "sonic" in lower_title:
        return "sonic_mania_plus" if ("plus" in lower_fn or "plus" in lower_title) else "sonic_mania"

    if title:
        clean = "".join(c if c.isalnum() or c in " _-" else "" for c in title)
        slug = "_".join(clean.lower().split())
        if slug:
            return slug

    if title_id:
        return title_id.lower()

    return Path(filename).stem.lower()


def is_already_extracted(games_dir: Path, content_id: str, title_id: str) -> Optional[Path]:
    """Check if a game with the given content_id or title_id is already extracted."""
    if not games_dir.is_dir():
        return None
    seen_dirs = set()
    for pattern in ["eboot.elf", "eboot.bin"]:
        for item in games_dir.rglob(pattern):
            p = item.parent
            if p in seen_dirs:
                continue
            seen_dirs.add(p)
            # Check if parent matches content_id or title_id
            if content_id and p.name == content_id:
                return p
            if title_id and p.name == title_id:
                return p
            # Check param.sfo in parent
            sfo_path = p / "sce_sys" / "param.sfo"
            if sfo_path.is_file():
                try:
                    sfo = parse_sfo(sfo_path.read_bytes())
                    if title_id and sfo.get("TITLE_ID") == title_id:
                        return p
                except Exception:
                    pass
    return None


def extract_archive(
    archive_path: Path,
    output_dir: Path,
    password: str = "SuperPSX",
) -> List[Path]:
    """Extract compressed archive (.zip, .7z, .rar) with real-time progress."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = archive_path.suffix.lower()
    extracted_files = []
    arch_size = archive_path.stat().st_size if archive_path.is_file() else 0

    log(f"Extracting archive: {archive_path.name} (password: {password})...")

    # Try 7zz / unar first if available (supports RAR, 7z, ZIP with passwords reliably)
    tool_7z = shutil.which("7zz") or shutil.which("7z") or shutil.which("7za")
    tool_unar = shutil.which("unar")

    if tool_7z:
        tracker = ExtractionProgressTracker(
            target_dir=output_dir,
            expected_bytes=arch_size,
            label=f"Unpacking {archive_path.name}",
        )
        tracker.start()
        cmd = [tool_7z, "x", f"-p{password}", "-y", "-bsp1", f"-o{output_dir}", str(archive_path)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            if proc.stdout:
                for line in iter(proc.stdout.readline, ''):
                    l_str = line.strip()
                    if not l_str:
                        continue
                    if l_str.endswith("%"):
                        tracker.set_file_info(f"[{l_str}]")
                    elif "+" in l_str:
                        parts = l_str.split("+", 1)
                        if len(parts) == 2:
                            tracker.set_status(parts[1].strip())
                    elif "error" in l_str.lower():
                        tracker.error_lines.append(l_str)
        finally:
            if proc.stdout:
                proc.stdout.close()
            res = proc.wait()
            tracker.finish(success=(res == 0))

        if res == 0:
            log_success(f"Extracted {archive_path.name} via 7z")
        else:
            # Try without password
            tracker = ExtractionProgressTracker(
                target_dir=output_dir,
                expected_bytes=arch_size,
                label=f"Unpacking {archive_path.name} (unencrypted)",
            )
            tracker.start()
            cmd_nopass = [tool_7z, "x", "-y", "-bsp1", f"-o{output_dir}", str(archive_path)]
            proc2 = subprocess.Popen(cmd_nopass, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                if proc2.stdout:
                    for line in iter(proc2.stdout.readline, ''):
                        l_str = line.strip()
                        if not l_str:
                            continue
                        if l_str.endswith("%"):
                            tracker.set_file_info(f"[{l_str}]")
                        elif "+" in l_str:
                            parts = l_str.split("+", 1)
                            if len(parts) == 2:
                                tracker.set_status(parts[1].strip())
            finally:
                if proc2.stdout:
                    proc2.stdout.close()
                res2 = proc2.wait()
                tracker.finish(success=(res2 == 0))

            if res2 == 0:
                log_success(f"Extracted {archive_path.name} via 7z (unencrypted)")
            else:
                log_fail(f"7z extraction failed: {res2}")

    elif tool_unar:
        tracker = ExtractionProgressTracker(
            target_dir=output_dir,
            expected_bytes=arch_size,
            label=f"Unpacking {archive_path.name}",
        )
        tracker.start()
        cmd = [tool_unar, "-p", password, "-o", str(output_dir), str(archive_path)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            if proc.stdout:
                for line in iter(proc.stdout.readline, ''):
                    l_str = line.strip()
                    if l_str:
                        tracker.set_status(l_str)
        finally:
            if proc.stdout:
                proc.stdout.close()
            res = proc.wait()
            tracker.finish(success=(res == 0))
        if res == 0:
            log_success(f"Extracted {archive_path.name} via unar")
        else:
            log_fail(f"unar extraction failed with code {res}")

    elif ext == ".zip":
        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                members = zf.infolist()
                total_bytes = sum(m.file_size for m in members)
                tracker = ExtractionProgressTracker(
                    target_dir=output_dir,
                    expected_bytes=total_bytes,
                    label=f"Extracting {archive_path.name}",
                )
                tracker.start()
                try:
                    for idx, member in enumerate(members):
                        tracker.set_file_info(f"[{idx+1}/{len(members)}]")
                        tracker.set_status(member.filename)
                        try:
                            zf.extract(member, path=output_dir, pwd=password.encode("utf-8"))
                        except RuntimeError:
                            zf.extract(member, path=output_dir)
                    tracker.finish(success=True)
                    log_success(f"Extracted {archive_path.name} via zipfile")
                except Exception as e:
                    tracker.finish(success=False)
                    log_fail(f"zipfile extraction failed: {e}")
        except Exception as e:
            log_fail(f"zipfile open failed: {e}")
    else:
        log_fail(f"No suitable decompression tool found to unpack {archive_path.name}. Please install 7zz or unar.")

    # Gather extracted files
    for root, _, files in os.walk(output_dir):
        for f in files:
            extracted_files.append(Path(root) / f)

    return extracted_files


def extract_pkg(
    pkg_path: Path,
    games_dir: Path,
    force: bool = False,
) -> Optional[Path]:
    """
    Extract a single PS4 PKG into games/<slug>/<content_id>.
    Converts eboot.bin to eboot.elf, configures VFS environment,
    and displays a real-time progress indicator during unpacking.
    """
    info = inspect_pkg(pkg_path)
    if not info["is_pkg"]:
        log_warn(f"{pkg_path.name} is not a valid PS4 PKG container.")
        return None

    content_id = info["content_id"] or "CUSA_UNKNOWN"
    title_id = info["title_id"] or "CUSA_UNKNOWN"
    slug = make_game_slug(info, pkg_path.name)
    title_str = info["title"] or slug

    target_parent = games_dir / slug
    final_dir = target_parent / content_id

    # Check if already extracted in games_dir
    existing = is_already_extracted(games_dir, content_id, title_id)
    if not force and existing:
        log(f"Game already extracted: {title_str} ({existing})", prefix="✓", color="\033[92m")
        # Ensure eboot.elf is ready
        eboot_elf = existing / "eboot.elf"
        eboot_bin = existing / "eboot.bin"
        if not eboot_elf.is_file() and eboot_bin.is_file():
            convert_eboot_bin_to_elf(eboot_bin, eboot_elf)
        (existing / "debug.log").touch(exist_ok=True)
        (existing / "vfs_root").mkdir(parents=True, exist_ok=True)
        return existing

    pkg_tool = ensure_pkg_tool()

    log(f"Extracting PKG: {pkg_path.name}")
    log(f"  Title: {title_str} (ID: {title_id}, Content: {content_id})")
    log(f"  Destination: {final_dir}")

    target_parent.mkdir(parents=True, exist_ok=True)
    extracted_title_dir = target_parent / title_id
    extracted_title_dir.mkdir(parents=True, exist_ok=True)

    # Pre-create internal directories so unpatched ps4-pkg-tool will not fail fopen
    (extracted_title_dir / "sce_module").mkdir(parents=True, exist_ok=True)
    (extracted_title_dir / "sce_sys").mkdir(parents=True, exist_ok=True)

    # ps4-pkg-tool extracts to <output_dir>/<title_id>
    cmd = [str(pkg_tool), str(pkg_path), str(target_parent)]
    log(f"Running: {' '.join(cmd)}")

    pkg_size = pkg_path.stat().st_size if pkg_path.is_file() else 0
    tracker = ExtractionProgressTracker(
        target_dir=target_parent,
        expected_bytes=pkg_size,
        label=f"Extracting {pkg_path.name}",
    )
    tracker.start()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        if proc.stdout:
            for line in iter(proc.stdout.readline, ''):
                line_clean = line.strip()
                if not line_clean:
                    continue
                if line_clean.startswith("Extracting: "):
                    tracker.set_status(line_clean[12:])
                elif line_clean.startswith("Progress: "):
                    tracker.set_file_info(f"[{line_clean[10:]}]")
                elif "failed" in line_clean.lower() or "error" in line_clean.lower():
                    tracker.error_lines.append(line_clean)
    finally:
        if proc.stdout:
            proc.stdout.close()
        ret = proc.wait()
        tracker.finish(success=(ret == 0))

    if ret != 0:
        err_msg = "\n".join(tracker.error_lines) if tracker.error_lines else f"Process exited with return code {ret}"
        log_fail(f"ps4-pkg-tool failed with return code {ret}:\n{err_msg}")
        return None

    # ps4-pkg-tool creates <target_parent>/<title_id>
    if extracted_title_dir.is_dir() and extracted_title_dir != final_dir:
        if final_dir.exists():
            shutil.rmtree(final_dir)
        extracted_title_dir.rename(final_dir)
        # Create a title_id symlink pointing to content_id for convenient lookup
        try:
            if not extracted_title_dir.exists():
                os.symlink(content_id, extracted_title_dir)
        except OSError:
            pass
    elif not final_dir.is_dir():
        # Check if ps4-pkg-tool extracted directly into target_parent
        if (target_parent / "eboot.bin").is_file():
            temp_extracted = target_parent / "extracted_tmp"
            temp_extracted.mkdir(parents=True, exist_ok=True)
            for item in target_parent.iterdir():
                if item != temp_extracted:
                    shutil.move(str(item), str(temp_extracted / item.name))
            temp_extracted.rename(final_dir)

    if not final_dir.is_dir():
        log_fail(f"Could not locate extracted game directory for {pkg_path.name}")
        return None

    # Post-extraction setup:
    # 1. Convert eboot.bin to eboot.elf using unpack_self
    eboot_bin = final_dir / "eboot.bin"
    eboot_elf = final_dir / "eboot.elf"
    if eboot_bin.is_file() and not eboot_elf.is_file():
        convert_eboot_bin_to_elf(eboot_bin, eboot_elf)

    # 2. Touch debug.log and ensure vfs_root/ exists
    (final_dir / "debug.log").touch(exist_ok=True)
    (final_dir / "vfs_root").mkdir(parents=True, exist_ok=True)

    log_success(f"Successfully extracted: {title_str} -> {final_dir.relative_to(REPO_ROOT) if final_dir.is_relative_to(REPO_ROOT) else final_dir}")
    return final_dir


def convert_eboot_bin_to_elf(eboot_bin: Path, eboot_elf: Path):
    """Unpack SELF eboot.bin to ELF eboot.elf."""
    if not unpack_self:
        log_warn("unpack_self module not available; skipping eboot.elf reconstruction.")
        return

    try:
        with open(eboot_bin, "rb") as f:
            magic = f.read(4)
        if magic == b"O\x15=\x1d":
            log(f"Unpacking SELF container: {eboot_bin.name} -> {eboot_elf.name}...")
            unpack_self(eboot_bin, eboot_elf)
            log_success(f"Created runnable binary: {eboot_elf.name}")
        elif magic == b"\x7fELF":
            # Already an ELF!
            shutil.copyfile(eboot_bin, eboot_elf)
            log_success(f"eboot.bin was already ELF format, copied to {eboot_elf.name}")
    except Exception as e:
        log_warn(f"Failed to unpack {eboot_bin} to {eboot_elf}: {e}")


def extract_all_in_folder(
    folder: Path,
    games_dir: Path,
    password: str = "SuperPSX",
    force: bool = False,
) -> List[Path]:
    """Scan folder for PKGs and archives, extracting all new games."""
    extracted_games = []

    # 1. Check for compressed archives (.zip, .7z, .rar)
    archive_exts = {".zip", ".7z", ".rar", ".tar.gz", ".tgz"}
    archive_files = [f for f in folder.iterdir() if f.is_file() and any(f.name.lower().endswith(ext) for ext in archive_exts)]

    for arch in archive_files:
        temp_out = folder / f"_tmp_{arch.stem}"
        arch_extracted = extract_archive(arch, temp_out, password=password)
        # Search extracted files for any PKG files
        for item in arch_extracted:
            if item.suffix.lower() == ".pkg":
                dest = extract_pkg(item, games_dir, force=force)
                if dest:
                    extracted_games.append(dest)
        # Clean up temp folder if it only contained the PKG
        shutil.rmtree(temp_out, ignore_errors=True)

    # 2. Check for PKG files directly in folder
    pkg_files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() == ".pkg"]
    for pkg in pkg_files:
        dest = extract_pkg(pkg, games_dir, force=force)
        if dest:
            extracted_games.append(dest)

    return extracted_games


def extract_game_target(
    target: Path,
    games_dir: Optional[Path] = None,
    password: str = "SuperPSX",
    force: bool = False,
) -> List[Path]:
    """Extract a specified file (PKG or archive) or directory."""
    if games_dir is None:
        games_dir = REPO_ROOT / "games"

    target = target.resolve()
    if target.is_dir():
        return extract_all_in_folder(target, games_dir, password=password, force=force)
    elif target.is_file():
        ext = target.suffix.lower()
        if ext == ".pkg":
            res = extract_pkg(target, games_dir, force=force)
            return [res] if res else []
        elif ext in {".zip", ".7z", ".rar", ".tar.gz", ".tgz"}:
            temp_out = target.parent / f"_tmp_{target.stem}"
            arch_extracted = extract_archive(target, temp_out, password=password)
            results = []
            for item in arch_extracted:
                if item.suffix.lower() == ".pkg":
                    dest = extract_pkg(item, games_dir, force=force)
                    if dest:
                        results.append(dest)
            shutil.rmtree(temp_out, ignore_errors=True)
            return results
        else:
            log_fail(f"Unsupported file type: {target.name}")
            return []
    else:
        log_fail(f"Target path does not exist: {target}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="PS4 Game and PKG Extractor (Supports 'SuperPSX' password for archives)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=str(REPO_ROOT / "games"),
        help="Path to PKG, archive (.zip/.7z/.rar), or directory (default: games/)",
    )
    parser.add_argument(
        "--password",
        "-p",
        default="SuperPSX",
        help="Password for encrypted archives (default: 'SuperPSX')",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=str(REPO_ROOT / "games"),
        help="Base games output directory (default: games/)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-extraction of already extracted games",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files and targets without extracting",
    )

    args = parser.parse_args()

    target_path = Path(args.target)
    games_out = Path(args.output_dir)

    print(f"\n\033[96m{'=' * 64}\033[0m")
    print(f"\033[1m\033[96m       PS4 Game Package & Archive Extractor       \033[0m")
    print(f"\033[96m{'=' * 64}\033[0m")
    log(f"Target: {target_path}")
    log(f"Output: {games_out}")
    log(f"Archive Password: {args.password}")

    if args.dry_run:
        log("DRY-RUN mode enabled.")
        if target_path.is_dir():
            for p in target_path.glob("*.pkg"):
                info = inspect_pkg(p)
                print(f"  PKG: {p.name} -> {info.get('title', 'Unknown')} ({info.get('content_id', '')})")
            for p in target_path.iterdir():
                if p.suffix.lower() in {".zip", ".7z", ".rar"}:
                    print(f"  Archive: {p.name}")
        return

    results = extract_game_target(
        target=target_path,
        games_dir=games_out,
        password=args.password,
        force=args.force,
    )

    print(f"\n\033[1mExtraction Summary:\033[0m")
    if results:
        for r in results:
            log_success(f"Game Ready: {r.name} ({r})")
    else:
        log("No new games were extracted.", prefix="ℹ")


if __name__ == "__main__":
    main()
