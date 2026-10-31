#!/usr/bin/env python3
"""
PS4 Game and Package Extractor

Automates extraction of PS4 games from PKG files and archives (.zip, .7z, .rar).
Handles password-protected archives (default: 'SuperPSX') and converts
extracted SELF eboot.bin files into runnable eboot.elf binaries using unpack_self.
"""

import os
import sys
import shutil
import struct
import tarfile
import zipfile
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
    for item in games_dir.rglob("eboot.bin"):
        p = item.parent
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
    """Extract compressed archive (.zip, .7z, .rar) using password if protected."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = archive_path.suffix.lower()
    extracted_files = []

    log(f"Extracting archive: {archive_path.name} (password: {password})...")

    # Try 7zz / unar first if available (supports RAR, 7z, ZIP with passwords reliably)
    tool_7z = shutil.which("7zz") or shutil.which("7z") or shutil.which("7za")
    tool_unar = shutil.which("unar")

    if tool_7z:
        cmd = [tool_7z, "x", f"-p{password}", "-y", f"-o{output_dir}", str(archive_path)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            log_success(f"Extracted {archive_path.name} via 7z")
        else:
            # Try without password
            cmd_nopass = [tool_7z, "x", "-y", f"-o{output_dir}", str(archive_path)]
            res2 = subprocess.run(cmd_nopass, capture_output=True, text=True)
            if res2.returncode == 0:
                log_success(f"Extracted {archive_path.name} via 7z (unencrypted)")
            else:
                log_fail(f"7z extraction failed: {res.stderr or res.stdout}")

    elif tool_unar:
        cmd = [tool_unar, "-p", password, "-o", str(output_dir), str(archive_path)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            log_success(f"Extracted {archive_path.name} via unar")
        else:
            log_fail(f"unar extraction failed: {res.stderr or res.stdout}")

    elif ext == ".zip":
        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                try:
                    zf.extractall(path=output_dir, pwd=password.encode("utf-8"))
                    log_success(f"Extracted {archive_path.name} via zipfile")
                except RuntimeError:
                    # Retry without password
                    zf.extractall(path=output_dir)
                    log_success(f"Extracted {archive_path.name} via zipfile (unencrypted)")
        except Exception as e:
            log_fail(f"zipfile extraction failed: {e}")
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
    Converts eboot.bin to eboot.elf and sets up environment.
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
        return existing

    pkg_tool = ensure_pkg_tool()

    log(f"Extracting PKG: {pkg_path.name}")
    log(f"  Title: {title_str} (ID: {title_id}, Content: {content_id})")
    log(f"  Destination: {final_dir}")

    target_parent.mkdir(parents=True, exist_ok=True)

    # ps4-pkg-tool extracts to <output_dir>/<title_id>
    cmd = [str(pkg_tool), str(pkg_path), str(target_parent)]
    log(f"Running: {' '.join(cmd)}")

    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        log_fail(f"ps4-pkg-tool failed with return code {proc.returncode}:\n{proc.stderr or proc.stdout}")
        return None

    # ps4-pkg-tool creates <target_parent>/<title_id>
    extracted_title_dir = target_parent / title_id
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
