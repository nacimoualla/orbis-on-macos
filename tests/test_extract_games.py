#!/usr/bin/env python3
"""
Tests for PS4 Game and Archive Extraction (extract_games.py & run_workflow.py integration).
"""

import os
import sys
import shutil
import zipfile
import tempfile
import subprocess
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from extract_games import (
    parse_sfo,
    inspect_pkg,
    make_game_slug,
    is_already_extracted,
    extract_archive,
    extract_pkg,
    extract_game_target,
    find_pkg_tool,
)


class TestExtractGames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.games_dir = REPO_ROOT / "games"

    def test_find_pkg_tool(self):
        tool = find_pkg_tool()
        self.assertIsNotNone(tool, "ps4-pkg-tool executable must be discoverable")
        self.assertTrue(tool.is_file(), f"{tool} must exist")
        self.assertTrue(os.access(tool, os.X_OK), f"{tool} must be executable")

    def test_inspect_pkg_headers(self):
        undertale_pkg = self.games_dir / "[SPSX]-Undertale-CUSA09415-EUR-Game-(5.05+)-PS4.pkg"
        sonic_pkg = self.games_dir / "[SuperPSX]-Sonic.Mania.Plus-CUSA07010-EUR-Game-(5.05+)-PS4.pkg"

        if undertale_pkg.is_file():
            u_info = inspect_pkg(undertale_pkg)
            self.assertTrue(u_info["is_pkg"])
            self.assertEqual(u_info["content_id"], "EP3746-CUSA09415_00-TFSHVCUTPS4EU084")
            self.assertEqual(u_info["title_id"], "CUSA09415")
            self.assertEqual(u_info["title"], "Undertale")

        if sonic_pkg.is_file():
            s_info = inspect_pkg(sonic_pkg)
            self.assertTrue(s_info["is_pkg"])
            self.assertEqual(s_info["content_id"], "EP0177-CUSA07010_00-SONICMANIA000000")
            self.assertEqual(s_info["title_id"], "CUSA07010")
            self.assertEqual(s_info["title"], "Sonic Mania")

    def test_make_game_slug(self):
        self.assertEqual(
            make_game_slug({"content_id": "EP3746-CUSA09415_00-TFSHVCUTPS4EU084", "title": "Undertale"}, "undertale.pkg"),
            "undertale"
        )
        self.assertEqual(
            make_game_slug({"content_id": "EP0177-CUSA07010_00-SONICMANIA000000", "title": "Sonic Mania"}, "[SuperPSX]-Sonic.Mania.Plus-CUSA07010-EUR-Game-(5.05+)-PS4.pkg"),
            "sonic_mania_plus"
        )
        self.assertEqual(
            make_game_slug({"content_id": "ED1633-NPXX51362_00-TER1520000000000", "title": "Flappy Bird"}, "flappy.pkg"),
            "flappy_bird"
        )
        self.assertEqual(
            make_game_slug({"content_id": "EP4395-CUSA00909_00-THOMASWASALONEFG", "title": "Thomas Was Alone"}, "thomas.pkg"),
            "thomas_was_alone"
        )

    def test_is_already_extracted(self):
        # flappy_bird should be already extracted
        flappy_dest = is_already_extracted(self.games_dir, "ED1633-NPXX51362_00-TER1520000000000", "NPXX51362")
        self.assertIsNotNone(flappy_dest)
        self.assertTrue((flappy_dest / "eboot.elf").is_file())

        # Undertale should be already extracted
        undertale_dest = is_already_extracted(self.games_dir, "EP3746-CUSA09415_00-TFSHVCUTPS4EU084", "CUSA09415")
        self.assertIsNotNone(undertale_dest)
        self.assertTrue((undertale_dest / "eboot.elf").is_file())

        # Sonic Mania Plus should be already extracted
        sonic_dest = is_already_extracted(self.games_dir, "EP0177-CUSA07010_00-SONICMANIA000000", "CUSA07010")
        self.assertIsNotNone(sonic_dest)
        self.assertTrue((sonic_dest / "eboot.elf").is_file())

    def test_archive_extraction_with_password(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Create a zip archive
            zip_file = tmppath / "test.zip"
            with zipfile.ZipFile(zip_file, "w") as zf:
                zf.writestr("test_payload.txt", "superpsx_secret_content")

            # Encrypt with 7zz if available
            tool_7z = shutil.which("7zz") or shutil.which("7z")
            if tool_7z:
                archive_7z = tmppath / "test_encrypted.7z"
                subprocess.run(
                    [tool_7z, "a", "-pSuperPSX", "-y", str(archive_7z), str(zip_file)],
                    check=True,
                    capture_output=True,
                )
                out_extracted = tmppath / "out_extracted"
                extracted = extract_archive(archive_7z, out_extracted, password="SuperPSX")
                self.assertTrue(len(extracted) > 0)
                # Verify extracted content exists
                self.assertTrue((out_extracted / "test.zip").is_file())

    def test_run_workflow_extract_subcommand(self):
        cmd = [sys.executable, str(SCRIPTS_DIR / "run_workflow.py"), "extract"]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"run_workflow extract failed: {proc.stderr}")
        self.assertIn("Extracting PS4 Games", proc.stdout)
        self.assertIn("Extraction Results", proc.stdout)


if __name__ == "__main__":
    unittest.main()
