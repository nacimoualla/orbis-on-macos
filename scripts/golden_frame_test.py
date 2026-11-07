#!/usr/bin/env python3
"""
Golden-frame regression test:
Runs ps4-runner headless-ish with PS4_RUN_SECONDS and PS4_GPU_READBACK on fixed frames
for Flappy Bird, Sonic Mania, and Dragon Ball Xenoverse 2 (logos and opening movie),
compares against stored reference images with tolerance, and records guest fps per
scene from PS4_FLIP_STATS into a CSV file.
"""

import argparse
import csv
import datetime
import os
import re
import shutil
import subprocess
import sys
import tempfile

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

TARGET_WIDTH = 160
TARGET_HEIGHT = 90

SCENES = [
    {
        "game_id": "flappy_bird",
        "name": "Flappy Bird",
        "rel_eboot": "flappy_bird/ED1633-NPXX51362_00-TER1520000000000/eboot.elf",
        "run_seconds": 3,
        "readback_every": 1,
        "readback_from": 0,
        "targets": [
            {
                "scene": "flappy_bird",
                "frame": 10,
                "ref_file": "flappy_bird_frame10.png",
            }
        ],
    },
    {
        "game_id": "sonic_mania",
        "name": "Sonic Mania",
        "rel_eboot": "sonic_mania_plus/EP0177-CUSA07010_00-SONICMANIA000000/eboot.elf",
        "run_seconds": 3,
        "readback_every": 1,
        "readback_from": 0,
        "targets": [
            {
                "scene": "sonic_mania",
                "frame": 2,
                "ref_file": "sonic_mania_frame2.png",
            }
        ],
    },
    {
        "game_id": "dragon_ball_xenoverse_2",
        "name": "Dragon Ball Xenoverse 2",
        "rel_eboot": "dragon_ball_xenoverse_2/UP0700-CUSA05350_00-DRAGONBALLXV2000/eboot.elf",
        "run_seconds": 25,
        "readback_every": 30,
        "readback_from": 0,
        "targets": [
            {
                "scene": "dbxv2_logos",
                "frame": 300,
                "ref_file": "dbxv2_logos_frame300.png",
            },
            {
                "scene": "dbxv2_opening_movie",
                "frame": 810,
                "ref_file": "dbxv2_opening_movie_frame810.png",
            },
        ],
    },
]


def load_ppm(path):
    """Load PPM P6 file and return (width, height, bytes)."""
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"P6":
            raise ValueError(f"Invalid PPM header in {path}: {magic}")
        line = f.readline().strip()
        while line.startswith(b"#"):
            line = f.readline().strip()
        w, h = map(int, line.split())
        max_val = f.readline().strip()
        data = f.read()
        return w, h, data


def downscale_image(ppm_path, out_width=TARGET_WIDTH, out_height=TARGET_HEIGHT):
    """Downscales PPM to (out_width, out_height). Returns (data_bytes, pil_or_ppm)."""
    if HAS_PIL:
        img = Image.open(ppm_path).convert("RGB")
        resized = img.resize((out_width, out_height), Image.Resampling.BILINEAR)
        return resized.tobytes(), resized
    else:
        w, h, data = load_ppm(ppm_path)
        block_w = w // out_width
        block_h = h // out_height
        out = bytearray(out_width * out_height * 3)
        for oy in range(out_height):
            for ox in range(out_width):
                r_sum = g_sum = b_sum = 0
                count = block_w * block_h
                for by in range(block_h):
                    iy = oy * block_h + by
                    for bx in range(block_w):
                        ix = ox * block_w + bx
                        idx = (iy * w + ix) * 3
                        r_sum += data[idx]
                        g_sum += data[idx + 1]
                        b_sum += data[idx + 2]
                oidx = (oy * out_width + ox) * 3
                out[oidx] = r_sum // count
                out[oidx + 1] = g_sum // count
                out[oidx + 2] = b_sum // count
        return bytes(out), None


def load_reference(ref_path):
    """Load reference image bytes."""
    if HAS_PIL:
        img = Image.open(ref_path).convert("RGB")
        return img.tobytes()
    else:
        # Fallback if saved as PPM
        _, _, data = load_ppm(ref_path)
        return data


def save_reference(ref_path, downscaled_bytes, pil_img):
    """Save reference image."""
    os.makedirs(os.path.dirname(os.path.abspath(ref_path)), exist_ok=True)
    if HAS_PIL and pil_img is not None:
        pil_img.save(ref_path)
    else:
        ppm_path = ref_path if ref_path.endswith(".ppm") else ref_path + ".ppm"
        with open(ppm_path, "wb") as f:
            f.write(f"P6\n{TARGET_WIDTH} {TARGET_HEIGHT}\n255\n".encode())
            f.write(downscaled_bytes)


def compute_mae(bytes1, bytes2):
    """Compute normalized Mean Absolute Error in [0.0, 1.0]."""
    if len(bytes1) != len(bytes2):
        return 1.0
    total_diff = sum(abs(b1 - b2) for b1, b2 in zip(bytes1, bytes2))
    return total_diff / (len(bytes1) * 255.0)


def parse_fps_metrics(stderr_text):
    """Parse [fps] lines from stderr and return summary stats."""
    # Matches: [fps] guest 57 fps, worst frame 49.5 ms, frames >20ms: 2
    guest_fps_matches = re.findall(
        r"\[fps\] guest (\d+) fps(?:, worst frame ([\d\.]+) ms)?(?:, frames >20ms: (\d+))?",
        stderr_text,
    )
    fps_list = []
    worst_frame = 0.0
    total_hitches = 0
    for match in guest_fps_matches:
        fps_list.append(int(match[0]))
        if match[1]:
            worst_frame = max(worst_frame, float(match[1]))
        if match[2]:
            total_hitches += int(match[2])

    # Also capture hitch lines: [fps] hitch 56.0 ms before flip #240
    hitch_matches = re.findall(r"\[fps\] hitch ([\d\.]+) ms", stderr_text)
    for h in hitch_matches:
        worst_frame = max(worst_frame, float(h))

    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0.0
    min_fps = min(fps_list) if fps_list else 0
    max_fps = max(fps_list) if fps_list else 0

    return {
        "avg_fps": round(avg_fps, 1),
        "min_fps": min_fps,
        "max_fps": max_fps,
        "worst_frame_ms": round(worst_frame, 1),
        "hitches": total_hitches,
    }


def find_eboot(games_dir, rel_path):
    """Find eboot binary either directly or through search paths."""
    candidates = [
        os.path.join(games_dir, rel_path),
        os.path.join(games_dir, rel_path.replace(".elf", ".bin")),
        os.path.join("/Users/nacimoualla/OpenResearch/ps4-to-macos/games", rel_path),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def record_to_csv(csv_path, rows):
    """Write or append rows to CSV."""
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    fieldnames = [
        "timestamp",
        "scene",
        "target_frame",
        "status",
        "mae",
        "guest_fps_avg",
        "guest_fps_min",
        "guest_fps_max",
        "worst_frame_ms",
        "hitches",
    ]
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Golden-frame regression test")
    parser.add_argument("--runner", default="build/ps4-runner", help="Path to ps4-runner")
    parser.add_argument("--games-dir", default="games", help="Path to games directory")
    parser.add_argument(
        "--ref-dir", default="tests/golden_frames", help="Path to reference images"
    )
    parser.add_argument(
        "--csv", default="build/fps_metrics.csv", help="Output CSV for FPS stats"
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.08, help="Max allowed MAE (default 0.08)"
    )
    parser.add_argument(
        "--record", action="store_true", help="Record reference images instead of verifying"
    )
    parser.add_argument(
        "--target", default="all", help="Specific game to test (flappy, sonic, dbxv2, or all)"
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runner_bin = os.path.abspath(args.runner)
    if not os.path.isfile(runner_bin):
        # Check relative to repo root
        alt = os.path.join(repo_root, args.runner)
        if os.path.isfile(alt):
            runner_bin = alt
        else:
            print(f"[ERROR] ps4-runner not found at {args.runner}", file=sys.stderr)
            return 1

    games_dir = os.path.abspath(args.games_dir)
    ref_dir = os.path.abspath(args.ref_dir)
    csv_path = os.path.abspath(args.csv)

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    all_passed = True
    csv_rows = []

    for game in SCENES:
        if args.target != "all" and args.target not in (game["game_id"], game["name"].lower()):
            continue

        print(f"\n=======================================================")
        print(f"Running Golden-Frame Test: {game['name']}")
        print(f"=======================================================")

        eboot = find_eboot(games_dir, game["rel_eboot"])
        if not eboot:
            print(f"[SKIP] Game binary not found: {game['rel_eboot']}")
            continue

        with tempfile.TemporaryDirectory() as tmp_dir:
            env = os.environ.copy()
            env["PS4_RUN_SECONDS"] = str(game["run_seconds"])
            env["PS4_GPU_READBACK"] = tmp_dir
            env["PS4_GPU_READBACK_EVERY"] = str(game["readback_every"])
            env["PS4_GPU_READBACK_FROM"] = str(game["readback_from"])
            env["PS4_FLIP_STATS"] = "1"
            # Ensure deterministic execution with no auto input
            env.pop("PS4_AUTO_PRESS", None)
            env.pop("PS4_AUTO_TAP", None)
            env.pop("PS4_AUTO_MOVE", None)

            cmd = [runner_bin, "--window", eboot]
            print(f"Executing: PS4_RUN_SECONDS={game['run_seconds']} {' '.join(cmd)}")
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)

            fps_stats = parse_fps_metrics(proc.stderr)
            print(
                f"[FPS Stats] avg={fps_stats['avg_fps']} min={fps_stats['min_fps']} "
                f"max={fps_stats['max_fps']} worst_frame={fps_stats['worst_frame_ms']}ms "
                f"hitches={fps_stats['hitches']}"
            )

            # Check each target frame for this scene
            for target in game["targets"]:
                frame_num = target["frame"]
                scene_name = target["scene"]
                ref_file = target["ref_file"]
                ref_path = os.path.join(ref_dir, ref_file)

                # Locate dumped frame
                frame_candidates = [
                    os.path.join(tmp_dir, f"frame_{frame_num}_gcn.ppm"),
                    os.path.join(tmp_dir, f"frame_{frame_num}_fixed.ppm"),
                ]
                dumped_path = None
                for c in frame_candidates:
                    if os.path.isfile(c):
                        dumped_path = c
                        break

                if not dumped_path:
                    # Look for any dumped frame in tmp_dir to diagnose
                    existing = sorted(os.listdir(tmp_dir))
                    print(
                        f"  ✗ Frame {frame_num} not dumped for {scene_name}! "
                        f"(Dumped frames in run: {existing[:10]}...)",
                        file=sys.stderr,
                    )
                    all_passed = False
                    csv_rows.append(
                        {
                            "timestamp": now_iso,
                            "scene": scene_name,
                            "target_frame": frame_num,
                            "status": "MISSING_FRAME",
                            "mae": 1.0,
                            "guest_fps_avg": fps_stats["avg_fps"],
                            "guest_fps_min": fps_stats["min_fps"],
                            "guest_fps_max": fps_stats["max_fps"],
                            "worst_frame_ms": fps_stats["worst_frame_ms"],
                            "hitches": fps_stats["hitches"],
                        }
                    )
                    continue

                down_bytes, pil_img = downscale_image(dumped_path)

                if args.record or not os.path.exists(ref_path):
                    save_reference(ref_path, down_bytes, pil_img)
                    print(f"  ✓ Recorded reference image: {ref_path}")
                    status = "RECORDED"
                    mae = 0.0
                else:
                    ref_bytes = load_reference(ref_path)
                    mae = compute_mae(down_bytes, ref_bytes)
                    if mae <= args.tolerance:
                        print(
                            f"  ✓ {scene_name} (frame {frame_num}) matches reference (MAE={mae:.4f} <= tol {args.tolerance})"
                        )
                        status = "PASS"
                    else:
                        print(
                            f"  ✗ {scene_name} (frame {frame_num}) MISMATCH (MAE={mae:.4f} > tol {args.tolerance})",
                            file=sys.stderr,
                        )
                        all_passed = False
                        status = "FAIL"

                csv_rows.append(
                    {
                        "timestamp": now_iso,
                        "scene": scene_name,
                        "target_frame": frame_num,
                        "status": status,
                        "mae": round(mae, 4),
                        "guest_fps_avg": fps_stats["avg_fps"],
                        "guest_fps_min": fps_stats["min_fps"],
                        "guest_fps_max": fps_stats["max_fps"],
                        "worst_frame_ms": fps_stats["worst_frame_ms"],
                        "hitches": fps_stats["hitches"],
                    }
                )

    if csv_rows:
        record_to_csv(csv_path, csv_rows)
        print(f"\n[FPS Metrics] Recorded {len(csv_rows)} rows to {csv_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
