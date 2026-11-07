#!/usr/bin/env python3
"""
PS4 PM4 Log Collector

Collects and analyzes data from PS4 PM4 translator logs.
Supports specifying log file via command-line argument or LOG_FILE env var.

Usage:
    python3 scripts/collect_logs.py /path/to/logfile.log
    LOG_FILE=/path/to/log.py scripts/collect_logs.py
    scripts/collect_logs.py --log /path/to/logs --pattern "0x[0-9A-F]" --count
"""

import sys
import os
import re
import argparse
from collections import Counter, defaultdict


# Safe No-Ops opcodes from the PM4 translator project
SAFE_NOOPS = {
    0x0B, 0x22, 0x23, 0x32, 0x42, 0x43, 0x4A,
    0x59, 0x5F, 0x60, 0x61, 0x73, 0x77, 0x78,
    0x79, 0x80, 0x81, 0x83, 0x84, 0x85, 0x86,
    0x88, 0x8B,
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect and analyze PS4 PM4 log data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--log",
        "-l",
        type=str,
        help="Path to the log file to analyze (overrides LOG_FILE env var)",
        default=None,
    )
    parser.add_argument(
        "--pattern",
        "-p",
        type=str,
        help="Regex pattern to search for in log lines",
        default=None,
    )
    parser.add_argument(
        "--keyword",
        "-k",
        type=str,
        help="Simple keyword to search for (case-insensitive)",
        default=None,
    )
    parser.add_argument(
        "--count",
        "-c",
        action="store_true",
        help="Count occurrences of each unique matching line",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show basic statistics about the log file",
    )
    parser.add_argument(
        "--opcodes",
        action="store_true",
        help="Extract and analyze PM4 opcode values",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract specific data fields from log lines",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path (default: stdout)",
        default=None,
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["text", "csv", "json"],
        default="text",
        help="Output format (default: text)",
    )
    return parser.parse_args()


def get_log_file_path(args):
    """Get the log file path from arguments or environment variable."""
    log_file = args.log or os.environ.get("LOG_FILE", None)

    if not log_file:
        print("❌ No log file specified!")
        print(__doc__)
        sys.exit(2)

    if not os.path.isfile(log_file):
        print(f"❌ File not found: {log_file}")
        sys.exit(1)

    return log_file


def read_log_file(file_path):
    """Read and return the contents of the log file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.readlines()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        sys.exit(1)


def analyze_log_basic(lines):
    """Perform basic analysis of the log file."""
    total_lines = len(lines)
    non_empty = [l for l in lines if l.strip()]
    unique_lines = set(l.strip() for l in non_empty)
    empty_lines = total_lines - len(non_empty)

    return {
        "total_lines": total_lines,
        "non_empty_lines": len(non_empty),
        "unique_lines": len(unique_lines),
        "empty_lines": empty_lines,
        "file_size": os.path.getsize(log_file_path) if "log_file_path" in dir() else 0,
    }


def extract_opcodes_from_lines(lines):
    """Extract hex opcode values from log lines (PM4 context)."""
    opcode_pattern = re.compile(r"0[xX][0-9a-fA-F]{1,8}")
    opcodes = []

    for line in lines:
        matches = opcode_pattern.findall(line)
        for match in matches:
            try:
                val = int(match, 16)
                opcodes.append(val)
            except ValueError:
                pass

    return opcodes


def count_safe_noops(opcodes):
    """Count how many opcodes are Safe No-Ops."""
    safe_count = sum(1 for op in opcodes if op in SAFE_NOOPS)
    return safe_count, len(opcodes) - safe_count


def search_lines(lines, pattern, keyword):
    """Search for patterns or keywords in log lines."""
    results = []

    for line_num, line in enumerate(lines, 1):
        line_stripped = line.strip()

        matched = False

        if pattern:
            if re.search(pattern, line_stripped, re.IGNORECASE):
                results.append((line_num, line_stripped))
                matched = True

        if keyword and not matched:
            if keyword.lower() in line_stripped.lower():
                results.append((line_num, line_stripped))
                matched = True

        if not pattern and not keyword:
            if line_stripped:
                results.append((line_num, line_stripped))

    return results


def output_results(results, format_type, output_file=None):
    """Output results in the specified format."""
    if format_type == "text":
        if output_file:
            with open(output_file, "w") as f:
                for line_num, line in results:
                    f.write(f"Line {line_num}: {line}\n")
        else:
            for line_num, line in results:
                print(f"Line {line_num}: {line}")

    elif format_type == "csv":
        if output_file:
            with open(output_file, "w") as f:
                f.write("line_number,content\n")
                for line_num, line in results:
                    f.write(f"{line_num},\"{line}\"\n")
        else:
            print("line_number,content")
            for line_num, line in results:
                print(f"{line_num},\"{line}\"")

    elif format_type == "json":
        import json

        if output_file:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
        else:
            import json
            print(json.dumps(results, indent=2))


def main():
    """Main function to collect and analyze log data."""
    args = parse_args()
    log_file = get_log_file_path(args)
    lines = read_log_file(log_file)

    # Get basic statistics
    stats = analyze_log_basic(lines)

    # Determine search criteria
    pattern = args.pattern
    keyword = args.keyword

    # Search for matching content
    if args.stats or not (pattern or keyword or args.count or args.opcodes or args.extract):
        # Show basic stats and a few lines
        matches = [
            (i, lines[i].strip()) for i in range(min(5, len(lines))) if lines[i].strip()
        ]
    else:
        matches = search_lines(lines, pattern, keyword)

    # Output statistics
    if args.stats:
        print(f"=== Log Statistics ===")
        print(f"Total lines:   {stats['total_lines']}")
        print(f"Non-empty:     {stats['non_empty_lines']}")
        print(f"Unique:        {stats['unique_lines']}")
        print(f"Empty:         {stats['empty_lines']}")
        print(f"File:          {log_file}")

    # Handle opcode extraction
    if args.opcodes:
        opcodes = extract_opcodes_from_lines(lines)
        safe, unsafe = count_safe_noops(opcodes)

        print(f"\n=== Opcode Analysis ===")
        print(f"Total opcodes found: {len(opcodes)}")
        print(f"Safe No-Ops:         {safe}")
        print(f"Other opcodes:       {unsafe}")

        # Count specific safe ops
        safe_counts = Counter(op for op in opcodes if op in SAFE_NOOPS)
        if safe_counts:
            print(f"\nSafe No-Op breakdown:")
            for op in sorted(safe_counts.keys(), key=lambda x: safe_counts[x], reverse=True):
                print(f"  0x{op:02X} (SAFE): {safe_counts[op]} occurrences")

        # Count unsafe ops
        unsafe_counts = Counter(op for op in opcodes if op not in SAFE_NOOPS)
        if unsafe_counts:
            print(f"\nTop unsafe opcodes:")
            for op, count in unsafe_counts.most_common(10):
                is_safe = "SAFE" if op in SAFE_NOOPS else "unsafe"
                print(f"  0x{op:02X} ({is_safe}): {count} occurrences")

    # Handle counting
    if args.count and matches:
        counter = Counter(line for _, line in matches)
        print(f"\n=== Occurrence Counts ===")
        for line, count in counter.most_common(20):
            display = line[:120] if len(line) > 120 else line
            print(f"  {count:4d}x: {display}")

    # Handle search results
    if matches:
        print(f"\n=== Found {len(matches)} matching lines ===")
        output_results(matches, args.format, args.output)
    else:
        if not any(
            [args.stats, args.opcodes, args.count, pattern, keyword]
        ):
            # Default: show first few lines
            print(f"\n=== First few lines from {log_file} ===")
            for i, line in enumerate(lines[:5]):
                if line.strip():
                    print(f"  {i+1}: {line.strip()[:100]}")

        if not matches and not args.stats and not args.opcodes:
            print("ℹ️ No matches found with the specified criteria.")


if __name__ == "__main__":
    main()
