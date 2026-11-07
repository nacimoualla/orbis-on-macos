#!/usr/bin/env python3
"""Summarize macOS sample call trees by thread, waiting samples and leaf function."""
import argparse
from collections import Counter
from pathlib import Path
import re

WAIT = re.compile(r'(?:mach_msg(?:2)?_trap|__psynch_|__ulock_wait|semaphore_(?:timed)?wait|'
                  r'__semwait_signal|kevent|__select|__poll|__workq_kernreturn|'
                  r'__wait4|__sigsuspend|nanosleep|pthread_cond_wait|pthread_mutex_lock)')
NODE = re.compile(r'^([ +!:|`-]*)(\d+)\s+(.+?)\s+\(in\s+')
THREAD = re.compile(r'^\s*(\d+)\s+(Thread_\S+.*)$')


def parse(text):
    threads = []
    current = None
    stack = []

    def finish_node():
        _, count, function, children = stack.pop()
        # Parent self samples are real samples too; a parent can have both
        # direct samples and child call trees in sample's aggregated output.
        count = max(0, count - children)
        if count:
            current['leaves'][function] += count
            current['waiting' if WAIT.search(function) else 'busy'] += count

    def finish_thread():
        while stack:
            finish_node()

    for line in text.splitlines():
        match = THREAD.match(line)
        if match:
            finish_thread()
            current = dict(name=match[2], samples=int(match[1]), leaves=Counter(), busy=0, waiting=0)
            threads.append(current)
            continue
        match = NODE.match(line)
        if match and current is not None:
            depth, count, function = len(match[1]), int(match[2]), match[3]
            while stack and depth <= stack[-1][0]:
                finish_node()
            if stack:
                stack[-1][3] += count
            stack.append([depth, count, function, 0])
        elif current is not None and line.strip() and not line.startswith((' ', '+', '!', ':', '|')):
            finish_thread()
            current = None
    finish_thread()
    return threads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sample', type=Path)
    parser.add_argument('--top', type=int, default=5)
    args = parser.parse_args()
    threads = parse(args.sample.read_text(errors='replace'))
    if not threads:
        parser.error('no macOS sample Thread_ call trees found')
    for thread in threads:
        total = thread['busy'] + thread['waiting']
        print(f"{thread['name']}: busy {thread['busy']}/{total} ({100 * thread['busy'] / max(total, 1):.1f}%), "
              f"waiting {thread['waiting']}/{total}; thread samples {thread['samples']}")
        for function, count in thread['leaves'].most_common(args.top):
            print(f'  {count:7d}  {function}')


if __name__ == '__main__':
    main()
