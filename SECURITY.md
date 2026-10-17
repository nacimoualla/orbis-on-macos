# Security Policy

## Supported Versions

This project is pre-release research software. Only the latest `main` is supported
with security updates.

| Version | Supported |
| ------- | --------- |
| `main` (latest) | ✅ |
| Older commits / forks | ❌ best-effort only |

## Reporting a Vulnerability

**Do not open a public issue.** Instead:

1. Contact the maintainer privately via the email listed on the
   [maintainer's GitHub profile](https://github.com/nacimoualla), with
   `[orbis-on-macos security]` in the subject.
2. Include: affected file/commit, steps to reproduce, and impact (e.g. host file
   write outside VFS sandbox, JIT code-cache escape, guest→host breakout).

We will acknowledge receipt, investigate, and coordinate a fix and disclosure
timeline with you. Please allow reasonable time before public disclosure.

## Scope Notes

- The VFS sandbox (`/app0`, `/data`, `/savedata` under `vfs_root/`) must never
  allow guest path traversal to host files — traversal bugs are treated as
  security issues.
- The JIT emits executable host code from guest input; JIT bugs that break W^X
  (`MAP_JIT` + `pthread_jit_write_protect`) or corrupt host memory are in scope.
- Game binaries themselves are untrusted input. Do not run titles you do not
  legally own and trust.
