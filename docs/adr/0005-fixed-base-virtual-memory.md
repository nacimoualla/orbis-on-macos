# Fixed Base Virtual Memory Allocation

To emulate the PS4's flat 8GB unified address space without the steep runtime penalty of software page table lookups, we chose to reserve a dedicated multi-gigabyte virtual address window in the host address space via `vm_allocate` / `mmap`. Guest 64-bit pointers map to host addresses via a single constant base register offset, enabling JIT-generated ARM64 machine code to perform guest loads and stores with single base-offset instructions.
