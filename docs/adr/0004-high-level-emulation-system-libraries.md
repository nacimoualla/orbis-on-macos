# High-Level Emulation (HLE) for System Libraries

To ensure legal clean-room distribution without requiring users to supply copyrighted Sony firmware dumps, we chose High-Level Emulation (HLE) in host C++ for core libraries (`libkernel`, `libc`, `libScePad`, `libSceVideoOut`) instead of Low-Level Emulation (LLE). This architecture intercepts PRX import tables at dynamic linking time, routing calls directly to native macOS frameworks (Metal, CoreAudio, GameController) with zero translation overhead and no dependency on internal proprietary kernel daemons.
