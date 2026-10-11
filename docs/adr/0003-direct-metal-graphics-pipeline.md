# Direct Metal Graphics Translation over MoltenVK

To translate the PS4's GNM/GNMX low-level graphics commands on macOS, we chose direct translation to Apple's Metal API instead of bridging through Vulkan via MoltenVK. Direct Metal translation eliminates runtime driver layering overhead and directly exploits Apple Silicon hardware features such as unified memory zero-copy buffers, tile shading, and native Argument Buffers.
