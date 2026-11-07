#pragma once
#include "ps4/common.hpp"

namespace ps4 {

enum PadButton : u32 {
    kPadSelect        = 0x0001, // SHARE
    kPadL3            = 0x0002,
    kPadR3            = 0x0004,
    kPadStart         = 0x0008, // OPTIONS
    kPadUp            = 0x0010,
    kPadRight         = 0x0020,
    kPadDown          = 0x0040,
    kPadLeft          = 0x0080,
    kPadL2            = 0x0100, // digital
    kPadR2            = 0x0200, // digital
    kPadL1            = 0x0400,
    kPadR1            = 0x0800,
    kPadTriangle      = 0x1000,
    kPadCircle        = 0x2000,
    kPadCross         = 0x4000,
    kPadSquare        = 0x8000,
    kPadPS            = 0x00010000,
    kPadTouchPadClick = 0x00100000,
    kPadIntercepted   = 0x00200000,
};

struct PadState {
    u32 buttons{0};
    u8 lx{128}, ly{128};
    u8 rx{128}, ry{128};
    u8 l2{0}, r2{0};
    u16 touch_x{960}, touch_y{471};
    bool touch_down{false};
    bool connected{true};
};

struct MouseState {
    i32 delta_x{0};
    i32 delta_y{0};
    i32 wheel{0};
    u32 buttons{0}; // bit 0: left, bit 1: right, bit 2: middle
    bool connected{true};
};

} // namespace ps4
