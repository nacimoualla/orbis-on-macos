#include <cassert>
#include <iostream>
#include "ps4/input.hpp"
#include "ps4/nid_registry.hpp"
#include "ps4/memory.hpp"

int main() {
    std::cout << "[INPUT] Testing PadState and MouseState in NidRegistry...\n";

    ps4::NidRegistry reg;
    reg.register_default_libraries();
    ps4::MemoryManager mem;
    assert(mem.initialize());

    // Test 1: Default PadState
    ps4::PadState default_st = reg.get_pad_state();
    assert(default_st.buttons == 0);
    assert(default_st.lx == 128 && default_st.ly == 128);
    assert(default_st.rx == 128 && default_st.ry == 128);
    assert(default_st.touch_x == 960 && default_st.touch_y == 471);
    assert(!default_st.touch_down);
    assert(default_st.connected);
    std::cout << "  ✓ Test 1: Default PadState verified\n";

    // Test 2: Custom PadState with analog sticks and touchpad
    ps4::PadState st{};
    st.buttons = ps4::kPadCross | ps4::kPadL1;
    st.lx = 0;   // Hard Left
    st.ly = 255; // Hard Down
    st.rx = 64;
    st.ry = 192;
    st.l2 = 200;
    st.r2 = 100;
    st.touch_x = 400;
    st.touch_y = 300;
    st.touch_down = true;

    reg.set_pad_state(st);
    ps4::PadState retrieved = reg.get_pad_state();
    assert(retrieved.buttons == (ps4::kPadCross | ps4::kPadL1));
    assert(retrieved.lx == 0 && retrieved.ly == 255);
    assert(retrieved.rx == 64 && retrieved.ry == 192);
    assert(retrieved.l2 == 200 && retrieved.r2 == 100);
    assert(retrieved.touch_x == 400 && retrieved.touch_y == 300);
    assert(retrieved.touch_down == true);
    std::cout << "  ✓ Test 2: Set and retrieve PadState verified\n";

    // Test 3: MouseState
    ps4::MouseState ms{};
    ms.delta_x = 15;
    ms.delta_y = -10;
    ms.wheel = 3;
    ms.buttons = 1; // Left click
    reg.set_mouse_state(ms);
    ps4::MouseState retrieved_ms = reg.get_mouse_state();
    assert(retrieved_ms.delta_x == 15);
    assert(retrieved_ms.delta_y == -10);
    assert(retrieved_ms.wheel == 3);
    assert(retrieved_ms.buttons == 1);
    std::cout << "  ✓ Test 3: Set and retrieve MouseState verified\n";

    // Test 4: libSceMouse HLE symbols
    {
        const auto* sym_open = reg.find_by_name("sceMouseOpen");
        assert(sym_open != nullptr);
        ps4::CpuContext ctx{};
        ctx.rdi = 1; // userId
        ctx.rsi = 0; // type
        ctx.rdx = 0; // index
        sym_open->handler(ctx, mem);
        assert(ctx.rax > 0); // valid mouse handle >= 1

        const auto* sym_read = reg.find_by_name("sceMouseRead");
        assert(sym_read != nullptr);
        ps4::GuestAddr mouse_buf = 0x20000;
        ctx.rdi = 1; // handle
        ctx.rsi = mouse_buf;
        ctx.rdx = 1; // num samples
        sym_read->handler(ctx, mem);
        assert(ctx.rax == 1); // 1 sample read

        // Verify serialized mouse fields
        ps4::u64 ts = 0;
        ps4::u32 btns = 0;
        ps4::i32 dx = 0, dy = 0, wh = 0;
        ps4::u8 conn = 0;
        mem.read_value(mouse_buf + 0x00, ts);
        mem.read_value(mouse_buf + 0x08, btns);
        mem.read_value(mouse_buf + 0x0C, dx);
        mem.read_value(mouse_buf + 0x10, dy);
        mem.read_value(mouse_buf + 0x14, wh);
        mem.read_value(mouse_buf + 0x18, conn);

        assert(ts > 0);
        assert(btns == 1);
        assert(dx == 15);
        assert(dy == -10);
        assert(wh == 3);
        assert(conn == 1);
        std::cout << "  ✓ Test 4: libSceMouse sceMouseOpen and sceMouseRead verified\n";
    }

    // Test 5: Verify ScePadData serialization via scePadRead
    {
        const auto* sym_pad_read = reg.find_by_name("scePadRead");
        assert(sym_pad_read != nullptr);
        ps4::GuestAddr pad_buf = 0x30000;
        ps4::CpuContext ctx{};
        ctx.rdi = 1; // handle
        ctx.rsi = pad_buf;
        ctx.rdx = 1; // num samples
        sym_pad_read->handler(ctx, mem);
        assert(ctx.rax == 1);

        ps4::u32 btns = 0;
        ps4::u8 lx = 0, ly = 0, rx = 0, ry = 0, l2 = 0, r2 = 0, touch_count = 0, conn = 0;
        ps4::u16 tx = 0, ty = 0;
        mem.read_value(pad_buf + 0x00, btns);
        mem.read_value(pad_buf + 0x04, lx);
        mem.read_value(pad_buf + 0x05, ly);
        mem.read_value(pad_buf + 0x06, rx);
        mem.read_value(pad_buf + 0x07, ry);
        mem.read_value(pad_buf + 0x08, l2);
        mem.read_value(pad_buf + 0x09, r2);
        mem.read_value(pad_buf + 0x34, touch_count);
        mem.read_value(pad_buf + 0x3C, tx);
        mem.read_value(pad_buf + 0x3E, ty);
        mem.read_value(pad_buf + 0x4C, conn);

        assert(btns == (ps4::kPadCross | ps4::kPadL1));
        assert(lx == 0 && ly == 255);
        assert(rx == 64 && ry == 192);
        assert(l2 == 200 && r2 == 100);
        assert(touch_count == 1);
        assert(tx == 400 && ty == 300);
        assert(conn == 1);
        std::cout << "  ✓ Test 5: ScePadData serialization verified\n";
    }

    std::cout << "[INPUT] All input state tests passed!\n";
    return 0;
}
