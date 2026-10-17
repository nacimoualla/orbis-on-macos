#pragma once

#include "ps4/common.hpp"
#include <string>
#include <vector>

namespace ps4 {

enum class Pm4PacketType : u8 {
    Type0 = 0, // Register write
    Type2 = 2, // NOP / Padding
    Type3 = 3  // Command packet
};

// Canonical GCN PM4 Type-3 Opcodes
enum Pm4Opcode : u8 {
    IT_NOP              = 0x10,
    IT_CLEAR_STATE      = 0x13,
    IT_DISPATCH_DIRECT  = 0x15,
    IT_DRAW_INDEX_2     = 0x27,
    IT_DRAW_INDEX_AUTO  = 0x2D,
    IT_WRITE_DATA       = 0x37,
    IT_WAIT_REG_MEM     = 0x3C,
    IT_EVENT_WRITE      = 0x46,
    IT_RELEASE_MEM      = 0x47,
    IT_DMA_DATA         = 0x50,
    IT_SET_CONTEXT_REG  = 0x69,
    IT_SET_SH_REG       = 0x76
};

// GCN Primitive types
enum class PrimitiveType : u8 {
    PointList     = 0x01,
    LineList      = 0x02,
    LineStrip     = 0x03,
    TriangleList  = 0x04,
    TriangleStrip = 0x05,
    RectList      = 0x08
};

// Context Register Offsets (relative to 0xA000)
enum ContextRegOffset : u16 {
    PA_SC_WINDOW_OFFSET    = 0x0080,
    PA_SC_SCREEN_SCISSOR_TL= 0x0082,
    PA_SC_SCREEN_SCISSOR_BR= 0x0083,
    PA_CL_VPORT_XSCALE     = 0x0084,
    PA_CL_VPORT_XOFFSET    = 0x0085,
    PA_CL_VPORT_YSCALE     = 0x0086,
    PA_CL_VPORT_YOFFSET    = 0x0087,
    CB_COLOR0_BASE         = 0x00B0,
    CB_COLOR0_VIEW         = 0x00B1,
    CB_COLOR0_INFO         = 0x00B2,
    CB_COLOR_CLEAR_WORD0   = 0x00DC,
    CB_COLOR_CLEAR_WORD1   = 0x00DD
};

// Shader Register Offsets (relative to 0x2C00 / SPI)
enum ShaderRegOffset : u16 {
    SPI_SHADER_USER_DATA_VS_0 = 0x0200,
    SPI_SHADER_USER_DATA_VS_1 = 0x0201,
    SPI_SHADER_USER_DATA_VS_2 = 0x0202,
    SPI_SHADER_USER_DATA_VS_3 = 0x0203
};

// Standard 3D colored vertex structure (28 bytes)
#pragma pack(push, 4)
struct Vertex3D {
    f32 position[3]; // X, Y, Z in normalized device coordinates [-1, 1]
    f32 color[4];    // R, G, B, A normalized colors [0, 1]
};
#pragma pack(pop)

struct Pm4Header {
    Pm4PacketType type{Pm4PacketType::Type2};
    u8 opcode{0};
    u16 count{0}; // Number of payload DWORDs
};

struct Pm4Packet {
    Pm4Header header;
    std::vector<u32> payload;
};

inline Pm4Header decode_pm4_header(u32 dword) {
    Pm4Header hdr{};
    const u32 type_val = (dword >> 30) & 0x03;
    if (type_val == 0) {
        hdr.type = Pm4PacketType::Type0;
        hdr.count = static_cast<u16>(((dword >> 16) & 0x3FFF) + 1);
        hdr.opcode = 0; // Base register is in lower 16 bits
    } else if (type_val == 2) {
        hdr.type = Pm4PacketType::Type2;
        hdr.count = 0;
        hdr.opcode = 0;
    } else if (type_val == 3) {
        hdr.type = Pm4PacketType::Type3;
        hdr.count = static_cast<u16>(((dword >> 16) & 0x3FFF) + 1);
        hdr.opcode = static_cast<u8>((dword >> 8) & 0xFF);
    }
    return hdr;
}

inline u32 encode_pm4_type3_header(u8 opcode, u16 payload_count) {
    const u32 count_field = (payload_count > 0 ? payload_count - 1 : 0) & 0x3FFF;
    return (0x03U << 30) | (count_field << 16) | (static_cast<u32>(opcode) << 8);
}

inline u32 encode_pm4_type0_header(u16 base_reg, u16 payload_count) {
    const u32 count_field = (payload_count > 0 ? payload_count - 1 : 0) & 0x3FFF;
    return (count_field << 16) | (base_reg & 0xFFFF);
}

inline u32 encode_pm4_type2_nop() {
    return 0x80000000U;
}

} // namespace ps4
