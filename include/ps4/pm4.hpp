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
    IT_NOP                  = 0x10,
    IT_SET_BASE             = 0x11,
    IT_CLEAR_STATE          = 0x12,
    IT_INDEX_BUFFER_SIZE    = 0x13,
    IT_DISPATCH_DIRECT      = 0x15,
    IT_DISPATCH_INDIRECT    = 0x16,
    IT_INDEX_TYPE           = 0x20,
    IT_DRAW_INDEX           = 0x27,
    IT_DRAW_INDEX_2         = 0x28,
    IT_INDEX_BASE           = 0x2A,
    IT_DRAW_INDEX_AUTO      = 0x2D,
    IT_DRAW_INDIRECT        = 0x2E,
    IT_DRAW_INDEX_INDIRECT  = 0x2E,
    IT_NUM_INSTANCES        = 0x2F,
    IT_INDEX_BASE_MAX       = 0x30,
    IT_DRAW_INDEX_INDIRECT_MULTI = 0x33,
    IT_DRAW_INDEX_OFFSET_2  = 0x35,
    IT_WRITE_DATA           = 0x37,
    IT_WAIT_REG_MEM         = 0x3C,
    IT_INDIRECT_BUFFER      = 0x3F,
    IT_EVENT_WRITE          = 0x46,
    IT_RELEASE_MEM          = 0x47,
    IT_DMA_DATA             = 0x50,
    IT_ACQUIRE_MEM          = 0x58,
    IT_SET_CONFIG_REG       = 0x68,
    IT_SET_CONTEXT_REG      = 0x69,
    IT_SET_SH_REG           = 0x76,
    IT_LOAD_CONFIG_REG      = 0x79
};

inline const char* pm4_opcode_to_name(u8 op) {
    switch (op) {
        case IT_NOP: return "NOP";
        case IT_SET_BASE: return "SET_BASE";
        case IT_CLEAR_STATE: return "CLEAR_STATE";
        case IT_INDEX_BUFFER_SIZE: return "INDEX_BUFFER_SIZE";
        case IT_DISPATCH_DIRECT: return "DISPATCH_DIRECT";
        case IT_DISPATCH_INDIRECT: return "DISPATCH_INDIRECT";
        case IT_INDEX_TYPE: return "INDEX_TYPE";
        case IT_DRAW_INDEX: return "DRAW_INDEX";
        case IT_DRAW_INDEX_2: return "DRAW_INDEX_2";
        case IT_DRAW_INDEX_OFFSET_2: return "DRAW_INDEX_OFFSET_2";
        case IT_INDEX_BASE: return "INDEX_BASE";
        case IT_DRAW_INDEX_AUTO: return "DRAW_INDEX_AUTO";
        case IT_DRAW_INDIRECT: return "DRAW_INDIRECT";
        case IT_NUM_INSTANCES: return "NUM_INSTANCES";
        case IT_INDEX_BASE_MAX: return "INDEX_BASE_MAX";
        case IT_DRAW_INDEX_INDIRECT_MULTI: return "DRAW_INDEX_INDIRECT_MULTI";
        case IT_WRITE_DATA: return "WRITE_DATA";
        case IT_WAIT_REG_MEM: return "WAIT_REG_MEM";
        case IT_INDIRECT_BUFFER: return "INDIRECT_BUFFER";
        case IT_EVENT_WRITE: return "EVENT_WRITE";
        case IT_RELEASE_MEM: return "RELEASE_MEM";
        case IT_DMA_DATA: return "DMA_DATA";
        case IT_ACQUIRE_MEM: return "ACQUIRE_MEM";
        case IT_SET_CONFIG_REG: return "SET_CONFIG_REG";
        case IT_SET_CONTEXT_REG: return "SET_CONTEXT_REG";
        case IT_SET_SH_REG: return "SET_SH_REG";
        case IT_LOAD_CONFIG_REG: return "LOAD_CONFIG_REG";
        default: return "UNKNOWN";
    }
}

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
    CB_COLOR0_BASE_GFX7    = 0x0318,
    CB_COLOR0_PITCH_GFX7   = 0x0319,
    CB_COLOR0_SLICE_GFX7   = 0x031A,
    CB_COLOR0_VIEW_GFX7    = 0x031B,
    CB_COLOR0_INFO_GFX7    = 0x031C,
    CB_COLOR0_ATTRIB_GFX7  = 0x031D,
    CB_COLOR0_CLEAR_WORD0_GFX7 = 0x0323,
    CB_COLOR0_CLEAR_WORD1_GFX7 = 0x0324,
    CB_COLOR_CLEAR_WORD0   = 0x00DC,
    CB_COLOR_CLEAR_WORD1   = 0x00DD,
    VGT_PRIMITIVE_TYPE_REG = 0x0258,
    VGT_DMA_MAX_SIZE_REG   = 0x029E,
    VGT_DMA_INDEX_TYPE_REG = 0x029F
};

// Shader Register Offsets (relative to 0x2C00 / SPI)
enum ShaderRegOffset : u16 {
    SPI_SHADER_PGM_LO_PS      = 0x0008,
    SPI_SHADER_PGM_HI_PS      = 0x0009,
    SPI_SHADER_PGM_RSRC1_PS   = 0x000A,
    SPI_SHADER_PGM_RSRC2_PS   = 0x000B,
    SPI_SHADER_USER_DATA_PS_0        = 0x000C,
    SPI_SHADER_PGM_LO_VS             = 0x0048,
    SPI_SHADER_PGM_HI_VS             = 0x0049,
    SPI_SHADER_PGM_RSRC1_VS          = 0x004A,
    SPI_SHADER_PGM_RSRC2_VS          = 0x004B,
    SPI_SHADER_USER_DATA_VS_0        = 0x004C,
    SPI_SHADER_USER_DATA_VS_0_LEGACY = 0x0200,
    SPI_SHADER_USER_DATA_VS_1_LEGACY = 0x0201,
    SPI_SHADER_USER_DATA_VS_2_LEGACY = 0x0202,
    SPI_SHADER_USER_DATA_VS_3_LEGACY = 0x0203,
    SPI_SHADER_USER_DATA_PS_0_LEGACY = 0x020C
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
    u16 base_reg{0}; // Type-0 base register offset
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
        hdr.opcode = 0;
        hdr.base_reg = static_cast<u16>(dword & 0xFFFF);
    } else if (type_val == 2) {
        hdr.type = Pm4PacketType::Type2;
        hdr.count = 0;
        hdr.opcode = 0;
        hdr.base_reg = 0;
    } else if (type_val == 3) {
        hdr.type = Pm4PacketType::Type3;
        hdr.count = static_cast<u16>(((dword >> 16) & 0x3FFF) + 1);
        hdr.opcode = static_cast<u8>((dword >> 8) & 0xFF);
        hdr.base_reg = 0;
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
