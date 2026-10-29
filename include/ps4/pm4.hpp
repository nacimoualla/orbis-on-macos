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

// PM4 Type-3 IT opcodes for GCN 1.1 / CIK (Sea Islands), the family of the PS4
// "Liverpool" GPU. Values match AMD's CIK PM4 spec / Mesa sid.h. Each value
// appears exactly once: there are no aliases and no emulator-private opcodes.
enum Pm4Opcode : u8 {
    IT_NOP                       = 0x10,
    IT_SET_BASE                  = 0x11,
    IT_CLEAR_STATE               = 0x12,
    IT_INDEX_BUFFER_SIZE         = 0x13,
    IT_DISPATCH_DIRECT           = 0x15,
    IT_DISPATCH_INDIRECT         = 0x16,
    IT_ATOMIC_GDS                = 0x1D,
    IT_ATOMIC_MEM                = 0x1E,
    IT_OCCLUSION_QUERY           = 0x1F,
    IT_SET_PREDICATION           = 0x20,
    IT_REG_RMW                   = 0x21,
    IT_COND_EXEC                 = 0x22,
    IT_PRED_EXEC                 = 0x23,
    IT_DRAW_INDIRECT             = 0x24,
    IT_DRAW_INDEX_INDIRECT       = 0x25,
    IT_INDEX_BASE                = 0x26,
    IT_DRAW_INDEX_2              = 0x27,
    IT_CONTEXT_CONTROL           = 0x28,
    IT_INDEX_TYPE                = 0x2A,
    IT_DRAW_INDIRECT_MULTI       = 0x2C,
    IT_DRAW_INDEX_AUTO           = 0x2D,
    IT_NUM_INSTANCES             = 0x2F,
    IT_DRAW_INDEX_MULTI_AUTO     = 0x30,
    IT_INDIRECT_BUFFER_CNST      = 0x33,
    IT_STRMOUT_BUFFER_UPDATE     = 0x34,
    IT_DRAW_INDEX_OFFSET_2       = 0x35,
    IT_DRAW_PREAMBLE             = 0x36,
    IT_WRITE_DATA                = 0x37,
    IT_DRAW_INDEX_INDIRECT_MULTI = 0x38,
    IT_MEM_SEMAPHORE             = 0x39,
    IT_COPY_DW                   = 0x3B,
    IT_WAIT_REG_MEM              = 0x3C,
    IT_INDIRECT_BUFFER           = 0x3F,
    IT_COPY_DATA                 = 0x40,
    IT_CP_DMA                    = 0x41,
    IT_PFP_SYNC_ME               = 0x42,
    IT_SURFACE_SYNC              = 0x43,
    IT_ME_INITIALIZE             = 0x44,
    IT_COND_WRITE                = 0x45,
    IT_EVENT_WRITE               = 0x46,
    IT_EVENT_WRITE_EOP           = 0x47,
    IT_EVENT_WRITE_EOS           = 0x48,
    IT_RELEASE_MEM               = 0x49,
    IT_PREAMBLE_CNTL             = 0x4A,
    IT_DMA_DATA                  = 0x50,
    IT_CONTEXT_REG_RMW           = 0x51,
    IT_ACQUIRE_MEM               = 0x58,
    IT_REWIND                    = 0x59,
    IT_LOAD_UCONFIG_REG          = 0x5E,
    IT_LOAD_SH_REG               = 0x5F,
    IT_LOAD_CONFIG_REG           = 0x60,
    IT_LOAD_CONTEXT_REG          = 0x61,
    IT_SET_CONFIG_REG            = 0x68,
    IT_SET_CONTEXT_REG           = 0x69,
    IT_SET_CONTEXT_REG_INDIRECT  = 0x73,
    IT_SET_SH_REG                = 0x76,
    IT_SET_SH_REG_OFFSET         = 0x77,
    IT_SET_QUEUE_REG             = 0x78,
    IT_SET_UCONFIG_REG           = 0x79,
    IT_SCRATCH_RAM_WRITE         = 0x7D,
    IT_SCRATCH_RAM_READ          = 0x7E,
    IT_LOAD_CONST_RAM            = 0x80,
    IT_WRITE_CONST_RAM           = 0x81,
    IT_DUMP_CONST_RAM            = 0x83,
    IT_INCREMENT_CE_COUNTER      = 0x84,
    IT_INCREMENT_DE_COUNTER      = 0x85,
    IT_WAIT_ON_CE_COUNTER        = 0x86,
    IT_WAIT_ON_DE_COUNTER_DIFF   = 0x88,
    IT_SWITCH_BUFFER             = 0x8B
};

inline const char* pm4_opcode_to_name(u8 op) {
    switch (op) {
        case IT_NOP: return "NOP";
        case IT_SET_BASE: return "SET_BASE";
        case IT_CLEAR_STATE: return "CLEAR_STATE";
        case IT_INDEX_BUFFER_SIZE: return "INDEX_BUFFER_SIZE";
        case IT_DISPATCH_DIRECT: return "DISPATCH_DIRECT";
        case IT_DISPATCH_INDIRECT: return "DISPATCH_INDIRECT";
        case IT_ATOMIC_GDS: return "ATOMIC_GDS";
        case IT_ATOMIC_MEM: return "ATOMIC_MEM";
        case IT_OCCLUSION_QUERY: return "OCCLUSION_QUERY";
        case IT_SET_PREDICATION: return "SET_PREDICATION";
        case IT_REG_RMW: return "REG_RMW";
        case IT_COND_EXEC: return "COND_EXEC";
        case IT_PRED_EXEC: return "PRED_EXEC";
        case IT_DRAW_INDIRECT: return "DRAW_INDIRECT";
        case IT_DRAW_INDEX_INDIRECT: return "DRAW_INDEX_INDIRECT";
        case IT_INDEX_BASE: return "INDEX_BASE";
        case IT_DRAW_INDEX_2: return "DRAW_INDEX_2";
        case IT_CONTEXT_CONTROL: return "CONTEXT_CONTROL";
        case IT_INDEX_TYPE: return "INDEX_TYPE";
        case IT_DRAW_INDIRECT_MULTI: return "DRAW_INDIRECT_MULTI";
        case IT_DRAW_INDEX_AUTO: return "DRAW_INDEX_AUTO";
        case IT_NUM_INSTANCES: return "NUM_INSTANCES";
        case IT_DRAW_INDEX_MULTI_AUTO: return "DRAW_INDEX_MULTI_AUTO";
        case IT_INDIRECT_BUFFER_CNST: return "INDIRECT_BUFFER_CNST";
        case IT_STRMOUT_BUFFER_UPDATE: return "STRMOUT_BUFFER_UPDATE";
        case IT_DRAW_INDEX_OFFSET_2: return "DRAW_INDEX_OFFSET_2";
        case IT_DRAW_PREAMBLE: return "DRAW_PREAMBLE";
        case IT_WRITE_DATA: return "WRITE_DATA";
        case IT_DRAW_INDEX_INDIRECT_MULTI: return "DRAW_INDEX_INDIRECT_MULTI";
        case IT_MEM_SEMAPHORE: return "MEM_SEMAPHORE";
        case IT_COPY_DW: return "COPY_DW";
        case IT_WAIT_REG_MEM: return "WAIT_REG_MEM";
        case IT_INDIRECT_BUFFER: return "INDIRECT_BUFFER";
        case IT_COPY_DATA: return "COPY_DATA";
        case IT_CP_DMA: return "CP_DMA";
        case IT_PFP_SYNC_ME: return "PFP_SYNC_ME";
        case IT_SURFACE_SYNC: return "SURFACE_SYNC";
        case IT_ME_INITIALIZE: return "ME_INITIALIZE";
        case IT_COND_WRITE: return "COND_WRITE";
        case IT_EVENT_WRITE: return "EVENT_WRITE";
        case IT_EVENT_WRITE_EOP: return "EVENT_WRITE_EOP";
        case IT_EVENT_WRITE_EOS: return "EVENT_WRITE_EOS";
        case IT_RELEASE_MEM: return "RELEASE_MEM";
        case IT_PREAMBLE_CNTL: return "PREAMBLE_CNTL";
        case IT_DMA_DATA: return "DMA_DATA";
        case IT_CONTEXT_REG_RMW: return "CONTEXT_REG_RMW";
        case IT_ACQUIRE_MEM: return "ACQUIRE_MEM";
        case IT_REWIND: return "REWIND";
        case IT_LOAD_UCONFIG_REG: return "LOAD_UCONFIG_REG";
        case IT_LOAD_SH_REG: return "LOAD_SH_REG";
        case IT_LOAD_CONFIG_REG: return "LOAD_CONFIG_REG";
        case IT_LOAD_CONTEXT_REG: return "LOAD_CONTEXT_REG";
        case IT_SET_CONFIG_REG: return "SET_CONFIG_REG";
        case IT_SET_CONTEXT_REG: return "SET_CONTEXT_REG";
        case IT_SET_CONTEXT_REG_INDIRECT: return "SET_CONTEXT_REG_INDIRECT";
        case IT_SET_SH_REG: return "SET_SH_REG";
        case IT_SET_SH_REG_OFFSET: return "SET_SH_REG_OFFSET";
        case IT_SET_QUEUE_REG: return "SET_QUEUE_REG";
        case IT_SET_UCONFIG_REG: return "SET_UCONFIG_REG";
        case IT_SCRATCH_RAM_WRITE: return "SCRATCH_RAM_WRITE";
        case IT_SCRATCH_RAM_READ: return "SCRATCH_RAM_READ";
        case IT_LOAD_CONST_RAM: return "LOAD_CONST_RAM";
        case IT_WRITE_CONST_RAM: return "WRITE_CONST_RAM";
        case IT_DUMP_CONST_RAM: return "DUMP_CONST_RAM";
        case IT_INCREMENT_CE_COUNTER: return "INCREMENT_CE_COUNTER";
        case IT_INCREMENT_DE_COUNTER: return "INCREMENT_DE_COUNTER";
        case IT_WAIT_ON_CE_COUNTER: return "WAIT_ON_CE_COUNTER";
        case IT_WAIT_ON_DE_COUNTER_DIFF: return "WAIT_ON_DE_COUNTER_DIFF";
        case IT_SWITCH_BUFFER: return "SWITCH_BUFFER";
        default: return "UNKNOWN";
    }
}

// Register-space bases (dword offsets). SET_*_REG packets carry the register
// offset relative to these bases in payload[0][15:0].
constexpr u32 kPm4ConfigRegBase  = 0x2000;  // SET_CONFIG_REG  (0x8000 bytes)
constexpr u32 kPm4ShRegBase      = 0x2C00;  // SET_SH_REG      (0xB000 bytes)
constexpr u32 kPm4ContextRegBase = 0xA000;  // SET_CONTEXT_REG (0x28000 bytes)
constexpr u32 kPm4UconfigRegBase = 0xC000;  // SET_UCONFIG_REG (0x30000 bytes, CIK+)

// CIK user-config registers (offsets relative to kPm4UconfigRegBase).
enum UconfigRegOffset : u16 {
    UCONFIG_VGT_PRIMITIVE_TYPE = 0x0242,   // 0x30908
    UCONFIG_VGT_INDEX_TYPE     = 0x0243,   // 0x3090C
    UCONFIG_VGT_NUM_INDICES    = 0x024C,   // 0x30930
    UCONFIG_VGT_NUM_INSTANCES  = 0x024D,   // 0x30934
};

// VGT_DMA_INDEX_TYPE / INDEX_TYPE packet fields.
constexpr u32 kIndexTypeMask = 0x3;        // 0=16-bit, 1=32-bit, 2=8-bit (VI+, invalid on CIK)

// DRAW_INITIATOR (VGT_DRAW_INITIATOR) SOURCE_SELECT values.
constexpr u32 kDiSrcSelDma       = 0;
constexpr u32 kDiSrcSelImmediate = 1;
constexpr u32 kDiSrcSelAutoIndex = 2;

// GCN primitive types (VGT_PRIMITIVE_TYPE.PRIM_TYPE / DI_PT_* values).
enum class PrimitiveType : u8 {
    None          = 0x00,
    PointList     = 0x01,
    LineList      = 0x02,
    LineStrip     = 0x03,
    TriangleList  = 0x04,
    TriangleFan   = 0x05,
    TriangleStrip = 0x06,
    Patch         = 0x09,
    LineListAdj   = 0x0A,
    LineStripAdj  = 0x0B,
    TriListAdj    = 0x0C,
    TriStripAdj   = 0x0D,
    RectList      = 0x11,
    LineLoop      = 0x12,
    QuadList      = 0x13,
    QuadStrip     = 0x14,
    Polygon       = 0x15
};

// Context Register Offsets (relative to 0xA000)
enum ContextRegOffset : u16 {
    PA_SC_WINDOW_OFFSET    = 0x0080,
    PA_SC_SCREEN_SCISSOR_TL= 0x000C,
    PA_SC_SCREEN_SCISSOR_BR= 0x000D,
    PA_CL_VPORT_XSCALE     = 0x010F,
    PA_CL_VPORT_XOFFSET    = 0x0110,
    PA_CL_VPORT_YSCALE     = 0x0111,
    PA_CL_VPORT_YOFFSET    = 0x0112,
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
