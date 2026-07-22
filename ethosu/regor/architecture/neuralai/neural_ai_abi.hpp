//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace regor::neuralai
{

constexpr uint32_t ModelMagic = 0x4D49414EU;       // "NAIM" in little-endian storage
constexpr uint32_t InvocationMagic = 0x5649414EU;  // "NAIV" in little-endian storage
constexpr uint16_t AbiMajor = 1;
constexpr uint16_t AbiMinor = 0;
constexpr uint32_t TargetId = 1;
constexpr uint32_t Alignment = 32;

enum class SectionType : uint32_t
{
    Commands = 1,
    Constants = 2,
    Tensors = 3,
    Bindings = 4,
    QParams = 5,
    DebugMap = 6,
};

enum class DataType : uint16_t
{
    Int8 = 1,
    Int32 = 4,
};

enum class TensorLayout : uint16_t
{
    NHWC = 1,
    Row32 = 2,
    C32Blocked = 3,
};

enum class BindingDirection : uint16_t
{
    Input = 1,
    Output = 2,
    L2Temporary = 3,
};

enum class Region : uint16_t
{
    ModelConstants = 1,
    ModelCommands = 2,
    InputBinding = 3,
    OutputBinding = 4,
    L2TemporaryBinding = 5,
    TCDMScratch = 6,
    DTCMRuntime = 7,
};

enum class CommandType : uint16_t
{
    End = 0,
    Barrier = 1,
    DMA1D = 2,
    DMA2D = 3,
    DMA3D = 4,
    RQLoad = 5,
    Gemm32 = 6,
    Gemm32Accum = 7,
    Gemm32Requant = 8,
    LineBufferJob = 9,
    PointwiseC32 = 10,
    DepthwiseC32 = 11,
    AFULut = 12,
    AFUBinary = 13,
    AFUGlobalAvgPool = 14,
    SpatzRequant = 15,
    SpatzAdd = 16,
    SpatzMul = 17,
    CopyLayout = 18,
    MaxPool = 19,
    UpsampleNearest = 20,
    RollingReset = 21,
    RollingProduce = 22,
    RollingConsumeRelease = 23,
    DMASubmit1D = 24,
    DMASubmit2D = 25,
    DMASubmit3D = 26,
    DMAWait = 27,
};

enum CommandFlags : uint32_t
{
    CommandFlagNone = 0,
    CommandFlagOptional = 1U << 0,
    CommandFlagSkippable = 1U << 1,
};

struct ModelHeaderV1
{
    uint32_t magic;
    uint16_t abiMajor;
    uint16_t abiMinor;
    uint32_t targetId;
    uint32_t flags;
    uint32_t totalBytes;
    uint32_t sectionCount;
    uint32_t sectionTableOffset;
    uint32_t entryCommandOffset;
    uint32_t commandCount;
    uint32_t requiredTCDMBytes;
    uint32_t requiredTCDMAlignment;
    uint32_t inputCount;
    uint32_t outputCount;
    uint32_t reserved[3];
};

struct SectionV1
{
    uint32_t type;
    uint32_t flags;
    uint32_t offset;
    uint32_t size;
    uint32_t alignment;
    uint32_t elementCount;
    uint32_t crc32;
    uint32_t reserved;
};

struct InvocationV1
{
    uint32_t magic;
    uint16_t abiMajor;
    uint16_t abiMinor;
    uint32_t totalBytes;
    uint32_t modelBase;
    uint32_t modelBytes;
    uint32_t bindingTableBase;
    uint32_t bindingCount;
    uint32_t flags;
    uint32_t reserved[8];
};

struct RefV1
{
    uint16_t region;
    uint16_t index;
    uint32_t offset;
};

struct TensorV1
{
    uint32_t tensorId;
    uint32_t flags;
    uint16_t dataType;
    uint16_t layout;
    uint16_t rank;
    uint16_t reserved0;
    uint32_t dimensions[4];
    uint32_t byteSize;
    uint32_t alignment;
    uint32_t scratchOffset;
    uint32_t qparamIndex;
    uint32_t reserved[4];
};

struct BindingV1
{
    uint16_t direction;
    uint16_t index;
    uint16_t dataType;
    uint16_t layout;
    uint16_t rank;
    uint16_t reserved0;
    uint32_t tensorId;
    uint32_t dimensions[4];
    uint32_t byteSize;
    uint32_t scaleBits;
    int32_t zeroPoint;
    uint32_t flags;
    uint32_t reserved[4];
};

struct BindingAddressV1
{
    uint16_t direction;
    uint16_t index;
    uint32_t base;
    uint32_t byteSize;
    uint32_t flags;
};

struct QParamV1
{
    int32_t bias;
    int32_t multiplier;
    uint32_t shift;
    int32_t zeroPoint;
    int32_t clampMin;
    int32_t clampMax;
    uint32_t reserved[2];
};

struct CommandHeaderV2
{
    uint16_t type;
    uint16_t sizeBytes;
    uint32_t flags;
    uint32_t layerId;
    uint32_t tileId;
};

struct CommandRQLoadV2
{
    CommandHeaderV2 header;
    uint32_t qparamIndex;
    uint32_t qparamCount;
    uint32_t qparamBlock;
    uint32_t reserved;
};

struct CommandDMA1DV2
{
    CommandHeaderV2 header;
    RefV1 source;
    RefV1 destination;
    uint32_t length;
    uint32_t direction;
    uint32_t reserved[6];
};

struct CommandDMA2DV2
{
    CommandHeaderV2 header;
    RefV1 source;
    RefV1 destination;
    uint32_t length;
    uint32_t sourceStride2;
    uint32_t destinationStride2;
    uint32_t repetitions2;
    uint32_t direction;
    uint32_t reserved[3];
};

struct CommandDMA3DV2
{
    CommandHeaderV2 header;
    RefV1 source;
    RefV1 destination;
    uint32_t length;
    uint32_t sourceStride2;
    uint32_t destinationStride2;
    uint32_t repetitions2;
    uint32_t sourceStride3;
    uint32_t destinationStride3;
    uint32_t repetitions3;
    uint32_t direction;
};

struct CommandGemm32V2
{
    CommandHeaderV2 header;
    RefV1 weights;
    RefV1 ifm;
    RefV1 partialSums;
    RefV1 ofm;
    uint32_t dimM;
    uint32_t ofmRowStride;
    uint32_t partialSumRowStride;
    uint32_t qparamBlock;
    uint32_t reserved[8];
};

enum class CopyLayoutMode : uint16_t
{
    NHWCToRow32 = 1,
    Row32ToNHWC = 2,
    NHWCToC32 = 3,
    C32ToNHWC = 4,
};

struct CommandCopyLayoutV2
{
    CommandHeaderV2 header;
    RefV1 source;
    RefV1 destination;
    uint16_t mode;
    uint16_t sourceLayout;
    uint16_t destinationLayout;
    uint16_t dataType;
    uint32_t dimensions[4];
    uint32_t validChannels;
    uint32_t sourceRowStride;
    uint32_t destinationRowStride;
    uint32_t reserved[7];
};

static_assert(sizeof(ModelHeaderV1) == 64);
static_assert(sizeof(SectionV1) == 32);
static_assert(sizeof(InvocationV1) == 64);
static_assert(sizeof(RefV1) == 8);
static_assert(sizeof(TensorV1) == 64);
static_assert(sizeof(BindingV1) == 64);
static_assert(sizeof(BindingAddressV1) == 16);
static_assert(sizeof(QParamV1) == 32);
static_assert(sizeof(CommandHeaderV2) == 16);
static_assert(sizeof(CommandRQLoadV2) == 32);
static_assert(offsetof(CommandRQLoadV2, qparamIndex) == 16);
static_assert(offsetof(CommandRQLoadV2, qparamBlock) == 24);
static_assert(sizeof(CommandDMA1DV2) == 64);
static_assert(sizeof(CommandDMA2DV2) == 64);
static_assert(sizeof(CommandDMA3DV2) == 64);
static_assert(sizeof(CommandGemm32V2) == 96);
static_assert(sizeof(CommandCopyLayoutV2) == 96);

using SerializedModelHeaderV1 = std::array<uint8_t, sizeof(ModelHeaderV1)>;
using SerializedSectionV1 = std::array<uint8_t, sizeof(SectionV1)>;
using SerializedInvocationV1 = std::array<uint8_t, sizeof(InvocationV1)>;

SerializedModelHeaderV1 Serialize(const ModelHeaderV1 &value);
SerializedSectionV1 Serialize(const SectionV1 &value);
SerializedInvocationV1 Serialize(const InvocationV1 &value);

bool IsAligned(uint32_t value, uint32_t alignment);
bool IsValidRange(uint32_t offset, uint32_t size, uint32_t totalBytes);
bool ValidateModelLayout(const ModelHeaderV1 &header, const SectionV1 *sections, size_t sectionCount, std::string &error);

}  // namespace regor::neuralai
