//
// SPDX-FileCopyrightText: Copyright 2021-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the License); you may
// not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#pragma once

#include "common/common.hpp"
#include "common/logging.hpp"

#include "architecture/ethos_u_register_cs_generator.hpp"
#include "architecture/ethos_u_scaling.hpp"
#include "common/data_type.hpp"
#include "compiler/high_level_command_stream.hpp"
#include "compiler/op_type.hpp"
#include "ethos_u55.hpp"

#include <cstdint>
#include <deque>
#include <unordered_map>
#include <vector>

namespace regor
{
class EthosU55Emitter
{
public:
    EthosU55Emitter() = default;
    void Emit(uint32_t instr);
    void Emit(uint64_t instr);
    void Clear();
    int Position() const { return int(_stream.size()); }
    const std::vector<uint32_t> &CommandStream() const { return _stream; }

private:
    bool SetRegister(uint16_t reg, uint64_t value);
    static bool IsCmd0(uint16_t key);
    static bool IsCmd1(uint16_t key);
    static bool IsOp(uint16_t key);
    std::vector<uint32_t> _stream;
    std::unordered_map<uint16_t, uint64_t> _registers;
};


// Specifies the addresses and dimensions of the tiles of a feature map.
// A feature map can use 1 to 4 tiles
struct TileBox
{
    int height0;         // The height of tile 0
    int height1;         // The height of tile 1
    int width0;          // The width of tile 0, and tile 2 (if used)
    Address address[4];  // Tile addresses
};

enum BasePointerIndex
{
    WeightTensor = 0,       // base address index for the Weight tensor
    ScratchTensor = 1,      // base address index for the Scratch_tensor in the TensorArena
    ScratchFastTensor = 2,  // base address for the Scratch_fast_tensor
    Mem2Mem = 3,            // base address slot for memory to memory transfer
};

enum class AccessDirection
{
    Read = 0,
    Write = 1,
};

enum class RCSIfmScaleMode : uint8_t
{
    OPA_OPB_16 = 0,
    OPA_32 = 1,
    OPB_32 = 2,
};

struct MemoryAccess
{
    AccessDirection direction;
    MemArea memArea;
    Address start;
    Address end;

    MemoryAccess(AccessDirection direction_, MemArea area_, Address start_, Address end_) :
            direction(direction_), memArea(area_), start(start_), end(end_)
    {
    }

    bool Conflicts(const MemoryAccess &other) const
    {
        bool overlaps = Overlaps(start, end, other.start, other.end) && memArea == other.memArea;
        return overlaps && (direction != AccessDirection::Read || other.direction != AccessDirection::Read);
    }
};

using MemoryAccesses = std::vector<MemoryAccess>;

struct LutSlot
{
    const HLCOperation *hlcOp = nullptr;
    int lastUsed = 0;
};

/// <summary>
/// Generates register command streams for Ethos-U55.
/// </summary>
class EthosU55RCSGenerator : public EthosURegisterCSGenerator<EthosU55RCSGenerator>
{
private:
    ArchEthosU55 *_arch;
    // For stripes that use LUT: the LUT slot to be used
    std::unordered_map<const HLCStripe *, int> _stripeToLutSlot;
    std::vector<LutSlot> _lutSlots;
    EthosU55Emitter _emit;

public:
    EthosU55RCSGenerator(ArchEthosU55 *arch);

    //----------------------------------------------------------------------
    // Print
    //----------------------------------------------------------------------

    int Disassemble(const uint32_t *in, std::string &op, std::vector<std::pair<std::string, std::string>> &fields);

protected:
    //----------------------------------------------------------------------
    // Helper functions
    //----------------------------------------------------------------------
    void Emit(uint32_t instr);
    void Emit(uint64_t instr);

    static int GetDoubleBufferOffset(HLCWeights *weights, int rangeIndex);
    static void CheckAddressRange(ArchitectureMemory *memory, Address address, int size);
    static void CheckAddresses(const HLCFeatureMap &fm);
    // Calculates the rolling buffer address of the given coordinate.
    static Address AddressForCoordinate(const HLCFeatureMap &fm, const Shape &strides, const Shape &coord);
    // Calculates tile sizes/addresses of a feature map
    static TileBox GetTiles(const HLCFeatureMap &fm, const Shape &strides, const Box &area);
    MemoryAccess ToMemoryAccess(const HLCFeatureMap &fm, const Box &area, AccessDirection direction);
    // Returns region number used in NPU_SET_..._REGION
    uint32_t ToRegion(const MemArea &memArea);
    // Checks if the feature map is a scalar, and if so, returns the
    // quantized value in scalarValue.
    static bool IsScalar(const HLCFeatureMap &fm, int32_t &scalarValue);
    // Calculates waits for KERNEL_WAIT/DMA_WAIT, returns -1 if no wait is needed
    // - opAccesses contains the memory accesses for the current operation
    // - outstanding contains the memory accesses for ongoing "other" operations
    //   (DMA operations if the current op is an NPU operation, NPU operations if the current op is a DMA operation)
    // Note: NPU - NPU dependency is handled via blockdep
    static int CalcCommandWaits(const MemoryAccesses &opAccesses, std::deque<MemoryAccesses> &outstanding);
    // Returns LUT slot to be used for the given LUT operation.
    // Sets alreadyInLutMem to true if the LUT is already in SHRAM.
    int AllocateLutSlot(std::vector<LutSlot> &lutSlots, const HLCOperation *op, int sizeInSlots, int timestamp, bool &alreadyInLutMem);
    //----------------------------------------------------------------------
    // Scaling (OFM/OPA/OPB_SCALE)
    //----------------------------------------------------------------------

    // Generates OFM_SCALE register for pooling operations
    void GenerateOFMScalingForPooling(HLCOperation *poolOp, bool useGlobalScale);
    // Generates OFM/OPA/OPB_SCALE registers for elementwise operators.
    // Returns the operator to scale
    RCSIfmScaleMode GenerateScalingForElementwise(HLCOperation *op, int ifm0Index);



    //----------------------------------------------------------------------
    // BLOCKDEP calculation
    //----------------------------------------------------------------------

    // Given the area and block size, adds the first/last jobs (depending on fromStart) to jobs.
    // - area: total amount of work to perform
    // - block: size of each job
    // - fromStart: if true, the first jobs are added, if false, the last jobs are added
    //   (in that case, the very last job is added last)
    void GetJobs(const Box &area, const Shape &block, int nrJobsToGet, bool fromStart, std::vector<Box> &jobs);
    // Calculates the value for the BLOCKDEP register
    int CalcBlockDep(const HLCStripe *prevStripe, const HLCStripe *stripe);



    //----------------------------------------------------------------------
    // Register generation
    //----------------------------------------------------------------------

    void GeneratePadding(const HLCPadding &padding);
    // Generates ACTIVATION registers
    void GenerateActivation(const HLCStripe *stripe, MemoryAccesses &memoryAccesses);
    // Generates KERNEL related registers
    void GenerateKernel(const Kernel &kernel, bool partKernel);
    // Generates IFM2_BROADCAST register for binary elementwise operations
    void GenerateIFM2Broadcast(const Shape &ifmShape, const Shape &ifm2Shape, bool reversedOperands, bool isScalar);
    // Generates IFM_PRECISION register
    void GenerateIFMPrecision(const HLCFeatureMap &fm, RCSIfmScaleMode scaleMode, HLCRoundMode roundMode);
    // Generates IFM2_PRECISION register
    void GenerateIFM2Precision(const HLCFeatureMap &fm);
    // Generates OFM_PRECISION register
    void GenerateOFMPrecision(const HLCFeatureMap &fm, bool useGlobalScale);
    // Generates common IFM registers
    void GenerateIFM(OpType opType, const HLCFeatureMap &fm, const Box &inputArea);
    // Generates common IFM2 registers
    void GenerateIFM2(OpType opType, const HLCFeatureMap &fm, const Box &inputArea, bool isScalar, int32_t scalarValue);
    // Generates OFM registers
    void GenerateOFM(OpType opType, const HLCFeatureMap &fm, const Box &outputArea);
    // Generates WEIGHT registers
    void GenerateWeights(const HLCStripe *stripe, MemoryAccesses &memoryAccesses);
    // Generates SCALE registers
    void GenerateScales(const HLCStripe *stripe, MemoryAccesses &memoryAccesses);
    // Generates OFM_BLK_HEIGHT/WIDTH/DEPTH registers
    void GenerateBlockConfig(const EthosU55OpConfig *config);
    // Generates IB_END/IB_START/AB_START/ACC_FORMAT registers
    void GenerateShramRegisters(const EthosU55OpConfig *config, bool hasIfm2);
    // Calculates and generates KERNEL_WAIT or DMA_WAIT register
    void GenerateWaits(bool isKernelWait, const MemoryAccesses &memoryAccesses, std::deque<MemoryAccesses> &outstandingAccesses);
    // Save current memory accesses to accessesToUpdate
    void UpdateMemoryAccesses(const MemoryAccesses &memoryAccesses, std::deque<MemoryAccesses> &accessesToUpdate, int maxWaits);

    struct Temporaries
    {
        std::vector<std::unique_ptr<HighLevelCommand>> cmds;
        std::vector<std::unique_ptr<ArchitectureOpConfig>> configs;
    };

    // Inserts DMA commands for copying LUTs from constant memory to LUT memory
    void InsertLUTDMACommand(int index, const HLCStripe *stripe, Temporaries &temps, std::vector<const HighLevelCommand *> &emitted);
    // Inserts DMA commands to handle TILE operations
    virtual void InsertTileDMACommand(const HLCStripe *stripe, Temporaries &temps, std::vector<const HighLevelCommand *> &emitted);
    // Inserts commands to handle transposing
    virtual void InsertTransposeCommand(const HLCStripe *stripe, Temporaries &temps, std::vector<const HighLevelCommand *> &emitted);

    //----------------------------------------------------------------------
    // Operations
    //----------------------------------------------------------------------
    struct AccessTracking
    {
        std::deque<MemoryAccesses> outstandingNpuAccesses;
        std::deque<MemoryAccesses> outstandingDmaAccesses;
        int maxOutstandingDMAOps;
        int maxOutstandingKernelOps;
    };

    // Generates NPU_OP_* command
    void GenerateOperationCode(OpType opType);
    void GenerateCommon(const HLCStripe *stripe, bool useGlobalScale, RCSIfmScaleMode opToScale,
        MemoryAccesses &memoryAccesses, int ifm0Index = 0);
    // Conv2D/Depthwise operations
    void GenerateConvolutionOp(const HLCStripe *stripe, MemoryAccesses &memoryAccesses);
    // MaxPool/AvgPool/ResizeBilinear or operations that are mapped to AvgPool
    void GeneratePoolingOp(const HLCStripe *stripe, MemoryAccesses &memoryAccesses);
    // Elementwise operations
    void GenerateElementwiseOp(const HLCStripe *stripe, MemoryAccesses &memoryAccesses);
    bool GenerateStripe(const HLCStripe *stripe, const HLCStripe *prevStripe, AccessTracking &accesses);
    void PrepareCommand(int index, HighLevelCommand *cmd, Temporaries &temps, std::vector<const HighLevelCommand *> &emitted);
    // Generates register commands for DMA operations
    virtual void GenerateDMA(const HLCDMA *dma, AccessTracking &accesses);

    virtual void GenerateInitialRegisterSetup()
    {
        // No special initial setup for Ethos U55
    }

public:
    std::vector<uint32_t> GenerateCommandStream(std::vector<std::unique_ptr<HighLevelCommand>> &highLevelCommandStream,
        CmdRanges *cmdRanges, bool verbose) override;

    static uint32_t IdRegister();
    static bool IsSupportedElementwise(const OpType opType);
};

}  // namespace regor
