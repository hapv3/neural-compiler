//
// SPDX-FileCopyrightText: Copyright 2021, 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "ethos_u65_register_cs_generator.hpp"

#include "ethos_u65.hpp"
#define NPU_NAMESPACE ethosu65
#include "ethos_u65_interface.hpp"

namespace regor
{

using namespace ethosu65;

EthosU65RCSGenerator::EthosU65RCSGenerator(ArchEthosU65 *arch) : EthosU55RCSGenerator(arch), _arch(arch)
{
}


void EthosU65RCSGenerator::InsertTileDMACommand(const HLCStripe *stripe, Temporaries &temps, std::vector<const HighLevelCommand *> &emitted)
{
    // reshape to 3D-tensor where the width-axis is being tiled
    static auto reshapeFunc = [](Shape &shape, int tiledAxis)
    {
        int height = 1;
        int channel = 1;
        // all axes before tiledAxis are reshaped to height
        for ( int i = 0; i < tiledAxis; i++ )
        {
            height *= shape[i];
        }
        // all axes after tiledAxis are reshaped to channel
        for ( int i = tiledAxis + 1; i < shape.Size(); i++ )
        {
            channel *= shape[i];
        }

        shape = {1, height, shape[tiledAxis], channel};
    };

    auto op = stripe->operation;
    assert(op->type == OpType::Tile);

    // convert tile-operation to multiple DMA operations
    auto &ifm = op->ifm[0];
    auto &ofm = op->ofm;
    // max-height for 2D/3D DMA operations
    constexpr int maxHeight = (1 << 16) - 1;

    assert(ifm.format == TensorFormat::NHWC);
    assert(ofm.format == TensorFormat::NHWC);

    const auto &tileParams = op->parameters.tile;

    reshapeFunc(ifm.shape, tileParams.axis);
    reshapeFunc(ofm.shape, tileParams.axis);

    int elemSize = DataTypeSizeBits(ifm.dataType) / 8;
    auto srcStrides = Shape::GetStridesForShape(ifm.shape, {1, 1, 1, elemSize});
    auto dstStrides = Shape::GetStridesForShape(ofm.shape, {1, 1, 1, elemSize});

    int srcheightOffset = 0;
    int dstheightOffset = 0;
    int height = ifm.shape.Height();
    while ( height > 0 )
    {
        int heightSlice = std::min(height, maxHeight);

        // create 2D/3D DMA that copies ifm to ofm
        for ( int i = 0; i < tileParams.multiplier; i++ )
        {
            int addrOffset = i * ifm.shape.Width() * srcStrides.Width();
            auto dma = std::make_unique<HLCDMA>();
            dma->srcMemArea = ifm.memArea;
            dma->srcAddress = ifm.address + srcheightOffset;
            dma->srcStrides = srcStrides;
            dma->length = ifm.shape.Depth() * elemSize;
            dma->sizes = Shape(heightSlice, ifm.shape.Width());
            dma->destMemArea = ofm.memArea;
            dma->destAddress = ofm.address + dstheightOffset + addrOffset;
            dma->destStrides = dstStrides;
            emitted.push_back(dma.get());
            temps.cmds.push_back(std::move(dma));
        }
        height -= heightSlice;
        srcheightOffset += heightSlice * srcStrides.Height();
        dstheightOffset += heightSlice * dstStrides.Height();
    }
}


// Generates register commands for DMA operations
void EthosU65RCSGenerator::GenerateDMA(const HLCDMA *dma, AccessTracking &accesses)
{
    MemoryAccesses memoryAccesses;

    auto srcRegionMode = dma_region_mode::EXTERNAL;
    auto destRegionMode = dma_region_mode::EXTERNAL;

    if ( dma->destMemArea == _arch->LUTMemory() )
    {
        destRegionMode = dma_region_mode::INTERNAL;
    }

    uint32_t size0 = dma->sizes.Size() > 0 ? dma->sizes[-1] : 1;
    uint32_t size1 = dma->sizes.Size() > 1 ? dma->sizes[-2] : 1;
    uint64_t srcStride0 = dma->srcStrides.Size() > 1 ? dma->srcStrides[-2] : 0;
    uint64_t srcStride1 = dma->srcStrides.Size() > 2 ? dma->srcStrides[-3] : 0;
    uint64_t destStride0 = dma->destStrides.Size() > 1 ? dma->destStrides[-2] : 0;
    uint64_t destStride1 = dma->destStrides.Size() > 2 ? dma->destStrides[-3] : 0;

    dma_stride_mode strideMode;
    if ( size1 > 1 ) strideMode = dma_stride_mode::D3;
    else if ( size0 > 1 ) strideMode = dma_stride_mode::D2;
    else strideMode = dma_stride_mode::D1;

    // TODO current implementation assumes 2D/3D mode for DMA-writes
    if ( strideMode != dma_stride_mode::D1 )
    {
        assert((srcStride0 == uint64_t(dma->length)) && "Ethos-U65 currently only supports 2D/3D mode for DMA writes");
        assert((srcStride1 == uint64_t(dma->length * size0)) && "Ethos-U65 currently only supports 2D/3D mode for DMA writes");
    }

    Emit(isa::npu_set_dma0_src_region_t(ToRegion(dma->srcMemArea), srcRegionMode, dma_stride_mode::D1));
    Emit(isa::npu_set_dma0_src_t(dma->srcAddress));
    Emit(isa::npu_set_dma0_dst_region_t(ToRegion(dma->destMemArea), destRegionMode, strideMode));
    Emit(isa::npu_set_dma0_dst_t(dma->destAddress));
    Emit(isa::npu_set_dma0_len_t(dma->length));

    if ( strideMode != dma_stride_mode::D1 )
    {
        Emit(isa::npu_set_dma0_size0_t(size0));
        uint64_t skip0 = destStride0 - dma->length;
        Emit(isa::npu_set_dma0_skip0_t(skip0));
    }

    if ( strideMode == dma_stride_mode::D3 )
    {
        Emit(isa::npu_set_dma0_size1_t(size1));
        uint64_t skip1 = destStride1 - (dma->length * size0);
        Emit(isa::npu_set_dma0_skip1_t(skip1));
    }

    if ( strideMode == dma_stride_mode::D1 )
    {
        // Address accesses for 1D mode
        CheckAddressRange(dma->srcMemArea.memory, dma->srcAddress, dma->length);
        CheckAddressRange(dma->destMemArea.memory, dma->destAddress, dma->length);
        memoryAccesses.emplace_back(AccessDirection::Read, dma->srcMemArea, dma->srcAddress, dma->srcAddress + dma->length);
        memoryAccesses.emplace_back(AccessDirection::Write, dma->destMemArea, dma->destAddress, dma->destAddress + dma->length);
    }
    else
    {
        // Address accesses for 2D and 3D mode
        CheckAddressRange(dma->srcMemArea.memory, dma->srcAddress, dma->srcStrides[0]);
        CheckAddressRange(dma->destMemArea.memory, dma->destAddress, dma->destStrides[0]);
        memoryAccesses.emplace_back(AccessDirection::Read, dma->srcMemArea, dma->srcAddress, dma->srcAddress + dma->srcStrides[0]);
        memoryAccesses.emplace_back(AccessDirection::Write, dma->destMemArea, dma->destAddress, dma->destAddress + dma->destStrides[0]);
    }

    // Track memory accesses
    GenerateWaits(false, memoryAccesses, accesses.outstandingDmaAccesses);
    GenerateWaits(true, memoryAccesses, accesses.outstandingNpuAccesses);
    UpdateMemoryAccesses(memoryAccesses, accesses.outstandingDmaAccesses, accesses.maxOutstandingDMAOps);

    Emit(isa::npu_op_dma_start_t());
}

void EthosU65RCSGenerator::GenerateInitialRegisterSetup()
{
    auto mode = _arch->_cores <= 1 ? parallel_mode::SINGLE_CORE : parallel_mode::DUAL_CORE_DEPTH;
    Emit(isa::npu_set_parallel_mode_t(mode));
}

}  // namespace regor
