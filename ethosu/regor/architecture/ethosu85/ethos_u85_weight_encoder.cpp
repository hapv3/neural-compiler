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

#include "ethos_u85_weight_encoder.hpp"

#include "common/logging.hpp"

#include "architecture/architecture.hpp"
#include "architecture/ethos_u_scaling.hpp"
#include "architecture/mlw_encode.hpp"
#include "common/shape.hpp"
#include "compiler/tensor_properties.hpp"
#include "ethos_u85.hpp"
#include "ethos_u85_scaling.hpp"

#include <mlw_encode.h>
#include <string>
#include <unordered_map>


namespace regor
{


EthosU85WeightEncoder::EthosUEncodingConfig::EthosUEncodingConfig(int cores, Flags<WeightFormat> weightFormat) :
        _cores(cores), _weightFormat(weightFormat)
{
}

void EthosU85WeightEncoder::EthosUEncodingConfig::Rehash()
{
    _depthOffsetHash = 0;
    for ( int offset : this->depthOffsets )
    {
        _depthOffsetHash = _depthOffsetHash * 31 ^ offset;
    }

    _hash = SimpleHash32(_depthOffsetHash, ifmType, ofmBlockDepth, ifmBlockDepth, traversal, acc, dilation, stride, ohwiStrides, ofmUBlock);
}

uint32_t EthosU85WeightEncoder::EthosUEncodingConfig::Hash()
{
    return _hash;
}

bool EthosU85WeightEncoder::EthosUEncodingConfig::Equals(IWeightEncodingConfig *other)
{
    EthosUEncodingConfig *p = static_cast<EthosUEncodingConfig *>(other);
    return std::tie(ofmBlockDepth, ifmBlockDepth, traversal, _depthOffsetHash, ifmType, dilation, ohwiStrides, _weightFormat) ==
           std::tie(p->ofmBlockDepth, p->ifmBlockDepth, p->traversal, p->_depthOffsetHash, p->ifmType, p->dilation, p->ohwiStrides, p->_weightFormat);
}

const std::vector<int> &EthosU85WeightEncoder::EthosUEncodingConfig::DepthOffsets()
{
    return this->depthOffsets;
}

Flags<WeightFormat> EthosU85WeightEncoder::EthosUEncodingConfig::Format()
{
    return _weightFormat;
}


std::unique_ptr<IWeightEncodingConfig> EthosU85WeightEncoder::GetEncodingConfig(ArchitectureOpConfig *opCfg, const WeightsRef &weights,
    const Kernel *kernel, DataType ifmType, const std::vector<int> &depthOffsets, Flags<WeightFormat> format)
{
    assert(opCfg);
    assert(kernel);
    std::unique_ptr<EthosUEncodingConfig> params = std::make_unique<EthosUEncodingConfig>(_arch->_cores, format);

    EthosU85OpConfig *opConfig = static_cast<EthosU85OpConfig *>(opCfg);
    params->ofmUBlock = opConfig->OfmUBlock();
    params->ofmBlockDepth = opConfig->OfmBlock().Depth();
    params->ifmBlockDepth = opConfig->IfmBlock().Depth();
    params->traversal = opConfig->Traversal();
    params->acc = opConfig->Acc();
    params->depthOffsets = depthOffsets;
    params->ifmType = ifmType;
    params->dilation = kernel->Dilation();
    params->stride = kernel->Stride();

    if ( !weights.isScales )
    {
        Shape ohwiStrides = weights.view->StrideBytes() * 8 / DataTypeSizeBits(weights.type);
        if ( weights.axisOrder == AxisOrder::IHWO )
        {
            ohwiStrides = ohwiStrides.Extract(3, 1, 2, 0);
        }
        else if ( weights.axisOrder == AxisOrder::HWCM )
        {
            ohwiStrides = ohwiStrides.Extract(2, 0, 1, 3);
        }
        params->ohwiStrides = std::move(ohwiStrides);
    }
    else
    {
        params->ohwiStrides = Shape{nullptr, 4, 0};
    }
    params->Rehash();

    return params;
}

int EthosU85WeightEncoder::StreamsRequired(IWeightEncodingConfig *config, const Shape & /*weightShape*/, int &scaleStreamsRequired)
{
    scaleStreamsRequired = 1;
    return config->Format() % WeightFormat::Fast ? 1 : _arch->_cores;
}

static int EncodeBias32(int64_t bias, int32_t scale, int shift, uint8_t data[10])
{
    assert(-(1LL << (32 - 1)) <= bias && bias < (1LL << (32 - 1)));  // signed 32-bit range
    assert(0 <= scale);                                              // unsigned 31-bit range
    assert(0 <= shift && shift < (1 << 6));                          // unsigned 6-bit range

    data[0] = uint8_t((bias >> (0 * 8)) & 0xFF);
    data[1] = uint8_t((bias >> (1 * 8)) & 0xFF);
    data[2] = uint8_t((bias >> (2 * 8)) & 0xFF);
    data[3] = uint8_t((bias >> (3 * 8)) & 0xFF);
    data[4] = uint8_t((scale >> (0 * 8)) & 0xFF);
    data[5] = uint8_t((scale >> (1 * 8)) & 0xFF);
    data[6] = uint8_t((scale >> (2 * 8)) & 0xFF);
    data[7] = uint8_t((scale >> (3 * 8)) & 0x7F);
    data[8] = uint8_t(shift & 0x3F);
    data[9] = 0;
    return 10;
}

static int EncodeBias48(int64_t bias, int32_t scale, int shift, uint8_t data[10])
{
    assert(-(1LL << (48 - 1)) <= bias && bias < (1LL << (48 - 1)));  // signed 48-bit range
    assert(0 <= scale && scale < (1L << 15));                        // unsigned 15-bit range
    assert(0 <= shift && shift < (1 << 6));                          // unsigned 6-bit range

    data[0] = uint8_t((bias >> (0 * 8)) & 0xFF);
    data[1] = uint8_t((bias >> (1 * 8)) & 0xFF);
    data[2] = uint8_t((bias >> (2 * 8)) & 0xFF);
    data[3] = uint8_t((bias >> (3 * 8)) & 0xFF);
    data[4] = uint8_t((bias >> (4 * 8)) & 0xFF);
    data[5] = uint8_t((bias >> (5 * 8)) & 0xFF);
    data[6] = uint8_t((scale >> (0 * 8)) & 0xFF);
    data[7] = uint8_t((scale >> (1 * 8)) & 0x7F);
    data[8] = uint8_t(shift & 0x3F);
    data[9] = 0;
    return 10;
}

struct SparsityTracker
{
    int _sparse_zeroes = 4;
    int _sparse_index = 0;
    uint32_t _sparse_pos = 0xFFFFFFFF;
    void Reset() { _sparse_pos = 0xFFFFFFFF; }

    void Check(uint32_t pos, int depth, int weight)
    {
        if ( _sparse_pos != pos )
        {
            _sparse_pos = pos;
            _sparse_zeroes = 0;
            _sparse_index = 0;
            if ( depth & 3 ) throw WeightsNotSparse();
        }

        if ( weight == 0 ) _sparse_zeroes++;
        else if ( weight > 127 || weight < -127 ) throw WeightsNotSparse();

        if ( (_sparse_index & 3) == 3 )
        {
            if ( _sparse_zeroes < 2 ) throw WeightsNotSparse();
            _sparse_zeroes = 0;
        }

        _sparse_index++;
    }
};

template<typename TYPE>
class EthosU85WeightOrdering : public WeightSourceCommon
{
protected:
    static constexpr int InterleaveDepth = 4;
    // Transform
    WeightTransformParam *_param;
    WeightTransformFunc _transform;
    // Loop Limits
    Point2i _stride;
    int _ofmBlockDepth;
    int _ifmBlockDepth;
    short _ofmUBlockDepth;
    short _ifmUBlockDepth;
    short _decompX;
    short _decompY;
    short _subKernelRound;
    short _dwPaddingCount;
    // Saved state
    int _ofmBlockZ = 0;
    int _ifmBlockZ = 0;
    int _subKernelX = 0;
    int _subKernelY = 0;
    int _ifmUBlockOuter = 0;
    int _ifmUBlockInner = 0;
    int _ofmUBlockZ = 0;
    int _ifmUBlockZ = 0;
    int _subKernelElements = 0;
    int _strideX = 0;
    int _strideY = 0;
    int _kernelX = 0;
    int _kernelY = 0;
    int _ofmUBlockInner = 0;
    int _ofmUBlockOuter = 0;
    int _ifmLoopInc = 0;
    int _padding = 0;
    EthosU85Traversal _traversal;
    bool _sparse;
    SparsityTracker _sparsity;

public:
    EthosU85WeightOrdering(int cores, int macs, Point2i stride, const Point2i &dilation, int ofmBlockDepth, int ifmBlockDepth, int ifmBitDepth,
        int ofmUBlockDepth, WeightTransformFunc func, WeightTransformParam *param, EthosU85Traversal traversal, bool sparse)
    {
        const bool ifm16bit = (ifmBitDepth == 16);
        _streams = cores;
        _transform = func;
        _param = param;
        _traversal = traversal;
        _sparse = sparse;
        _stride = stride;

        _ofmBlockDepth = ofmBlockDepth;
        _ifmBlockDepth = ifmBlockDepth;
        _ofmUBlockDepth = short(ofmUBlockDepth);

        if ( traversal == EthosU85Traversal::PartKernel )
        {
            _subKernelRound = (ifm16bit || sparse) ? 10 : 5;
            _ifmUBlockDepth = ifm16bit && !sparse ? 8 : _ifmBlockDepth;
        }
        else
        {
            if ( traversal == EthosU85Traversal::DepthFirst )
            {
                _stride = Point2i(1, 1);
                _subKernelRound = 1;
                _ifmUBlockDepth = _ifmBlockDepth;
            }
            else if ( traversal == EthosU85Traversal::Depthwise )
            {
                _subKernelRound = 10;
                _ifmUBlockDepth = 1;
            }
        }

        _decompX = short(8 / dilation.x);
        _decompY = short(8 / dilation.y);
        _dwPaddingCount = (!ifm16bit && macs <= 256) ? 0 : (macs <= 512) ? 2 : 6;

        _ifmLoopInc = -_ifmBlockDepth;
    }

    void SetSource(const void *buffer, int depthOffset, const Shape &ohwiShape, const Shape &ohwiStrides, int streamIndex) override
    {
        SetSourceCommon(buffer, depthOffset, ohwiShape, ohwiStrides, streamIndex, false);
        assert(_streamIndex == streamIndex);
        _ofmUBlockZ = _streamIndex * InterleaveDepth;
        _sparsity.Reset();
    }

public:
    int Get(int16_t *output, int count) override
    {
        if ( _traversal == EthosU85Traversal::Depthwise )
        {
            assert(!_sparse);
            return GetNext<false, true>(output, count);
        }
        else if ( _sparse )
        {
            return GetNext<true, false>(output, count);
        }

        return GetNext<false, false>(output, count);
    }

    template<bool IS_SPARSE, bool IS_DEPTHWISE>
    int GetNext(int16_t *output, int count)
    {
        if ( _ofmBlockZ >= _ofmDepth )
        {
            return 0;
        }

        int ofmBlockZ, ifmBlockZ;
        int ifmUBlockOuter, ifmUBlockInner;
        int ifmUBlockZ, ofmUBlockZ, ofmUBlockInner, ofmUBlockOuter;
        int subKernelX, subKernelY;
        int strideX, strideY;
        int kernelX, kernelY;
        int padding;
        int16_t *write = output;

        const TYPE *buffer = reinterpret_cast<const TYPE *>(_source);

        for ( ofmBlockZ = _ofmBlockZ; ofmBlockZ < _ofmDepth; ofmBlockZ += _ofmBlockDepth )
        {
            _ifmLoopInc = -_ifmLoopInc;
            int clippedOfmBlockDepth = std::min(_ofmBlockDepth, _ofmDepth - ofmBlockZ);
            // IFM blocks required for the brick
            for ( ifmBlockZ = _ifmBlockZ; ifmBlockZ < (IS_DEPTHWISE ? 1 : _ifmDepth) && ifmBlockZ >= 0; ifmBlockZ += _ifmLoopInc )
            {
                _ifmBlockZ = ifmBlockZ;
                int clippedIfmBlockDepth = std::min(_ifmBlockDepth, _ifmDepth - ifmBlockZ);

                // Weight decomposition
                // Subkernel splitting (W)
                for ( subKernelX = _subKernelX; subKernelX < _kernelW; subKernelX += _decompX )
                {
                    int subWidth = std::min<int>(_kernelW - subKernelX, _decompX);
                    // Subkernel Splitting (H)
                    for ( subKernelY = _subKernelY; subKernelY < _kernelH; subKernelY += _decompY )
                    {
                        int subHeight = std::min<int>(_kernelH - subKernelY, _decompY);
                        int ifmBlockDepthOuter = IS_DEPTHWISE ? 1 : clippedIfmBlockDepth;
                        for ( ifmUBlockOuter = _ifmUBlockOuter; ifmUBlockOuter < ifmBlockDepthOuter; ifmUBlockOuter += _ifmUBlockDepth )
                        {
                            // OFM uBlocks in OFM-block over depth
                            for ( ofmUBlockOuter = _ofmUBlockOuter; ofmUBlockOuter < clippedOfmBlockDepth; ofmUBlockOuter += _ofmUBlockDepth )
                            {
                                // Part kernel first works across the kernel H/W and needs padding
                                if ( !_subKernelElements )
                                {
                                    int subKernelElements = subWidth * subHeight;
                                    _subKernelElements = RoundAway<int>(subKernelElements, _subKernelRound);
                                }
                                for ( strideY = _strideY; strideY < _stride.y; ++strideY )
                                {
                                    int stridedKernelH = (subHeight + _stride.y - 1 - strideY) / _stride.y;
                                    for ( strideX = _strideX; strideX < _stride.x; ++strideX )
                                    {
                                        int stridedKernelW = (subWidth + _stride.x - 1 - strideX) / _stride.x;
                                        for ( kernelY = _kernelY; kernelY < stridedKernelH; ++kernelY )
                                        {
                                            int y = kernelY;
                                            for ( kernelX = _kernelX; kernelX < stridedKernelW; ++kernelX )
                                            {
                                                int x = kernelY % 2 == 0 ? kernelX : stridedKernelW - 1 - kernelX;
                                                _subKernelElements--;
                                                int ifmUBlockInnerStep = IS_DEPTHWISE ? 1 : (IS_SPARSE ? 16 : 8);
                                                for ( ifmUBlockInner = _ifmUBlockInner; ifmUBlockInner < _ifmUBlockDepth; ifmUBlockInner += ifmUBlockInnerStep )
                                                {
                                                    // Feed OFM uBlock elements
                                                    for ( ofmUBlockZ = _ofmUBlockZ; ofmUBlockZ < _ofmUBlockDepth; ofmUBlockZ += InterleaveDepth * _streams )
                                                    {
                                                        for ( ofmUBlockInner = _ofmUBlockInner; ofmUBlockInner < InterleaveDepth; ofmUBlockInner++ )
                                                        {
                                                            // Source IFM uBlock elements (only 1 element deep if
                                                            // depthwise)
                                                            for ( ifmUBlockZ = _ifmUBlockZ; ifmUBlockZ < ifmUBlockInnerStep; ifmUBlockZ++ )
                                                            {
                                                                // Source position within the current subkernel
                                                                int wx = subKernelX + strideX + x * _stride.x;
                                                                int wy = subKernelY + strideY + y * _stride.y;
                                                                // Source IFM/OFM slices
                                                                int ifm_z = ifmBlockZ + ifmUBlockOuter + ifmUBlockInner + ifmUBlockZ;
                                                                int ofm_z = ofmBlockZ + ofmUBlockOuter + ofmUBlockInner + ofmUBlockZ;
                                                                int weight = 0;
                                                                if ( ifm_z < _ifmDepth && ofm_z < _ofmDepth )
                                                                {
                                                                    _param->o = ofm_z;
                                                                    _param->h = wy;
                                                                    _param->w = wx;
                                                                    _param->i = ifm_z;
                                                                    weight = int(buffer[WeightIndex(ofm_z, wy, wx, ifm_z)]);
                                                                    weight = _transform(_param, weight);
                                                                }

                                                                if constexpr ( IS_SPARSE )
                                                                    _sparsity.Check((unsigned(wy) << 16) | wx, ifm_z, weight);

                                                                *write++ = int16_t(weight);

                                                                if ( --count == 0 )
                                                                {
                                                                    // Save state
                                                                    _subKernelElements++;
                                                                    _ifmUBlockZ = ifmUBlockZ + 1;
                                                                    _ofmUBlockInner = ofmUBlockInner;
                                                                    _ofmUBlockZ = ofmUBlockZ;
                                                                    _ifmUBlockInner = ifmUBlockInner;
                                                                    _kernelX = kernelX;
                                                                    _kernelY = kernelY;
                                                                    _strideX = strideX;
                                                                    _strideY = strideY;
                                                                    _ofmUBlockOuter = ofmUBlockOuter;
                                                                    _ifmUBlockOuter = ifmUBlockOuter;
                                                                    _subKernelY = subKernelY;
                                                                    _subKernelX = subKernelX;
                                                                    _ofmBlockZ = ofmBlockZ;
                                                                    _ifmLoopInc = -_ifmLoopInc;
                                                                    return int(intptr_t(write - output));
                                                                }
                                                            }
                                                            _ifmUBlockZ = 0;
                                                        }
                                                        _ofmUBlockInner = 0;
                                                    }
                                                    _ofmUBlockZ = _streamIndex * InterleaveDepth;
                                                }
                                                // Depthwise padding
                                                if ( IS_DEPTHWISE && _subKernelElements % _subKernelRound == 0 )
                                                {
                                                    int padCount = _dwPaddingCount * _ofmUBlockDepth / _streams;
                                                    for ( padding = _padding; padding < padCount; padding++ )
                                                    {
                                                        *write++ = 0;
                                                        if ( --count == 0 )
                                                        {
                                                            // Save state
                                                            _subKernelElements++;
                                                            _padding = padding + 1;
                                                            _ifmUBlockInner = ifmUBlockInner;  // Will skip loop above
                                                            _kernelX = kernelX;
                                                            _kernelY = kernelY;
                                                            _strideX = strideX;
                                                            _strideY = strideY;
                                                            _ofmUBlockOuter = ofmUBlockOuter;
                                                            _ifmUBlockOuter = ifmUBlockOuter;
                                                            _subKernelY = subKernelY;
                                                            _subKernelX = subKernelX;
                                                            _ofmBlockZ = ofmBlockZ;
                                                            _ifmLoopInc = -_ifmLoopInc;
                                                            return int(intptr_t(write - output));
                                                        }
                                                    }
                                                    _padding = 0;
                                                }
                                                _ifmUBlockInner = 0;
                                            }
                                            _kernelX = 0;
                                        }
                                        _kernelY = 0;
                                    }
                                    _strideX = 0;
                                }
                                // Padding
                                if ( _subKernelElements > 0 )
                                {
                                    int padCount = _subKernelElements + (IS_DEPTHWISE ? _dwPaddingCount : 0);
                                    padCount = padCount * _ifmUBlockDepth * _ofmUBlockDepth / _streams;
                                    for ( padding = _padding; padding < padCount; padding++ )
                                    {
                                        *write++ = 0;
                                        if ( --count == 0 )
                                        {
                                            // Save state
                                            _padding = padding + 1;
                                            _strideY = strideY;  // Will skip loop above
                                            _ofmUBlockOuter = ofmUBlockOuter;
                                            _ifmUBlockOuter = ifmUBlockOuter;
                                            _subKernelY = subKernelY;
                                            _subKernelX = subKernelX;
                                            _ofmBlockZ = ofmBlockZ;
                                            _ifmLoopInc = -_ifmLoopInc;
                                            return int(intptr_t(write - output));
                                        }
                                    }
                                    _padding = 0;
                                }
                                _subKernelElements = 0;
                                _strideY = 0;
                            }
                            _ofmUBlockOuter = 0;
                        }
                        _ifmUBlockOuter = 0;
                    }
                    _subKernelY = 0;
                }
                _subKernelX = 0;
            }
        }
        _ifmLoopInc = -_ifmBlockDepth;
        _ifmBlockZ = 0;
        _ofmBlockZ = 0;
        // Return weights generated (less than requested count == EOS)
        return int(intptr_t(write - output));
    }
};


template<typename TYPE>
class EthosU85WeightOrderingFwd : public WeightSourceCommon
{
protected:
    int _weightBlockSize;
    int _blockSizeEmitted;
    EthosU85Traversal _traversal;
    std::vector<std::unique_ptr<EthosU85WeightOrdering<TYPE>>> _weightSource;

public:
    EthosU85WeightOrderingFwd(int cores, int macs, Point2i stride, const Point2i &dilation, int ofmBlockDepth, int ifmBlockDepth,
        int ifmBitDepth, int ofmUBlockDepth, WeightTransformFunc func, WeightTransformParam *param, EthosU85Traversal traversal, bool sparse)
    {
        assert(traversal != EthosU85Traversal::Depthwise);
        const bool isPartKernel = (traversal == EthosU85Traversal::PartKernel);
        const bool ifm16bit = ifmBitDepth == 16;

        const int ifmUBlockDepth = isPartKernel && ifm16bit && !sparse ? 8 : ifmBlockDepth;
        const int subKernelRound = isPartKernel ? (!ifm16bit && !sparse ? 5 : 10) : 1;
        _streams = cores;
        _weightBlockSize = ofmUBlockDepth * ifmUBlockDepth * subKernelRound / _streams;
        _blockSizeEmitted = 0;
        _traversal = traversal;
        for ( int stream = 0; stream < _streams; ++stream )
        {
            _weightSource.push_back(std::make_unique<EthosU85WeightOrdering<TYPE>>(cores, macs, stride, dilation,
                ofmBlockDepth, ifmBlockDepth, ifmBitDepth, ofmUBlockDepth, func, param, traversal, sparse));
        }
    }

    void SetSource(const void *buffer, int depthOffset, const Shape &ohwiShape, const Shape &ohwiStrides, int streamIndex) override
    {
        SetSourceCommon(buffer, depthOffset, ohwiShape, ohwiStrides, streamIndex, false);
        _streamIndex = streamIndex;
        for ( int stream = 0; stream < _streams; ++stream )
        {
            _weightSource[stream]->SetSource(buffer, depthOffset, ohwiShape, ohwiStrides, stream);
        }
    }

public:
    int Get(int16_t *output, int count) override
    {
        // Interleave weight sources
        int offset = 0;
        int result = _weightBlockSize;
        int stream;
        while ( offset < count && result )
        {
            for ( stream = _streamIndex; stream < _streams; ++stream )
            {
                int blockSize = std::min(_weightBlockSize - _blockSizeEmitted, count - offset);
                result = _weightSource[stream]->Get(output + offset, blockSize);
                offset += result;
                if ( offset == count )
                {  // Output filled
                    if ( result < _weightBlockSize )
                    {  // Incomplete block, save state
                        _blockSizeEmitted = result;
                        _streamIndex = stream;
                    }
                    else
                    {  // Complete block, start on next
                        _blockSizeEmitted = 0;
                        _streamIndex = stream + 1;
                    }
                    return offset;
                }
                _blockSizeEmitted = 0;
            }
            _streamIndex = 0;
        }
        // Return weights generated (less than requested count == EOS)
        return offset;
    }
};


std::unique_ptr<IVolumeWeightSource> EthosU85WeightEncoder::GetWeightSource(
    IWeightEncodingConfig *config, DataType weightType, WeightTransformFunc func, WeightTransformParam *param)
{
    EthosUEncodingConfig *cfg = static_cast<EthosUEncodingConfig *>(config);

    int ofmUBlockDepth = cfg->ofmUBlock.Depth();

    int ifmBitDepth = DataTypeSizeBits(cfg->ifmType);
    bool isFast = cfg->Format() % WeightFormat::Fast;
    bool isSparse = cfg->Format() % WeightFormat::Sparse2_4;

    if ( weightType == DataType::UInt8 )
    {
        if ( isFast && _arch->_cores > 1 )  // No interleaving needed for FWD if only one stream
        {
            return std::make_unique<EthosU85WeightOrderingFwd<uint8_t>>(_arch->_cores, _arch->_macs, cfg->stride, cfg->dilation,
                cfg->ofmBlockDepth, cfg->ifmBlockDepth, ifmBitDepth, ofmUBlockDepth, func, param, cfg->traversal, isSparse);
        }
        else
        {
            assert(!(isFast && cfg->traversal == EthosU85Traversal::Depthwise));
            return std::make_unique<EthosU85WeightOrdering<uint8_t>>(_arch->_cores, _arch->_macs, cfg->stride, cfg->dilation,
                cfg->ofmBlockDepth, cfg->ifmBlockDepth, ifmBitDepth, ofmUBlockDepth, func, param, cfg->traversal, isSparse);
        }
    }
    else if ( weightType == DataType::Int8 )
    {
        if ( isFast && _arch->_cores > 1 )  // No interleaving needed for FWD if only one stream
        {
            return std::make_unique<EthosU85WeightOrderingFwd<int8_t>>(_arch->_cores, _arch->_macs, cfg->stride, cfg->dilation,
                cfg->ofmBlockDepth, cfg->ifmBlockDepth, ifmBitDepth, ofmUBlockDepth, func, param, cfg->traversal, isSparse);
        }
        else
        {
            assert(!(isFast && cfg->traversal == EthosU85Traversal::Depthwise));
            return std::make_unique<EthosU85WeightOrdering<int8_t>>(_arch->_cores, _arch->_macs, cfg->stride, cfg->dilation,
                cfg->ofmBlockDepth, cfg->ifmBlockDepth, ifmBitDepth, ofmUBlockDepth, func, param, cfg->traversal, isSparse);
        }
    }

    assert(false && "No weight source for this datatype");
    return nullptr;
}


template<typename TYPE, DataType IFM_TYPE>
class EthosU85ScaleSource : public IVolumeScaleSource
{
private:
    const TYPE *_buffer = nullptr;
    const QuantizedScale *_scales = nullptr;
    int _biasIndex = 0;
    int _biasCount = 0;
    int _depthLength = 0;
    int _bufferSize = 0;
    int _streamIndex = 0;
    int _cores = 0;
    int _uBlockDepth = 0;
    Quantization _quantization;

public:
    EthosU85ScaleSource(int cores, int uBlockDepth, Quantization quantization) :
            _cores(cores), _uBlockDepth(uBlockDepth), _quantization(std::move(quantization))
    {
        assert(!_quantization.scales.empty());
        auto invalidScale = std::find_if(std::begin(_quantization.scales), std::end(_quantization.scales),
            [](const auto q) { return q.shift < 0 || q.shift >= 64; });
        assert(invalidScale == std::end(_quantization.scales));
    }

    int Elements()
    {
        assert(_biasCount >= 0);
        return _biasCount;
    }

    int Get(int64_t *biasBuffer, QuantizedScale *quantBuffer, int count)
    {
        count = std::min(count, _biasCount);
        const size_t scaleSize = _quantization.scales.size();

        for ( int i = 0; i < count; i++ )
        {
            int index = _biasIndex + i;
            if ( _depthLength > 0 )
            {
                _depthLength--;
                *biasBuffer++ = _buffer ? static_cast<int64_t>(_buffer[index % _bufferSize]) : 0;
                *quantBuffer++ = _quantization.scales[index % scaleSize];
            }
            else
            {
                *biasBuffer++ = 0;
                *quantBuffer++ = QuantizedScale(0, 0);
            }
            _biasCount--;
        }

        _biasIndex += count;
        return count;
    }

    void SetSource(const void *buffer, int biasCount, int depthOffset, int depthLength, int streamIndex)
    {
        assert(streamIndex >= 0 && streamIndex < _cores);
        bool isBroadcast = biasCount == 1;
        assert(depthOffset + depthLength <= biasCount || isBroadcast);
        _buffer = reinterpret_cast<const TYPE *>(buffer);
        _depthLength = depthLength;
        _biasIndex = depthOffset + streamIndex;                            // Where to start in the buffer
        _biasCount = RoundAway(depthLength, RoundAway(_uBlockDepth, 16));  // How many biases to generate
        _bufferSize = biasCount;
    }
};


std::unique_ptr<IVolumeScaleSource> EthosU85WeightEncoder::GetScaleSource(
    IWeightEncodingConfig *config, DataType scaleType, const Quantization &explicitQuant)
{
    EthosUEncodingConfig *cfg = static_cast<EthosUEncodingConfig *>(config);
    assert(explicitQuant.type == QuantizationType::EXPLICIT);

    if ( scaleType == DataType::Int32 )
    {
        if ( cfg->ifmType == DataType::Int8 )
        {
            return std::make_unique<EthosU85ScaleSource<int32_t, DataType::Int8>>(_arch->_cores, cfg->ofmUBlock.Depth(), explicitQuant);
        }
        else if ( cfg->ifmType == DataType::UInt8 )
        {
            return std::make_unique<EthosU85ScaleSource<int32_t, DataType::UInt8>>(_arch->_cores, cfg->ofmUBlock.Depth(), explicitQuant);
        }
        else if ( cfg->ifmType == DataType::Int16 )
        {
            return std::make_unique<EthosU85ScaleSource<int32_t, DataType::Int16>>(_arch->_cores, cfg->ofmUBlock.Depth(), explicitQuant);
        }
        else if ( cfg->ifmType == DataType::UInt16 )
        {
            return std::make_unique<EthosU85ScaleSource<int32_t, DataType::UInt16>>(_arch->_cores, cfg->ofmUBlock.Depth(), explicitQuant);
        }
        else if ( cfg->ifmType == DataType::Int32 )
        {
            return std::make_unique<EthosU85ScaleSource<int32_t, DataType::Int32>>(_arch->_cores, cfg->ofmUBlock.Depth(), explicitQuant);
        }
        else if ( cfg->ifmType == DataType::Int48 )
        {
            return std::make_unique<EthosU85ScaleSource<int32_t, DataType::Int48>>(_arch->_cores, cfg->ofmUBlock.Depth(), explicitQuant);
        }
        else if ( cfg->ifmType == DataType::Int64 )
        {
            return std::make_unique<EthosU85ScaleSource<int32_t, DataType::Int64>>(_arch->_cores, cfg->ofmUBlock.Depth(), explicitQuant);
        }
    }
    else if ( scaleType == DataType::Int48 && DataTypeSizeBits(cfg->ifmType) == 16 )
    {
        return std::make_unique<EthosU85ScaleSource<int48_t, DataType::Int16>>(_arch->_cores, cfg->ofmUBlock.Depth(), explicitQuant);
    }
    else if ( scaleType == DataType::Int64 && DataTypeSizeBits(cfg->ifmType) == 16 )
    {
        return std::make_unique<EthosU85ScaleSource<int64_t, DataType::Int16>>(_arch->_cores, cfg->ofmUBlock.Depth(), explicitQuant);
    }

    return nullptr;
}

Quantization EthosU85WeightEncoder::MakeExplicit(const Quantization &ifmQ, const Quantization &weightQ,
    const Quantization &ofmQ, DataType scaleType, DataType ifmType, OpType opType)
{
    if ( DataTypeSizeBits(ifmType) == 16 ) ifmType = DataType::Int16;

    return RescalePerChannel(ifmQ, weightQ, ofmQ, scaleType, ifmType, opType);
}


WeightsInfo EthosU85WeightEncoder::EncodeWeights(IWeightEncodingConfig *config, IWeightSource *source, std::vector<uint8_t> &result)
{
    EthosUEncodingConfig *cfg = static_cast<EthosUEncodingConfig *>(config);
    auto fn = (cfg->Format() % WeightFormat::Fast) ? mle_encode_fwd_proxy : mle_encode_proxy;
    unsigned flags = MLW_ENCODE_FLAG_NONE;
    if ( cfg->Format().All(WeightFormat::Fast, WeightFormat::Sparse2_4) ) flags |= MLW_ENCODE_NO_PALETTE_LUT;
    auto res = fn(source, 128 * 1024, result, flags);
    WeightsInfo weightsInfo;
    weightsInfo.sourceSize = res.elements_read;
    weightsInfo.encodedSize = res.bytes_written;
    weightsInfo.zeroCount = res.zero_count;
    if ( res.distinct_values > 0 )
    {
        weightsInfo.distinctValues = res.distinct_values;
        for ( int i = 0; i < 8; i++ )
        {
            weightsInfo.weightsUsed[i] = res.distinct_weights[i];
        }
    }
    return weightsInfo;
}


int EthosU85WeightEncoder::EncodeScales(IWeightEncodingConfig *config, IScaleSource *source, std::vector<uint8_t> &result, bool measureOnly)
{
    EthosUEncodingConfig *cfg = static_cast<EthosUEncodingConfig *>(config);

    constexpr int BUFFER_SIZE = 8;
    constexpr int SCALE_ELEMENT_SIZE = 10;
    auto EncodeBias = cfg->acc == EthosU85Accumulator::Acc32 ? EncodeBias32 : EncodeBias48;

    if ( measureOnly )
    {
        return source->Elements() * SCALE_ELEMENT_SIZE;  // Must be accurate
    }

    int64_t scaleBuffer[BUFFER_SIZE];
    QuantizedScale quantBuffer[BUFFER_SIZE];

    int start = int(result.size());
    int write = start;
    result.reserve(start + source->Elements() * SCALE_ELEMENT_SIZE);
    while ( true )
    {
        int count = source->Get(scaleBuffer, quantBuffer, BUFFER_SIZE);
        result.resize(write + (count * SCALE_ELEMENT_SIZE));

        for ( int i = 0; i < count; i++ )
        {
            write += EncodeBias(scaleBuffer[i], quantBuffer[i].scale, quantBuffer[i].shift, &result[write]);
        }
        if ( count < BUFFER_SIZE )
        {
            break;
        }
    }

    return write - start;
}

}  // namespace regor
