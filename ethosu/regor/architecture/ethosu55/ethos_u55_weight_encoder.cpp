//
// SPDX-FileCopyrightText: Copyright 2021-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "ethos_u55_weight_encoder.hpp"

#include "common/logging.hpp"

#include "architecture/architecture.hpp"
#include "architecture/ethos_u_scaling.hpp"
#include "architecture/mlw_encode.hpp"
#include "common/buffer_view.hpp"
#include "common/shape.hpp"
#include "compiler/tensor_properties.hpp"
#include "ethos_u55.hpp"
#include "ethos_u55_scaling.hpp"

#include <mlw_encode.h>

namespace regor
{


EthosU55WeightEncoder::EthosUEncodingConfig::EthosUEncodingConfig(int cores) : _cores(cores)
{
}

void EthosU55WeightEncoder::EthosUEncodingConfig::Rehash()
{
    _depthOffsetHash = 0;
    for ( int offset : this->depthOffsets )
    {
        _depthOffsetHash = _depthOffsetHash * 31 ^ offset;
    }

    _hash = SimpleHash32(ofmBlockDepth, traversal, _depthOffsetHash, ifmType, dilation, ohwiStrides);
}

uint32_t EthosU55WeightEncoder::EthosUEncodingConfig::Hash()
{
    return _hash;
}

bool EthosU55WeightEncoder::EthosUEncodingConfig::Equals(IWeightEncodingConfig *other)
{
    EthosUEncodingConfig *p = static_cast<EthosUEncodingConfig *>(other);
    return std::tie(ofmBlockDepth, traversal, _depthOffsetHash, ifmType, dilation, ohwiStrides) ==
           std::tie(p->ofmBlockDepth, p->traversal, p->_depthOffsetHash, p->ifmType, p->dilation, ohwiStrides);
}

const std::vector<int> &EthosU55WeightEncoder::EthosUEncodingConfig::DepthOffsets()
{
    return this->depthOffsets;
}

Flags<WeightFormat> EthosU55WeightEncoder::EthosUEncodingConfig::Format()
{
    return WeightFormat::Default;
}


std::unique_ptr<IWeightEncodingConfig> EthosU55WeightEncoder::GetEncodingConfig(ArchitectureOpConfig *opCfg,
    const WeightsRef &weights, const Kernel *kernel, DataType ifmType, const std::vector<int> &depthOffsets, Flags<WeightFormat>)
{
    assert(opCfg);
    assert(kernel);
    std::unique_ptr<EthosUEncodingConfig> params = std::make_unique<EthosUEncodingConfig>(_arch->_cores);

    EthosU55OpConfig *opConfig = static_cast<EthosU55OpConfig *>(opCfg);
    params->ofmBlockDepth = opConfig->OfmBlock().Depth();
    params->traversal = opConfig->Traversal();
    params->depthOffsets = depthOffsets;
    params->ifmType = ifmType;
    params->dilation = kernel->Dilation();

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
        params->ohwiStrides[0] = params->ohwiStrides[0] * _arch->_cores;
    }
    else
    {
        params->ohwiStrides = Shape{nullptr, 4, 0};
    }
    params->Rehash();

    return params;
}

int EthosU55WeightEncoder::StreamsRequired(IWeightEncodingConfig *, const Shape &weightShape, int &scaleStreamsRequired)
{
    scaleStreamsRequired = std::min(weightShape[0], _arch->_cores);
    return scaleStreamsRequired;
}

static int EncodeBias(int64_t bias, int32_t scale, int shift, uint8_t data[10])
{
    assert(-(1LL << (40 - 1)) <= bias && bias < (1LL << (40 - 1)));  // signed 40-bit range
    assert(0 <= shift && shift < (1 << 6));                          // unsigned 6-bit range

    data[0] = uint8_t((bias >> (0 * 8)) & 0xFF);
    data[1] = uint8_t((bias >> (1 * 8)) & 0xFF);
    data[2] = uint8_t((bias >> (2 * 8)) & 0xFF);
    data[3] = uint8_t((bias >> (3 * 8)) & 0xFF);
    data[4] = uint8_t((bias >> (4 * 8)) & 0xFF);
    data[5] = uint8_t((scale >> (0 * 8)) & 0xFF);
    data[6] = uint8_t((scale >> (1 * 8)) & 0xFF);
    data[7] = uint8_t((scale >> (2 * 8)) & 0xFF);
    data[8] = uint8_t((scale >> (3 * 8)) & 0xFF);
    data[9] = uint8_t(shift & 0x3F);
    return 10;
}


template<typename TYPE>
class EthosUWeightOrdering : public WeightSourceCommon
{
protected:
    // Transform
    WeightTransformParam *_param;
    WeightTransformFunc _transform;
    // Loop Limits
    int _ofmBlockDepth;
    int _ifmBlockDepth;
    short _ofmUBlockDepth;
    short _ifmUBlockDepth;
    short _decompX;
    short _decompY;
    short _subKernelRound;
    // Saved state
    int _ofmBlockZ = 0;
    int _ifmBlockZ = 0;
    int _subKernelX = 0;
    int _subKernelY = 0;
    int _ifmUBlockOuter = 0;
    int _ifmUBlockInner = 0;
    int _ofmUBlockZ = 0;
    int _ifmUBlockZ = 0;
    int _kernelElement = 0;
    int _ofmUBlock = 0;
    EthosUTraversal _traversal;

public:
    EthosUWeightOrdering(int cores, const Point2i &dilation, int ofmBlockDepth, int ifmBitDepth, int ofmUBlockDepth,
        int ifmUBlockDepth, WeightTransformFunc func, WeightTransformParam *param, EthosUTraversal traversal)
    {
        _streams = cores;
        _ofmBlockDepth = ofmBlockDepth;
        _ifmBlockDepth = ((traversal == EthosUTraversal::PartKernel) || (ifmBitDepth == 16)) ? 16 : 32;
        _ofmUBlockDepth = short(ofmUBlockDepth);
        _ifmUBlockDepth = short(ifmUBlockDepth);
        _decompX = short(8 / dilation.x);
        _decompY = short(8 / dilation.y);
        if ( traversal == EthosUTraversal::Depthwise )
        {
            _subKernelRound = 4;
        }
        else if ( traversal == EthosUTraversal::PartKernel )
        {
            _subKernelRound = (ifmBitDepth == 16) ? 2 : 4;
        }
        else
        {
            _subKernelRound = 1;
        }
        _transform = func;
        _param = param;
        _traversal = traversal;
    }

    void SetSource(const void *buffer, int depthOffset, const Shape &ohwiShape, const Shape &ohwiStrides, int streamIndex) override
    {
        SetSourceCommon(buffer, depthOffset + streamIndex, ohwiShape, ohwiStrides, streamIndex, true);
    }

public:
    int Get(int16_t *output, int count) override
    {
        if ( _traversal == EthosUTraversal::Depthwise ) return GetNext<false, true>(output, count);
        else if ( _traversal == EthosUTraversal::PartKernel ) return GetNext<true, false>(output, count);
        return GetNext<false, false>(output, count);
    }

    template<bool IS_PARTKERNEL, bool IS_DEPTHWISE>
    int GetNext(int16_t *output, int count)
    {
        if ( _ofmBlockZ >= _ofmDepth )
        {
            return 0;
        }

        int ofmBlockZ, ifmBlockZ;
        int ifmUBlockOuter, ifmUBlockInner;
        int ifmUBlockZ, ofmUBlockZ, ofmUBlock;
        int subKernelX, subKernelY;
        int kernelElement;
        int16_t *write = output;

        const TYPE *buffer = reinterpret_cast<const TYPE *>(_source);
        int streamBlockDepth = (_ofmBlockDepth + _streams - 1 - _streamIndex) / _streams;

        for ( ofmBlockZ = _ofmBlockZ; ofmBlockZ < _ofmDepth; ofmBlockZ += streamBlockDepth )
        {
            int clippedOfmBlockDepth = std::min(streamBlockDepth, _ofmDepth - ofmBlockZ);
            // IFM blocks required for the brick
            for ( ifmBlockZ = _ifmBlockZ; ifmBlockZ < (IS_DEPTHWISE ? 1 : _ifmDepth); ifmBlockZ += _ifmBlockDepth )
            {
                int clippedIfmBlockDepth;
                if ( IS_DEPTHWISE )
                {
                    clippedIfmBlockDepth = _ifmUBlockDepth;
                }
                else
                {
                    clippedIfmBlockDepth = IS_PARTKERNEL ? std::min(_ifmBlockDepth, _ifmDepth - ifmBlockZ) : _ifmBlockDepth;
                }

                // Weight decomposition
                // Subkernel Splitting (H)
                for ( subKernelY = _subKernelY; subKernelY < _kernelH; subKernelY += _decompY )
                {
                    int subHeight = std::min<int>(_kernelH - subKernelY, _decompY);
                    // Subkernel splitting (W)
                    for ( subKernelX = _subKernelX; subKernelX < _kernelW; subKernelX += _decompX )
                    {
                        int subWidth = std::min<int>(_kernelW - subKernelX, _decompX);
                        int subKernelElements = subWidth * subHeight;

                        // Part-kernel first works across the kernel H/W and needs padding
                        subKernelElements = RoundAway<int>(subKernelElements, _subKernelRound);

                        int ifmBlockDepthOuter = IS_PARTKERNEL ? clippedIfmBlockDepth : 1;
                        int ifmBlockDepthInner = IS_PARTKERNEL ? 1 : clippedIfmBlockDepth;

                        for ( ifmUBlockOuter = _ifmUBlockOuter; ifmUBlockOuter < ifmBlockDepthOuter; ifmUBlockOuter += _ifmUBlockDepth )
                        {
                            // OFM uBlocks in OFM-block over depth
                            for ( ofmUBlock = _ofmUBlock; ofmUBlock < clippedOfmBlockDepth; ofmUBlock += _ofmUBlockDepth )
                            {
                                // HW Kernel element traversal - cannot be a H/W loop due to element
                                // padding requirement on depthwise/part-kernel configurations
                                for ( kernelElement = _kernelElement; kernelElement < subKernelElements; kernelElement++ )
                                {
                                    int kx = kernelElement % subWidth;
                                    int ky = kernelElement / subWidth;
                                    // IFM uBlocks in IFM-block over depth (only 1 uBlock if depthwise)
                                    // In case of part-kernel-first IFM uBlock traversal have already been handled
                                    // and this loop is ignored.
                                    for ( ifmUBlockInner = _ifmUBlockInner; ifmUBlockInner < ifmBlockDepthInner; ifmUBlockInner += _ifmUBlockDepth )
                                    {
                                        int ifmUBlock = ifmUBlockInner + ifmUBlockOuter;
                                        // Feed OFM uBlock elements
                                        for ( ofmUBlockZ = _ofmUBlockZ; ofmUBlockZ < _ofmUBlockDepth; ofmUBlockZ++ )
                                        {
                                            // Source IFM uBlock elements (only 1 element deep if depthwise)
                                            for ( ifmUBlockZ = _ifmUBlockZ; ifmUBlockZ < (IS_DEPTHWISE ? 1 : _ifmUBlockDepth); ifmUBlockZ++ )
                                            {
                                                // Source position within the current subkernel
                                                int wx = subKernelX + kx;
                                                int wy = subKernelY + ky;
                                                // Source IFM/OFM slices
                                                int ifm_z = ifmBlockZ + ifmUBlock + ifmUBlockZ;
                                                int ofm_z = ofmBlockZ + ofmUBlock + ofmUBlockZ;
                                                if ( (ifm_z < _ifmDepth) && (ofm_z < _ofmDepth) && (ky < subHeight) )
                                                {
                                                    _param->o = ofm_z;
                                                    _param->h = wy;
                                                    _param->w = wx;
                                                    _param->i = ifm_z;
                                                    int weight = int(buffer[WeightIndex(ofm_z, wy, wx, ifm_z)]);
                                                    *write = int16_t(_transform(_param, weight));
                                                }
                                                else
                                                {
                                                    *write = 0;
                                                }
                                                write++;
                                                if ( --count == 0 )
                                                {
                                                    // Save state
                                                    _ifmUBlockZ = ifmUBlockZ + 1;
                                                    _ofmUBlockZ = ofmUBlockZ;
                                                    _ifmUBlockInner = ifmUBlockInner;
                                                    _kernelElement = kernelElement;
                                                    _ofmUBlock = ofmUBlock;
                                                    _ifmUBlockOuter = ifmUBlockOuter;
                                                    _subKernelX = subKernelX;
                                                    _subKernelY = subKernelY;
                                                    _ifmBlockZ = ifmBlockZ;
                                                    _ofmBlockZ = ofmBlockZ;
                                                    // Return weights generated (less than requested count == EOS)
                                                    return int(intptr_t(write - output));
                                                }
                                            }
                                            _ifmUBlockZ = 0;
                                        }
                                        _ofmUBlockZ = 0;
                                    }
                                    _ifmUBlockInner = 0;
                                }
                                _kernelElement = 0;
                            }
                            _ofmUBlock = 0;
                        }
                        _ifmUBlockOuter = 0;
                    }
                    _subKernelX = 0;
                }
                _subKernelY = 0;
            }
            _ifmBlockZ = 0;
        }
        _ofmBlockZ = 0;
        return int(intptr_t(write - output));
    }
};


std::unique_ptr<IVolumeWeightSource> EthosU55WeightEncoder::GetWeightSource(
    IWeightEncodingConfig *config, DataType weightType, WeightTransformFunc func, WeightTransformParam *param)
{
    int ofmUBlockDepth = _arch->_ofmUBlock.Depth();
    int ifmUBlockDepth = _arch->_ifmUBlock.Depth();

    EthosUEncodingConfig *cfg = static_cast<EthosUEncodingConfig *>(config);
    int ifmBitDepth = DataTypeSizeBits(cfg->ifmType);

    if ( weightType == DataType::UInt8 )
    {
        return std::make_unique<EthosUWeightOrdering<uint8_t>>(_arch->_cores, cfg->dilation, cfg->ofmBlockDepth,
            ifmBitDepth, ofmUBlockDepth, ifmUBlockDepth, func, param, cfg->traversal);
    }
    else if ( weightType == DataType::Int8 )
    {
        return std::make_unique<EthosUWeightOrdering<int8_t>>(_arch->_cores, cfg->dilation, cfg->ofmBlockDepth,
            ifmBitDepth, ofmUBlockDepth, ifmUBlockDepth, func, param, cfg->traversal);
    }

    assert(false && "No weight source for this datatype");
    return nullptr;
}


template<typename TYPE>
class EthosUScaleSource : public IVolumeScaleSource
{
private:
    const TYPE *_buffer = nullptr;
    const QuantizedScale *_scales = nullptr;
    int _biasIndex = 0;
    int _biasCount = 0;
    int _bufferSize = 0;
    int _streamIndex = 0;
    int _streams = 0;
    Quantization _quantization;

public:
    EthosUScaleSource(int cores, Quantization quantization) : _streams(cores), _quantization(std::move(quantization))
    {
        assert(!_quantization.scales.empty());
        // assert that no scale is out of range
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
            int index = _biasIndex + (i * _streams);
            *biasBuffer++ = _buffer ? static_cast<int64_t>(_buffer[index % _bufferSize]) : 0;
            *quantBuffer++ = _quantization.scales[index % scaleSize];
            _biasCount--;
        }

        _biasIndex += (count * _streams);
        return count;
    }

    void SetSource(const void *buffer, int biasCount, int depthOffset, int depthLength, int streamIndex)
    {
        assert(streamIndex >= 0 && streamIndex < _streams);
        bool isBroadcast = biasCount == 1;
        assert(depthOffset + depthLength <= biasCount || isBroadcast);
        assert(uintptr_t(buffer) % alignof(TYPE) == 0);
        _buffer = reinterpret_cast<const TYPE *>(buffer);
        _biasIndex = depthOffset + streamIndex;                              // Where to start in the buffer
        _biasCount = (depthLength + _streams - 1 - streamIndex) / _streams;  // How many biases to generate
        _bufferSize = biasCount;
    }
};


std::unique_ptr<IVolumeScaleSource> EthosU55WeightEncoder::GetScaleSource(
    IWeightEncodingConfig *config, DataType scaleType, const Quantization &explicitQuant)
{
    EthosUEncodingConfig *cfg = static_cast<EthosUEncodingConfig *>(config);
    assert(explicitQuant.type == QuantizationType::EXPLICIT);

    if ( scaleType == DataType::Int32 )
    {
        return std::make_unique<EthosUScaleSource<int32_t>>(_arch->_cores, explicitQuant);
    }
    else if ( scaleType == DataType::Int64 )
    {
        return std::make_unique<EthosUScaleSource<int64_t>>(_arch->_cores, explicitQuant);
    }

    return nullptr;
}

Quantization EthosU55WeightEncoder::MakeExplicit(const Quantization &ifmQ, const Quantization &weightQ,
    const Quantization &ofmQ, DataType scaleType, DataType ifmType, OpType opType)
{
    return RescalePerChannel(ifmQ, weightQ, ofmQ, scaleType, ifmType, opType);
}


WeightsInfo EthosU55WeightEncoder::EncodeWeights(IWeightEncodingConfig *config, IWeightSource *source, std::vector<uint8_t> &result)
{
    [[maybe_unused]] EthosUEncodingConfig *cfg = static_cast<EthosUEncodingConfig *>(config);
    assert(cfg->Format() == WeightFormat::Default);
    unsigned flags = MLW_ENCODE_FLAG_NONE;
    auto res = mle_encode_proxy(source, 128 * 1024, result, flags);
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


int EthosU55WeightEncoder::EncodeScales(IWeightEncodingConfig *config, IScaleSource *source, std::vector<uint8_t> &result, bool measureOnly)
{
    UNUSED(config);
    constexpr int BUFFER_SIZE = 8;
    constexpr int SCALE_ELEMENT_SIZE = 10;

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
