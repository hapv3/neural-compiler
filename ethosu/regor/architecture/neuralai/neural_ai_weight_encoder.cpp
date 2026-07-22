//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "neural_ai_weight_encoder.hpp"

#include "common/common.hpp"
#include "common/numeric_util.hpp"

#include "architecture/neuralai/neural_ai_abi.hpp"
#include "compiler/quantization.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace regor::neuralai
{

std::vector<uint8_t> PackGEMM32Weights(const int8_t *weightsKN, int depthK, int depthN)
{
    if ( !weightsKN || depthK <= 0 || depthN <= 0 )
    {
        throw std::invalid_argument("Neural-AI GEMM weights require positive K and N dimensions");
    }

    constexpr int tile = 32;
    const int kGroups = RoundAway(depthK, tile) / tile;
    const int nGroups = RoundAway(depthN, tile) / tile;
    std::vector<uint8_t> packed(size_t(kGroups) * nGroups * tile * tile, 0);
    size_t output = 0;
    for ( int nGroup = 0; nGroup < nGroups; ++nGroup )
    {
        for ( int kGroup = 0; kGroup < kGroups; ++kGroup )
        {
            for ( int kLane = 0; kLane < tile; ++kLane )
            {
                const int k = kGroup * tile + kLane;
                for ( int nLane = 0; nLane < tile; ++nLane )
                {
                    const int n = nGroup * tile + nLane;
                    if ( k < depthK && n < depthN )
                    {
                        packed[output] = uint8_t(weightsKN[k * depthN + n]);
                    }
                    ++output;
                }
            }
        }
    }
    return packed;
}

}  // namespace regor::neuralai

namespace regor
{
namespace
{

void Append32(std::vector<uint8_t> &output, uint32_t value)
{
    output.push_back(uint8_t(value));
    output.push_back(uint8_t(value >> 8));
    output.push_back(uint8_t(value >> 16));
    output.push_back(uint8_t(value >> 24));
}

class MatrixWeightSource final : public IVolumeWeightSource
{
private:
    WeightTransformFunc _transform;
    WeightTransformParam *_param;
    std::vector<int16_t> _packed;
    int _sourceElements = 0;
    int _position = 0;

public:
    MatrixWeightSource(WeightTransformFunc transform, WeightTransformParam *param) :
            _transform(transform), _param(param)
    {
    }

    int Elements() override { return int(_packed.size()); }
    int SourceElements() const { return _sourceElements; }

    int Get(int16_t *buffer, int count) override
    {
        const int available = int(_packed.size()) - _position;
        count = std::min(count, available);
        std::copy_n(_packed.data() + _position, count, buffer);
        _position += count;
        return count;
    }

    void SetSource(const void *buffer, int depthOffset, const Shape &ohwiShape, const Shape &ohwiStrides,
        int streamIndex) override
    {
        if ( streamIndex != 0 || !buffer || ohwiShape.Batch() <= 0 || ohwiShape.Depth() <= 0 )
        {
            throw WeightEncodeException("Invalid Neural-AI matrix weight source");
        }
        const auto *source = static_cast<const int8_t *>(buffer);
        const int outputDepth = ohwiShape.Batch();
        const int kernelElements = ohwiShape.Height() * ohwiShape.Width() * ohwiShape.Depth();
        const int kGroups = RoundAway(kernelElements, 32) / 32;
        const int nGroups = RoundAway(outputDepth, 32) / 32;
        _sourceElements = outputDepth * kernelElements;
        _position = 0;
        _packed.assign(size_t(kGroups) * nGroups * 32 * 32, 0);

        size_t output = 0;
        for ( int nGroup = 0; nGroup < nGroups; ++nGroup )
        {
            for ( int kGroup = 0; kGroup < kGroups; ++kGroup )
            {
                for ( int kLane = 0; kLane < 32; ++kLane )
                {
                    const int k = kGroup * 32 + kLane;
                    for ( int nLane = 0; nLane < 32; ++nLane )
                    {
                        const int n = nGroup * 32 + nLane;
                        if ( k < kernelElements && n < outputDepth )
                        {
                            const int h = k / (ohwiShape.Width() * ohwiShape.Depth());
                            const int wi = k % (ohwiShape.Width() * ohwiShape.Depth());
                            const int w = wi / ohwiShape.Depth();
                            const int i = wi % ohwiShape.Depth();
                            const int o = depthOffset + n;
                            int value = source[Shape(o, h, w, i).Dot(ohwiStrides)];
                            if ( _transform )
                            {
                                _param->o = o;
                                _param->h = h;
                                _param->w = w;
                                _param->i = i;
                                value = _transform(_param, value);
                            }
                            _packed[output] = int16_t(value);
                        }
                        ++output;
                    }
                }
            }
        }
    }
};

class QParamSource final : public IVolumeScaleSource
{
private:
    DataType _biasType;
    Quantization _quantization;
    const void *_biases = nullptr;
    int _biasCount = 0;
    int _depthOffset = 0;
    int _depthLength = 0;

    int64_t Bias(int index) const
    {
        if ( !_biases ) return 0;
        index %= _biasCount;
        return _biasType == DataType::Int64 ? static_cast<const int64_t *>(_biases)[index] :
                                              static_cast<const int32_t *>(_biases)[index];
    }

public:
    QParamSource(DataType biasType, const Quantization &quantization) :
            _biasType(biasType), _quantization(quantization)
    {
    }

    int Elements() override { return RoundAway(_depthLength, 32); }

    int Get(int64_t *biasBuffer, QuantizedScale *quantBuffer, int count) override
    {
        count = std::min(count, _depthLength);
        for ( int index = 0; index < count; ++index )
        {
            const int channel = _depthOffset + index;
            biasBuffer[index] = Bias(channel);
            quantBuffer[index] = _quantization.scales[channel % _quantization.scales.size()];
        }
        return count;
    }

    void SetSource(const void *buffer, int biasCount, int depthOffset, int depthLength, int streamIndex) override
    {
        if ( streamIndex != 0 || biasCount <= 0 || depthOffset < 0 || depthLength <= 0 )
        {
            throw WeightEncodeException("Invalid Neural-AI quantization source");
        }
        _biases = buffer;
        _biasCount = biasCount;
        _depthOffset = depthOffset;
        _depthLength = depthLength;
    }

    int Encode(std::vector<uint8_t> &result, bool measureOnly)
    {
        const int elements = Elements();
        if ( measureOnly ) return elements * int(sizeof(neuralai::QParamV1));
        const int clampMin = _quantization.quantMin.empty() ? -128 : int(_quantization.quantMin[0]);
        const int clampMax = _quantization.quantMax.empty() ? 127 : int(_quantization.quantMax[0]);
        for ( int index = 0; index < elements; ++index )
        {
            const bool padding = index >= _depthLength;
            const int channel = _depthOffset + std::min(index, _depthLength - 1);
            const QuantizedScale scale = padding ? QuantizedScale(0, 0) :
                                                   _quantization.scales[channel % _quantization.scales.size()];
            const int64_t bias = padding ? 0 : Bias(channel);
            const int zeroPoint = padding || _quantization.zeroPoints.empty() ? 0 :
                                      int(_quantization.zeroPoints[channel % _quantization.zeroPoints.size()]);
            if ( bias < std::numeric_limits<int32_t>::min() || bias > std::numeric_limits<int32_t>::max() ||
                 scale.shift < 0 || scale.shift > 31 )
            {
                throw WeightEncodeException("Neural-AI quantization parameter is out of range");
            }
            Append32(result, uint32_t(int32_t(bias)));
            Append32(result, uint32_t(scale.scale));
            Append32(result, uint32_t(scale.shift));
            Append32(result, uint32_t(zeroPoint));
            Append32(result, uint32_t(clampMin));
            Append32(result, uint32_t(clampMax));
            Append32(result, 0);
            Append32(result, 0);
        }
        return elements * int(sizeof(neuralai::QParamV1));
    }
};

}  // namespace

uint32_t NeuralAIWeightEncoder::EncodingConfig::Hash()
{
    return SimpleHash32(_ifmType, _format);
}

bool NeuralAIWeightEncoder::EncodingConfig::Equals(IWeightEncodingConfig *other)
{
    auto *config = static_cast<EncodingConfig *>(other);
    return _ifmType == config->_ifmType && _format == config->_format;
}

std::unique_ptr<IWeightEncodingConfig> NeuralAIWeightEncoder::GetEncodingConfig(
    ArchitectureOpConfig *opCfg, const Kernel *, DataType ifmType, Flags<WeightFormat> format)
{
    if ( !opCfg || ifmType != DataType::Int8 || format != WeightFormat::Default )
    {
        throw WeightEncodeException("Unsupported Neural-AI weight encoding configuration");
    }
    return std::make_unique<EncodingConfig>(ifmType, format);
}

int NeuralAIWeightEncoder::StreamsRequired(IWeightEncodingConfig *, const Shape &, int &scaleStreamsRequired)
{
    scaleStreamsRequired = 1;
    return 1;
}

std::unique_ptr<IVolumeWeightSource> NeuralAIWeightEncoder::GetWeightSource(
    IWeightEncodingConfig *, DataType weightType, WeightTransformFunc func, WeightTransformParam *param)
{
    if ( weightType != DataType::Int8 ) throw WeightEncodeException("Neural-AI weights must be INT8");
    return std::make_unique<MatrixWeightSource>(func, param);
}

std::unique_ptr<IVolumeScaleSource> NeuralAIWeightEncoder::GetScaleSource(
    IWeightEncodingConfig *, DataType scaleType, const Quantization &explicitQuant)
{
    if ( (scaleType != DataType::Int32 && scaleType != DataType::Int64) || explicitQuant.scales.empty() )
    {
        throw WeightEncodeException("Unsupported Neural-AI scale source");
    }
    return std::make_unique<QParamSource>(scaleType, explicitQuant);
}

WeightsInfo NeuralAIWeightEncoder::EncodeWeights(
    IWeightEncodingConfig *, IWeightSource *source, std::vector<uint8_t> &result)
{
    auto *matrixSource = static_cast<MatrixWeightSource *>(source);
    WeightsInfo info;
    info.sourceSize = matrixSource->SourceElements();
    const size_t start = result.size();
    std::vector<int16_t> values(1024);
    int count = 0;
    while ( (count = source->Get(values.data(), int(values.size()))) != 0 )
    {
        for ( int index = 0; index < count; ++index )
        {
            const int value = values[index];
            if ( value < -128 || value > 127 ) throw WeightEncodeException("Neural-AI weight is out of INT8 range");
            result.push_back(uint8_t(int8_t(value)));
            if ( value == 0 ) ++info.zeroCount;
            const int used = value + 256;
            info.weightsUsed[used / 64].set(used % 64);
        }
    }
    info.encodedSize = int(result.size() - start);
    return info;
}

int NeuralAIWeightEncoder::EncodeScales(
    IWeightEncodingConfig *, IScaleSource *source, std::vector<uint8_t> &result, bool measureOnly)
{
    return static_cast<QParamSource *>(source)->Encode(result, measureOnly);
}

}  // namespace regor
