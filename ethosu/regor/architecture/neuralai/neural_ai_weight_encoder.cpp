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
    bool _linebufferK3;
    std::vector<int16_t> _packed;
    int _sourceElements = 0;
    int _position = 0;

public:
    MatrixWeightSource(WeightTransformFunc transform, WeightTransformParam *param, bool linebufferK3) :
            _transform(transform), _param(param), _linebufferK3(linebufferK3)
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
        const int kernelHeight = ohwiShape.Height();
        const int kernelWidth = ohwiShape.Width();
        const int inputChannels = ohwiShape.Depth();
        const int kernelElements = kernelHeight * kernelWidth * inputChannels;
        const int kGroups = RoundAway(kernelElements, 32) / 32;
        const int nGroups = RoundAway(outputDepth, 32) / 32;
        const bool groupedK3 = _linebufferK3 && inputChannels > 32;
        const int groupedK = groupedK3 ?
            kernelHeight * kernelWidth * RoundAway(inputChannels, 32) : kGroups * 32;
        const int encodedKGroups = groupedK3 ? groupedK / 32 : kGroups;
        _sourceElements = outputDepth * kernelElements;
        _position = 0;
        _packed.assign(size_t(encodedKGroups) * nGroups * 32 * 32, 0);

        size_t output = 0;
        for ( int nGroup = 0; nGroup < nGroups; ++nGroup )
        {
            if ( groupedK3 )
            {
                const int inputGroups = RoundAway(inputChannels, 32) / 32;
                for ( int inputGroup = 0; inputGroup < inputGroups; ++inputGroup )
                {
                    for ( int h = 0; h < kernelHeight; ++h )
                    {
                        for ( int w = 0; w < kernelWidth; ++w )
                        {
                            for ( int iLane = 0; iLane < 32; ++iLane )
                            {
                                const int i = inputGroup * 32 + iLane;
                                for ( int nLane = 0; nLane < 32; ++nLane )
                                {
                                    const int n = nGroup * 32 + nLane;
                                    if ( i < inputChannels && n < outputDepth )
                                    {
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
            }
            else
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
    }
};

class DepthwiseWeightSource final : public IVolumeWeightSource
{
private:
    WeightTransformFunc _transform;
    WeightTransformParam *_param;
    std::vector<int16_t> _packed;
    int _sourceElements = 0;
    int _position = 0;

public:
    DepthwiseWeightSource(WeightTransformFunc transform, WeightTransformParam *param) :
            _transform(transform), _param(param)
    {
    }

    int Elements() override { return int(_packed.size()); }
    int SourceElements() const { return _sourceElements; }

    int Get(int16_t *buffer, int count) override
    {
        count = std::min(count, int(_packed.size()) - _position);
        std::copy_n(_packed.data() + _position, count, buffer);
        _position += count;
        return count;
    }

    void SetSource(const void *buffer, int depthOffset, const Shape &ohwiShape,
        const Shape &ohwiStrides, int streamIndex) override
    {
        if ( streamIndex != 0 || !buffer || depthOffset < 0 || ohwiShape.Batch() <= 0 ||
             ohwiShape.Height() <= 0 || ohwiShape.Width() <= 0 || ohwiShape.Depth() <= 0 )
            throw WeightEncodeException("Invalid Neural-AI depthwise weight source");
        const int channels = std::max(ohwiShape.Batch(), ohwiShape.Depth());
        if ( ohwiShape.Batch() != 1 && ohwiShape.Depth() != 1 )
            throw WeightEncodeException("Depthwise weights must have one channel axis");
        const auto *source = static_cast<const int8_t *>(buffer);
        const int kh = ohwiShape.Height();
        const int kw = ohwiShape.Width();
        const int groups = RoundAway(channels, 32) / 32;
        _sourceElements = channels * kh * kw;
        _position = 0;
        _packed.assign(size_t(groups) * kh * kw * 32, 0);

        size_t output = 0;
        for ( int group = 0; group < groups; ++group )
        {
            for ( int h = 0; h < kh; ++h )
            {
                for ( int w = 0; w < kw; ++w )
                {
                    for ( int lane = 0; lane < 32; ++lane )
                    {
                        const int channel = group * 32 + lane;
                        if ( channel < channels )
                        {
                            const int globalChannel = depthOffset + channel;
                            const int o = ohwiShape.Batch() > 1 ? globalChannel : 0;
                            const int i = ohwiShape.Depth() > 1 ? globalChannel : 0;
                            int value = source[Shape(o, h, w, i).Dot(ohwiStrides)];
                            if ( _transform )
                            {
                                _param->o = globalChannel;
                                _param->h = h;
                                _param->w = w;
                                _param->i = globalChannel;
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
        // The Neural-AI requantizer exposes one clamp pair for each 32-lane
        // block.  Keep the compiler-side contract explicit: channel scales
        // may vary, but activation bounds must be uniform within a block so
        // an emitted package cannot be rejected later by the firmware decoder.
        for ( int blockStart = 0; blockStart < elements; blockStart += 32 )
        {
            const int firstChannel = _depthOffset + std::min(blockStart, _depthLength - 1);
            const int64_t firstRequestedMin = _quantization.quantMin.empty() ? -128 :
                _quantization.quantMin[firstChannel % _quantization.quantMin.size()];
            const int64_t firstRequestedMax = _quantization.quantMax.empty() ? 127 :
                _quantization.quantMax[firstChannel % _quantization.quantMax.size()];
            const int64_t firstClampMin = std::clamp<int64_t>(firstRequestedMin, -128, 127);
            const int64_t firstClampMax = std::clamp<int64_t>(firstRequestedMax, -128, 127);
            for ( int lane = blockStart; lane < std::min(blockStart + 32, elements); ++lane )
            {
                const int channel = _depthOffset + std::min(lane, _depthLength - 1);
                const int64_t requestedMin = _quantization.quantMin.empty() ? -128 :
                    _quantization.quantMin[channel % _quantization.quantMin.size()];
                const int64_t requestedMax = _quantization.quantMax.empty() ? 127 :
                    _quantization.quantMax[channel % _quantization.quantMax.size()];
                const int64_t clampMin = std::clamp<int64_t>(requestedMin, -128, 127);
                const int64_t clampMax = std::clamp<int64_t>(requestedMax, -128, 127);
                if ( clampMin > clampMax )
                    throw WeightEncodeException("Neural-AI quantization clamp minimum exceeds maximum");
                if ( clampMin != firstClampMin || clampMax != firstClampMax )
                    throw WeightEncodeException("Neural-AI requires uniform activation clamps per C32 qparam block");
            }
        }
        for ( int index = 0; index < elements; ++index )
        {
            const bool padding = index >= _depthLength;
            const int channel = _depthOffset + std::min(index, _depthLength - 1);
            const QuantizedScale scale = padding ? QuantizedScale(0, 0) :
                                                   _quantization.scales[channel % _quantization.scales.size()];
            const int64_t bias = padding ? 0 : Bias(channel);
            const int zeroPoint = padding || _quantization.zeroPoints.empty() ? 0 :
                                      int(_quantization.zeroPoints[channel % _quantization.zeroPoints.size()]);
            const int clampChannel = _depthLength == 0 ? 0 :
                _depthOffset + std::min(index, _depthLength - 1);
            const int64_t requestedClampMin = _quantization.quantMin.empty() ? -128 :
                _quantization.quantMin[clampChannel % _quantization.quantMin.size()];
            const int64_t requestedClampMax = _quantization.quantMax.empty() ? 127 :
                _quantization.quantMax[clampChannel % _quantization.quantMax.size()];
            if ( requestedClampMin > requestedClampMax )
                throw WeightEncodeException("Neural-AI quantization clamp minimum exceeds maximum");
            const int64_t clampMin = std::clamp<int64_t>(requestedClampMin, -128, 127);
            const int64_t clampMax = std::clamp<int64_t>(requestedClampMax, -128, 127);
            if ( bias < std::numeric_limits<int32_t>::min() || bias > std::numeric_limits<int32_t>::max() ||
                 scale.shift < 0 || scale.shift > 31 || clampMin < std::numeric_limits<int32_t>::min() ||
                 clampMin > std::numeric_limits<int32_t>::max() || clampMax < std::numeric_limits<int32_t>::min() ||
                 clampMax > std::numeric_limits<int32_t>::max() || clampMin > clampMax )
            {
                throw WeightEncodeException("Neural-AI quantization parameter is out of range");
            }
            Append32(result, uint32_t(int32_t(bias)));
            Append32(result, uint32_t(scale.scale));
            Append32(result, uint32_t(scale.shift));
            Append32(result, uint32_t(zeroPoint));
            Append32(result, uint32_t(int32_t(clampMin)));
            Append32(result, uint32_t(int32_t(clampMax)));
            Append32(result, 0);
            Append32(result, 0);
        }
        return elements * int(sizeof(neuralai::QParamV1));
    }
};

}  // namespace

uint32_t NeuralAIWeightEncoder::EncodingConfig::Hash()
{
    return SimpleHash32(_ifmType, _format, _mode);
}

bool NeuralAIWeightEncoder::EncodingConfig::Equals(IWeightEncodingConfig *other)
{
    auto *config = static_cast<EncodingConfig *>(other);
    return _ifmType == config->_ifmType && _format == config->_format && _mode == config->_mode;
}

std::unique_ptr<IWeightEncodingConfig> NeuralAIWeightEncoder::GetEncodingConfig(
    ArchitectureOpConfig *opCfg, const Kernel *, DataType ifmType, Flags<WeightFormat> format)
{
    if ( !opCfg || ifmType != DataType::Int8 || format != WeightFormat::Default )
    {
        throw WeightEncodeException("Unsupported Neural-AI weight encoding configuration");
    }
    const auto *neuralConfig = static_cast<const NeuralAIOpConfig *>(opCfg);
    return std::make_unique<EncodingConfig>(ifmType, format,
        neuralConfig ? neuralConfig->Mode() : NeuralAIOpMode::Unsupported);
}

int NeuralAIWeightEncoder::StreamsRequired(IWeightEncodingConfig *, const Shape &, int &scaleStreamsRequired)
{
    scaleStreamsRequired = 1;
    return 1;
}

std::unique_ptr<IVolumeWeightSource> NeuralAIWeightEncoder::GetWeightSource(
    IWeightEncodingConfig *config, DataType weightType, WeightTransformFunc func, WeightTransformParam *param)
{
    if ( weightType != DataType::Int8 ) throw WeightEncodeException("Neural-AI weights must be INT8");
    const auto *encoding = static_cast<const EncodingConfig *>(config);
    if ( encoding->Mode() == NeuralAIOpMode::DepthwiseC32S1Requant ||
         encoding->Mode() == NeuralAIOpMode::DepthwiseC32S2Requant )
        return std::make_unique<DepthwiseWeightSource>(func, param);
    const NeuralAIOpMode mode = encoding->Mode();
    const bool linebufferK3 = mode == NeuralAIOpMode::Conv2DRgbLinebufRequant ||
        mode == NeuralAIOpMode::Conv2DLinebufC32S1Requant ||
        mode == NeuralAIOpMode::Conv2DLinebufC32S2Requant ||
        mode == NeuralAIOpMode::Conv2DLinebufC32TailRequant;
    return std::make_unique<MatrixWeightSource>(func, param, linebufferK3);
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
    IWeightEncodingConfig *config, IWeightSource *source, std::vector<uint8_t> &result)
{
    const auto *encoding = static_cast<const EncodingConfig *>(config);
    const int sourceElements = encoding->Mode() == NeuralAIOpMode::DepthwiseC32S1Requant ||
            encoding->Mode() == NeuralAIOpMode::DepthwiseC32S2Requant ?
        static_cast<DepthwiseWeightSource *>(source)->SourceElements() :
        static_cast<MatrixWeightSource *>(source)->SourceElements();
    WeightsInfo info;
    info.sourceSize = sourceElements;
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
