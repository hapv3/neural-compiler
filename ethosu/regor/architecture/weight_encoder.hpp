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

#pragma once

#include "common/scaling.hpp"
#include "compiler/scheduler_operation.hpp"
#include "mlw_encode.hpp"

#include <bitset>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace regor
{

/// <summary>
/// Contains info about encoded weights/scales for one core/depth offset combination
/// </summary>
struct WeightRange
{
    int offset = 0;
    int scaleBytes = 0;
    int weightOffset = 0;
    int weightBytes = 0;
    int index = 0;

    int TotalBytes() const { return scaleBytes + weightBytes; }
};

/// Produces key for indexing WeightTensor::encodedRanges
constexpr inline int WeightKey(int stream, int depth)
{
    return (depth << 8) + stream;
}

struct WeightTransformParam
{
    int o, h, w, i;
};

typedef int (*WeightTransformFunc)(const WeightTransformParam *param, int weight);

struct IWeightEncodingConfig
{
    virtual ~IWeightEncodingConfig() = default;
    virtual uint32_t Hash() = 0;
    virtual bool Equals(IWeightEncodingConfig *other) = 0;
    virtual const std::vector<int> &DepthOffsets() = 0;
    virtual Flags<WeightFormat> Format() = 0;
};

struct IVolumeWeightSource : public IWeightSource
{
    virtual ~IVolumeWeightSource() = default;
    virtual void SetSource(const void *buffer, int depthOffset, const Shape &ohwiShape, const Shape &ohwiStrides, int streamIndex) = 0;
};

struct IScaleSource
{
    virtual ~IScaleSource() = default;
    virtual int Elements() = 0;
    virtual int Get(int64_t *biasBuffer, QuantizedScale *quantBuffer, int count) = 0;
};

struct IVolumeScaleSource : public IScaleSource
{
    virtual ~IVolumeScaleSource() = default;
    virtual void SetSource(const void *buffer, int biasCount, int depthOffset, int depthLength, int streamIndex) = 0;
};


/// <summary>
/// Contains encoded weights and biases.
/// </summary>
class NpuWeightTensor : public SchedulerTensor
{
public:
    virtual ~NpuWeightTensor() = default;

    /** Required size in bytes if double buffering is applied */
    int maxRangeBytes = 0;
    int doubleBufferSize = 0;
    int doubleBufferOffset = 0;
    int totalSourceBytes = 0;
    int totalWeightBytes = 0;
    int subStreams = 0;
    int distinctWeights = 0;
    int zeroCount;
    std::unordered_map<int, WeightRange> encodedRanges;
    std::unique_ptr<IWeightEncodingConfig> config;
};

struct WeightScaleTensors
{
    /** Encoded weights, may be null */
    std::shared_ptr<NpuWeightTensor> npuWeightsTensor;
    /** Combined scaling parameters in the weights tensor **/
    uint32_t scaleHash;
    /** Encoded scales, may be null */
    std::shared_ptr<NpuWeightTensor> npuScalesTensor;
};


struct WeightsRef
{
    BufferView *view = nullptr;
    AxisOrder axisOrder = AxisOrder::Unknown;
    DataType type = DataType::None;
    bool isScales = false;
};

struct WeightsInfo
{
    int sourceSize = 0;
    int encodedSize = 0;
    int zeroCount = 0;
    int distinctValues = 0;
    int streams = 0;
    std::bitset<64> weightsUsed[8];
};

/// <summary>
/// Encodes weights and biases.
/// </summary>
class WeightEncoder
{
public:
    virtual ~WeightEncoder() = default;

    virtual std::unique_ptr<IWeightEncodingConfig> GetEncodingConfig(ArchitectureOpConfig *opCfg, const WeightsRef &weights,
        const Kernel *kernel, DataType ifmType, const std::vector<int> &depthOffsets, Flags<WeightFormat> format) = 0;

    virtual int StreamsRequired(IWeightEncodingConfig *config, const Shape &ofmShape, int &scaleStreamsRequired) = 0;

    virtual std::unique_ptr<IVolumeWeightSource> GetWeightSource(
        IWeightEncodingConfig *config, DataType weightType, WeightTransformFunc func, WeightTransformParam *param) = 0;

    virtual std::unique_ptr<IVolumeScaleSource>
    GetScaleSource(IWeightEncodingConfig *config, DataType scaleType, const Quantization &explicitQuant) = 0;

    virtual Quantization MakeExplicit(const Quantization &ifmQ, const Quantization &weightQ, const Quantization &ofmQ,
        DataType scaleType, DataType ifmType, OpType opType) = 0;

    virtual WeightsInfo EncodeWeights(IWeightEncodingConfig *config, IWeightSource *source, std::vector<uint8_t> &result) = 0;
    virtual int EncodeScales(IWeightEncodingConfig *config, IScaleSource *source, std::vector<uint8_t> &result, bool measureOnly) = 0;
};

// IVolumeWeightSource common implementation
class WeightSourceCommon : public IVolumeWeightSource
{

protected:
    const void *_source;
    int16_t _streams = 1;
    int16_t _streamIndex = 0;
    int _ofmDepth = 0;
    int _ifmDepth = 0;
    int _kernelH = 0;
    int _kernelW = 0;
    int _ohwiStrides[4];

protected:
    void SetSourceCommon(const void *buffer, int depthOffset, const Shape &ohwiShape, const Shape &ohwiStrides, int streamIndex, bool separated)
    {
        assert(streamIndex < _streams);
        _streamIndex = streamIndex;

        int streamOffset = Shape(depthOffset, 0, 0, 0).Dot(ohwiStrides);
        _source = reinterpret_cast<const uint8_t *>(buffer) + streamOffset;
        _ifmDepth = ohwiShape[-1];
        _ofmDepth = separated ? (ohwiShape[0] + _streams - 1 - streamIndex) / _streams : ohwiShape[0];
        _kernelH = ohwiShape.Height();
        _kernelW = ohwiShape.Width();

        // Bring in values for better cache locality
        _ohwiStrides[0] = ohwiStrides[0] * (separated ? _streams : 1);
        _ohwiStrides[1] = ohwiStrides[1];
        _ohwiStrides[2] = ohwiStrides[2];
        _ohwiStrides[3] = ohwiStrides[3];
    }

    int Elements() override { return _ofmDepth * _ifmDepth * _kernelH * _kernelW; }

    inline int WeightIndex(int ofm_z, int wy, int wx, int ifm_z) const
    {
        return ofm_z * _ohwiStrides[0] + wy * _ohwiStrides[1] + wx * _ohwiStrides[2] + ifm_z * _ohwiStrides[3];
    }
};

struct WeightEncodeException : public std::runtime_error
{
    WeightEncodeException() : std::runtime_error("weight encode") {}
};

struct WeightsNotSparse : public WeightEncodeException
{
    WeightsNotSparse() {}
};

}  // namespace regor
