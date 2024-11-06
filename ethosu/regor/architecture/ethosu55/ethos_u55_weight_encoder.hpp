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

#include "architecture/architecture.hpp"
#include "architecture/ethos_u_scaling.hpp"
#include "architecture/mlw_encode.hpp"
#include "architecture/weight_encoder.hpp"
#include "common/shape.hpp"
#include "ethos_u55.hpp"

namespace regor
{

/// <summary>
/// Encodes weights and biases. Common implementation for Ethos U55 and Ethos U65.
/// </summary>
class EthosU55WeightEncoder : public WeightEncoder
{
private:
    struct EthosUEncodingConfig : IWeightEncodingConfig
    {
    private:
        uint32_t _hash = 0;
        uint32_t _depthOffsetHash = 0;
        int _cores = 0;

    public:
        DataType ifmType = DataType::None;
        int ofmBlockDepth = 0;
        EthosUTraversal traversal = EthosUTraversal::DepthFirst;
        std::vector<int> depthOffsets;
        Point2i dilation;
        Shape ohwiStrides;

    public:
        EthosUEncodingConfig(int cores);
        void Rehash();
        uint32_t Hash() override;
        bool Equals(IWeightEncodingConfig *other) override;
        const std::vector<int> &DepthOffsets() override;
        Flags<WeightFormat> Format() override;
    };

public:
    EthosU55WeightEncoder(ArchEthosU55 *arch) : _arch(arch) {}

public:
    std::unique_ptr<IWeightEncodingConfig> GetEncodingConfig(ArchitectureOpConfig *opCfg, const WeightsRef &weights,
        const Kernel *kernel, DataType ifmType, const std::vector<int> &depthOffsets, Flags<WeightFormat> format);

    int StreamsRequired(IWeightEncodingConfig *config, const Shape &weightShape, int &scaleStreamsRequired);

    std::unique_ptr<IVolumeWeightSource> GetWeightSource(
        IWeightEncodingConfig *config, DataType weightType, WeightTransformFunc func, WeightTransformParam *param);

    std::unique_ptr<IVolumeScaleSource> GetScaleSource(IWeightEncodingConfig *config, DataType scaleType, const Quantization &explicitQuant);

    Quantization MakeExplicit(const Quantization &ifmQ, const Quantization &weightQ, const Quantization &ofmQ,
        DataType scaleType, DataType ifmType, OpType opType);

    WeightsInfo EncodeWeights(IWeightEncodingConfig *config, IWeightSource *source, std::vector<uint8_t> &result);

    int EncodeScales(IWeightEncodingConfig *config, IScaleSource *source, std::vector<uint8_t> &result, bool measureOnly);

private:
    ArchEthosU55 *_arch;
};

}  // namespace regor
