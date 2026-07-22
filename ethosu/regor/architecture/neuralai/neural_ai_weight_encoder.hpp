//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "architecture/weight_encoder.hpp"

#include <cstdint>
#include <vector>

namespace regor::neuralai
{

std::vector<uint8_t> PackGEMM32Weights(const int8_t *weightsKN, int depthK, int depthN);

}  // namespace regor::neuralai

namespace regor
{

class NeuralAIWeightEncoder final : public WeightEncoder
{
public:
    class EncodingConfig final : public IWeightEncodingConfig
    {
    private:
        DataType _ifmType;
        Flags<WeightFormat> _format;

    public:
        EncodingConfig(DataType ifmType, Flags<WeightFormat> format) : _ifmType(ifmType), _format(format) {}
        uint32_t Hash() override;
        bool Equals(IWeightEncodingConfig *other) override;
        Flags<WeightFormat> Format() override { return _format; }
    };

    std::unique_ptr<IWeightEncodingConfig> GetEncodingConfig(
        ArchitectureOpConfig *opCfg, const Kernel *kernel, DataType ifmType, Flags<WeightFormat> format) override;
    int StreamsRequired(IWeightEncodingConfig *config, const Shape &weightShape, int &scaleStreamsRequired) override;
    std::unique_ptr<IVolumeWeightSource> GetWeightSource(
        IWeightEncodingConfig *config, DataType weightType, WeightTransformFunc func,
        WeightTransformParam *param) override;
    std::unique_ptr<IVolumeScaleSource> GetScaleSource(
        IWeightEncodingConfig *config, DataType scaleType, const Quantization &explicitQuant) override;
    WeightsInfo EncodeWeights(
        IWeightEncodingConfig *config, IWeightSource *source, std::vector<uint8_t> &result) override;
    int EncodeScales(
        IWeightEncodingConfig *config, IScaleSource *source, std::vector<uint8_t> &result, bool measureOnly) override;
};

}  // namespace regor
