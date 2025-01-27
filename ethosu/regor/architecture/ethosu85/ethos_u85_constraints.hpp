//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "../architecture_constraints.hpp"
#include "ethos_u85.hpp"

namespace regor
{

class EthosU85Constraints : public IArchitectureConstraints
{
public:
    EthosU85Constraints(ArchEthosU85 *arch) : _arch(arch) {}

    bool SupportsLeakyRelu(bool quantized, DataType type) override;
    bool SupportsMatMul(OpType opType) override;
    TransposeSupport SupportsTranspose(OpType opType, TransposeType transposeType) override;
    bool SupportsReverse(OpType opType, ReverseType reverseTypeMask) override;
    bool SupportsFusedRescale(OpType opType, TensorUsage tensorUsage, DataType rescaleFromType, DataType rescaleToType,
        DataType opFromType, DataType opToType, const Quantization &quantization) override;
    bool SupportsRescale(DataType fromType, DataType toType) override;
    bool SupportsAccumulatorSaveRestore() override { return true; }
    bool SupportsGather(OpType opType) override;
    bool SupportsScatter(OpType opType) override;
    bool SupportsResize(const ResizeSupportQuery &query) override;
    bool SupportsArgMax(OpType opType) override;
    bool SupportsCast(OpType opType, DataType ifmType, DataType ofmType) override;
    bool SupportsNonMatchingShapes(const Shape &ifmShape, const Shape &ifm2Shape, const Shape &ofmShape) override;
    bool SupportsNegativeStrides() override { return false; };
    bool SupportsNot() override { return true; };
    Flags<QueryResult> OperatorQuery(OpType opType, const ArchOperatorQuery *query, ArchRequirements *req) override;

private:
    ArchEthosU85 *_arch;
};

}  // namespace regor
