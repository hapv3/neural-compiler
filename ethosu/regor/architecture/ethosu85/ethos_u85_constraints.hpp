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
private:
    ArchEthosU85 *_arch;

public:
    EthosU85Constraints(ArchEthosU85 *arch) : _arch(arch) {}
    bool SupportsFusedRescale(OpType opType, TensorUsage tensorUsage, DataType rescaleFromType, DataType rescaleToType,
        DataType opFromType, DataType opToType, const Quantization &quantization) override;
    bool SupportsAccumulatorSaveRestore() override { return true; }
    bool SupportsNegativeStrides() override { return false; };
    bool SupportsElementwiseLeakyRelu(bool quantized, DataType type) override { return true; };
    bool SupportsRescale(DataType fromType, DataType toType) override;
    Flags<QueryResult> OperatorQuery(OpType opType, const ArchOperatorQuery *query, ArchRequirements *req) override;
    bool SupportedZeroPoint(int64_t zp, TensorUsage usage, DataType dtype, OpType opType) override;

private:
    bool SupportedDtypes(OpType opType, DataType ifmType, DataType ifm2Type, DataType ofmType);
    bool SupportsFusedReverse(OpType opType, ReverseType reverseTypeMask);
    TransposeSupport SupportsFusedTranspose(OpType opType, TransposeType transposeType);
};

}  // namespace regor
