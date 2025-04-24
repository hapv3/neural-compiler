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

#include "architecture/architecture_constraints.hpp"
#include "ethos_u55.hpp"

namespace regor
{

class EthosU55Constraints : public IArchitectureConstraints
{
private:
    ArchEthosU55 *_arch = nullptr;

public:
    EthosU55Constraints(ArchEthosU55 *arch);

    bool SupportsFusedRescale(OpType opType, TensorUsage tensorUsage, DataType rescaleFromType, DataType rescaleToType,
        DataType opFromType, DataType opToType, const Quantization &quantization) override;
    bool SupportsAccumulatorSaveRestore() override { return false; }
    bool SupportsNegativeStrides() override { return true; };
    bool SupportsElementwiseLeakyRelu(bool quantized, DataType type) override;
    bool SupportsRescale(DataType fromType, DataType toType) override;
    Flags<QueryResult> OperatorQuery(OpType opType, const ArchOperatorQuery *query, ArchRequirements *req) override;
    bool SupportedZeroPoint(int64_t zp, TensorUsage usage, DataType dtype, OpType opType) override;

private:
    bool SupportedDtypes(OpType opType, DataType ifmType, DataType ifm2Type, DataType ofmType);
    bool SupportsFusedReverse(OpType opType, ReverseType reverseTypeMask);
    TransposeSupport SupportsFusedTranspose(OpType opType, TransposeType transposeType);
};

}  // namespace regor
