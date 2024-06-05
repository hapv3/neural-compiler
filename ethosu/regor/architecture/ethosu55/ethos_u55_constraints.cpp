//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "ethos_u55_constraints.hpp"

#include "ethos_u55_register_cs_generator.hpp"

namespace regor
{

bool EthosU55Constraints::SupportsLeakyRelu(bool quantized, DataType type)
{
    return quantized == false && type == DataType::Int16;
}

bool EthosU55Constraints::SupportsMatMul(OpType opType)
{
    UNUSED(opType);
    return false;
}

bool EthosU55Constraints::SupportsTransposeHW(OpType opType, TransposeType transposeType)
{
    UNUSED(opType);
    UNUSED(transposeType);
    return IsNone(transposeType);
}

bool EthosU55Constraints::SupportsTranspose(OpType opType, TransposeType transposeType, Shape ifmShape)
{
    UNUSED(ifmShape);
    if ( SupportsTransposeHW(opType, transposeType) )
    {
        return true;
    }
    return false;
}

bool EthosU55Constraints::SupportsReverse(OpType opType, ReverseType reverseType)
{
    UNUSED(opType);
    UNUSED(reverseType);
    return reverseType == ReverseType::None;
}

bool EthosU55Constraints::SupportsGather(OpType opType)
{
    UNUSED(opType);
    return false;
}

bool EthosU55Constraints::SupportsScatter(OpType opType)
{
    UNUSED(opType);
    return false;
}
bool EthosU55Constraints::SupportsResize(const ResizeSupportQuery &query)
{
    UNUSED(query);
    return false;
}

bool EthosU55Constraints::SupportsSigmoidTanhLutInt16(OpType opType)
{
    UNUSED(opType);
    return false;
}

bool EthosU55Constraints::SupportsArgMax(OpType opType)
{
    UNUSED(opType);
    return false;
}

}  // namespace regor
