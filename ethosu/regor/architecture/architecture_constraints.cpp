//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "architecture_constraints.hpp"

#include "common/bit_flags.hpp"

BEGIN_ENUM_TABLE(regor::ArchRequirement)
    ADD_ENUM_NAME(None)
    ADD_ENUM_NAME(Tensor)
    ADD_ENUM_NAME(OpSubstitution)
    ADD_ENUM_NAME(Decompose)
END_ENUM_TABLE()

BEGIN_ENUM_TABLE(regor::ArchProperty)
    ADD_ENUM_NAME(None)
    ADD_ENUM_NAME(TensorAxis)
    ADD_ENUM_NAME(TensorDims)
    ADD_ENUM_NAME(KernelStride)
    ADD_ENUM_NAME(KernelDilation)
    ADD_ENUM_NAME(DepthMultiplier)
    ADD_ENUM_NAME(TransposeMask)
    ADD_ENUM_NAME(ReduceAxis)
    ADD_ENUM_NAME(Scaling)
    ADD_ENUM_NAME(NonConstantWeights)
END_ENUM_TABLE()

BEGIN_ENUM_TABLE(regor::QueryResult)
    ADD_ENUM_NAME(None)
    ADD_ENUM_NAME(Unsupported)
    ADD_ENUM_NAME(Native)
    ADD_ENUM_NAME(Constrained)
    ADD_ENUM_NAME(HasRequirements)
    ADD_ENUM_NAME(Emulated)
END_ENUM_TABLE()
