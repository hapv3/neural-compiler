//
// SPDX-FileCopyrightText: Copyright 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "common/data_type.hpp"

#include "common/logging.hpp"

#include "common/bit_flags.hpp"

BEGIN_ENUM_TABLE(regor::DataType)
    ADD_ENUM_NAME(None)
    ADD_ENUM_NAME(Bits4)
    ADD_ENUM_NAME(Bits8)
    ADD_ENUM_NAME(Bits16)
    ADD_ENUM_NAME(Bits32)
    ADD_ENUM_NAME(Bits64)
    ADD_ENUM_NAME(Bits128)
    ADD_ENUM_NAME(Signed)
    ADD_ENUM_NAME(Asymmetric)
    ADD_ENUM_NAME(Int)
    ADD_ENUM_NAME(SignedInt)
    ADD_ENUM_NAME(Int4)
    ADD_ENUM_NAME(Int8)
    ADD_ENUM_NAME(Int16)
    ADD_ENUM_NAME(Int32)
    ADD_ENUM_NAME(Int48)
    ADD_ENUM_NAME(Int64)
    ADD_ENUM_NAME(UInt8)
    ADD_ENUM_NAME(UInt16)
    ADD_ENUM_NAME(UInt32)
    ADD_ENUM_NAME(UInt64)
    ADD_ENUM_NAME(QInt)
    ADD_ENUM_NAME(QInt4)
    ADD_ENUM_NAME(QInt8)
    ADD_ENUM_NAME(QInt12)
    ADD_ENUM_NAME(QInt16)
    ADD_ENUM_NAME(QInt32)
    ADD_ENUM_NAME(QUInt)
    ADD_ENUM_NAME(QUInt4)
    ADD_ENUM_NAME(QUInt8)
    ADD_ENUM_NAME(QUInt12)
    ADD_ENUM_NAME(QUInt16)
    ADD_ENUM_NAME(QUInt32)
    ADD_ENUM_NAME(Float)
    ADD_ENUM_NAME(Float16)
    ADD_ENUM_NAME(Float32)
    ADD_ENUM_NAME(Float64)
    ADD_ENUM_NAME(Bool)
    ADD_ENUM_NAME(Bool8)
    ADD_ENUM_NAME(Complex)
    ADD_ENUM_NAME(Complex64)
    ADD_ENUM_NAME(Complex128)
    ADD_ENUM_NAME(VariablySized)
    ADD_ENUM_NAME(String)
    ADD_ENUM_NAME(Resource)
    ADD_ENUM_NAME(Variant)
END_ENUM_TABLE()
