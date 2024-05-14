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

#include "common/reverse_type.hpp"

#include "common/logging.hpp"

#include "common/bit_flags.hpp"

BEGIN_ENUM_TABLE(ReverseType)
    ADD_ENUM_NAME(None)
    ADD_ENUM_NAME(H)
    ADD_ENUM_NAME(W)
    ADD_ENUM_NAME(C)
END_ENUM_TABLE()
