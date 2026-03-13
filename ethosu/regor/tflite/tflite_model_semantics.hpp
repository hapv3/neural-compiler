//
// SPDX-FileCopyrightText: Copyright 2023, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "common/ordered_map.hpp"
#include "tflite_schema_generated.hpp"

namespace tflite
{

class TFLiteModelSemantics
{
public:
    explicit TFLiteModelSemantics(const Model *model, const uint8_t *input, size_t size) :
            m_model(model), m_input(input), m_size(size)
    {
    }

    void Check();

private:
    const Model *m_model;
    const uint8_t *m_input;
    size_t m_size;
};
}  // namespace tflite
