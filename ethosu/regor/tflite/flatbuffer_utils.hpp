//
// SPDX-FileCopyrightText: Copyright 2021, 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include <flatbuffers/flatbuffers.h>

namespace FlatbufferUtils
{
// Load a vector (if present) from a flatbuffer into a local copy.
// Intended for small vectors only - large vectors should be left in place and mapped using a Buffer class instead.
template<typename T>
static std::vector<T> LoadVector(const flatbuffers::Vector<T> *source)
{
    std::vector<T> destination;
    if ( source )
    {
        destination.insert(destination.begin(), source->begin(), source->end());
    }
    return destination;
}

// Copy a vector (if present) from one flatbuffer to another, returning the offset into the destination buffer.
template<typename T>
static flatbuffers::Offset<flatbuffers::Vector<T>>
CopyVector(flatbuffers::FlatBufferBuilder &destination, const flatbuffers::Vector<T> *source)
{
    if ( source )
    {
        return destination.CreateVector<T>(source->data(), source->size());
    }
    return 0;
}
}  // namespace FlatbufferUtils
