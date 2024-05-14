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

#include <vector>

/// <summary>
/// Mechanism for treating partial sections of a vector as complete sequences
/// to allow sub-array processing.
/// Does not track invalidation (source vector cannot change)
/// </summary>
template<typename TYPE>
class vector_span
{
    using iterator = typename std::vector<TYPE>::iterator;
    using const_iterator = typename std::vector<TYPE>::const_iterator;

private:
    iterator _start;
    iterator _end;

public:
    vector_span() = default;

    vector_span(std::vector<TYPE> &vec)
    {
        _start = vec.begin();
        _end = vec.end();
    }

    vector_span(std::vector<TYPE> &vec, int start, int end)
    {
        _start = vec.begin() + start;
        _end = vec.begin() + end;
    }

    vector_span(const std::vector<TYPE> &vec, int start, int end)
    {
        auto posStart = vec.begin() + start;
        auto posEnd = vec.begin() + end;
        // Use vec.erase to convert from const to non-const iterators (erases nothing!)
        _start = const_cast<std::vector<TYPE> &>(vec).erase(posStart, posStart);
        _end = const_cast<std::vector<TYPE> &>(vec).erase(posEnd, posEnd);
    }

    TYPE &front() { return *_start; }
    const TYPE &front() const { return *_start; }

    TYPE &back() { return *(_end - 1); }
    const TYPE &back() const { return *(_end - 1); }

    // Iterate just the values
    iterator begin() { return _start; }
    iterator end() { return _end; }
    const_iterator begin() const { return _start; }
    const_iterator end() const { return _end; }

    int size() const { return int(std::distance(_start, _end)); }

    TYPE &operator[](int index) { return *(_start + index); }
};
