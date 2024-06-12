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

#pragma once

#include "common.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

namespace regor
{

///< summary>
/// Generic Hash value storage template
///</summary>
template<int BITS>
struct HashT
{
    enum
    {
        WORDS = (BITS + 31) / 32
    };

    union
    {
        uint8_t v8[BITS / 8];
        uint32_t v32[WORDS];
    };

    HashT()
    {
        for ( int i = 0; i < WORDS; i++ )
            v32[i] = 0;
    }

    HashT(const uint32_t *values, int count)
    {
        count = std::min(count, WORDS);
        int i = 0;
        while ( i < count )
            v32[i] = values[i++];
        while ( i < WORDS )
            v32[i++] = 0;
    }

    HashT(std::initializer_list<uint32_t> list)
    {
        const int max = std::min(int(WORDS), int(list.size()));
        int i = 0;
        for ( auto v : list )
        {
            v32[i++] = v;
            if ( i == max ) break;
        }
        while ( i < WORDS )
            v32[i++] = 0;
    }

    bool operator<(const HashT<BITS> &b) const { return memcmp(v8, b.v8, BITS / 8) < 0; }

    bool operator==(const HashT<BITS> &b) const { return memcmp(v8, b.v8, BITS / 8) == 0; }

    bool operator!=(const HashT<BITS> &b) const { return !((*this) == b); }

    uint8_t *Buffer() { return this->v8; }
    const uint8_t *Buffer() const { return this->v8; }
    int Size() const { return BITS / 8; }

    constexpr uint32_t Hash32() const
    {
        uint32_t hash = v32[0];
        for ( int i = 1; i < WORDS; i++ )
        {
            hash = (hash * 31) + v32[i];
        }
        return hash;
    }
};

template<typename VALUE>
uint32_t SimpleHash32(const VALUE &value)
{
    return uint32_t(value);
}

template<typename VALUE, typename... REST>
uint32_t SimpleHash32(const VALUE &value, REST &&...rest)
{
    return SimpleHash32(std::forward<REST>(rest)...) * 31 + uint32_t(value);
}

template<typename VALUE>
uint64_t SimpleHash64(const VALUE &value)
{
    return uint64_t(value);
}

template<typename VALUE, typename... REST>
uint64_t SimpleHash64(const VALUE &value, REST &&...rest)
{
    return SimpleHash64(std::forward<REST>(rest)...) * 31 + uint64_t(value);
}

template<typename TYPE>
uint32_t HashVector32(const std::vector<TYPE> &values)
{
    uint32_t hash = REGOR_FNV_SEED;
    for ( auto const x : values )
    {
        hash = FNVHashBytes(reinterpret_cast<const char *>(&x), sizeof(x), hash);
    }
    return hash;
}

using Hash128 = HashT<128>;

///< summary>
/// MD5 Hash implementation
///</summary>
class MD5
{
    enum
    {
        BLOCKSIZE = 64
    };

private:
    uint32_t _hash[4];
    uint32_t _block[BLOCKSIZE / 4];
    uint64_t _totalBytes = 0;
    bool _done = false;

public:
    MD5();

public:
    void Reset();
    void Combine(const uint8_t *data, int length);
    int Get(uint8_t *hash, int length);
    void Get(Hash128 &hash) { Get(hash.Buffer(), hash.Size()); }
};

}  // namespace regor

namespace std
{

template<>
struct hash<regor::Hash128>
{
    std::size_t operator()(const regor::Hash128 &h) const
    {
        std::size_t value = h.v32[0];
        for ( int i = 1; i < h.Size() / 4; i++ )
        {
            value = (value * 31) + h.v32[i];
        }
        return value;
    }
};

}  // namespace std
