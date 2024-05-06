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

#include "hash.hpp"

#include <algorithm>
#include <cstring>

namespace regor
{

// clang-format off

// Shift values
static const int s_Shift[] = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
};

// K-values
static const uint32_t s_K[] = {
    0xD76AA478, 0xE8C7B756, 0x242070DB, 0xC1BDCEEE, 0xF57C0FAF, 0x4787C62A, 0xA8304613, 0xFD469501,
    0x698098D8, 0x8B44F7AF, 0xFFFF5BB1, 0x895CD7BE, 0x6B901122, 0xFD987193, 0xA679438E, 0x49B40821,
    0xF61E2562, 0xC040B340, 0x265E5A51, 0xE9B6C7AA, 0xD62F105D, 0x02441453, 0xD8A1E681, 0xE7D3FBC8,
    0x21E1CDE6, 0xC33707D6, 0xF4D50D87, 0x455A14ED, 0xA9E3E905, 0xFCEFA3F8, 0x676F02D9, 0x8D2A4C8A,
    0xFFFA3942, 0x8771F681, 0x6D9D6122, 0xFDE5380C, 0xA4BEEA44, 0x4BDECFA9, 0xF6BB4B60, 0xBEBFBC70,
    0x289B7EC6, 0xEAA127FA, 0xD4EF3085, 0x04881D05, 0xD9D4D039, 0xE6DB99E5, 0x1FA27CF8, 0xC4AC5665,
    0xF4292244, 0x432AFF97, 0xAB9423A7, 0xFC93A039, 0x655B59C3, 0x8F0CCC92, 0xFFEFF47D, 0x85845DD1,
    0x6FA87E4F, 0xFE2CE6E0, 0xA3014314, 0x4E0811A1, 0xF7537E82, 0xBD3AF235, 0x2AD7D2BB, 0xEB86D391,
};

// clang-format on

static inline uint32_t Rol(uint32_t value, int shift)
{
    assert(shift < 32);
    return (value << shift) | (value >> (32 - shift));
}

static void HashStep(uint32_t hash[4], uint32_t block[16])
{
    uint32_t a = hash[0];
    uint32_t b = hash[1];
    uint32_t c = hash[2];
    uint32_t d = hash[3];

    int i = 0, f, g;

    // Step 1
    for ( ; i < 16; i++ )
    {
        f = (b & c) | ((~b) & d);
        g = i;

        f = Rol(f + a + s_K[i] + block[g], s_Shift[i]);
        a = d;
        d = c;
        c = b;
        b += f;
    }

    // Step 2
    for ( ; i < 32; i++ )
    {
        f = (d & b) | ((~d) & c);
        g = (5 * i + 1) % 16;

        f = Rol(f + a + s_K[i] + block[g], s_Shift[i]);
        a = d;
        d = c;
        c = b;
        b += f;
    }

    // Step 3
    for ( ; i < 48; i++ )
    {
        f = b ^ c ^ d;
        g = (3 * i + 5) % 16;

        f = Rol(f + a + s_K[i] + block[g], s_Shift[i]);
        a = d;
        d = c;
        c = b;
        b += f;
    }

    // Step 4
    for ( ; i < 64; i++ )
    {
        f = c ^ (b | ~d);
        g = (7 * i) % 16;

        f = Rol(f + a + s_K[i] + block[g], s_Shift[i]);
        a = d;
        d = c;
        c = b;
        b += f;
    }

    // Remember hash state
    hash[0] += a;
    hash[1] += b;
    hash[2] += c;
    hash[3] += d;
}

MD5::MD5()
{
    Reset();
}

void MD5::Reset()
{
    _hash[0] = 0x67452301;
    _hash[1] = 0xEFCDAB89;
    _hash[2] = 0x98BADCFE;
    _hash[3] = 0x10325476;
    _totalBytes = 0;
    _done = false;
}

void MD5::Combine(const uint8_t *data, int length)
{
    if ( data != nullptr )
    {
        while ( length > 0 )
        {
            // Attempt to fill up the internal buffer block
            int used = int(_totalBytes % BLOCKSIZE);
            int toCopy = std::min<int>(BLOCKSIZE - used, length);

            if ( toCopy > 0 )
            {
                uint8_t *p = reinterpret_cast<uint8_t *>(_block) + used;
                std::copy_n(data, toCopy, p);
                data += toCopy;
                _totalBytes += toCopy;
                used += toCopy;
            }

            // Start when the block buffer is full
            if ( used == BLOCKSIZE )
            {
                HashStep(_hash, _block);
            }

            length -= toCopy;
        }
    }
}

int MD5::Get(uint8_t *hash, int length)
{
    if ( hash == nullptr )
    {
        return int(sizeof(_hash));
    }

    if ( !_done )
    {
        const int BLOCKPAD = BLOCKSIZE - sizeof(uint64_t);

        int used = int(_totalBytes % BLOCKSIZE);
        uint8_t *p = reinterpret_cast<uint8_t *>(_block) + used;

        // Terminate
        *p = 0x80;
        used++;

        // Zero-fill rest of block
        if ( used < BLOCKSIZE )
        {
            p = reinterpret_cast<uint8_t *>(_block) + used;
            memset(p, 0, BLOCKSIZE - used);
        }

        // Wrap to next block if size fields don't fit
        if ( used > BLOCKPAD )
        {
            HashStep(_hash, _block);
            p = reinterpret_cast<uint8_t *>(_block);
            memset(p, 0, BLOCKPAD);
        }

        // Pad to 448 bits (mod 512) and append message length (in bits)
        _block[14] = uint32_t(_totalBytes) << 3;
        _block[15] = uint32_t(_totalBytes >> 29);
        HashStep(_hash, _block);

        _done = true;
    }

    // Return copy of the hash
    length = std::min(length, int(sizeof(_hash)));
    std::copy_n(reinterpret_cast<uint8_t *>(_hash), length, hash);
    return length;
}

}  // namespace regor
