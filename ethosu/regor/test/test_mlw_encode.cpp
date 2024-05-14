//
// SPDX-FileCopyrightText: Copyright 2021-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "common/common.hpp"

#include "architecture/mlw_encode.hpp"
#include "randomize.hpp"

#include <mlw_decode.h>
#include <mlw_encode.h>
#include <catch_all.hpp>

namespace
{

enum class sc_compression_schemes_t : int
{
    DIRECT_MODE,
    DIRECT_MODE_WITH_RLE,
    DIRECT_TRUNCATED_MODE_WITH_RLE,
    PALETTE_DIRECT_MODE,
    PALETTE_MODE_WITH_RLE,
    PALETTE_TRUNCATED_MODE_WITH_RLE,
    PALETTE_DIRECT_MODE_WITH_RLE,
    UNCOMPRESSED_MODE,
    RANDOM_MODE,
    PALETTE_MODE,
    PALETTE_TRUNCATED_MODE,
    DIRECT_TRUNCATED_MODE,

    CNT
};

class MlwTb
{
public:
    const std::vector<int16_t> &GenWeightSymbols(sc_compression_schemes_t comp_scheme, int total_num_weights)
    {
        int slice_div = urandom_range(1, 10);
        int num_slices = (total_num_weights / slice_div) + (((total_num_weights % slice_div) != 0) ? 1 : 0);
        unsigned weights = 0;
        constexpr int signext_shift = 16 - 9;
        // Make sure an all zeroes stream is tested with RLE
        bool all_zeroes = comp_scheme == sc_compression_schemes_t::DIRECT_MODE_WITH_RLE && urandom_range(0, 1);
        _unmapped_weights.clear();
        _unmapped_weights.reserve(total_num_weights);

        for ( int i = 0; i < num_slices; i++ )
        {
            int non_zeros = 0;
            sc_compression_schemes_t sel_compression = GetMinMax(comp_scheme);
            for ( int s = 0; s < slice_div; s++ )
            {
                if ( all_zeroes )
                {
                    weights = 0;
                }
                else if ( sel_compression == sc_compression_schemes_t::DIRECT_TRUNCATED_MODE_WITH_RLE )
                {
                    weights = (non_zeros < 3) ? urandom_range(0, 1) * urandom_range(_min, _max) : 0;
                }
                else
                {
                    weights = urandom_range(_min, _max);
                }
                if ( weights != 0 ) non_zeros++;
                // Encoder range is -255 to 255. The value -1 in 2's complement is represented as -256
                // Hence add +1 if the weight value is -1
                if ( weights == 256 )
                {
                    weights++;
                }
                if ( int(_unmapped_weights.size()) < total_num_weights )
                {
                    auto w = int16_t(int16_t(weights << signext_shift) >> signext_shift);
                    _unmapped_weights.push_back(w);
                }
            }
        }
        return _unmapped_weights;
    }

private:
    sc_compression_schemes_t GetMinMax(sc_compression_schemes_t comp_scheme)
    {
        int zeros;
        sc_compression_schemes_t sel_compression;

        if ( comp_scheme == sc_compression_schemes_t::RANDOM_MODE )
        {
            sel_compression = sc_compression_schemes_t(urandom_range(0, unsigned(sc_compression_schemes_t::CNT) - 1));
        }
        else
        {
            sel_compression = comp_scheme;
        }
        switch ( sel_compression )
        {
            case sc_compression_schemes_t::UNCOMPRESSED_MODE:
                _min = urandom_range(0, 127);
                _max = urandom_range(_min, 510);
                break;
            case sc_compression_schemes_t::DIRECT_MODE:
                _min = urandom_range(0, 200);
                _max = urandom_range(_min, _min + 10);
                break;
            case sc_compression_schemes_t::DIRECT_MODE_WITH_RLE:
                zeros = urandom_range(0, 1);
                _min = urandom_range(0, zeros * 127);
                _max = urandom_range(_min, zeros * 255);
                break;
            case sc_compression_schemes_t::DIRECT_TRUNCATED_MODE:
                _min = urandom_range(0, 200);
                _max = urandom_range(_min, _min + 3);
                break;
            case sc_compression_schemes_t::DIRECT_TRUNCATED_MODE_WITH_RLE:
                zeros = urandom_range(0, 1);
                _min = urandom_range(0, zeros * 127);
                _max = urandom_range(_min, _min + 3);
                break;
            case sc_compression_schemes_t::PALETTE_MODE:
                _min = urandom() % 3 == 0 ? urandom_range(0, 5) : urandom_range(0, 16);
                _max = urandom_range(_min, _min + 3);
                break;
            case sc_compression_schemes_t::PALETTE_DIRECT_MODE:
                // Fall through
            case sc_compression_schemes_t::PALETTE_MODE_WITH_RLE:
                _min = urandom_range(0, 220);
                _max = urandom_range(_min, _min + 31);
                break;
            case sc_compression_schemes_t::PALETTE_DIRECT_MODE_WITH_RLE:
                _min = urandom_range(0, 7);
                _max = urandom_range(_min, _min + 2);
                break;
            case sc_compression_schemes_t::PALETTE_TRUNCATED_MODE:
                _min = urandom() % 7 == 0 ? urandom_range(0, 16) : urandom_range(0, 200);
                _max = urandom_range(_min, _min + 2);
                break;
            case sc_compression_schemes_t::PALETTE_TRUNCATED_MODE_WITH_RLE:
                // Fall through
            case sc_compression_schemes_t::RANDOM_MODE:
                // Fall through
            case sc_compression_schemes_t::CNT:
                // Fall through
            default:
                _min = 0;
                _max = 0;
                break;
        }
        return sel_compression;
    }

    int _min = 0;
    int _max = 0;
    std::vector<int16_t> _unmapped_weights;
};

}  // namespace


struct LinearWeightSource : public IWeightSource
{
    const int16_t *_buffer;
    int _index = 0;
    int _size = 0;

    LinearWeightSource(const int16_t *inbuf, int size) : _buffer(inbuf), _size(size) {}

    int Elements() { return _size - _index; }

    int Get(int16_t *buffer, int count)
    {
        count = std::min(count, _size - _index);
        std::copy(_buffer + _index, _buffer + _index + count, buffer);
        _index += count;
        return count;
    }
};

TEST_CASE("mlw_encode")
{
    MlwTb tb;
    extern bool OptNightly;
    int iter_cnt = OptNightly ? 1024 : 2;
    int long_stream_cnt = 0;
    constexpr int min_long_stream = 2;
    constexpr int max_long_stream = 4;

    for ( int iter = 0; iter < iter_cnt; iter++ )
    {
        CAPTURE(iter);
        // Make sure RLE is tested
        sc_compression_schemes_t mode = iter < 16 ? sc_compression_schemes_t::DIRECT_MODE_WITH_RLE : sc_compression_schemes_t::RANDOM_MODE;
        bool long_stream = (long_stream_cnt < max_long_stream) && (urandom_range(0, 4) == 0 || iter >= (iter_cnt - min_long_stream));
        if ( long_stream )
        {
            long_stream_cnt++;
        }
        int len = long_stream ? urandom_range(1024 * 1024, 4096 * 1024) : urandom_range(16, 2048 * 10);
        CAPTURE(mode);
        CAPTURE(len);
        const std::vector<int16_t> &inbuf_enc = tb.GenWeightSymbols(mode, len);
        std::vector<uint8_t> outbuf_enc;

        // Test direct API
        ml_encode_result_t enc_res;
        mle_context_t *ctx = mle_create_context(MLW_ENCODE_SYNTAX_ETHOSU);
        int outbuf_size_enc = mle_encode(ctx, &enc_res, inbuf_enc.data(), int(inbuf_enc.size()), MLW_ENCODE_FLAG_NONE);
        mle_destroy_context(ctx);
        REQUIRE(enc_res.encoded_length > 0);

        ml_decode_result_t dec_res;
        ml_decode_ethosu_stream(&dec_res, enc_res.encoded_data, enc_res.encoded_length);
        mle_free(&enc_res);
        REQUIRE(size_t(dec_res.decoded_length) == inbuf_enc.size());
        for ( unsigned i = 0; i < inbuf_enc.size(); i++ )
        {
            if ( dec_res.decoded_data[i] != inbuf_enc[i] )
            {
                CAPTURE(i);
                REQUIRE(dec_res.decoded_data[i] == inbuf_enc[i]);
            }
        }
        mld_free(&dec_res);

        // Test interface API
        LinearWeightSource source(inbuf_enc.data(), int(inbuf_enc.size()));
        outbuf_enc.clear();
        auto proxy_res = mle_encode_proxy(&source, 1024 * 1024, outbuf_enc, MLW_ENCODE_FLAG_NONE);
        REQUIRE(proxy_res.bytes_written > 0);
        REQUIRE(size_t(proxy_res.elements_read) == inbuf_enc.size());

        ml_decode_ethosu_stream(&dec_res, outbuf_enc.data(), proxy_res.bytes_written);
        REQUIRE(size_t(dec_res.decoded_length) == inbuf_enc.size());
        for ( unsigned i = 0; i < inbuf_enc.size(); i++ )
        {
            if ( dec_res.decoded_data[i] != inbuf_enc[i] )
            {
                CAPTURE(i);
                REQUIRE(dec_res.decoded_data[i] == inbuf_enc[i]);
            }
        }
        mld_free(&dec_res);
    }
}
