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

#include "mlw_encode.hpp"

#include "common/common.hpp"

#include "common/bit_flags.hpp"

#include <mlw_encode.h>
#include <cassert>

BEGIN_ENUM_TABLE(WeightFormat)
    ADD_ENUM_NAME(Default)
    ADD_ENUM_NAME(Fast)
    ADD_ENUM_NAME(Sparse2_4)
END_ENUM_TABLE()

thread_local static std::vector<uint8_t> *sResult = nullptr;

static void *reallocFunc(void *ptr, size_t reserve, int purpose)
{
    UNUSED(purpose);
    assert(sResult);
    assert(purpose == MLW_ENCODE_ALLOC_STREAM0);
    size_t offset = ptr ? static_cast<uint8_t *>(ptr) - sResult->data() : sResult->size();
    sResult->resize(reserve + offset);
    return reserve ? static_cast<void *>(sResult->data() + offset) : nullptr;
}

MlwEncodeResult mle_encode_proxy(IWeightSource *source, int chunkSize, std::vector<uint8_t> &output, unsigned encodeFlags)
{
    assert(sResult == nullptr);
    sResult = &output;
    MlwEncodeResult encodeResult;
    auto output_size = output.size();
    ml_encode_result_t res;
    ml_ethosu_encode_params_t params;
    params.encoder_flags = encodeFlags;
    params.source_buffering_hint = chunkSize;
    params.realloc_func = reallocFunc;

    auto weight_func = [](int32_t query, ml_source_state_t *state, int16_t *buffer, int32_t size, void *user_arg)
    {
        UNUSED(query);
        assert(query == MLW_SOURCE_QUERY_WEIGHTS);
        IWeightSource *src = reinterpret_cast<IWeightSource *>(user_arg);
        int source_size = src->Get(buffer, size);
        state->eos = source_size < size;
        return source_size;
    };

    try
    {
        mle_context_t *ctx = nullptr;
        auto ret = ml_encode_ethosu_stream(&res, &params, weight_func, source, &ctx);
        if ( ret < 0 ) throw std::runtime_error("mlw encode failed");
        encodeResult.zero_count = mle_context_query_zeroes(ctx);
        encodeResult.distinct_values = mle_context_query_weights_used(ctx, encodeResult.distinct_weights);
        mle_destroy_context(ctx);
    }
    catch ( const std::runtime_error & )
    {
        sResult = nullptr;
        throw;
    }
    sResult = nullptr;
    res.encoded_data = nullptr;  // Data owned by output
    mle_free(&res);
    output.resize(output_size + res.encoded_length);
    encodeResult.bytes_written = res.encoded_length;
    encodeResult.elements_read = res.source_length;
    return encodeResult;
}

MlwEncodeResult mle_encode_fwd_proxy(IWeightSource *source, int chunkSize, std::vector<uint8_t> &output, unsigned encodeFlags)
{
    assert(sResult == nullptr);
    sResult = &output;
    MlwEncodeResult encodeResult;
    auto output_size = output.size();
    int count = 0;
    int totalCount = 0;

    std::vector<int16_t> weights;

    try
    {
        do
        {
            weights.resize(weights.size() + chunkSize);
            count = source->Get(weights.data() + totalCount, chunkSize);
            totalCount += count;
        } while ( count == chunkSize );
        weights.resize(totalCount);
    }
    catch ( const std::runtime_error & )
    {
        sResult = nullptr;
        throw;
    }

    ml_encode_result_t res;
    res.encoded_length = 0;
    res.source_length = 0;
    mle_context_t *ctx = mle_create_context(MLW_ENCODE_SYNTAX_ETHOSU_FWD);
    mle_context_set_allocator(ctx, reallocFunc);
    mle_encode(ctx, &res, weights.data(), totalCount, encodeFlags);
    encodeResult.zero_count = mle_context_query_zeroes(ctx);
    encodeResult.distinct_values = mle_context_query_weights_used(ctx, encodeResult.distinct_weights);
    res.encoded_data = nullptr;  // Data owned by output
    mle_destroy_context(ctx);
    mle_free(&res);
    output.resize(output_size + res.encoded_length);
    sResult = nullptr;
    encodeResult.bytes_written = res.encoded_length;
    encodeResult.elements_read = res.source_length;
    return encodeResult;
}
