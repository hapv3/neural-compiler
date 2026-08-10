//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace regor
{

struct TfLiteModelInventory
{
    std::string json;
    std::string error;

    explicit operator bool() const { return error.empty(); }
};

/// Build a deterministic, model-source inventory. This describes frontend
/// contents only; it does not classify operators as supported by a target.
TfLiteModelInventory BuildTfLiteModelInventory(
    const uint8_t *data, size_t size, std::string_view artifactName = {});

struct TfLiteTopologyMicrograph
{
    std::vector<uint8_t> model;
    std::string provenanceJson;
    std::string error;

    explicit operator bool() const { return error.empty(); }
};

/// Extract a deterministic TFLite micro-graph containing the selected source
/// operators. Boundary tensors become graph inputs/outputs, while source tensor
/// contracts, constant buffers, operator options, and topology are preserved.
TfLiteTopologyMicrograph BuildTfLiteTopologyMicrograph(const uint8_t *data, size_t size,
    unsigned subgraphIndex, const std::vector<unsigned> &operatorIndices, std::string_view artifactName = {},
    int inputHeight = 0, int inputWidth = 0);

}  // namespace regor
