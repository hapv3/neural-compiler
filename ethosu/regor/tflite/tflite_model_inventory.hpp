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

}  // namespace regor
