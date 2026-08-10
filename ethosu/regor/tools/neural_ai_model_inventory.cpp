//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "tflite/tflite_model_inventory.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

int main(int argc, char **argv)
{
    if ( argc != 2 )
    {
        std::cerr << "Usage: neural-ai-model-inventory MODEL.tflite\n";
        return 2;
    }

    const std::filesystem::path path(argv[1]);
    std::ifstream stream(path, std::ios::binary);
    if ( !stream )
    {
        std::cerr << "Failed to open model: " << path << '\n';
        return 2;
    }
    std::vector<uint8_t> data(std::istreambuf_iterator<char>(stream), {});
    const auto inventory = regor::BuildTfLiteModelInventory(data.data(), data.size(), path.filename().string());
    if ( !inventory )
    {
        std::cerr << inventory.error << '\n';
        return 1;
    }
    std::cout << inventory.json;
    return 0;
}
