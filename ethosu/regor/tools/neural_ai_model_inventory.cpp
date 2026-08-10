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
#include <sstream>
#include <string>
#include <vector>

namespace
{

bool ParseSelection(const std::string &text, unsigned &subgraph, std::vector<unsigned> &operators)
{
    const auto colon = text.find(':');
    if ( colon == std::string::npos || colon == 0 || colon + 1 == text.size() ) return false;
    try
    {
        size_t consumed = 0;
        subgraph = unsigned(std::stoul(text.substr(0, colon), &consumed));
        if ( consumed != colon ) return false;
        std::istringstream stream(text.substr(colon + 1));
        std::string token;
        while ( std::getline(stream, token, ',') )
        {
            if ( token.empty() ) return false;
            consumed = 0;
            const auto index = std::stoul(token, &consumed);
            if ( consumed != token.size() ) return false;
            operators.push_back(unsigned(index));
        }
    }
    catch ( const std::exception & )
    {
        return false;
    }
    return !operators.empty();
}

}  // namespace

int main(int argc, char **argv)
{
    if ( argc != 2 && argc != 6 )
    {
        std::cerr << "Usage:\n"
                  << "  neural-ai-model-inventory MODEL.tflite\n"
                  << "  neural-ai-model-inventory MODEL.tflite --micrograph SUBGRAPH:OP[,OP...] --output OUTPUT.tflite\n";
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
    if ( argc == 2 )
    {
        const auto inventory = regor::BuildTfLiteModelInventory(data.data(), data.size(), path.filename().string());
        if ( !inventory )
        {
            std::cerr << inventory.error << '\n';
            return 1;
        }
        std::cout << inventory.json;
        return 0;
    }

    if ( std::string(argv[2]) != "--micrograph" || std::string(argv[4]) != "--output" )
    {
        std::cerr << "Expected --micrograph SUBGRAPH:OP[,OP...] --output OUTPUT.tflite\n";
        return 2;
    }
    unsigned subgraph = 0;
    std::vector<unsigned> operators;
    if ( !ParseSelection(argv[3], subgraph, operators) )
    {
        std::cerr << "Invalid micrograph selection; expected SUBGRAPH:OP[,OP...]\n";
        return 2;
    }
    const auto micrograph = regor::BuildTfLiteTopologyMicrograph(
        data.data(), data.size(), subgraph, operators, path.filename().string());
    if ( !micrograph )
    {
        std::cerr << micrograph.error << '\n';
        return 1;
    }
    const std::filesystem::path outputPath(argv[5]);
    std::ofstream output(outputPath, std::ios::binary);
    if ( !output )
    {
        std::cerr << "Failed to open output model: " << outputPath << '\n';
        return 2;
    }
    output.write(reinterpret_cast<const char *>(micrograph.model.data()), std::streamsize(micrograph.model.size()));
    if ( !output )
    {
        std::cerr << "Failed to write output model: " << outputPath << '\n';
        return 2;
    }
    std::cout << micrograph.provenanceJson;
    return 0;
}
