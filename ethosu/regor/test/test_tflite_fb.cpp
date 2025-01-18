//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "architecture/ethosu85/ethos_u85.hpp"
#include "tflite/tflite_reader.hpp"
#include "tflite/tflite_writer.hpp"
#include "util.hpp"

#include <fmt/format.h>
#include <catch_all.hpp>

#include "regor.h"

using namespace regor;


TEST_CASE("test_tflite_fb - load/store")
{
    // Create arch
    auto arch = CreateArchDefault<ArchEthosU85>();
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");

    SECTION("Buffer offset")
    {
        // Get a simple passthrough model
        size_t psSize, psOffset;
        const void *tfliteOp;
        const void *tfliteModel;
        const auto passthrough = [&](size_t &size, size_t &offset, const void *&model, const void *&op)
        {
            flatbuffers::FlatBufferBuilder builder;
            std::vector<flatbuffers::Offset<tflite::OperatorCode>> codes;
            std::vector<flatbuffers::Offset<tflite::Operator>> serialised_operations;
            std::vector<flatbuffers::Offset<tflite::SubGraph>> serialised_subgraphs;
            serialised_operations.push_back(tflite::CreateOperator(
                builder, 0, 0, 0, tflite::BuiltinOptions::AddOptions, tflite::CreateAddOptions(builder).Union()));
            serialised_subgraphs.push_back(tflite::CreateSubGraphDirect(builder, nullptr, nullptr, nullptr, &serialised_operations));
            codes.push_back(tflite::CreateOperatorCodeDirect(builder, 0, nullptr, 1, tflite::BuiltinOperator::ADD));
            const auto ps = tflite::CreateModelDirect(builder, 3, &codes, &serialised_subgraphs);
            tflite::FinishModelBuffer(builder, ps);
            const uint8_t *base = builder.ReleaseRaw(size, offset);

            const tflite::Model *m = tflite::GetModel(&base[offset]);
            assert(m->operator_codes());
            auto tflite_subgraphs = m->subgraphs();
            assert(tflite_subgraphs->size() == 1);
            auto tflite_operators = (*tflite_subgraphs)[0]->operators();
            assert(tflite_operators->size() == 1);
            op = (*tflite_operators)[0];
            model = m;

            return std::unique_ptr<const uint8_t[]>(base);
        }(psSize, psOffset, tfliteModel, tfliteOp);

        // The same model in graph flavor
        const auto graphs = [&]()
        {
            std::vector<std::shared_ptr<Operation>> ops;
            std::vector<Operation *> rawOps;
            auto cifmStorageShape = Shape(1, 1, 1, 128);
            std::vector<int8_t> cifmData(cifmStorageShape.Elements(), 0);
            for ( size_t i = 0; i < cifmData.size(); i++ )
            {
                cifmData[i] = i;
            }
            auto cifm = CreateTensor("CIFM", cifmStorageShape, DataType::Int8, std::move(cifmData));
            auto ifm = CreateTensor("IFM", cifmStorageShape.WithWidth(10), DataType::Int8);
            auto ofm = CreateTensor("OFM", cifmStorageShape.WithWidth(10), DataType::Int8);
            auto op = CreateOperation(OpType::Add, TensorUsage::IFM, ifm, TensorUsage::IFM1, cifm, TensorUsage::OFM, ofm);
            op->SetPassthrough(tfliteOp);
            rawOps.push_back(op.get());
            ops.push_back(std::move(op));

            // Create graph with ops
            std::vector<std::unique_ptr<Graph>> ret;
            auto gr = CreateGraph(ops);
            gr->SetScheduledOrder(std::move(rawOps));
            gr->SetPassthrough(tfliteModel);
            ret.push_back(std::move(gr));

            return ret;
        }();

        // These integers are output by the call to TfLiteWriter::Serialise below
        int64_t output_buffer_offset = 0;
        size_t output_buffer_size = 0;
        for ( size_t i = 0; i < 2; i++ )
        {
            // First iteration : FlatBuffer will have a 2GB limit
            // Second iteration : FlatBuffer will have a previous_size-1 limit to ensure offset buffers are used
            TfLiteWriter writer(output_buffer_size > 0 ? output_buffer_size - 1 : size_t{1U << 31});

            auto fb = writer.Serialise(graphs, {{}}, output_buffer_offset, output_buffer_size);

            TfLiteReader reader;
            std::vector<std::unique_ptr<Graph>> readerGraphs;

            reader.LoadGraphs(&fb[output_buffer_offset], output_buffer_size, readerGraphs, nullptr, arch->Constraints());

            REQUIRE(readerGraphs.size() == 1);
            std::vector<Operation *> operations;
            readerGraphs[0]->GetAllOperations(operations);
            REQUIRE(operations.size() == 1);
            const auto *cten = operations[0]->IFM(1);
            REQUIRE(cten->Name() == "CIFM");
            REQUIRE(cten->IsConstant());
            auto v = cten->View();
            REQUIRE(v.Elements() == 128);
            BufferReader<int8_t> tensorReader = v.Values<int8_t>();
            int8_t j = 0;
            for ( const auto &e : tensorReader )
            {
                REQUIRE(e == j++);
            }
        }
    }
}
