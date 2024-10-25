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

#include "optimiser_utils.hpp"

#include "operation.hpp"

namespace regor::GraphOptimisation
{

// Find specified tensor in Inputs() / Outputs() vectors.
// returns true if found in given vector.
bool IsTensorInVector(const std::vector<std::shared_ptr<Tensor>> &tensorVec, const Tensor *const tensorToFind)
{
    auto pos = std::find_if(
        tensorVec.begin(), tensorVec.end(), [&](const std::shared_ptr<Tensor> &t) { return t.get() == tensorToFind; });

    return (pos != tensorVec.end());
}

// Insert a MemoryCopy operation after given ifm tensor. Returns a copy op shared_ptr.
// Will make a clone of ifm as ofm  and connects any other consumers of the ifm to it.
std::shared_ptr<Operation> InsertCopyOpAfterTensor(const std::shared_ptr<Tensor> &ifm, const Quantization &quantization)
{
    std::shared_ptr<Tensor> copyTensor = ifm->Clone();
    auto copyOp = std::make_shared<Operation>(OpType::MemoryCopy);
    copyOp->ConnectInput(TensorUsage::IFM0, ifm).Set(quantization);
    auto name = ifm->Name();
    name.append("_copy");
    copyTensor->SetName(name);
    copyOp->ConnectOutput(TensorUsage::OFM, copyTensor).Set(quantization);

    std::vector<std::shared_ptr<Operation>> ifmReaders(ifm->Readers());
    for ( const auto &opReader : ifmReaders )
    {
        auto *cons = opReader.get();
        if ( cons != copyOp.get() )
        {
            auto idx = 0;
            auto usage = MakeTensorUsage(TensorUsage::IFM, 0);
            auto *consIfmConn = cons->Input(usage);

            while ( consIfmConn != nullptr )
            {
                if ( consIfmConn->tensor.get() == ifm.get() )
                {
                    cons->ConnectInput(usage, copyTensor).Set(quantization);
                }
                usage = MakeTensorUsage(TensorUsage::IFM, ++idx);
                consIfmConn = cons->Input(usage);
            }
        }
    }
    return copyOp;
}

// Connects output to operations in given list. Will not replace connection shape.
// Parameters:
// - producerList: List of producers.
// - tensorToReplace: if OFM on consumer match this tensor, replace it.
// - newTensor: The new output tensor to connect.
void ReplaceProducerOutput(std::vector<std::shared_ptr<Operation>> producerList, const Tensor *const tensorToReplace,
    std::shared_ptr<Tensor> newTensor)
{
    // Not passed by reference. Original can be modified in loop.
    for ( const auto &producer : producerList )
    {
        Operation *prod = producer.get();
        auto idx = 0;
        auto usage = MakeTensorUsage(TensorUsage::OFM, 0);
        auto prodOfmConn = prod->Output(usage);

        while ( prodOfmConn != nullptr )
        {
            if ( prodOfmConn->tensor.get() == tensorToReplace )
            {
                // Do not want to replace the shape. Only the tensor and add writers.
                // As ConnectOutput but do not replace shape.
                newTensor->AddWriter(prod->shared_from_this());
                if ( prodOfmConn->tensor != newTensor )
                {
                    prodOfmConn->tensor->RemoveWriter(prod->shared_from_this());
                }
                prodOfmConn->tensor = newTensor;
            }
            usage = MakeTensorUsage(TensorUsage::OFM, ++idx);
            prodOfmConn = prod->Output(usage);
        }
    }
}


// Connects input to operations in given list. Will not replace connection shape.
// Parameters:
// - exemptOperation: operation to exempt.
// - consumerList: List of consumers.
// - tensorToReplace: if input tensor on consumer match this tensor, replace it.
// - newTensor: The new input tensor to connect.
void ReplaceConsumerInput(const Operation *const exemptOperation, std::vector<std::shared_ptr<Operation>> consumerList,
    const Tensor *const tensorToReplace, std::shared_ptr<Tensor> newTensor)
{
    // Not passed by reference. Original can be modified in loop.
    for ( const auto &consumer : consumerList )
    {
        if ( consumer.get() == exemptOperation ) continue;

        for ( const auto &consInput : consumer->Inputs().pairs() )
        {
            if ( consInput.second.tensor.get() == tensorToReplace )
            {
                // Do not want to replace the shape. Only the tensor and add writers.
                // As ConnectInput but do not replace shape.
                newTensor->AddReader(consumer);
                auto *consInputConnection = consumer->Input(consInput.first);
                if ( consInputConnection->tensor != newTensor )
                {
                    consInputConnection->tensor->RemoveReader(consumer);
                    consInputConnection->tensor = newTensor;
                }
            }
        }
    }
}

}  // namespace regor::GraphOptimisation
