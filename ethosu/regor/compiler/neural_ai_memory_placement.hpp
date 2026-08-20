//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "architecture/architecture.hpp"

#include <memory>
#include <vector>

namespace regor
{

struct SchedulerTensor;
class SchedulerOperation;

struct NeuralAIMemoryPlacementStats
{
    int l2Tensors = 0;
    int tcdmTensors = 0;
};

// Returns true when every producer and consumer of an internal tensor has a
// reviewed Neural-AI L2 staging path. This is deliberately conservative: it
// describes placement eligibility, but does not mutate scheduler memory roles.
bool IsNeuralAIL2SpillCandidate(const SchedulerTensor *tensor);

// Applies the conservative policy to complete equivalence groups. A group is
// placed in L2 only when every nonconstant member is independently eligible.
NeuralAIMemoryPlacementStats ApplyNeuralAIMemoryPlacement(
    const std::vector<std::unique_ptr<SchedulerOperation>> &operations,
    const MemArea &l2Memory, const MemArea &tcdmMemory);

}  // namespace regor
