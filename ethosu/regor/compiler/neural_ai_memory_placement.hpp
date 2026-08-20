//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace regor
{

struct SchedulerTensor;

// Returns true when every producer and consumer of an internal tensor has a
// reviewed Neural-AI L2 staging path. This is deliberately conservative: it
// describes placement eligibility, but does not mutate scheduler memory roles.
bool IsNeuralAIL2SpillCandidate(const SchedulerTensor *tensor);

}  // namespace regor
