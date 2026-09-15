//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "neural_ai_command_scheduler.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <set>

namespace regor::neuralai
{
namespace
{

constexpr int DMAQueues = 2;
constexpr int SystolicQueue = 2;
constexpr int QueueCount = 3;
constexpr int64_t InfiniteCycles = std::numeric_limits<int64_t>::max();

bool UsesResource(const SemanticCommand &command, SemanticResource resource)
{
    return (command.resources & ResourceMask(resource)) != 0;
}

bool IsDMA(const SemanticCommand &command)
{
    return UsesResource(command, SemanticResource::DMAExternalToLocal) ||
           UsesResource(command, SemanticResource::DMALocalToExternal);
}

bool IsSystolic(const SemanticCommand &command)
{
    return UsesResource(command, SemanticResource::Systolic);
}

bool QueueOrders(const SemanticCommand &predecessor, const SemanticCommand &successor)
{
    return predecessor.asyncCapable && successor.asyncCapable && IsDMA(predecessor) &&
           IsDMA(successor) && predecessor.queueId == successor.queueId;
}

bool IssueOrders(const SemanticCommand &predecessor, const SemanticCommand &successor)
{
    // Only a hardware FIFO provides a general issue-order guarantee. Memory,
    // state and control dependencies otherwise require completion ordering;
    // an engine-specific shadow register must not silently weaken that rule
    // because loading the shadow value can have additional runtime side
    // effects (for example a shared DMA transfer).
    return QueueOrders(predecessor, successor);
}

int AsyncQueue(const SemanticCommand &command)
{
    if ( IsDMA(command) && command.queueId < DMAQueues ) return int(command.queueId);
    if ( IsSystolic(command) && command.queueId == 0 ) return SystolicQueue;
    return -1;
}

CommandType SubmitType(CommandType type)
{
    if ( type == CommandType::DMA1D ) return CommandType::DMASubmit1D;
    if ( type == CommandType::DMA2D ) return CommandType::DMASubmit2D;
    if ( type == CommandType::DMA3D ) return CommandType::DMASubmit3D;
    if ( type == CommandType::LineBufferJob ) return CommandType::LineBufferSubmit;
    if ( type == CommandType::LineBufferBinary ) return CommandType::LineBufferBinarySubmit;
    return type;
}

void Write16(std::vector<uint8_t> &bytes, int offset, uint16_t value)
{
    bytes[offset] = uint8_t(value);
    bytes[offset + 1] = uint8_t(value >> 8);
}

void Write32(std::vector<uint8_t> &bytes, int offset, uint32_t value)
{
    for ( int byte = 0; byte < 4; ++byte ) bytes[offset + byte] = uint8_t(value >> (byte * 8));
}

uint32_t Read32(const std::vector<uint8_t> &bytes, int offset)
{
    return uint32_t(bytes[offset]) | (uint32_t(bytes[offset + 1]) << 8) |
           (uint32_t(bytes[offset + 2]) << 16) | (uint32_t(bytes[offset + 3]) << 24);
}

bool Overlaps(const SemanticMemoryAccess &lhs, const SemanticMemoryAccess &rhs)
{
    return lhs.region == rhs.region && lhs.index == rhs.index &&
           lhs.begin < rhs.end && rhs.begin < lhs.end;
}

std::vector<uint8_t> WaitEncoding(int queue, uint32_t layerId, uint32_t tileId)
{
    std::vector<uint8_t> bytes(sizeof(CommandHeaderV2) * 2, 0);
    const CommandType type = queue == SystolicQueue ?
        CommandType::SystolicWait : CommandType::DMAWait;
    Write16(bytes, 0, uint16_t(type));
    Write16(bytes, 2, uint16_t(bytes.size()));
    Write32(bytes, 8, layerId);
    Write32(bytes, 12, tileId);
    if ( queue < DMAQueues ) Write32(bytes, 16, uint32_t(queue));
    return bytes;
}

int64_t SaturatingAdd(int64_t lhs, int64_t rhs)
{
    if ( rhs > InfiniteCycles - lhs ) return InfiniteCycles;
    return lhs + rhs;
}

int64_t SaturatingMultiply(int64_t lhs, int64_t rhs)
{
    if ( lhs == 0 || rhs == 0 ) return 0;
    if ( lhs > InfiniteCycles / rhs ) return InfiniteCycles;
    return lhs * rhs;
}

uint32_t TCDMPhaseConflicts(const SemanticCommand &lhs, const SemanticCommand &rhs)
{
    uint32_t conflicts = 0;
    for ( const SemanticMemoryAccess &left : lhs.memoryAccesses )
    {
        if ( left.bankPhase == UINT16_MAX ) continue;
        for ( const SemanticMemoryAccess &right : rhs.memoryAccesses )
            if ( right.bankPhase == left.bankPhase ) ++conflicts;
    }
    return conflicts;
}

}  // namespace

bool OptimizeQParamResidency(std::vector<SemanticCommand> &commands,
    QParamResidencyStats &stats, std::string &error)
{
    stats = {};
    error.clear();
    std::vector<SemanticCommand> compacted;
    compacted.reserve(commands.size());
    uint64_t residentIdentity = 0;
    bool resident = false;
    for ( SemanticCommand &command : commands )
    {
        if ( command.type != CommandType::RQLoad )
        {
            compacted.push_back(std::move(command));
            continue;
        }
        ++stats.inputLoads;
        const auto quantization = std::find_if(command.stateAccesses.begin(),
            command.stateAccesses.end(), [](const SemanticStateAccess &access)
            {
                return access.state == SemanticState::Quantization &&
                       access.mode == SemanticAccessMode::Write;
            });
        if ( quantization == command.stateAccesses.end() || quantization->valueIdentity == 0 )
        {
            error = "Neural-AI qparam residency encountered an RQ_LOAD without an identity";
            return false;
        }
        if ( resident && residentIdentity == quantization->valueIdentity )
        {
            ++stats.redundantLoads;
            continue;
        }
        resident = true;
        residentIdentity = quantization->valueIdentity;
        ++stats.retainedLoads;
        compacted.push_back(std::move(command));
    }
    commands = std::move(compacted);
    for ( int index = 0; index < int(commands.size()); ++index )
        commands[index].sourceCommandIndex = uint32_t(index);
    return true;
}

bool OptimizeAFULutResidency(std::vector<SemanticCommand> &commands,
    AFULutResidencyStats &stats, std::string &error)
{
    stats = {};
    error.clear();
    uint64_t residentIdentity = 0;
    bool resident = false;
    for ( SemanticCommand &command : commands )
    {
        auto lutState = std::find_if(command.stateAccesses.begin(),
            command.stateAccesses.end(), [](const SemanticStateAccess &access)
            { return access.state == SemanticState::AFULut; });
        if ( command.type == CommandType::AFULut )
        {
            ++stats.commands;
            if ( lutState == command.stateAccesses.end() || lutState->valueIdentity == 0 )
            {
                error = "Neural-AI AFU LUT residency encountered a command without an identity";
                return false;
            }
            const bool encodedReuse = (command.flags & CommandFlagAFULutReuse) != 0;
            if ( encodedReuse && (!resident || residentIdentity != lutState->valueIdentity) )
            {
                error = "Neural-AI AFU LUT residency encountered an invalid resident-table reference";
                return false;
            }
            if ( encodedReuse || (resident && residentIdentity == lutState->valueIdentity) )
            {
                command.flags |= CommandFlagAFULutReuse;
                Write32(command.encoding, 4, command.flags);
                lutState->mode = SemanticAccessMode::Read;
                if ( command.memoryAccesses.size() == 3 ) command.memoryAccesses.pop_back();
                else if ( !encodedReuse )
                {
                    error = "Neural-AI AFU LUT residency encountered an invalid LUT memory access";
                    return false;
                }
                ++stats.reused;
            }
            else
            {
                command.flags &= ~CommandFlagAFULutReuse;
                Write32(command.encoding, 4, command.flags);
                lutState->mode = SemanticAccessMode::Write;
                ++stats.loads;
            }
            resident = true;
            residentIdentity = lutState->valueIdentity;
            continue;
        }
        if ( lutState != command.stateAccesses.end() &&
             lutState->mode == SemanticAccessMode::Write )
        {
            if ( resident ) ++stats.invalidations;
            resident = false;
            residentIdentity = 0;
        }
    }
    return true;
}

bool OptimizeDMAResidency(std::vector<SemanticCommand> &commands,
    DMAResidencyStats &stats, std::string &error)
{
    struct ResidentTransfer
    {
        CommandType type = CommandType::End;
        SemanticMemoryAccess source;
        SemanticMemoryAccess destination;
        std::vector<uint8_t> descriptor;
    };

    stats = {};
    error.clear();
    std::vector<ResidentTransfer> resident;
    std::vector<SemanticCommand> compacted;
    compacted.reserve(commands.size());
    for ( SemanticCommand &command : commands )
    {
        const bool load = (command.type == CommandType::DMA1D ||
                           command.type == CommandType::DMA2D ||
                           command.type == CommandType::DMA3D) &&
            command.queueId == uint32_t(DMADirection::ExternalToLocal) &&
            command.memoryAccesses.size() == 2 &&
            command.memoryAccesses[0].mode == SemanticAccessMode::Read &&
            command.memoryAccesses[1].mode == SemanticAccessMode::Write &&
            command.memoryAccesses[1].region == uint16_t(Region::TCDMScratch);
        std::vector<uint8_t> descriptor;
        if ( load )
        {
            ++stats.inputLoads;
            descriptor.assign(command.encoding.begin() + sizeof(CommandHeaderV2),
                command.encoding.end());
            const auto match = std::find_if(resident.begin(), resident.end(),
                [&](const ResidentTransfer &entry)
                {
                    const SemanticMemoryAccess &source = command.memoryAccesses[0];
                    const SemanticMemoryAccess &destination = command.memoryAccesses[1];
                    return entry.type == command.type && entry.descriptor == descriptor &&
                        entry.source.region == source.region && entry.source.index == source.index &&
                        entry.source.begin == source.begin && entry.source.end == source.end &&
                        entry.destination.region == destination.region &&
                        entry.destination.index == destination.index &&
                        entry.destination.begin == destination.begin &&
                        entry.destination.end == destination.end;
                });
            if ( match != resident.end() )
            {
                const uint64_t length = Read32(command.encoding, 32);
                uint64_t transfers = 1;
                if ( command.type == CommandType::DMA2D )
                    transfers = Read32(command.encoding, 44);
                else if ( command.type == CommandType::DMA3D )
                    transfers = uint64_t(Read32(command.encoding, 44)) *
                        Read32(command.encoding, 56);
                stats.redundantBytes += length * transfers;
                if ( command.memoryAccesses[0].region == uint16_t(Region::ModelConstants) )
                    ++stats.redundantConstantLoads;
                else
                    ++stats.redundantFeatureLoads;
                continue;
            }
        }

        for ( const SemanticMemoryAccess &access : command.memoryAccesses )
        {
            if ( access.mode != SemanticAccessMode::Write ) continue;
            resident.erase(std::remove_if(resident.begin(), resident.end(),
                [&](const ResidentTransfer &entry)
                { return Overlaps(access, entry.source) || Overlaps(access, entry.destination); }),
                resident.end());
        }
        if ( load )
        {
            ++stats.retainedLoads;
            if ( Overlaps(command.memoryAccesses[0], command.memoryAccesses[1]) )
            {
                error = "Neural-AI DMA residency encountered an overlapping load";
                return false;
            }
            resident.push_back({command.type, command.memoryAccesses[0],
                command.memoryAccesses[1], std::move(descriptor)});
        }
        compacted.push_back(std::move(command));
    }
    commands = std::move(compacted);
    for ( int index = 0; index < int(commands.size()); ++index )
        commands[index].sourceCommandIndex = uint32_t(index);
    return true;
}

bool ScheduleGlobalCommandStream(const std::vector<SemanticCommand> &commands,
    const CommandDependencyGraph &dependencies, std::vector<uint8_t> &scheduledBytes,
    GlobalCommandScheduleStats &stats, std::string &error)
{
    scheduledBytes.clear();
    stats = {};
    error.clear();
    if ( dependencies.predecessors.size() != commands.size() ||
         dependencies.successors.size() != commands.size() )
    {
        error = "Neural-AI global scheduler dependency graph size mismatch";
        return false;
    }
    if ( std::any_of(commands.begin(), commands.end(), [](const SemanticCommand &command)
         { return command.asynchronous || command.type == CommandType::DMAWait ||
                  command.type == CommandType::SystolicWait; }) )
    {
        error = "Neural-AI global scheduler requires a blocking input stream";
        return false;
    }

    const int count = int(commands.size());
    std::vector<int> indegree(count);
    std::vector<bool> emitted(count, false);
    std::vector<std::vector<uint32_t>> completionPredecessors(count);
    for ( int index = 0; index < count; ++index )
        indegree[index] = int(dependencies.predecessors[index].size());
    for ( const CommandDependency &edge : dependencies.edges )
    {
        if ( !IssueOrders(commands[edge.predecessor], commands[edge.successor]) )
            completionPredecessors[edge.successor].push_back(edge.predecessor);
    }

    std::vector<int64_t> criticalPath(count, 1);
    for ( int index = count; index-- > 0; )
    {
        int64_t successorPath = 0;
        for ( uint32_t successor : dependencies.successors[index] )
            successorPath = std::max(successorPath, criticalPath[successor]);
        criticalPath[index] = SaturatingAdd(
            std::max<int64_t>(1, commands[index].estimatedCycles), successorPath);
    }

    std::set<uint32_t> ready;
    for ( int index = 0; index < count; ++index )
        if ( indegree[index] == 0 ) ready.insert(uint32_t(index));

    std::array<std::vector<uint32_t>, QueueCount> pending;
    std::array<int64_t, QueueCount> queueTail{};
    std::vector<int> pendingQueue(count, -1);
    std::vector<int64_t> startTime(count, 0);
    std::vector<int64_t> completionTime(count, 0);
    std::vector<uint32_t> originalOrder;
    originalOrder.reserve(count);
    int64_t currentTime = 0;

    const auto pendingCount = [&]()
    {
        int result = 0;
        for ( const auto &queue : pending ) result += int(queue.size());
        return result;
    };
    const auto queuesBlocking = [&](uint32_t index)
    {
        std::array<bool, QueueCount> blocked{};
        const SemanticCommand &candidate = commands[index];
        if ( candidate.controlFence )
            for ( int queue = 0; queue < QueueCount; ++queue ) blocked[queue] = !pending[queue].empty();
        for ( uint32_t predecessor : completionPredecessors[index] )
        {
            const int queue = pendingQueue[predecessor];
            if ( queue >= 0 ) blocked[queue] = true;
        }
        if ( IsSystolic(candidate) && !pending[SystolicQueue].empty() )
            blocked[SystolicQueue] = true;
        const int queue = AsyncQueue(candidate);
        if ( queue >= 0 && int(pending[queue].size()) >= int(candidate.queueCapacity) )
            blocked[queue] = true;
        return blocked;
    };
    const auto hasBlockedQueue = [](const std::array<bool, QueueCount> &blocked)
    {
        return std::any_of(blocked.begin(), blocked.end(), [](bool value) { return value; });
    };
    const auto bankPenaltyRejects = [&](uint32_t index)
    {
        if ( pendingCount() == 0 ) return false;
        const SemanticCommand &candidate = commands[index];
        const int64_t candidateCycles = std::max<int64_t>(1, candidate.estimatedCycles);
        int64_t hiddenCycles = 0;
        int64_t predictedStall = 0;
        const int64_t candidateEnd = SaturatingAdd(currentTime, candidateCycles);
        for ( const auto &queue : pending )
        {
            for ( uint32_t active : queue )
            {
                const int64_t overlapBegin = std::max(currentTime, startTime[active]);
                const int64_t overlapEnd = std::min(candidateEnd, completionTime[active]);
                if ( overlapBegin >= overlapEnd ) continue;
                const int64_t overlapCycles = overlapEnd - overlapBegin;
                const uint32_t conflicts = TCDMPhaseConflicts(candidate, commands[active]);
                const int64_t weightedConflicts = SaturatingAdd(
                    SaturatingMultiply(overlapCycles, conflicts), 15) / 16;
                predictedStall = SaturatingAdd(predictedStall, weightedConflicts);
                hiddenCycles = std::max(hiddenCycles, overlapEnd - currentTime);
            }
        }
        if ( hiddenCycles == 0 || predictedStall == 0 ) return false;
        return predictedStall >= hiddenCycles;
    };
    const auto canOverlapAfter = [&](uint32_t asynchronous, uint32_t candidate)
    {
        if ( commands[candidate].controlFence ) return false;
        if ( IsSystolic(commands[asynchronous]) && IsSystolic(commands[candidate]) ) return false;
        if ( std::find(completionPredecessors[candidate].begin(),
                 completionPredecessors[candidate].end(), asynchronous) !=
             completionPredecessors[candidate].end() ) return false;
        return TCDMPhaseConflicts(commands[asynchronous], commands[candidate]) < 16;
    };
    const auto hasOverlapOpportunity = [&](uint32_t index)
    {
        for ( uint32_t candidate : ready )
            if ( candidate != index && canOverlapAfter(index, candidate) ) return true;
        for ( uint32_t successor : dependencies.successors[index] )
        {
            if ( !canOverlapAfter(index, successor) ) continue;
            const bool unlocks = std::all_of(dependencies.predecessors[successor].begin(),
                dependencies.predecessors[successor].end(), [&](uint32_t predecessor)
                { return predecessor == index || emitted[predecessor]; });
            if ( unlocks ) return true;
        }
        return false;
    };
    const auto emitWait = [&](int queue)
    {
        const uint32_t owner = pending[queue].front();
        const int64_t tail = std::max(currentTime, queueTail[queue]);
        currentTime = SaturatingAdd(currentTime, std::max<int64_t>(1, tail - currentTime));
        const std::vector<uint8_t> wait = WaitEncoding(
            queue, commands[owner].layerId, commands[owner].tileId);
        scheduledBytes.insert(scheduledBytes.end(), wait.begin(), wait.end());
        for ( uint32_t active : pending[queue] ) pendingQueue[active] = -1;
        pending[queue].clear();
        queueTail[queue] = currentTime;
        ++stats.insertedWaits;
    };

    int emittedCommands = 0;
    while ( emittedCommands < count )
    {
        int selected = -1;
        int64_t selectedPriority = -1;
        bool sawBankDeferral = false;
        for ( uint32_t candidate : ready )
        {
            const auto blocked = queuesBlocking(candidate);
            if ( hasBlockedQueue(blocked) ) continue;
            if ( bankPenaltyRejects(candidate) )
            {
                sawBankDeferral = true;
                continue;
            }
            int64_t priority = criticalPath[candidate];
            if ( pendingCount() == 0 )
            {
                const bool startsOverlap = commands[candidate].asyncCapable &&
                    hasOverlapOpportunity(candidate);
                priority = startsOverlap ? InfiniteCycles - candidate : -int64_t(candidate);
            }
            if ( selected < 0 || priority > selectedPriority ||
                 (priority == selectedPriority && candidate < uint32_t(selected)) )
            {
                selected = int(candidate);
                selectedPriority = priority;
            }
        }
        if ( selected < 0 )
        {
            if ( pendingCount() == 0 )
            {
                error = "Neural-AI global scheduler found no ready command";
                return false;
            }
            int waitQueue = -1;
            int64_t bestPriority = -1;
            for ( uint32_t candidate : ready )
            {
                const auto blocked = queuesBlocking(candidate);
                for ( int queue = 0; queue < QueueCount; ++queue )
                {
                    if ( !blocked[queue] || pending[queue].empty() ) continue;
                    if ( criticalPath[candidate] > bestPriority )
                    {
                        waitQueue = queue;
                        bestPriority = criticalPath[candidate];
                    }
                }
            }
            if ( waitQueue < 0 )
            {
                for ( int queue = 0; queue < QueueCount; ++queue )
                    if ( !pending[queue].empty() &&
                         (waitQueue < 0 || queueTail[queue] < queueTail[waitQueue]) )
                        waitQueue = queue;
            }
            if ( waitQueue < 0 )
            {
                error = "Neural-AI global scheduler could not resolve pending work";
                return false;
            }
            if ( sawBankDeferral ) ++stats.bankConflictDeferrals;
            emitWait(waitQueue);
            continue;
        }

        const uint32_t index = uint32_t(selected);
        ready.erase(index);
        SemanticCommand scheduled = commands[index];
        if ( scheduled.type == CommandType::RQLoad && !pending[SystolicQueue].empty() )
            ++stats.qparamPreloads;
        const int queue = AsyncQueue(scheduled);
        const bool submit = scheduled.asyncCapable && queue >= 0 &&
            (pendingCount() != 0 || hasOverlapOpportunity(index));
        if ( submit )
        {
            const CommandType submitType = SubmitType(scheduled.type);
            if ( submitType == scheduled.type )
            {
                error = "Neural-AI global scheduler has no submit encoding for an asynchronous command";
                return false;
            }
            scheduled.type = submitType;
            scheduled.asynchronous = true;
            Write16(scheduled.encoding, 0, uint16_t(submitType));
            const int64_t start = std::max(currentTime, queueTail[queue]);
            startTime[index] = start;
            completionTime[index] = SaturatingAdd(start,
                std::max<int64_t>(1, scheduled.estimatedCycles));
            queueTail[queue] = completionTime[index];
            pending[queue].push_back(index);
            pendingQueue[index] = queue;
            currentTime = SaturatingAdd(currentTime, 1);
            ++stats.asynchronousSubmits;
        }
        else
        {
            currentTime = SaturatingAdd(currentTime,
                std::max<int64_t>(1, scheduled.estimatedCycles));
        }
        scheduledBytes.insert(scheduledBytes.end(),
            scheduled.encoding.begin(), scheduled.encoding.end());
        originalOrder.push_back(index);
        emitted[index] = true;
        ++emittedCommands;
        for ( uint32_t successor : dependencies.successors[index] )
        {
            if ( --indegree[successor] == 0 ) ready.insert(successor);
        }
    }
    for ( int queue = 0; queue < QueueCount; ++queue )
        if ( !pending[queue].empty() ) emitWait(queue);

    for ( int position = 0; position < count; ++position )
        if ( originalOrder[position] != uint32_t(position) ) ++stats.reorderedCommands;
    stats.estimatedCycles = currentTime;
    return true;
}

}  // namespace regor::neuralai
