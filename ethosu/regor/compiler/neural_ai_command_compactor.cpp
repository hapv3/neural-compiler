//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "neural_ai_command_compactor.hpp"

#include "architecture/neuralai/neural_ai_abi.hpp"

#include <algorithm>
#include <limits>
#include <optional>

namespace regor::neuralai
{
namespace
{

uint16_t Read16(const std::vector<uint8_t> &bytes, uint32_t offset)
{
    return uint16_t(bytes[offset]) | uint16_t(uint16_t(bytes[offset + 1]) << 8);
}

uint32_t Read32(const std::vector<uint8_t> &bytes, uint32_t offset)
{
    return uint32_t(bytes[offset]) | (uint32_t(bytes[offset + 1]) << 8) |
           (uint32_t(bytes[offset + 2]) << 16) | (uint32_t(bytes[offset + 3]) << 24);
}

void Write16(std::vector<uint8_t> &bytes, uint32_t offset, uint16_t value)
{
    bytes[offset] = uint8_t(value);
    bytes[offset + 1] = uint8_t(value >> 8);
}

void Write32(std::vector<uint8_t> &bytes, uint32_t offset, uint32_t value)
{
    for ( int byte = 0; byte < 4; ++byte ) bytes[offset + byte] = uint8_t(value >> (byte * 8));
}

uint32_t Align32(uint32_t value)
{
    return (value + Alignment - 1) & ~(Alignment - 1);
}

struct Patch
{
    uint32_t bodyWordOffset = 0;
    uint32_t delta = 0;
};

struct Candidate
{
    uint32_t begin = 0;
    uint32_t end = 0;
    uint32_t bodyCommands = 0;
    uint32_t iterations = 0;
    uint32_t bodyBytes = 0;
    uint32_t descriptorBytes = 0;
    uint32_t savedBytes = 0;
    uint32_t encodedReduction = 0;
    std::vector<Patch> patches;
};

struct Score
{
    uint64_t savedBytes = 0;
    uint32_t encodedReduction = 0;
};

bool Better(const Score &lhs, const Score &rhs)
{
    return lhs.savedBytes > rhs.savedBytes ||
           (lhs.savedBytes == rhs.savedBytes && lhs.encodedReduction > rhs.encodedReduction);
}

std::optional<Candidate> BuildCandidate(const std::vector<SemanticCommand> &commands,
    uint32_t begin, uint32_t bodyCommands, uint32_t executableCommands)
{
    if ( begin + bodyCommands * 2 > executableCommands ) return std::nullopt;
    Candidate candidate;
    candidate.begin = begin;
    candidate.bodyCommands = bodyCommands;
    candidate.iterations = 2;
    std::vector<std::vector<uint32_t>> deltas;
    uint32_t bodyWordBase = 0;
    for ( uint32_t child = 0; child < bodyCommands; ++child )
    {
        const auto &first = commands[begin + child].encoding;
        const auto &second = commands[begin + bodyCommands + child].encoding;
        if ( first.size() != second.size() || first.size() % 4 != 0 ||
             commands[begin + child].type != commands[begin + bodyCommands + child].type )
            return std::nullopt;
        std::vector<uint32_t> childDeltas(first.size() / 4, 0);
        for ( uint32_t word = 0; word < childDeltas.size(); ++word )
        {
            childDeltas[word] = Read32(second, word * 4) - Read32(first, word * 4);
            if ( childDeltas[word] == 0 ) continue;
            if ( word < 2 || candidate.patches.size() == MaxAffineLoopPatches )
                return std::nullopt;
            candidate.patches.push_back({bodyWordBase + word, childDeltas[word]});
        }
        candidate.bodyBytes += uint32_t(first.size());
        bodyWordBase += uint32_t(first.size() / 4);
        deltas.push_back(std::move(childDeltas));
    }
    candidate.descriptorBytes = Align32(uint32_t(sizeof(CommandAffineLoopV2) +
        candidate.patches.size() * sizeof(CommandAffinePatchV2)));
    if ( candidate.descriptorBytes + candidate.bodyBytes > MaxAffineLoopRecordBytes )
        return std::nullopt;

    while ( begin + (candidate.iterations + 1) * bodyCommands <= executableCommands )
    {
        bool affine = true;
        for ( uint32_t child = 0; child < bodyCommands && affine; ++child )
        {
            const auto &previous = commands[begin + (candidate.iterations - 1) * bodyCommands + child];
            const auto &next = commands[begin + candidate.iterations * bodyCommands + child];
            if ( previous.type != next.type || previous.encoding.size() != next.encoding.size() )
            {
                affine = false;
                break;
            }
            for ( uint32_t word = 0; word < deltas[child].size(); ++word )
            {
                if ( Read32(next.encoding, word * 4) -
                         Read32(previous.encoding, word * 4) != deltas[child][word] )
                {
                    affine = false;
                    break;
                }
            }
        }
        if ( !affine ) break;
        ++candidate.iterations;
    }
    const uint64_t repeatedBytes = uint64_t(candidate.iterations - 1) * candidate.bodyBytes;
    if ( repeatedBytes <= candidate.descriptorBytes ) return std::nullopt;
    const uint64_t saved = repeatedBytes - candidate.descriptorBytes;
    if ( saved == 0 || saved > std::numeric_limits<uint32_t>::max() ) return std::nullopt;
    candidate.savedBytes = uint32_t(saved);
    candidate.encodedReduction = (candidate.iterations - 1) * bodyCommands - 1;
    candidate.end = begin + candidate.iterations * bodyCommands;
    return candidate;
}

bool ValidateBody(const std::vector<uint8_t> &encoded, uint32_t bodyOffset,
    uint32_t bodyBytes, uint32_t bodyCommands, std::vector<uint32_t> &commandOffsets)
{
    commandOffsets.clear();
    uint32_t offset = bodyOffset;
    const uint32_t end = bodyOffset + bodyBytes;
    for ( uint32_t child = 0; child < bodyCommands; ++child )
    {
        if ( offset > end || end - offset < sizeof(CommandHeaderV2) ) return false;
        const uint16_t type = Read16(encoded, offset);
        const uint16_t size = Read16(encoded, offset + 2);
        if ( type == uint16_t(CommandType::End) || type == uint16_t(CommandType::AffineLoop) ||
             size < sizeof(CommandHeaderV2) || size % Alignment != 0 || size > end - offset )
            return false;
        commandOffsets.push_back(offset - bodyOffset);
        offset += size;
    }
    return offset == end;
}

}  // namespace

bool CompactAffineCommandStream(const std::vector<SemanticCommand> &commands,
    std::vector<uint8_t> &encoded, CommandCompactionStats &stats, std::string &error)
{
    encoded.clear();
    stats = {};
    error.clear();
    if ( commands.empty() || commands.back().type != CommandType::End )
    {
        error = "Neural-AI command compaction requires a final END command";
        return false;
    }
    const uint32_t executableCommands = uint32_t(commands.size() - 1);
    stats.logicalCommands = executableCommands;
    std::vector<std::vector<Candidate>> candidates(executableCommands);
    for ( uint32_t begin = 0; begin < executableCommands; ++begin )
    {
        for ( uint32_t body = 1; body <= MaxAffineLoopBodyCommands; ++body )
        {
            auto candidate = BuildCandidate(commands, begin, body, executableCommands);
            if ( candidate ) candidates[begin].push_back(std::move(*candidate));
        }
    }

    std::vector<Score> best(executableCommands + 1);
    std::vector<uint32_t> previous(executableCommands + 1, 0);
    std::vector<int32_t> selected(executableCommands + 1, -1);
    for ( uint32_t begin = 0; begin < executableCommands; ++begin )
    {
        if ( !Better(best[begin + 1], best[begin]) )
        {
            best[begin + 1] = best[begin];
            previous[begin + 1] = begin;
            selected[begin + 1] = -1;
        }
        for ( int candidateIndex = 0; candidateIndex < int(candidates[begin].size()); ++candidateIndex )
        {
            const Candidate &candidate = candidates[begin][candidateIndex];
            const Score score{best[begin].savedBytes + candidate.savedBytes,
                best[begin].encodedReduction + candidate.encodedReduction};
            if ( Better(score, best[candidate.end]) )
            {
                best[candidate.end] = score;
                previous[candidate.end] = begin;
                selected[candidate.end] = candidateIndex;
            }
        }
    }
    std::vector<const Candidate *> chosen(executableCommands, nullptr);
    for ( uint32_t end = executableCommands; end != 0; )
    {
        const uint32_t begin = previous[end];
        if ( selected[end] >= 0 ) chosen[begin] = &candidates[begin][selected[end]];
        end = begin;
    }

    for ( const auto &command : commands ) stats.bytesBefore += uint32_t(command.encoding.size());
    for ( uint32_t commandIndex = 0; commandIndex < executableCommands; )
    {
        const Candidate *candidate = chosen[commandIndex];
        if ( candidate == nullptr )
        {
            encoded.insert(encoded.end(), commands[commandIndex].encoding.begin(),
                commands[commandIndex].encoding.end());
            ++stats.encodedCommands;
            ++commandIndex;
            continue;
        }
        const uint32_t loopOffset = uint32_t(encoded.size());
        encoded.resize(loopOffset + candidate->descriptorBytes, 0);
        Write16(encoded, loopOffset, uint16_t(CommandType::AffineLoop));
        Write16(encoded, loopOffset + 2, uint16_t(candidate->descriptorBytes));
        Write32(encoded, loopOffset + 8, commands[commandIndex].layerId);
        Write32(encoded, loopOffset + 12, commands[commandIndex].tileId);
        Write32(encoded, loopOffset + 16, candidate->iterations);
        Write32(encoded, loopOffset + 20, candidate->bodyCommands);
        Write32(encoded, loopOffset + 24, candidate->bodyBytes);
        Write32(encoded, loopOffset + 28, uint32_t(candidate->patches.size()));
        uint32_t patchOffset = loopOffset + sizeof(CommandAffineLoopV2);
        for ( const Patch &patch : candidate->patches )
        {
            Write32(encoded, patchOffset, patch.bodyWordOffset);
            Write32(encoded, patchOffset + 4, patch.delta);
            patchOffset += sizeof(CommandAffinePatchV2);
        }
        for ( uint32_t child = 0; child < candidate->bodyCommands; ++child )
        {
            const auto &body = commands[commandIndex + child].encoding;
            encoded.insert(encoded.end(), body.begin(), body.end());
        }
        stats.encodedCommands += candidate->bodyCommands + 1;
        ++stats.loops;
        stats.loopedCommands += candidate->bodyCommands * candidate->iterations;
        commandIndex = candidate->end;
    }
    encoded.insert(encoded.end(), commands.back().encoding.begin(), commands.back().encoding.end());
    stats.bytesAfter = uint32_t(encoded.size());
    if ( stats.bytesBefore < stats.bytesAfter ||
         stats.bytesBefore - stats.bytesAfter != best.back().savedBytes )
    {
        error = "Neural-AI command compaction accounting mismatch";
        return false;
    }
    return true;
}

bool ExpandAffineCommandStream(const std::vector<uint8_t> &encoded,
    std::vector<uint8_t> &expanded, uint32_t &logicalCommands, std::string &error)
{
    expanded.clear();
    logicalCommands = 0;
    error.clear();
    uint32_t offset = 0;
    bool ended = false;
    while ( offset < encoded.size() )
    {
        if ( encoded.size() - offset < sizeof(CommandHeaderV2) ) break;
        const CommandType type = CommandType(Read16(encoded, offset));
        const uint32_t size = Read16(encoded, offset + 2);
        if ( size < sizeof(CommandHeaderV2) || size % Alignment != 0 ||
             size > encoded.size() - offset )
            break;
        if ( type != CommandType::AffineLoop )
        {
            expanded.insert(expanded.end(), encoded.begin() + offset,
                encoded.begin() + offset + size);
            offset += size;
            if ( type == CommandType::End )
            {
                ended = offset == encoded.size();
                break;
            }
            ++logicalCommands;
            continue;
        }
        if ( size < sizeof(CommandAffineLoopV2) || Read32(encoded, offset + 4) != 0 ) break;
        const uint32_t iterations = Read32(encoded, offset + 16);
        const uint32_t bodyCommands = Read32(encoded, offset + 20);
        const uint32_t bodyBytes = Read32(encoded, offset + 24);
        const uint32_t patchCount = Read32(encoded, offset + 28);
        if ( iterations < 2 || bodyCommands == 0 || bodyCommands > MaxAffineLoopBodyCommands ||
             patchCount > MaxAffineLoopPatches )
            break;
        const uint32_t expectedSize = Align32(uint32_t(sizeof(CommandAffineLoopV2) +
            patchCount * sizeof(CommandAffinePatchV2)));
        if ( size != expectedSize ||
             size + bodyBytes > MaxAffineLoopRecordBytes ||
             size + bodyBytes > encoded.size() - offset )
            break;
        std::vector<uint32_t> commandOffsets;
        const uint32_t bodyOffset = offset + size;
        if ( !ValidateBody(encoded, bodyOffset, bodyBytes, bodyCommands, commandOffsets) ) break;
        std::vector<uint8_t> body(encoded.begin() + bodyOffset,
            encoded.begin() + bodyOffset + bodyBytes);
        uint32_t previousPatch = 0;
        bool validPatches = true;
        for ( uint32_t patchIndex = 0; patchIndex < patchCount; ++patchIndex )
        {
            const uint32_t patchOffset = offset + sizeof(CommandAffineLoopV2) +
                patchIndex * sizeof(CommandAffinePatchV2);
            const uint32_t word = Read32(encoded, patchOffset);
            const uint32_t byte = word * 4;
            if ( word > std::numeric_limits<uint32_t>::max() / 4 || byte > bodyBytes ||
                 bodyBytes - byte < 4 ||
                 (patchIndex != 0 && word <= previousPatch) )
            {
                validPatches = false;
                break;
            }
            bool patchable = false;
            for ( uint32_t childOffset : commandOffsets )
            {
                const uint32_t childBytes = Read16(body, childOffset + 2);
                if ( byte >= childOffset + 8 && byte + 4 <= childOffset + childBytes )
                {
                    patchable = true;
                    break;
                }
            }
            if ( !patchable )
            {
                validPatches = false;
                break;
            }
            previousPatch = word;
        }
        if ( !validPatches ) break;
        for ( uint32_t iteration = 0; iteration < iterations; ++iteration )
        {
            expanded.insert(expanded.end(), body.begin(), body.end());
            logicalCommands += bodyCommands;
            for ( uint32_t patchIndex = 0; patchIndex < patchCount; ++patchIndex )
            {
                const uint32_t patchOffset = offset + sizeof(CommandAffineLoopV2) +
                    patchIndex * sizeof(CommandAffinePatchV2);
                const uint32_t byte = Read32(encoded, patchOffset) * 4;
                Write32(body, byte, Read32(body, byte) + Read32(encoded, patchOffset + 4));
            }
        }
        offset += size + bodyBytes;
    }
    if ( !ended )
    {
        expanded.clear();
        logicalCommands = 0;
        error = "Neural-AI affine command stream is malformed";
        return false;
    }
    return true;
}

}  // namespace regor::neuralai
