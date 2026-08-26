//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

namespace regor::neuralai
{

struct MemoryContent
{
    uint64_t identity = 0;
    uint32_t generation = 0;

    bool operator==(const MemoryContent &other) const
    {
        return identity == other.identity && generation == other.generation;
    }
};

// Tracks the logical content held by byte ranges in a command-visible memory.
// The content offset makes subranges and local-to-local copies comparable
// without requiring them to reside at the same physical address.
class CommandMemoryState
{
public:
    void Clear() { _spans.clear(); }

    void Invalidate(uint32_t address, uint32_t size)
    {
        Replace(address, size, nullptr, 0);
    }

    void Write(uint32_t address, uint32_t size, MemoryContent content,
        uint64_t contentOffset = 0)
    {
        Replace(address, size, &content, contentOffset);
    }

    bool Contains(uint32_t address, uint32_t size, MemoryContent content,
        uint64_t contentOffset = 0) const
    {
        if ( size == 0 ) return true;
        const uint64_t begin = address;
        const uint64_t end = begin + size;
        uint64_t cursor = begin;
        for ( const Span &span : _spans )
        {
            if ( span.end <= cursor ) continue;
            if ( span.begin > cursor || !(span.content == content) ) return false;
            const uint64_t expected = contentOffset + cursor - begin;
            const uint64_t actual = span.contentOffset + cursor - span.begin;
            if ( actual != expected ) return false;
            cursor = std::min(end, span.end);
            if ( cursor == end ) return true;
        }
        return false;
    }

    std::optional<uint32_t> Find(MemoryContent content, uint64_t contentOffset,
        uint32_t size) const
    {
        if ( size == 0 ) return uint32_t(0);
        const uint64_t contentEnd = contentOffset + size;
        for ( const Span &span : _spans )
        {
            const uint64_t spanContentEnd = span.contentOffset + span.end - span.begin;
            if ( span.content == content && contentOffset >= span.contentOffset &&
                 contentEnd <= spanContentEnd )
            {
                const uint64_t address = span.begin + contentOffset - span.contentOffset;
                if ( address <= UINT32_MAX ) return uint32_t(address);
            }
        }
        return std::nullopt;
    }

private:
    struct Span
    {
        uint64_t begin = 0;
        uint64_t end = 0;
        MemoryContent content;
        uint64_t contentOffset = 0;
    };

    void Replace(uint32_t address, uint32_t size, const MemoryContent *content,
        uint64_t contentOffset)
    {
        if ( size == 0 ) return;
        const uint64_t begin = address;
        const uint64_t end = begin + size;
        std::vector<Span> next;
        next.reserve(_spans.size() + 1);
        for ( const Span &span : _spans )
        {
            if ( span.end <= begin || span.begin >= end )
            {
                next.push_back(span);
                continue;
            }
            if ( span.begin < begin )
            {
                Span left = span;
                left.end = begin;
                next.push_back(left);
            }
            if ( span.end > end )
            {
                Span right = span;
                right.contentOffset += end - span.begin;
                right.begin = end;
                next.push_back(right);
            }
        }
        if ( content != nullptr ) next.push_back({begin, end, *content, contentOffset});
        std::sort(next.begin(), next.end(), [](const Span &lhs, const Span &rhs)
            { return lhs.begin < rhs.begin; });
        _spans.clear();
        for ( const Span &span : next )
        {
            if ( !_spans.empty() )
            {
                Span &last = _spans.back();
                if ( last.end == span.begin && last.content == span.content &&
                     last.contentOffset + last.end - last.begin == span.contentOffset )
                {
                    last.end = span.end;
                    continue;
                }
            }
            _spans.push_back(span);
        }
    }

    std::vector<Span> _spans;
};

}  // namespace regor::neuralai
