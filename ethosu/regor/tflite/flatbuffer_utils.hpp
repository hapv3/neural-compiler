//
// SPDX-FileCopyrightText: Copyright 2021, 2023, 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#pragma once

#include "common/logging.hpp"

#include <flatbuffers/flatbuffers.h>
#include <unordered_map>

namespace FlatbufferUtils
{

static flatbuffers::Offset<> CopyTable(flatbuffers::FlatBufferBuilder &destination, const flatbuffers::Table *source,
    const flatbuffers::TypeTable *typeTable);

// Load a vector (if present) from a flatbuffer into a local copy.
// Intended for small vectors only - large vectors should be left in place and mapped using a Buffer class instead.
template<typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
static std::vector<T> LoadVector(const flatbuffers::Vector<T> *source)
{
    std::vector<T> destination;
    if ( source )
    {
        destination.insert(destination.begin(), source->begin(), source->end());
    }
    return destination;
}

// Copy a vector of scalars (if present) from one flatbuffer to another
template<typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
static flatbuffers::Offset<flatbuffers::Vector<T>>
CopyVector(flatbuffers::FlatBufferBuilder &destination, const flatbuffers::Vector<T> *source)
{
    if ( source )
    {
        return destination.CreateVector<T>(source->data(), source->size());
    }

    // Zero offset means unset or non-existing vector
    return 0;
}

// Make a copy of a repeating table field, by field
static flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<>>> CopyVectorOfTables(flatbuffers::FlatBufferBuilder &destination,
    const flatbuffers::Table *source, flatbuffers::voffset_t field, const flatbuffers::TypeTable *types)
{
    std::vector<flatbuffers::Offset<>> dstVector;
    if ( source )
    {
        const auto *srcVector = source->GetPointer<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::Table>> *>(field);
        if ( srcVector )
        {
            for ( const auto *table : *srcVector )
            {
                dstVector.push_back(CopyTable(destination, table, types));
            }
        }
    }

    // Create a new vector
    return destination.CreateVector(dstVector);
}

// Make a copy of a repeating string field, by field
static flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> CopyVectorOfStrings(
    flatbuffers::FlatBufferBuilder &destination, const flatbuffers::Table *source, flatbuffers::voffset_t field)
{
    std::vector<flatbuffers::Offset<flatbuffers::String>> dstVector;
    if ( source )
    {
        const auto srcVector = source->GetPointer<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(field);
        if ( srcVector )
        {
            for ( const auto *str : *srcVector )
            {
                dstVector.push_back(destination.CreateString(str));
            }
        }
    }

    // Create a new vector
    return destination.CreateVector(dstVector);
}

// Copy a vector of scalars (if present) from one flatbuffer to another, by field
template<typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
static flatbuffers::Offset<flatbuffers::Vector<T>>
CopyVector(flatbuffers::FlatBufferBuilder &destination, const flatbuffers::Table *source, flatbuffers::voffset_t field)
{
    if ( source && source->CheckField(field) )
    {
        return CopyVector<T>(destination, source->GetPointer<flatbuffers::Vector<T> *>(field));
    }

    // Zero offset means unset or non-existing vector
    return 0;
}

// Copy a vector of scalars (if present) from one flatbuffer to another, by field
static flatbuffers::Offset<> CopyVectorOfScalars(flatbuffers::FlatBufferBuilder &destination,
    const flatbuffers::Table *source, flatbuffers::voffset_t field, flatbuffers::ElementaryType type)
{
    if ( source && source->CheckField(field) )
    {
        if ( type == flatbuffers::ET_BOOL ) return CopyVector<bool>(destination, source, field).Union();
        else if ( type == flatbuffers::ET_CHAR ) return CopyVector<int8_t>(destination, source, field).Union();
        else if ( type == flatbuffers::ET_UCHAR ) return CopyVector<uint8_t>(destination, source, field).Union();
        else if ( type == flatbuffers::ET_SHORT ) return CopyVector<int16_t>(destination, source, field).Union();
        else if ( type == flatbuffers::ET_USHORT ) return CopyVector<uint16_t>(destination, source, field).Union();
        else if ( type == flatbuffers::ET_INT ) return CopyVector<int32_t>(destination, source, field).Union();
        else if ( type == flatbuffers::ET_UINT ) return CopyVector<uint32_t>(destination, source, field).Union();
        else if ( type == flatbuffers::ET_LONG ) return CopyVector<int64_t>(destination, source, field).Union();
        else if ( type == flatbuffers::ET_ULONG ) return CopyVector<uint64_t>(destination, source, field).Union();
        else if ( type == flatbuffers::ET_FLOAT ) return CopyVector<float>(destination, source, field).Union();
        else if ( type == flatbuffers::ET_DOUBLE ) return CopyVector<double>(destination, source, field).Union();
        else if ( type == flatbuffers::ET_STRING ) return CopyVectorOfStrings(destination, source, field).Union();
        else assert(false && "Unsupported elementary type");
    }

    // Zero offset means unset or non-existing vector
    return 0;
}

// Copy a scalar (if present) from one flatbuffer to another, by field
template<typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
static void CopyScalar(flatbuffers::FlatBufferBuilder &destination, const flatbuffers::Table *source, flatbuffers::voffset_t field)
{
    if ( source && source->CheckField(field) )
    {
        destination.AddElement(field, source->GetField<T>(field, 0));
    }
}

// Copy a string (if present) from one flatbuffer to another, by field
static flatbuffers::Offset<flatbuffers::String>
CopyString(flatbuffers::FlatBufferBuilder &destination, const flatbuffers::Table *source, flatbuffers::voffset_t field)
{
    if ( source && source->CheckField(field) )
    {
        return destination.CreateString(source->GetPointer<const flatbuffers::String *>(field));
    }

    // Zero offset means unset or non-existing string
    return 0;
}

// Copy a table (if present) from one flatbuffer to another by field
static flatbuffers::Offset<> CopyTable(flatbuffers::FlatBufferBuilder &destination, const flatbuffers::Table *source,
    flatbuffers::voffset_t field, const flatbuffers::TypeTable *types)
{
    if ( source && source->CheckField(field) )
    {
        return CopyTable(destination, source->GetPointer<const flatbuffers::Table *>(field), types);
    }

    // Zero offset means unset or non-existing table
    return 0;
}

// Copy a table
static flatbuffers::Offset<> CopyTable(flatbuffers::FlatBufferBuilder &destination, const flatbuffers::Table *source,
    const flatbuffers::TypeTable *typeTable)
{
    // This function copies a flatbuffer Table by iterating over the type table two times. First time to copy
    // offsets to all non-scalar, repeating types and strings, then a second time to add all offsets and scalars.

    if ( !source ) return flatbuffers::Offset<>();

    // Can only copy tables
    assert(typeTable);
    assert(typeTable->st == flatbuffers::ST_TABLE);

    std::unordered_map<flatbuffers::voffset_t, flatbuffers::Offset<>> fieldToOffset;

    // Iterate over all types and create offsets for non-scalar items
    for ( flatbuffers::voffset_t i = 0; i < typeTable->num_elems; i++ )
    {
        const auto name = typeTable->names[i];
        const auto type = flatbuffers::ElementaryType(typeTable->type_codes[i].base_type);
        const auto isRepeating = typeTable->type_codes[i].is_repeating != 0;
        const auto sequenceRef = typeTable->type_codes[i].sequence_ref;
        const auto field = flatbuffers::FieldIndexToOffset(i);

        if ( isRepeating )
        {
            if ( sequenceRef >= 0 )
            {
                LOG_TRACE1("Copy repeating {} (vector of {})\n", name, flatbuffers::ElementaryTypeNames()[type]);

                const auto *sequenceTypeTable = typeTable->type_refs[sequenceRef]();
                assert(sequenceTypeTable);

                if ( type == flatbuffers::ET_SEQUENCE )
                    fieldToOffset[field] = CopyVectorOfTables(destination, source, field, sequenceTypeTable).Union();
                else fieldToOffset[field] = CopyVectorOfScalars(destination, source, field, type);
            }
            else
            {
                LOG_TRACE1("Copy repeating {} (vector of {})\n", name, flatbuffers::ElementaryTypeNames()[type]);

                // Vector of scalars/strings
                fieldToOffset[field] = CopyVectorOfScalars(destination, source, field, type);
            }
        }
        else if ( sequenceRef >= 0 )
        {
            const auto *sequenceTypeTable = typeTable->type_refs[sequenceRef]();
            assert(sequenceTypeTable);

            if ( sequenceTypeTable->st == flatbuffers::ST_TABLE )
            {
                LOG_TRACE1("Copy non-repeating table {} ({})\n", name, flatbuffers::ElementaryTypeNames()[type]);

                fieldToOffset[field] = CopyTable(destination, source, field, sequenceTypeTable);
            }
            else if ( sequenceTypeTable->st == flatbuffers::ST_UNION )
            {
                LOG_TRACE1("Copy non-repeating union {} ({})\n", name, flatbuffers::ElementaryTypeNames()[type]);

                if ( type == flatbuffers::ET_SEQUENCE )
                {
                    const auto unionType = source->GetField<uint8_t>(flatbuffers::FieldIndexToOffset(i - 1u), 0);
                    if ( unionType > 0 )
                        fieldToOffset[field] = CopyTable(destination, source, field, sequenceTypeTable->type_refs[unionType - 1]());
                    else fieldToOffset[field] = flatbuffers::Offset<>();  // Default NONE value
                }
            }
        }
        else if ( type == flatbuffers::ElementaryType::ET_STRING )
        {
            LOG_TRACE1("Copy non-repeating non-sequence {} ({})\n", name, flatbuffers::ElementaryTypeNames()[type]);

            fieldToOffset[field] = CopyString(destination, source, field).Union();
        }
    }

    const auto tableOffset = destination.StartTable();

    // Iterate over all types and add offsets and scalar types
    for ( flatbuffers::voffset_t i = 0; i < typeTable->num_elems; i++ )
    {
        const auto name = typeTable->names[i];
        const auto baseType = flatbuffers::ElementaryType(typeTable->type_codes[i].base_type);
        const auto isRepeating = typeTable->type_codes[i].is_repeating != 0;
        const auto sequenceRef = typeTable->type_codes[i].sequence_ref;
        const auto field = flatbuffers::FieldIndexToOffset(i);

        if ( fieldToOffset.count(field) == 0 )
        {
            LOG_TRACE1("Copying and adding scalar {} ({})\n", name, flatbuffers::ElementaryTypeNames()[baseType]);

            // At this point it's too late for repeating types
            assert(!isRepeating);

            if ( sequenceRef >= 0 )
            {
                // At this point it's too late for any sequence types except for ENUM and the UNION type
                const auto *sequenceTypeTable = typeTable->type_refs[sequenceRef]();
                assert(sequenceTypeTable);
                assert(sequenceTypeTable->st == flatbuffers::ST_ENUM ||
                       (sequenceTypeTable->st == flatbuffers::ST_UNION && baseType == flatbuffers::ET_UTYPE));
            }

            // Scalar
            if ( baseType == flatbuffers::ET_UTYPE ) CopyScalar<uint8_t>(destination, source, field);
            else if ( baseType == flatbuffers::ET_BOOL ) CopyScalar<bool>(destination, source, field);
            else if ( baseType == flatbuffers::ET_CHAR ) CopyScalar<int8_t>(destination, source, field);
            else if ( baseType == flatbuffers::ET_UCHAR ) CopyScalar<uint8_t>(destination, source, field);
            else if ( baseType == flatbuffers::ET_SHORT ) CopyScalar<int16_t>(destination, source, field);
            else if ( baseType == flatbuffers::ET_USHORT ) CopyScalar<uint16_t>(destination, source, field);
            else if ( baseType == flatbuffers::ET_INT ) CopyScalar<int32_t>(destination, source, field);
            else if ( baseType == flatbuffers::ET_UINT ) CopyScalar<uint32_t>(destination, source, field);
            else if ( baseType == flatbuffers::ET_LONG ) CopyScalar<int64_t>(destination, source, field);
            else if ( baseType == flatbuffers::ET_ULONG ) CopyScalar<uint64_t>(destination, source, field);
            else if ( baseType == flatbuffers::ET_FLOAT ) CopyScalar<float>(destination, source, field);
            else if ( baseType == flatbuffers::ET_DOUBLE ) CopyScalar<double>(destination, source, field);
            else assert(false && "Unsupported elementary type");
        }
        else
        {
            LOG_TRACE1("Adding offset {} ({})\n", name, flatbuffers::ElementaryTypeNames()[baseType]);

            destination.AddOffset(field, fieldToOffset[field]);
        }
    }

    return destination.EndTable(tableOffset);
}

// Copy a table
template<typename T, std::enable_if_t<std::is_base_of_v<flatbuffers::Table, T>, int> = 0>
static flatbuffers::Offset<T> CopyTable(flatbuffers::FlatBufferBuilder &destination, const T *source)
{
    const flatbuffers::Offset<> offset = CopyTable(
        destination, reinterpret_cast<const flatbuffers::Table *>(source), T::MiniReflectTypeTable());

    // Special thing here to create an Offset object with a type
    return flatbuffers::Offset<T>(offset.o);
}

// Copy a vector of tables
template<typename T, std::enable_if_t<std::is_base_of_v<flatbuffers::Table, T>, int> = 0>
static std::vector<flatbuffers::Offset<T>>
CopyVectorOfTables(flatbuffers::FlatBufferBuilder &destination, const flatbuffers::Vector<flatbuffers::Offset<T>> *source)
{
    std::vector<flatbuffers::Offset<T>> srcTables;
    if ( source )
    {
        for ( const auto *table : *source )
        {
            srcTables.push_back(CopyTable(destination, table));
        }
    }
    return srcTables;
}

}  // namespace FlatbufferUtils
