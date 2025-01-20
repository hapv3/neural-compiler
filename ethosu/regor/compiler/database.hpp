//
// SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "common/common.hpp"

#include "include/regor_database.hpp"

#include <memory>
#include <string>
#include <vector>

namespace regor
{



/// <summary>
/// Base database implementation
/// </summary>
class Database : public IDatabase
{
protected:
    struct DataRow
    {
        int uniqueId = 0;
        std::unique_ptr<std::string[]> fields;

        DataRow() = default;

        DataRow(DataRow &&other) noexcept : fields(std::move(other.fields)) { uniqueId = other.uniqueId; }

        DataRow &operator=(DataRow &&other) noexcept
        {
            this->uniqueId = other.uniqueId;
            this->fields = std::move(other.fields);
            return *this;
        }
    };

    struct DataTable
    {
        std::string name;
        std::vector<DataRow> rows;
        std::vector<std::string> columnNames;
        int columns = 0;
        bool isIndexed = true;
        DataTable(const char *tableName) : name(tableName) {}
    };


    struct RowIterator : public IRowIterator<std::string>
    {
    private:
        const std::string *_fields;
        int _uniqueId = 0;
        int _columns = 0;
        int _index = -1;

    public:
        RowIterator(int uniqueId, const std::string *fields, int columns) :
                _fields(fields), _uniqueId(uniqueId), _columns(columns)
        {
        }
        std::string Value() override { return _fields[_index]; }
        int Id() override { return _uniqueId; }
        int Column() override { return _index; }
        bool Next() override
        {
            _index++;
            return _index < _columns;
        }
        void Release() override { delete this; }
    };

    struct TableIterator : public ITableIterator
    {
    private:
        std::vector<DataTable> &_tables;
        int _index = -1;

    public:
        TableIterator(std::vector<DataTable> &tables) : _tables(tables) {}

    public:
        std::string Name() override
        {
            assert(_index >= 0);
            return _tables[_index].name;
        }
        int Rows() override
        {
            assert(_index >= 0);
            return int(_tables[_index].rows.size());
        }
        int Columns() override
        {
            assert(_index >= 0);
            return _tables[_index].columns;
        }
        IRowIterator<std::string> *ColumnNames() override
        {
            return new RowIterator(
                _tables[_index].isIndexed ? 1 : 0, _tables[_index].columnNames.data(), _tables[_index].columns);
        }
        IRowIterator<std::string> *Row(int row) override
        {
            const auto &entry = _tables[_index].rows[row];
            return new RowIterator(entry.uniqueId, entry.fields.get(), _tables[_index].columns);
        }
        bool Next() override
        {
            _index++;
            return _index < int(_tables.size());
        }
        void Release() override { delete this; }
    };

    std::vector<DataTable> _tables;

public:
    int AddTable(const char *name, bool isIndexed = true)
    {
        _tables.emplace_back(name);
        _tables.back().isIndexed = isIndexed;
        return int(_tables.size()) - 1;
    }

    void AddColumns(int tableId, std::initializer_list<const char *> names)
    {
        assert(tableId >= 0 && tableId < int(_tables.size()));
        DataTable *table = &_tables[tableId];
        table->columnNames.insert(table->columnNames.end(), names.begin(), names.end());
        table->columns = int(table->columnNames.size());
    }

    void AddColumns(int tableId, std::vector<std::string> names)
    {
        assert(tableId >= 0 && tableId < int(_tables.size()));
        DataTable *table = &_tables[tableId];
        table->columnNames.insert(table->columnNames.end(), names.begin(), names.end());
        table->columns = int(table->columnNames.size());
    }

    int AddRow(int tableId, int uniqueId, std::initializer_list<std::string> values)
    {
        assert(tableId >= 0 && tableId < int(_tables.size()));
        DataTable *table = &_tables[tableId];
        DataRow tmp;
        tmp.uniqueId = uniqueId;
        tmp.fields = std::unique_ptr<std::string[]>(new std::string[table->columns]);
        auto pos = values.begin();
        for ( int i = 0; i < table->columns && pos != values.end(); i++, pos++ )
        {
            tmp.fields[i] = *pos;
        }

        table->rows.push_back(std::move(tmp));
        return int(table->rows.size()) - 1;
    }

    int AddRow(int tableId, int uniqueId, std::vector<std::string> values)
    {
        assert(tableId >= 0 && tableId < int(_tables.size()));
        DataTable *table = &_tables[tableId];
        DataRow tmp;
        tmp.uniqueId = uniqueId;
        tmp.fields = std::unique_ptr<std::string[]>(new std::string[table->columns]);
        auto pos = values.begin();
        for ( int i = 0; i < table->columns && pos != values.end(); i++, pos++ )
        {
            tmp.fields[i] = *pos;
        }

        table->rows.push_back(std::move(tmp));
        return int(table->rows.size()) - 1;
    }

    void SetField(int tableId, int row, int column, const std::string &value)
    {
        assert(tableId >= 0 && tableId < int(_tables.size()));
        DataTable *table = &_tables[tableId];
        assert(row < int(table->rows.size()));
        assert(column < table->columns);
        table->rows[row].fields[column] = value;
    }

    // From IDatabase
    ITableIterator *Tables() override { return new TableIterator(_tables); }
};


}  // namespace regor
