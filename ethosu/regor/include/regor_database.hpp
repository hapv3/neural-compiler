//
// SPDX-FileCopyrightText: Copyright 2022-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include <cstdint>
#include <string>

namespace regor
{

struct IDatabaseIterator
{
    virtual ~IDatabaseIterator() = default;
    virtual bool Next() = 0;
    virtual void Release() = 0;
};

template<typename TYPE>
struct IRowIterator : public IDatabaseIterator
{
    virtual TYPE Value() = 0;
    virtual int Id() = 0;
    virtual int Column() = 0;
};

struct ITableIterator : public IDatabaseIterator
{
    virtual std::string Name() = 0;
    virtual int Rows() = 0;
    virtual int Columns() = 0;
    virtual IRowIterator<std::string> *ColumnNames() = 0;
    virtual IRowIterator<std::string> *Row(int row) = 0;
};

struct IDatabase
{
    virtual ~IDatabase() = default;
    virtual ITableIterator *Tables() = 0;
};

}  // namespace regor
