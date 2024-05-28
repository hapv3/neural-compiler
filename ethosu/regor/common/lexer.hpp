//
// SPDX-FileCopyrightText: Copyright 2021, 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <string>


/// <summary>
/// Text string lexer
/// </summary>
class Lexer
{
protected:
    const char *_source = nullptr;
    const char *_end = nullptr;
    const char *_pos = nullptr;

public:
    Lexer(const char *text, size_t length) : _source(text), _end(text + length) { _pos = _source; }

    bool SkipSpace()
    {
        const char *p = _pos;
        while ( p < _end && std::isblank(*p) )
        {
            p++;
        }
        _pos = p;
        return p < _end;
    }

    bool SkipWhite()
    {
        const char *p = _pos;
        while ( p < _end && std::isspace(*p) )
        {
            p++;
        }
        _pos = p;
        return p < _end;
    }

    bool SkipUntil(char term, bool consume)
    {
        const char *p = _pos;
        while ( (p < _end) && (*p != term) )
        {
            p++;
        }
        if ( consume && (p < _end) && (*p == term) )
        {
            p++;
        }
        _pos = p;
        return p < _end;
    }

    char Peek() const
    {
        assert(_pos < _end);
        return _pos < _end ? *_pos : '\0';
    }

    bool Expect(char c)
    {
        assert(_pos < _end);
        if ( *_pos == c )
        {
            _pos++;
            return true;
        }
        return false;
    }

    bool Expect(const char *text, size_t length = 0)
    {
        ptrdiff_t avail = std::max<ptrdiff_t>(0, _end - _pos);
        size_t compare = length == 0 ? std::char_traits<char>::length(text) : length;
        compare = std::min(compare, size_t(avail));
        if ( strncmp(_pos, text, compare) == 0 )
        {
            _pos += compare;
            return true;
        }
        return false;
    }

    bool GetIdent(std::string &ident, bool skipwhite = true)
    {
        if ( skipwhite && !SkipWhite() )
        {
            return false;
        }

        const char *p = _pos;
        if ( p < _end && *p != '_' && !std::isalpha(*p) )
        {
            return false;
        }

        const char *maxEnd = _end;
        while ( (p < maxEnd) && (*p == '_' || std::isalnum(*p)) )
        {
            p++;
        }

        ident.assign(_pos, p);
        _pos = p;
        return !ident.empty();
    }

    bool GetString(std::string &text, char quote, char escape, char term)
    {
        if ( _pos >= _end ) return false;

        text.reserve(16);
        text.clear();

        bool quoted = Expect(quote);

        const char *p = _pos;
        while ( (p < _end) && (*p != term) )
        {
            // Handle escaping first to allow quotes and terminator in string.
            if ( (*p == escape) && (p < _end) )
            {
                p++;
                if ( p == _end ) break;
            }
            else if ( quoted )
            {
                if ( *p == quote ) break;
            }
            text += *p;
            p++;
        }

        _pos = p;
        return quoted || !text.empty();  // Quoted strings are intentionally present (just empty)
    }
};
