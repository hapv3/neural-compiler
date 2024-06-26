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

#include "common.hpp"
#include "lexer.hpp"

#include <cassert>
#include <charconv>
#include <vector>


/// <summary>
/// INI file reader
/// </summary>
class IniReader : Lexer
{
    enum class ParseState
    {
        None,
        Section,
        Key,
        Value
    };

protected:
    ParseState _parseState = ParseState::None;
    bool _wasError = false;

public:
    IniReader(const char *src, size_t length) : Lexer(src, length) {}

    IniReader(const IniReader &other) = delete;

public:
    bool Begin(std::string &key)
    {
        assert(_parseState != ParseState::Value);

        if ( _parseState == ParseState::None )
        {
            if ( !SkipCommentSpace() )
            {
                return false;
            }

            // Expect section
            if ( !Expect('[') )
            {
                // Error state
                _wasError = true;
                return false;
            }
            _parseState = ParseState::Section;
        }

        if ( _parseState == ParseState::Section )
        {
            // Get section identifier
            if ( !GetString(key, 0, 0, ']') )
            {
                return false;
            }
            if ( !SkipWhite() )
            {
                return false;
            }
            if ( !Expect(']') )
            {
                // Error state
                _wasError = true;
                return false;
            }
            _parseState = ParseState::Key;
        }
        else if ( _parseState == ParseState::Key )
        {
            if ( !SkipCommentSpace() )
            {
                return false;
            }
            // Get key name
            if ( !GetIdent(key) )
            {
                _parseState = ParseState::None;
                return false;
            }
            while ( !EOS() && Expect('.') )
            {
                key += ".";
                std::string tmp;
                if ( GetIdent(tmp) )
                {
                    key += tmp;
                }
            }
            if ( !SkipWhite() )
            {
                return false;
            }
            if ( !Expect('=') )
            {
                // Error - attempt to recover by jumping to next line
                _wasError = true;
                SkipUntil('\n', true);
                return false;
            }
            _parseState = ParseState::Value;
        }

        return true;
    }

    void End()
    {
        if ( _parseState == ParseState::Value )  // End key/value pair
        {
            // End this line (doesn't work with quoted strings that span lines)
            SkipUntil('\n', true);
            _parseState = ParseState::Key;
        }
        else if ( _parseState == ParseState::Key )  // End section
        {
            // Skip until next section or EOF
            std::string skipped;
            while ( Begin(skipped) )
            {
                assert(_parseState != ParseState::Key);  // Prevent recursion depth > 2
                End();
            }
            _parseState = ParseState::None;
        }
    }

    bool Read(bool &value)
    {
        if ( !SkipSpace() )
        {
            return false;
        }

        assert(_parseState == ParseState::Value);

        if ( Expect("true") ) value = true;
        else if ( Expect("yes") ) value = true;
        else if ( Expect("1") ) value = true;
        else if ( Expect("false") ) value = false;
        else if ( Expect("no") ) value = false;
        else if ( Expect("0") ) value = false;
        else return false;

        // Allow comma-separated value reads
        if ( SkipSpace() ) Expect(',');
        return true;
    }

    bool Read(int64_t &value)
    {
        if ( !SkipSpace() )
        {
            return false;
        }

        assert(_parseState == ParseState::Value);

        auto [ptr, ec] = std::from_chars(_pos, _end, value);
        if ( _pos == ptr )
        {
            return false;
        }
        _pos = ptr;

        // Allow comma-separated value reads
        if ( SkipSpace() ) Expect(',');
        return true;
    }

    bool Read(int &value)
    {
        if ( !SkipSpace() )
        {
            return false;
        }

        assert(_parseState == ParseState::Value);

        auto [ptr, ec] = std::from_chars(_pos, _end, value);
        if ( _pos == ptr )
        {
            return false;
        }
        _pos = ptr;

        // Allow comma-separated value reads
        if ( SkipSpace() ) Expect(',');
        return true;
    }

    bool Read(float &value)
    {
        if ( !SkipSpace() )
        {
            return false;
        }

        assert(_parseState == ParseState::Value);

        char *end;
        errno = 0;
        value = std::strtof(_pos, &end);
        if ( errno || (value == HUGE_VAL) ) return false;
        if ( end == _pos )
        {
            return false;
        }
        _pos = end;

        // Allow comma-separated value reads
        if ( SkipSpace() ) Expect(',');
        return true;
    }

    bool Read(std::string &value)
    {
        if ( !SkipSpace() )
        {
            return false;
        }

        assert(_parseState == ParseState::Value);
        if ( GetString(value, '"', '\\', '\n') )
        {
            if ( !value.empty() && value.back() == '\r' )
            {
                value.pop_back();
            }
        }
        return true;
    }

    template<typename TYPE, std::enable_if_t<std::is_arithmetic<TYPE>::value, bool> = true>
    bool Read(std::vector<TYPE> &out)
    {
        assert(_parseState == ParseState::Value);

        TYPE tmp{};
        while ( Read(tmp) )
        {
            out.push_back(tmp);
        }
        return !out.empty();
    }

    template<typename TYPE>
    TYPE Get()
    {
        TYPE tmp = TYPE();
        Read(tmp);
        return tmp;
    }

    ssize_t Position() const { return _pos - _source; }

private:
    bool EOS() const { return _pos >= _end; }

    bool SkipCommentSpace()
    {
        // Skip whitespace and comments
        while ( true )
        {
            if ( !SkipWhite() )
            {
                return false;
            }
            if ( !Expect(';') )
            {
                break;
            }
            if ( !SkipUntil('\n', true) )
            {
                return false;
            }
        }
        return !EOS();
    }
};
