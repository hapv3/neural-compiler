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

#include "common/ordered_map.hpp"
#include "randomize.hpp"

#include <array>
#include <catch_all.hpp>
#include <iterator>
#include <unordered_map>
#include <vector>

using regor::ordered_map;

namespace
{
bool operator==(const std::array<int, 1> &lhs, const std::array<int, 1> &rhs)
{
    return lhs[0] == rhs[0];
}

template<typename KEY, typename VAL>
static void CheckInsertionOrder(const ordered_map<KEY, VAL> &uut, const std::unordered_map<KEY, VAL> &ref, const KEY expected_keys[])
{
    REQUIRE(uut.size() == ref.size());

    size_t i = 0;
    for ( auto pair : uut.pairs() )
    {
        REQUIRE(i < uut.size());
        CHECK(pair.first == expected_keys[i]);
        CHECK(pair.second == ref.at(expected_keys[i]));
        ++i;
    }
    REQUIRE(i == uut.size());

    i = 0;
    for ( auto value : uut )
    {
        REQUIRE(i < uut.size());
        CHECK(value == ref.at(expected_keys[i]));
        ++i;
    }
    REQUIRE(i == uut.size());

    i = 0;
    for ( auto &key : uut.keys() )
    {
        REQUIRE(i < uut.size());
        CHECK(key == expected_keys[i]);
        ++i;
    }
    REQUIRE(i == uut.size());
}

}  // namespace

TEST_CASE("ordered_map: Retains insertion order")
{
    using Key = char16_t;
    using Val = int;

    // 0: Using only operator[] insertion
    // 1: Using only insert() insertion
    // 2: Using only emplace() insertion
    // 3: Using a combination of all three
    for ( int test = 0; test < 4; ++test )
    {
        for ( size_t capacity = 1; capacity <= 256; capacity *= 2 )
        {
            ordered_map<Key, Val> uut{capacity};
            std::unordered_map<Key, Val> ref;
            Val val{};

            for ( auto c : u"An ordered_map will always retain insertion order" )
            {
                randomize(val);
                int rand = urandom_range(0, 3);

                if ( (test == 0) || ((test == 3) && (rand == 0)) )
                {
                    uut[c] = val;
                    ref[c] = val;
                }
                else if ( (test == 1) || ((test == 3) && (rand == 1)) )
                {
                    if ( ref.count(c) == 0 )
                    {
                        uut.insert(c, val);
                        ref[c] = val;
                    }
                }
                else
                {
                    uut.emplace(c, val);
                    ref.emplace(c, val);
                }
            }

            REQUIRE(uut.size() == ref.size());

            CheckInsertionOrder<Key, Val>(uut, ref, u"An orde_mapwilyst");

            uut.erase('r');
            uut.erase('e');
            uut.erase('m');
            uut.erase('o');
            uut.erase('v');
            uut.erase('e');

            ref.erase('r');
            ref.erase('e');
            ref.erase('m');
            ref.erase('o');
            ref.erase('v');
            ref.erase('e');

            CheckInsertionOrder<Key, Val>(uut, ref, u"An d_apwilyst");
        }
    }
}

TEST_CASE("ordered_map: Replaces")
{
    using Key = char16_t;
    using Val = int;

    ordered_map<Key, Val> uut;
    std::unordered_map<Key, Val> ref;
    Val val{};

    for ( auto c : u"Using replace() does not send you to the back of the line" )
    {
        randomize(val);
        ref[c] = val;
        uut[c] = val;
    }

    CheckInsertionOrder<Key, Val>(uut, ref, u"Using replac()dotyuhbkf");

    randomize(val);
    uut.replace(' ', '_', val);
    ref.erase(' ');
    ref['_'] = val;

    randomize(val);
    uut.replace('e', 'E', val);
    ref.erase('e');
    ref['E'] = val;

    randomize(val);
    uut.replace('U', 'e', val);
    ref.erase('U');
    ref['e'] = val;

    CheckInsertionOrder<Key, Val>(uut, ref, u"esing_rEplac()dotyuhbkf");
}

TEST_CASE("ordered_map: Reinserts")
{
    using Key = char16_t;
    using Val = std::array<int, 1>;

    for ( size_t capacity = 1; capacity <= 256; capacity *= 2 )
    {
        ordered_map<Key, Val> uut{capacity};
        std::unordered_map<Key, Val> ref;
        Val val{};

        // Reinsert the only element
        randomize(val);
        uut['U'] = val;
        ref['U'] = val;
        CheckInsertionOrder<Key, Val>(uut, ref, u"U");
        randomize(val);
        uut.reinsert('U', val);
        ref['U'] = val;
        CheckInsertionOrder<Key, Val>(uut, ref, u"U");

        for ( auto c : u"Using reinsert() sends you to the back of the line" )
        {
            randomize(val);
            ref[c] = val;
            uut[c] = val;
        }

        CheckInsertionOrder<Key, Val>(uut, ref, u"Using ret()dyouhbackfl");

        // Reinsert elements from the middle
        randomize(val);
        uut.reinsert(' ', val);
        ref[' '] = val;
        randomize(val);
        uut.reinsert('e', val);
        ref['e'] = val;

        CheckInsertionOrder<Key, Val>(uut, ref, u"Usingrt()dyouhbackfl\0 e");

        // Reinsert the first element
        randomize(val);
        uut.reinsert('U', val);
        ref['U'] = val;

        CheckInsertionOrder<Key, Val>(uut, ref, u"singrt()dyouhbackfl\0 eU");

        // Reinsert the last element
        randomize(val);
        uut.reinsert('U', val);
        ref['U'] = val;

        CheckInsertionOrder<Key, Val>(uut, ref, u"singrt()dyouhbackfl\0 eU");

        // Reinsert using rvalue reference
        randomize(val);
        ref['r'] = val;
        uut.reinsert('r', std::move(val));

        CheckInsertionOrder<Key, Val>(uut, ref, u"singt()dyouhbackfl\0 eUr");
    }
}

TEST_CASE("ordered_map: Reorders")
{
    using Key = char16_t;
    using Val = int;

    for ( size_t capacity = 1; capacity <= 256; capacity *= 2 )
    {
        ordered_map<Key, Val> uut{capacity};
        std::unordered_map<Key, Val> ref;
        Val val{};

        // Reorder the only element in the map
        randomize(val);
        uut['R'] = val;
        ref['R'] = val;
        CheckInsertionOrder<Key, Val>(uut, ref, u"R");
        uut.reorder_before('R', uut.find('R'));
        CheckInsertionOrder<Key, Val>(uut, ref, u"R");
        uut.reorder_after('R', uut.find('R'));
        CheckInsertionOrder<Key, Val>(uut, ref, u"R");

        for ( auto c : u"Reordering by key does not change value" )
        {
            randomize(val);
            ref[c] = val;
            uut[c] = val;
        }

        CheckInsertionOrder<Key, Val>(uut, ref, u"Reording bykstchavlu");

        // Reorder in the middle
        uut.reorder_after('r', uut.find('b'));   // after, move forwards
        uut.reorder_after('c', uut.find('e'));   // after, move backwards
        uut.reorder_before('l', uut.find('t'));  // before, move forwards
        uut.reorder_before('a', uut.find('y'));  // before, move backwards

        CheckInsertionOrder<Key, Val>(uut, ref, u"Recoding braykslthvu");

        // Reorder to the front
        uut.reorder_before(' ', uut.find('R'));

        CheckInsertionOrder<Key, Val>(uut, ref, u" Recodingbraykslthvu");

        // Reorder from the front
        uut.reorder_before(' ', uut.find('\0'));
        uut.reorder_after('R', uut.find('o'));

        CheckInsertionOrder<Key, Val>(uut, ref, u"ecoRdingbraykslthvu ");

        // Reorder from the back
        uut.reorder_before('\0', uut.find('R'));
        uut.reorder_after(' ', uut.find('R'));

        CheckInsertionOrder<Key, Val>(uut, ref, u"eco\0R dingbraykslthvu");

        // Reorder to the back
        uut.reorder_after('y', uut.find('u'));

        CheckInsertionOrder<Key, Val>(uut, ref, u"eco\0R dingbrakslthvuy");

        // Reorder adjacent
        uut.reorder_after('c', uut.find('o'));
        uut.reorder_before('r', uut.find('b'));

        CheckInsertionOrder<Key, Val>(uut, ref, u"eoc\0R dingrbakslthvuy");
    }
}

TEST_CASE("ordered_map: Swaps")
{
    using Key = char16_t;
    using Val = int;

    for ( size_t capacity = 1; capacity <= 256; capacity *= 2 )
    {
        ordered_map<Key, Val> uut{capacity};
        std::unordered_map<Key, Val> ref;
        Val val{};

        // Swap the only element with itself
        randomize(val);
        uut['S'] = val;
        ref['S'] = val;
        CheckInsertionOrder<Key, Val>(uut, ref, u"S");
        uut.swap_order(uut.find('S'), uut.find('S'));
        CheckInsertionOrder<Key, Val>(uut, ref, u"S");

        for ( auto c : u"Swapping by key does not change value" )
        {
            randomize(val);
            ref[c] = val;
            uut[c] = val;
        }

        CheckInsertionOrder<Key, Val>(uut, ref, u"Swaping bykedostchvlu");

        // Swap elements in the middle
        uut.swap_order(uut.find('w'), uut.find('k'));  // earlier element first
        uut.swap_order(uut.find('t'), uut.find('p'));  // earlier element second

        CheckInsertionOrder<Key, Val>(uut, ref, u"Skating bywedospchvlu");

        // Swap the first element
        uut.swap_order(uut.begin(), uut.find('s'));

        CheckInsertionOrder<Key, Val>(uut, ref, u"skating bywedoSpchvlu");

        // Swap the last element
        uut.swap_order(uut.find('w'), uut.find('\0'));

        CheckInsertionOrder<Key, Val>(uut, ref, u"skating by\0edoSpchvluw");

        // Swap the first and last elements
        uut.swap_order(uut.find('w'), uut.begin());

        CheckInsertionOrder<Key, Val>(uut, ref, u"wkating by\0edoSpchvlus");

        // Swap adjacent elements
        uut.swap_order(uut.find('c'), uut.find('h'));  // earlier element first

        CheckInsertionOrder<Key, Val>(uut, ref, u"wkating by\0edoSphcvlus");

        uut.swap_order(uut.find('b'), uut.find(' '));  // earlier element second

        CheckInsertionOrder<Key, Val>(uut, ref, u"wkatingb y\0edoSphcvlus");
    }
}

TEST_CASE("ordered_map: Clears")
{
    ordered_map<float, bool> uut;

    CHECK(uut.empty());
    CHECK(uut.size() == 0);
    for ( auto pair : uut.pairs() )
    {
        REQUIRE(false);
    }
    for ( auto value : uut )
    {
        REQUIRE(false);
    }

    // Clear when empty
    uut.clear();
    CHECK(uut.empty());

    // Clear when not empty
    uut[3.14f] = false;
    uut.insert(2.72f, true);
    uut.clear();

    bool val = false;
    CHECK(uut.find(3.14f) == uut.end());
    CHECK(uut.try_ref(3.14f) == nullptr);
    CHECK(uut.try_get(2.72f, val) == false);
    CHECK(val == false);

    CHECK(uut.empty());
    CHECK(uut.size() == 0);
    for ( auto pair : uut.pairs() )
    {
        REQUIRE(false);
    }
    for ( auto value : uut )
    {
        REQUIRE(false);
    }
}

TEST_CASE("ordered_map: Resizes (move assignable)")
{
    using Key = long long int;
    using Val = std::unique_ptr<bool>;

    static_assert(std::is_move_assignable_v<Val>);

    for ( size_t capacity = 1; capacity <= 256; capacity *= 2 )
    {
        ordered_map<Key, Val> uut{capacity};

        const Key lower = -10 * capacity;
        const Key upper = 10 * capacity;

        for ( Key i = lower; i < upper; ++i )
        {
            uut[i] = std::make_unique<bool>(i % 4);
        }

        for ( Key i = lower; i < upper; ++i )
        {
            CHECK(*uut[i] == bool(i % 4));
        }
    }
}

TEST_CASE("ordered_map: Resizes (trivially copyable)")
{
    using Key = long long int;

    struct Val
    {
        Key key;
        Val() = default;
        Val(Key k) : key(k) {}
        Val &operator=(const Val &val) = default;
        Val &operator=(Val &&val) = delete;
    };

    static_assert(!std::is_move_assignable_v<Val>);
    static_assert(std::is_trivially_copyable_v<Val>);

    for ( size_t capacity = 1; capacity <= 256; capacity *= 2 )
    {
        ordered_map<Key, Val> uut{capacity};

        const Key lower = -10 * capacity;
        const Key upper = 10 * capacity;

        for ( Key i = lower; i < upper; ++i )
        {
            uut.emplace(i, i);
        }

        for ( Key i = lower; i < upper; ++i )
        {
            CHECK(uut[i].key == i);
        }
    }
}

TEST_CASE("ordered_map: Resizes (not move assignable, not trivially copyable)")
{
    using Key = long long int;

    struct Val
    {
        Key key;
        Val() = default;
        Val(Key k) : key(k) {}
        Val(const Val &val) : key(val.key) {}
        Val &operator=(const Val &val) = default;
        Val &operator=(Val &&val) = delete;
    };

    static_assert(!std::is_move_assignable_v<Val>);
    static_assert(!std::is_trivially_copyable_v<Val>);

    for ( size_t capacity = 1; capacity <= 256; capacity *= 2 )
    {
        ordered_map<Key, Val> uut{capacity};

        const Key lower = -10 * capacity;
        const Key upper = 10 * capacity;

        for ( Key i = lower; i < upper; ++i )
        {
            uut.emplace(i, i);
        }

        for ( Key i = lower; i < upper; ++i )
        {
            CHECK(uut[i].key == i);
        }
    }
}

namespace
{
constexpr int test_length = 10000;
constexpr bool debug = false;

enum class MapLikeOp
{
    Size,
    Empty,
    SquareBracketsAccess,
    SquareBracketsInsert,
    SquareBracketsReplace,
    Insert,
    ReinsertInsert,
    ReinsertReplace,
    EmplaceSucceed,
    EmplaceFail,
    EraseSucceed,
    EraseFail,
    AtAccess,
    AtAccessConst,
    AtReplace,
    ContainsTrue,
    ContainsFalse,
    Clear,
    TryGetSucceed,
    TryGetFail,
    TryRefSucceed,
    TryRefFail,
    TryRefReplace,
    TryRefConstSucceed,
    TryRefConstFail,
    FindSucceed,
    FindFail,
    FindConstSucceed,
    FindConstFail,
    FindReplace,
    SwapOrder,
    Replace,
    CheckAll,
};

bool RequiresKeyHit(MapLikeOp op)
{
    switch ( op )
    {
        case MapLikeOp::SquareBracketsAccess:
        case MapLikeOp::SquareBracketsReplace:
        case MapLikeOp::AtAccess:
        case MapLikeOp::AtAccessConst:
        case MapLikeOp::AtReplace:
        case MapLikeOp::ContainsTrue:
        case MapLikeOp::TryGetSucceed:
        case MapLikeOp::TryRefSucceed:
        case MapLikeOp::TryRefConstSucceed:
        case MapLikeOp::TryRefReplace:
        case MapLikeOp::EraseSucceed:
        case MapLikeOp::EmplaceFail:
        case MapLikeOp::ReinsertReplace:
        case MapLikeOp::FindSucceed:
        case MapLikeOp::FindConstSucceed:
        case MapLikeOp::FindReplace:
        case MapLikeOp::SwapOrder:
        case MapLikeOp::Replace:
            return true;
        default:
            return false;
    }
}

bool RequiresKeyMiss(MapLikeOp op)
{
    switch ( op )
    {
        case MapLikeOp::SquareBracketsInsert:
        case MapLikeOp::EmplaceSucceed:
        case MapLikeOp::Insert:
        case MapLikeOp::ReinsertInsert:
        case MapLikeOp::ContainsFalse:
        case MapLikeOp::TryGetFail:
        case MapLikeOp::TryRefFail:
        case MapLikeOp::TryRefConstFail:
        case MapLikeOp::FindFail:
        case MapLikeOp::FindConstFail:
        case MapLikeOp::EraseFail:
            return true;
        default:
            return false;
    }
}

}  // namespace

TEST_CASE("ordered_map: Behaves like a map")
{
    using Key = unsigned char;
    using Val = int;

    for ( size_t capacity = 1; capacity <= 256; capacity *= 2 )
    {
        ordered_map<Key, Val> uut{capacity};
        std::unordered_map<Key, Val> ref;

        for ( int i = 0; i < test_length; ++i )
        {
            auto op = MapLikeOp(urandom_range(unsigned(MapLikeOp::Size), unsigned(MapLikeOp::CheckAll)));
            Key key;
            Val val;
            randomize(val);

            if ( i + 1 == test_length )
            {
                op = MapLikeOp::CheckAll;  // Override to always finish on a check
            }

            if ( RequiresKeyHit(op) )
            {
                if ( ref.empty() )
                {
                    continue;
                }
                key = std::next(ref.begin(), urandom_range(0, unsigned(ref.size() - 1u)))->first;
            }
            else if ( RequiresKeyMiss(op) )
            {
                randomize(key);
                if ( ref.count(key) )
                {
                    continue;
                }
            }
            else
            {
                key = 0;
            }

            INFO("op = " << unsigned(op) << "\tkey = " << unsigned(key) << ",\tval = " << val);

            switch ( op )
            {
                // Capacity
                case MapLikeOp::Size:
                    CHECK(uut.size() == ref.size());
                    break;
                case MapLikeOp::Empty:
                    CHECK(uut.empty() == ref.empty());
                    break;

                // Lookup existing key
                case MapLikeOp::SquareBracketsAccess:
                    CHECK(uut[key] == ref[key]);
                    break;
                case MapLikeOp::AtAccess:
                    CHECK(uut.at(key) == ref.at(key));
                    break;
                case MapLikeOp::AtAccessConst:
                {
                    const ordered_map<Key, Val> &const_uut{uut};
                    CHECK(const_uut.at(key) == ref.at(key));
                }
                break;
                case MapLikeOp::ContainsTrue:
                    CHECK(uut.contains(key));
                    break;
                case MapLikeOp::TryGetSucceed:
                    CHECK(uut.try_get(key, val));
                    CHECK(val == ref[key]);
                    break;
                case MapLikeOp::TryRefSucceed:
                    CHECK(uut.try_ref(key));
                    CHECK(*uut.try_ref(key) == ref[key]);
                    break;
                case MapLikeOp::TryRefConstSucceed:
                {
                    const ordered_map<Key, Val> &const_uut{uut};
                    CHECK(const_uut.try_ref(key));
                    CHECK(*const_uut.try_ref(key) == ref[key]);
                }
                break;
                case MapLikeOp::FindSucceed:
                    CHECK(*uut.find(key) == ref[key]);
                    break;
                case MapLikeOp::FindConstSucceed:
                {
                    const ordered_map<Key, Val> &const_uut{uut};
                    CHECK(*const_uut.find(key) == ref[key]);
                }
                break;

                // Lookup non-existing key
                case MapLikeOp::ContainsFalse:
                    CHECK(uut.contains(key) == false);
                    break;
                case MapLikeOp::TryGetFail:
                    CHECK(uut.try_get(key, val) == false);
                    break;
                case MapLikeOp::TryRefFail:
                    CHECK(uut.try_ref(key) == nullptr);
                    break;
                case MapLikeOp::TryRefConstFail:
                {
                    const ordered_map<Key, Val> &const_uut{uut};
                    CHECK(const_uut.try_ref(key) == nullptr);
                }
                break;
                case MapLikeOp::FindFail:
                    CHECK(uut.find(key) == uut.end());
                    break;
                case MapLikeOp::FindConstFail:
                {
                    const ordered_map<Key, Val> &const_uut{uut};
                    CHECK(const_uut.find(key) == uut.end());
                }
                break;

                // Modify the map
                case MapLikeOp::SquareBracketsInsert:   // key does not exist, create it
                case MapLikeOp::SquareBracketsReplace:  // key exists, replace it
                    ref[key] = val;
                    uut[key] = val;
                    break;
                case MapLikeOp::AtReplace:  // key must exist, replace it
                    ref[key] = val;
                    uut.at(key) = val;
                    break;
                case MapLikeOp::EmplaceSucceed:  // key does not exist, create it
                case MapLikeOp::EmplaceFail:     // key exists, no-op
                    ref.emplace(key, val);
                    uut.emplace(key, val);
                    break;
                case MapLikeOp::Insert:  // key does not exist, create it
                    ref[key] = val;
                    uut.insert(key, val);
                    break;
                case MapLikeOp::TryRefReplace:  // key exists, replace it
                {
                    ref[key] = val;
                    auto *p = uut.try_ref(key);
                    CHECK(p != nullptr);
                    *p = val;
                }
                break;
                case MapLikeOp::EraseSucceed:  // key exists, remove it
                case MapLikeOp::EraseFail:     // key does not exist, no-op
                    ref.erase(key);
                    CHECK((uut.erase(key) != 0) == (op == MapLikeOp::EraseSucceed));
                    break;
                case MapLikeOp::ReinsertInsert:   // key does not exist, create it
                case MapLikeOp::ReinsertReplace:  // key exists, replace it
                    ref[key] = val;
                    uut.reinsert(key, val);
                    break;
                case MapLikeOp::FindReplace:
                    ref[key] = val;
                    *uut.find(key) = val;
                    break;
                case MapLikeOp::SwapOrder:
                    uut.swap_order(uut.find(key),
                        uut.find(std::next(ref.begin(), urandom_range(0, unsigned(ref.size() - 1u)))->first));
                    break;
                case MapLikeOp::Replace:  // key must exist and new_key must not, remove key and create new_key
                {
                    Key new_key;
                    randomize(new_key);
                    if ( ref.count(new_key) == 0 )
                    {
                        ref.erase(key);
                        ref[new_key] = val;
                        uut.replace(key, new_key, val);
                    }
                }
                break;

                // Lookup every element
                case MapLikeOp::CheckAll:
                case MapLikeOp::Clear:  // Recently added elements may have never been checked, so check before clearing
                default:
                    CHECK(uut.size() == ref.size());
                    CHECK(uut.empty() == ref.empty());
                    for ( auto &pair : ref )
                    {
                        key = pair.first;
                        CHECK(uut.contains(key));
                        CHECK(uut[key] == ref[key]);
                    }
                    if ( op == MapLikeOp::Clear )
                    {
                        ref.clear();
                        uut.clear();
                    }
                    break;
            }

            if ( debug )
            {
                // Additional checks after every operation in debug mode to end the test closer to the point of failure.
                // Disabled by default in case the act of reading the uut after every write masks a bug.
                REQUIRE(uut.size() == ref.size());
                REQUIRE(uut.empty() == ref.empty());
                for ( auto &pair : ref )
                {
                    key = pair.first;
                    REQUIRE(uut.contains(key));
                    REQUIRE(uut[key] == ref[key]);
                }
            }
        }
    }
}

namespace
{
enum class VectorLikeOp
{
    Size,
    Empty,
    Begin,
    End,
    Front,
    Back,
    Erase,
    Clear,
    IteratorRead,
    IteratorReadConst,
    IteratorWrite,
    PairIteratorRead,
    PairIteratorReadConst,
    PairIteratorWrite,
    EmplaceBack,
    PushBackInsert,
    PushBackSquareBrackets,
    PushBackSeveral,
    CheckAll,
};

bool RequiresKeyVal(VectorLikeOp op)
{
    switch ( op )
    {
        case VectorLikeOp::IteratorWrite:
        case VectorLikeOp::PairIteratorWrite:
        case VectorLikeOp::EmplaceBack:
        case VectorLikeOp::PushBackInsert:
        case VectorLikeOp::PushBackSquareBrackets:
            return true;
        default:
            return false;
    }
}

bool RequiresPos(VectorLikeOp op)
{
    switch ( op )
    {
        case VectorLikeOp::IteratorRead:
        case VectorLikeOp::IteratorReadConst:
        case VectorLikeOp::IteratorWrite:
        case VectorLikeOp::PairIteratorRead:
        case VectorLikeOp::PairIteratorReadConst:
        case VectorLikeOp::PairIteratorWrite:
        case VectorLikeOp::Erase:
            return true;
        default:
            return false;
    }
}

bool RequiresContents(VectorLikeOp op)
{
    if ( RequiresPos(op) )
    {
        return true;
    }

    switch ( op )
    {
        case VectorLikeOp::Front:
        case VectorLikeOp::Back:
            return true;
        default:
            return false;
    }
}

}  // namespace

TEST_CASE("ordered_map: Behaves like a vector")
{
    using Key = size_t;
    using Val = Key;

    for ( size_t capacity = 1; capacity <= 256; capacity *= 2 )
    {
        ordered_map<Key, Val> uut{capacity};
        std::vector<Val> ref;
        int elements_to_add = 0;

        for ( int i = 0; i < test_length; ++i )
        {
            VectorLikeOp op{};
            Key keyval{};
            int pos = 0;
            auto iterator = uut.begin();
            auto pair_iterator = uut.pairs().begin();

            if ( i + 1 == test_length )
            {
                op = VectorLikeOp::CheckAll;  // Override to always finish on a check
            }
            else if ( elements_to_add > 0 )
            {
                op = VectorLikeOp(urandom_range(unsigned(VectorLikeOp::EmplaceBack), unsigned(VectorLikeOp::PushBackSquareBrackets)));
                --elements_to_add;
            }
            else
            {
                op = VectorLikeOp(urandom_range(unsigned(VectorLikeOp::Size), unsigned(VectorLikeOp::CheckAll)));
            }

            if ( ref.empty() && RequiresContents(op) )
            {
                continue;
            }

            if ( RequiresKeyVal(op) )
            {
                randomize(keyval);

                if ( std::find(ref.begin(), ref.end(), keyval) != ref.end() )
                {
                    continue;  // Unique keys only - not exercising map-like overwriting in this test case
                }
            }

            if ( RequiresPos(op) )
            {
                pos = int(urandom_range(0, unsigned(ref.size() - 1)));
                for ( int n = 0; n < pos; ++n )
                {
                    if ( urandom_range(0, 1) )
                    {
                        ++iterator;
                        ++pair_iterator;
                    }
                    else
                    {
                        iterator++;
                        pair_iterator++;
                    }
                }
            }

            INFO("op = " << unsigned(op) << "\tkeyval = " << keyval << "\tpos = " << pos);

            switch ( op )
            {
                // Capacity
                case VectorLikeOp::Size:
                    CHECK(uut.size() == ref.size());
                    break;
                case VectorLikeOp::Empty:
                    CHECK(uut.empty() == ref.empty());
                    break;

                // Begin/End/Front/Back
                case VectorLikeOp::Begin:
                {
                    const auto cbegin = uut.begin();
                    CHECK(uut.begin() == cbegin);
                    if ( !ref.empty() )
                    {
                        CHECK(*uut.begin() == *cbegin);
                        CHECK(*uut.begin() == *ref.begin());
                    }
                }
                break;
                case VectorLikeOp::End:
                {
                    const auto cend = uut.end();
                    CHECK(uut.end() == cend);
                }
                break;
                case VectorLikeOp::Front:
                    CHECK(uut.front() == ref.front());
                    break;
                case VectorLikeOp::Back:
                    CHECK(uut.back() == ref.back());
                    break;

                // Erase
                case VectorLikeOp::Erase:
                {
                    auto ref_iterator = ref.erase(ref.begin() + pos);
                    auto uut_iterator = uut.erase(iterator);
                    if ( ref_iterator == ref.end() )
                    {
                        CHECK(uut_iterator == uut.end());
                    }
                    else
                    {
                        CHECK(*uut_iterator == *ref_iterator);
                    }
                }
                break;

                // Iterator access
                case VectorLikeOp::IteratorRead:
                    CHECK(*iterator == ref[pos]);
                    break;
                case VectorLikeOp::IteratorReadConst:
                {
                    const ordered_map<Key, Val> &const_uut{uut};
                    auto const_iterator = const_uut.begin();
                    for ( int n = 0; n < pos; ++n )
                    {
                        if ( urandom_range(0, 1) )
                        {
                            ++const_iterator;
                        }
                        else
                        {
                            const_iterator++;
                        }
                    }
                    CHECK(*const_iterator == ref[pos]);
                }
                break;
                case VectorLikeOp::IteratorWrite:
                    ref[pos] = keyval;
                    *iterator = keyval;
                    break;
                case VectorLikeOp::PairIteratorRead:
                    CHECK((*pair_iterator).second == ref[pos]);
                    CHECK(pair_iterator.value() == ref[pos]);
                    break;
                case VectorLikeOp::PairIteratorReadConst:
                {
                    const ordered_map<Key, Val> &const_uut{uut};
                    auto const_pair_iterator = const_uut.pairs().begin();
                    for ( int n = 0; n < pos; ++n )
                    {
                        if ( urandom_range(0, 1) )
                        {
                            ++const_pair_iterator;
                        }
                        else
                        {
                            const_pair_iterator++;
                        }
                    }
                    CHECK((*const_pair_iterator).second == ref[pos]);
                    CHECK(const_pair_iterator.value() == ref[pos]);
                }
                break;
                case VectorLikeOp::PairIteratorWrite:
                    ref[pos] = keyval;
                    pair_iterator.value() = keyval;
                    break;

                // Add new elements
                case VectorLikeOp::EmplaceBack:
                    ref.emplace_back(keyval);
                    uut.emplace(keyval, keyval);
                    break;
                case VectorLikeOp::PushBackInsert:
                    ref.push_back(keyval);
                    uut.insert(keyval, keyval);
                    break;
                case VectorLikeOp::PushBackSquareBrackets:
                    ref.push_back(keyval);
                    uut[keyval] = keyval;
                    break;
                case VectorLikeOp::PushBackSeveral:  // Add several elements in a row
                    elements_to_add = int(urandom_range(1, unsigned(capacity)));
                    break;

                // Iterate through every element
                case VectorLikeOp::CheckAll:
                case VectorLikeOp::Clear:  // Recently added elements may have never been checked, so check before
                                           // clearing
                {
                    CHECK(uut.size() == ref.size());
                    CHECK(uut.empty() == ref.empty());
                    auto actual_it = uut.begin();
                    for ( auto expect : ref )
                    {
                        REQUIRE(actual_it != uut.end());
                        CHECK(*actual_it == expect);
                        actual_it++;
                    }
                    CHECK(actual_it == uut.end());
                    auto expect_it = ref.begin();
                    for ( auto actual : uut )
                    {
                        REQUIRE(expect_it != ref.end());
                        CHECK(*expect_it == actual);
                        expect_it++;
                    }
                    CHECK(expect_it == ref.end());
                    if ( op == VectorLikeOp::Clear )
                    {
                        ref.clear();
                        uut.clear();
                    }
                }
                break;
                default:
                    break;
            }

            if ( debug )
            {
                // Additional checks after every operation in debug mode to end the test closer to the point of failure.
                // Disabled by default in case the act of reading the uut after every write masks a bug.
                REQUIRE(uut.size() == ref.size());
                REQUIRE(uut.empty() == ref.empty());
                auto actual_it = uut.begin();
                for ( auto expect : ref )
                {
                    REQUIRE(actual_it != uut.end());
                    REQUIRE(*actual_it == expect);
                    actual_it++;
                }
                REQUIRE(actual_it == uut.end());
                auto expect_it = ref.begin();
                for ( auto actual : uut )
                {
                    REQUIRE(expect_it != ref.end());
                    REQUIRE(*expect_it == actual);
                    expect_it++;
                }
                REQUIRE(expect_it == ref.end());
            }
        }
    }
}

TEST_CASE("ordered_map: shared_ptr")
{
    // Pathological case to reproduce an error observed in real use, where an ordered map of shared pointers was
    // destroying a shared pointer twice upon its removal, resulting in a double decrement of the reference count.

    // An object which complains when destroyed unexpectedly
    struct Obj
    {
        bool &ok_to_destroy;
        bool &return_fail;

        Obj(bool &ok, bool &fail) : ok_to_destroy(ok), return_fail(fail) {}

        ~Obj() { return_fail = return_fail || !ok_to_destroy; }
    };

    bool ok = false;
    bool failed = false;

    ordered_map<int, std::shared_ptr<Obj>> uut(3);

    // Hold a shared_ptr to every object we create so that nothing should be destroyed until the end of the test
    std::vector<std::shared_ptr<Obj>> refs;

    // Overflow hash table so that k=0 is the start of a hash chain
    for ( int k = 0; k < 10; ++k )
    {
        std::shared_ptr<Obj> ptr = std::make_shared<Obj>(ok, failed);
        uut[k] = ptr;
        refs.push_back(ptr);
    }

    // Remove an element from the overflow table
    uut.erase(5);

    // Remove an element from the front of a hash chain
    uut.erase(0);

    ok = true;
    CHECK(!failed);
}

static const std::pair<int, std::string> defaultValues[] = {
    {444, "text4"},
    {111, "text1"},
    {555, "text5"},
    {222, "text2"},
    {333, "text3"},
};

TEST_CASE("ordered_map: initialised construction")
{
    ordered_map<int, std::string> map(defaultValues, std::size(defaultValues));
    REQUIRE(map.size() == std::size(defaultValues));
    auto pos = map.begin();
    for ( size_t i = 0; i < std::size(defaultValues); i++ )
    {
        REQUIRE(defaultValues[i].second == *pos++);
    }
}


TEST_CASE("ordered_map: .at behaviour")
{
    ordered_map<int, std::string> map(defaultValues, std::size(defaultValues));
    try
    {
        auto &text1 = map.at(111);
        map.erase(333);
        auto &text2 = map.at(333);  // throws, jumping past the REQUIRE(false)
        REQUIRE(false);
    }
    catch ( std::out_of_range & )
    {
    }
}

TEST_CASE("ordered_map: copy assignment")
{
    ordered_map<int, std::string> map(defaultValues, std::size(defaultValues));

    // Construction copy
    ordered_map<int, std::string> map2(map);
    REQUIRE(map2.size() == map.size());
    REQUIRE(map2.at(333) == "text3");

    // Assignment copy
    ordered_map<int, std::string> map3;
    map3.emplace(1, "overwritten");
    map3 = map2;
    REQUIRE(map3.at(222) == "text2");
    REQUIRE(!map3.contains(1));
}

TEST_CASE("ordered_map: reverse value iterator")
{
    ordered_map<int, std::string> map(defaultValues, std::size(defaultValues));
    auto pos = map.rbegin();
    for ( auto i = std::size(defaultValues) - 1; i < std::size(defaultValues); i-- )
    {
        REQUIRE(defaultValues[i].second == *pos++);
    }
}

TEST_CASE("ordered_map: key_of inverse query")
{
    ordered_map<int, std::string> map(defaultValues, std::size(defaultValues));
    int temp;
    REQUIRE(map.key_of("text2", temp));
    REQUIRE(temp == 222);
    REQUIRE(map.key_of("text5", temp));
    REQUIRE(temp == 555);
    REQUIRE(map.key_of("text1", temp));
    REQUIRE(temp == 111);
}

TEST_CASE("ordered_map: zero initial capacity")
{
    ordered_map<int, std::string> map;
    REQUIRE(map.capacity() == 0);
    map[1] = "hello";
    REQUIRE(map.capacity() > 0);
}

TEST_CASE("ordered_map: unsigned indexer")
{
    ordered_map<int, int, armstd::ordered_map_hash<int>, uint8_t> map(127);

    for ( int i = 0; i < 253; i++ )
    {
        map.emplace(i, i);
    }

    bool allPresent = true;
    for ( int i = 0; i < 253; i++ )
    {
        allPresent = allPresent && (map.at(i) == i);
    }

    REQUIRE(map.capacity() == 253);  // Internal sizing behaviour should clamp
    REQUIRE(allPresent);
}
