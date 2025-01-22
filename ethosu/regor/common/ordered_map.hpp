//
// SPDX-FileCopyrightText: Copyright 2021, 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace armstd
{

template<typename TYPE, typename ENABLE = void>
struct ordered_map_hash
{
    size_t operator()(const TYPE &value, size_t limit) const
    {
        // A better hash would map the incoming value to log2(limit) bits
        // before applying the modulo.
        // Disabled for hash performance
        // coverity[cert_int33_c_violation]
        return std::hash<TYPE>()(value) % limit;
    }
};

template<typename TYPE>
struct ordered_map_hash<TYPE, typename std::enable_if_t<std::is_pointer_v<TYPE>>>
{
    size_t operator()(TYPE value, size_t limit) const
    {
        uintptr_t hash = uintptr_t(value);
        // Pointers tend to have alignments that have lower bits set zero.
        // TODO: use alignof(std::remove_pointer_t<TYPE>) to determine lower zero count.
        hash ^= (hash >> 3) & 7;
        hash ^= (hash >> 15) * 3;
        // Disabled for hash performance
        // coverity[cert_int33_c_violation]
        return size_t(hash % limit);
    }
};

template<typename TYPE>
struct ordered_map_hash<TYPE, typename std::enable_if_t<std::is_integral_v<TYPE> || std::is_enum_v<TYPE>>>
{
    size_t operator()(TYPE value, size_t limit) const
    {
        // A C++ std::hash implementation need not return a linear mapping for arithmetic values because it is
        // implementation defined. The hash function required for this ordered map should perform exactly the same
        // across all platforms and ideally produce sensible hashes for small-valued incrementing keys (not all
        // compilers do this).
        size_t hash = size_t(value);
        // Mix in upper bits if the type is large (enum bit flags, for example)
        if constexpr ( sizeof(value) > 2 )
        {
            hash ^= (size_t(value) >> 15) * 3;
            hash ^= (size_t(value) >> 24) * 5;
        }
        // Disabled for hash performance
        // coverity[cert_int33_c_violation]
        return hash % limit;
    }
};


template<typename KEY, typename VALUE, typename HASH = ordered_map_hash<KEY>, typename INDEXER = int16_t, bool PURE_HASH_CHAINS = true>
class ordered_map
{
protected:
    using this_class_t = ordered_map<KEY, VALUE, HASH, INDEXER, PURE_HASH_CHAINS>;
    static_assert(std::is_integral<INDEXER>::value, "indexer must be integer");
    static constexpr INDEXER HASH_FREE = INDEXER(-2);
    static constexpr INDEXER HASH_END = INDEXER(-1);
    static constexpr INDEXER NODE_UNLINKED = INDEXER(-1);
    static constexpr size_t DEFAULT_INITIAL_SIZE = 3;
    static constexpr size_t MAX_INDEX_CAPACITY = size_t(std::numeric_limits<INDEXER>::max() - 2);

    // Node stores two arrays in one allocation.
    //  - Items array, mapping key->untyped storage
    //  - The traversal order list
    struct Node
    {
        alignas(alignof(VALUE)) uint8_t value[sizeof(VALUE)];  // Untyped value storage (place first)
        KEY key;                                               // Map key
        INDEXER order_next = NODE_UNLINKED;                    // order forwards traversal
        INDEXER order_prev = NODE_UNLINKED;                    // order backwards traversal
        INDEXER hash_next = HASH_FREE;  // Same hash collision relocation (-2=free, -1=used/end, otherwise=next bucket)

        Node() = default;

        VALUE &Value() { return *reinterpret_cast<VALUE *>(reinterpret_cast<void *>(&value)); }

        const VALUE &Value() const { return *reinterpret_cast<const VALUE *>(reinterpret_cast<const void *>(&value)); }

        void copy_links(Node &other)
        {
            this->order_next = other.order_next;
            this->order_prev = other.order_prev;
            this->hash_next = other.hash_next;
        }
    };

    std::unique_ptr<Node[]> _items;       // Bulk node store
    INDEXER _capacity = 0;                // Total allocated capacity
    INDEXER _itemCount = 0;               // Number of inserted/valid items
    INDEXER _tableSize = 0;               // Hash table size
    INDEXER _orderBegin = NODE_UNLINKED;  // First item in insertion order
    INDEXER _orderLast = NODE_UNLINKED;   // Last item in insertion order
    INDEXER _allocWatermark = 0;          // Location of last overflow allocation

public:
    ordered_map(size_t initialCapacity = 0)
    {
        if ( initialCapacity ) resize(initialCapacity);
    }

    ordered_map(std::initializer_list<std::pair<KEY, VALUE>> init)
    {
        resize(init.size());
        for ( auto &pair : init )
        {
            emplace(pair.first, pair.second);
        }
    }

    ordered_map(const std::pair<KEY, VALUE> *pairs, size_t count)
    {
        resize(count);
        while ( count-- )
        {
            emplace(pairs->first, pairs->second);
            pairs++;
        }
    }

    ordered_map(const ordered_map &other) { *this = other; }

    ordered_map(ordered_map &&other) { *this = std::move(other); }

    ordered_map &operator=(const ordered_map &other)
    {
        // Can't selectively delete operator= via enable_if_t
        if constexpr ( !std::is_copy_constructible<VALUE>::value )
        {
            throw std::invalid_argument("value type non-copyable");
        }
        else if ( this != &other )
        {
            clear();

            // Duplicating a map will compact and rehash it
            // pad a little to prevent an immediate resize
            // if the user appends an item.
            resize(other._itemCount + 2);

            // Duplicate in source's insertion order
            int order = other._orderBegin;
            while ( order != NODE_UNLINKED )
            {
                auto *from = &other._items[order];
                int index = table_index_of(from->key);
                auto *to = allocate_node(index, from->key);
                to->key = from->key;
                ::new (reinterpret_cast<void *>(&to->Value())) VALUE(from->Value());
                order = from->order_next;
            }
        }
        return *this;
    }

    ordered_map &operator=(ordered_map &&other)
    {
        if ( this != &other )
        {
            clear();

            _items = std::move(other._items);
            _capacity = other._capacity;
            _itemCount = other._itemCount;
            _tableSize = other._tableSize;
            _orderBegin = other._orderBegin;
            _orderLast = other._orderLast;
            _allocWatermark = other._allocWatermark;
            other._itemCount = 0;
            other._capacity = 0;
            other._orderBegin = NODE_UNLINKED;
        }
        return *this;
    }

    ~ordered_map() { clear(); }

public:
    size_t size() const { return size_t(_itemCount); }
    bool empty() const { return _itemCount == 0; }
    size_t capacity() const { return size_t(_capacity); }
    bool contains(const KEY &key) const
    {
        if ( empty() ) return false;
        int index = table_index_of(key);
        return find_node(index, key) != nullptr;
    }

    void clear()
    {
        if ( !_items ) return;
        // Call destructors in order
        int order = _orderBegin;
        for ( int i = 0; i < _itemCount; i++ )
        {
            assert((i < _itemCount - 1) || (order == _orderLast));  // Consistency error
            _items[order].Value().~VALUE();
            order = _items[order].order_next;
        }
        assert(order == NODE_UNLINKED);  // Consistency error

        // Reset every entry in the hash table
        for ( int i = 0; i < _capacity; i++ )
        {
            _items[i].hash_next = HASH_FREE;
            _items[i].order_next = NODE_UNLINKED;
            _items[i].order_prev = NODE_UNLINKED;
        }

        _itemCount = 0;
        _orderBegin = NODE_UNLINKED;
        _orderLast = NODE_UNLINKED;
    }

    template<class... Args>
    VALUE &emplace(const KEY &key, Args &&...args)
    {
        if ( !_tableSize ) resize(DEFAULT_INITIAL_SIZE);
        int index = table_index_of(key);
        Node *node = find_node(index, key);
        if ( node == nullptr )
        {
            node = allocate_node(index, key);
            ::new (reinterpret_cast<void *>(&node->value)) VALUE(std::forward<Args>(args)...);
        }
        return node->Value();
    }

    // insert(key, value)
    //  - Appends a new node for key at the end of the iteration order and copies or moves value into it
    //  - The key must not already be present in the map
    //  - If the map is at capacity then resizing will occur
    //  - If resizing occurs, all iterators and references are invalidated. Otherwise, they are unaffected.

    void insert(const KEY &key, const VALUE &value)
    {
        if ( !_tableSize ) resize(DEFAULT_INITIAL_SIZE);
        int index = table_index_of(key);
        Node *node = find_node(index, key);
        if ( node == nullptr )
        {
            node = allocate_node(index, key);
            ::new (reinterpret_cast<void *>(&node->value)) VALUE(value);  // Requires accessible copy constructor
        }
        else
        {
            assert(false && "key already present");
        }
    }

    void insert(const KEY &key, VALUE &&value)
    {
        if ( !_tableSize ) resize(DEFAULT_INITIAL_SIZE);
        int index = table_index_of(key);
        Node *node = find_node(index, key);
        if ( node == nullptr )
        {
            node = allocate_node(index, key);
            ::new (reinterpret_cast<void *>(&node->value)) VALUE(std::move(value));  // Requires accessible move
                                                                                     // constructor
        }
        else
        {
            assert(false && "key already present");
        }
    }

    // reinsert(key, value)
    //  - If key is not already present in the map:
    //    - Is equivalent to insert
    //    - Appends a new node for key at the end of the iteration order and copies or moves value into it
    //    - If the map is at capacity then resizing will occur
    //    - If resizing occurs, all iterators and references are invalidated. Otherwise, they are unaffected.
    //  - If key is already present in the map:
    //    - Moves the node at key to the back of the iteration order
    //    - Replaces the VALUE object at key with value
    //    - No hash chains are modified
    //    - All iterators remain valid, but iterators to the reinserted node will point to its new position

    void reinsert(const KEY &key, const VALUE &value)
    {
        if ( !_tableSize ) resize(DEFAULT_INITIAL_SIZE);
        int index = table_index_of(key);
        Node *node = find_node(index, key);
        if ( node == nullptr )
        {
            node = allocate_node(index, key);
            ::new (reinterpret_cast<void *>(&node->value)) VALUE(value);  // Requires accessible copy constructor
        }
        else  // Reinsert re-links at end of ordering chain
        {
            if ( node->order_next != NODE_UNLINKED )  // No need to relink if already at end of chain
            {
                index = int(node - _items.get());  // Get actual node index
                unlink_node_order(node);

                // relink node at end of chain
                _items[_orderLast].order_next = INDEXER(index);
                node->order_prev = _orderLast;
                node->order_next = NODE_UNLINKED;
                _orderLast = INDEXER(index);
            }

            // Re-assign value (last)
            node->Value() = value;
        }
    }

    void reinsert(const KEY &key, VALUE &&value)
    {
        if ( !_tableSize ) resize(DEFAULT_INITIAL_SIZE);
        int index = table_index_of(key);
        Node *node = find_node(index, key);
        if ( node == nullptr )
        {
            node = allocate_node(index, key);
            ::new (reinterpret_cast<void *>(&node->value)) VALUE(std::move(value));  // Requires accessible move
                                                                                     // constructor
        }
        else  // Reinsert re-links at end of ordering chain
        {
            if ( node->order_next != NODE_UNLINKED )  // No need to relink if already at end of chain
            {
                index = int(node - _items.get());  // Get actual node index
                unlink_node_order(node);

                // relink node at end of chain
                _items[_orderLast].order_next = INDEXER(index);
                node->order_prev = _orderLast;
                node->order_next = NODE_UNLINKED;
                _orderLast = INDEXER(index);
            }

            // Re-assign value (last)
            node->Value() = std::move(value);
        }
    }

    // replace(oldKey, newKey, value)
    //  - Replaces the node at oldKey with a new key and value but same position in iteration order
    //  - The new key must not already be present in the map
    //  - All iterators and references are invalidated
    //  - If the map is at its capacity, resizing will occur

    void replace(const KEY &oldKey, const KEY &newKey, const VALUE &value)
    {
        insert(newKey, value);
        swap_order(find(oldKey), find(newKey));
        erase(oldKey);
    }

    // operator[]
    //  - Analogous to unordered_map::operator[]
    //  - If insertion occurs, the new node is appended to the end of the iteration order
    //  - If insertion occurs and the map is at capacity then resizing will occur
    //  - If resizing occurs, all iterators and references are invalidated. Otherwise, they are unaffected.

    VALUE &operator[](const KEY &key)
    {
        if ( !_tableSize ) resize(DEFAULT_INITIAL_SIZE);
        int index = table_index_of(key);
        Node *node = find_node(index, key);
        if ( node == nullptr )
        {
            node = allocate_node(index, key);
            ::new (reinterpret_cast<void *>(&node->value)) VALUE();
        }
        return node->Value();
    }

    const VALUE &operator[](const KEY &key) const
    {
        int index = table_index_of(key);
        const Node *node = find_node(index, key);
        if ( node == nullptr ) throw std::out_of_range("missing key");
        return node->Value();
    }

    const VALUE &at(const KEY &key) const
    {
        if ( !_tableSize ) throw std::out_of_range("not initialised");
        int index = table_index_of(key);
        const Node *node = find_node(index, key);
        if ( node == nullptr ) throw std::out_of_range("missing key");
        return node->Value();
    }

    VALUE &at(const KEY &key) { return const_cast<VALUE &>(const_cast<const this_class_t *>(this)->at(key)); }

    bool try_get(const KEY &key, VALUE &value) const
    {
        if ( empty() ) return false;
        int index = table_index_of(key);
        const Node *node = find_node(index, key);
        if ( node != nullptr )
        {
            value = node->Value();
            return true;
        }
        return false;
    }

    const VALUE *try_ref(const KEY &key) const
    {
        if ( empty() ) return nullptr;
        int index = table_index_of(key);
        const Node *node = find_node(index, key);
        if ( node != nullptr )
        {
            return &node->Value();
        }
        return nullptr;
    }

    VALUE *try_ref(const KEY &key) { return const_cast<VALUE *>(const_cast<const this_class_t *>(this)->try_ref(key)); }

    const VALUE &front() const
    {
        if ( empty() ) throw std::out_of_range("no keys");
        return _items[_orderBegin].Value();
    }

    const VALUE &back() const
    {
        if ( empty() ) throw std::out_of_range("no keys");
        return _items[_orderLast].Value();
    }

    // Return all keys (in insertion order)
    std::vector<KEY> keys() const
    {
        std::vector<KEY> tmp;
        tmp.reserve(_itemCount);
        // Collect the keys in order
        int order = _orderBegin;
        for ( int i = 0; i < _itemCount; i++ )
        {
            const Node &item = _items[order];
            assert(item.hash_next != HASH_FREE);
            tmp.push_back(item.key);
            order = item.order_next;
        }
        assert(order == NODE_UNLINKED);
        return tmp;
    }

    template<bool PAIRS, bool REVERSE, bool IS_CONST = false>
    class iterator_base
    {
        friend this_class_t;
        using value_ref_t = typename std::conditional_t<IS_CONST, const VALUE &, VALUE &>;
        using value_ptr_t = typename std::conditional_t<IS_CONST, const VALUE *, VALUE *>;
        using node_t = typename std::conditional_t<IS_CONST, const Node *, Node *>;
        using iterator_base_t = iterator_base<PAIRS, REVERSE, IS_CONST>;

    protected:
        node_t _items;
        int _at;

    public:
        iterator_base() = default;
        iterator_base(const iterator_base_t &other) { *this = other; }
        iterator_base(node_t items, int start) : _items(items), _at(start) {}

        // Value-only access
        template<bool PP = PAIRS, typename std::enable_if_t<!PP, int> = 0>
        value_ref_t operator*() const
        {
            return _items[_at].Value();
        }

        template<bool PP = PAIRS, typename std::enable_if_t<!PP, int> = 0>
        value_ptr_t operator->()
        {
            return &_items[_at].Value();
        }

        // Pair access
        template<bool PP = PAIRS, typename std::enable_if_t<PP, int> = 0>
        const KEY &key() const
        {
            return _items[_at].key;
        }
        template<bool PP = PAIRS, typename std::enable_if_t<PP, int> = 0>
        value_ref_t value() const
        {
            return _items[_at].Value();
        }

        template<bool PP = PAIRS, typename std::enable_if_t<PP, int> = 0>
        std::pair<KEY, value_ref_t> operator*() const
        {
            return std::pair<KEY, value_ref_t>(key(), value());
        }

        iterator_base_t &operator++()
        {
            if constexpr ( REVERSE ) _at = _items[_at].order_prev;
            else _at = _items[_at].order_next;
            return *this;
        }
        iterator_base_t operator++(int)
        {
            int tmp = _at;
            if constexpr ( REVERSE ) _at = _items[_at].order_prev;
            else _at = _items[_at].order_next;
            return iterator_base_t(_items, tmp);
        }

        iterator_base_t &operator--()
        {
            if constexpr ( REVERSE ) _at = _items[_at].order_next;
            else _at = _items[_at].order_prev;
            return *this;
        }
        iterator_base_t operator--(int)
        {
            int tmp = _at;
            if constexpr ( REVERSE ) _at = _items[_at].order_next;
            else _at = _items[_at].order_prev;
            return iterator_base_t(_items, tmp);
        }

        iterator_base_t &operator=(const iterator_base_t &other)
        {
            _items = other._items;
            _at = other._at;
            return *this;
        }

        bool operator==(const iterator_base<PAIRS, REVERSE, false> &b) const
        {
            assert(_items == b._items);
            return _at == b._at;
        }

        bool operator==(const iterator_base<PAIRS, REVERSE, true> &b) const
        {
            assert(_items == b._items);
            return _at == b._at;
        }

        bool operator!=(const iterator_base<PAIRS, REVERSE, false> &b) const
        {
            assert(_items == b._items);
            return _at != b._at;
        }

        bool operator!=(const iterator_base<PAIRS, REVERSE, true> &b) const
        {
            assert(_items == b._items);
            return _at != b._at;
        }

    protected:
        Node *node() const { return const_cast<Node *>(&_items[_at]); }
    };

    using iterator = iterator_base<false, false>;
    using reverse_iterator = iterator_base<false, true>;
    using pair_iterator = iterator_base<true, false>;
    using const_iterator = iterator_base<false, false, true>;
    using const_reverse_iterator = iterator_base<false, true, true>;
    using const_pair_iterator = iterator_base<true, false, true>;

    // Forward value iterators
    iterator begin() { return iterator(_items.get(), _orderBegin); }
    iterator end() { return iterator(_items.get(), NODE_UNLINKED); }
    const_iterator begin() const { return const_iterator(_items.get(), _orderBegin); }
    const_iterator end() const { return const_iterator(_items.get(), NODE_UNLINKED); }

    // Reverse value iterators
    reverse_iterator rbegin() { return reverse_iterator(_items.get(), _orderLast); }
    reverse_iterator rend() { return reverse_iterator(_items.get(), NODE_UNLINKED); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(_items.get(), _orderLast); }
    const_reverse_iterator rend() const { return const_reverse_iterator(_items.get(), NODE_UNLINKED); }

    template<bool IS_CONST>
    class iterator_proxy
    {
    private:
        using outer_type_t = typename std::conditional_t<IS_CONST, const this_class_t &, this_class_t &>;
        outer_type_t _outer;

    public:
        iterator_proxy(outer_type_t outer) : _outer(outer) {}
        pair_iterator begin() { return pair_iterator(_outer._items.get(), _outer._orderBegin); }
        pair_iterator end() { return pair_iterator(_outer._items.get(), NODE_UNLINKED); }
        const_pair_iterator begin() const { return const_pair_iterator(_outer._items.get(), _outer._orderBegin); }
        const_pair_iterator end() const { return const_pair_iterator(_outer._items.get(), NODE_UNLINKED); }
    };

    iterator_proxy<false> pairs() { return iterator_proxy<false>(*this); }
    iterator_proxy<true> pairs() const { return iterator_proxy<true>(*this); }

    iterator find(const KEY &key)
    {
        int index = table_index_of(key);
        const Node *node = find_node(index, key);
        if ( node == nullptr )
        {
            return end();
        }
        return iterator(_items.get(), int(node - _items.get()));
    }

    const_iterator find(const KEY &key) const
    {
        int index = table_index_of(key);
        const Node *node = find_node(index, key);
        if ( node == nullptr )
        {
            return end();
        }
        return const_iterator(_items.get(), int(node - _items.get()));
    }

    // reorder_after(key, position) and reorder_before(key, position)
    //  - Moves a node to a new position in iteration order
    //    - reorder_after() moves the node to immediately after position in iteration order
    //    - reorder_before() moves the node to immediately before position in iteration order
    //  - Key must already be present in the map
    //  - end() can be used as the position for reorder_before() but not reorder_after()
    //  - No hash chains are modified
    //  - No VALUE objects are moved, copied, created or destroyed
    //  - Container size is not changed and no reallocation occurs
    //  - All iterators remain valid, but iterators to the reordered node will point to its new position

    void reorder_after(const KEY &key, iterator position)
    {
        int index = table_index_of(key);
        Node *node = find_node(index, key);
        reorder_node(node, position.node(), true);
    }

    void reorder_before(const KEY &key, iterator position)
    {
        int index = table_index_of(key);
        Node *node = find_node(index, key);
        if ( position == end() )
        {
            reorder_node(node, &_items[_orderLast], true);
        }
        else
        {
            reorder_node(node, position.node(), false);
        }
    }

    // erase(const KEY &key)
    //  - Analogous to std::unordered_map::erase()
    //  - Removes key from map, if present, and returns the number of elements removed
    //  - Unlike unordered_map::erase(), all iterators and references are invalidated (not just the removed element)

    int erase(const KEY &key)
    {
        if ( !_tableSize ) return 0;
        int index = table_index_of(key);
        int prevIndex = -1;  // local sentinel - not the indexer value
        Node *node = find_node(index, key, &prevIndex);
        if ( node != nullptr )
        {
            deallocate_node(node, prevIndex);
            return 1;
        }
        return 0;
    }

    // erase(iterator pos)
    //  - Analogous to std::vector::erase()
    //  - Removes the node at pos and returns an iterator to the next node in iteration order
    //  - Unlike vector::erase(), all iterators and references are invalidated (not just the removed element onwards)

    iterator erase(iterator pos)
    {
        assert(pos._at != NODE_UNLINKED);
        const auto &key = _items[pos._at].key;
        int index = table_index_of(key);  // Locate in hash chain (iterators walk the ordering chain)
        int prevIndex = -1;               // local sentinel - not the indexer value
        int nextOrder = _items[pos._at].order_next;
        Node *node = find_node(index, key, &prevIndex);
        assert(node != nullptr);
        deallocate_node(node, prevIndex);
        return iterator(_items.get(), nextOrder);
    }

    // swap_order()
    //  - Swaps the iteration order of two nodes specified by iterator
    //  - Both iterators must be dereferenceable
    //  - No hash chains are modified
    //  - No VALUE objects are moved, copied, created or destroyed
    //  - Container size is not changed and no reallocation occurs
    //  - All iterators remain valid, but iterators to either of the two swapped nodes will point to their new position

    void swap_order(iterator first, iterator second)
    {
        assert(first != end() && second != end());
        swap_node_order(first.node(), second.node());
    }

    bool key_of(const VALUE &value, KEY &key) const
    {
        if ( !_tableSize ) return false;
        int order = _orderBegin;
        while ( order != NODE_UNLINKED )
        {
            auto *node = &_items[order];
            if ( node->Value() == value )
            {
                key = node->key;
                return true;
            }
            order = node->order_next;
        }
        return false;
    }

private:
    ordered_map(this_class_t &other, size_t capacity)
    {
        resize(capacity);

        // Duplicate in source's insertion order
        int order = other._orderBegin;
        while ( order != NODE_UNLINKED )
        {
            auto *from = &other._items[order];
            int index = table_index_of(from->key);
            auto *to = allocate_node(index, from->key);
            copy_move_helper<KEY>()(to->key, from->key);
            ::new (reinterpret_cast<void *>(&to->value)) VALUE();
            copy_move_helper<VALUE>()(to->Value(), from->Value());
            order = from->order_next;
        }
    }

    void resize(size_t capacity)
    {
        // Resize capacity limited to chosen indexer
        capacity = std::min(capacity, MAX_INDEX_CAPACITY);
        INDEXER newTableSize = INDEXER(hashtable_size(capacity));
        if ( capacity <= size_t(_capacity) )
        {
            assert(_capacity != MAX_INDEX_CAPACITY && "Reached Indexer Limit");
            return;
        }

        // Same hash table size, just move old items into the new item storage
        if ( !_items || _tableSize == newTableSize )
        {
            std::unique_ptr<Node[]> newItems = std::make_unique<Node[]>(capacity);
            if ( _items )
            {
                // Probably resizing because we are full, so move everything
                for ( int i = 0; i < _capacity; i++ )
                {
                    auto *from = &_items[i];
                    auto *to = &newItems[i];
                    to->copy_links(*from);
                    copy_move_helper<KEY>()(to->key, from->key);
                    ::new (reinterpret_cast<void *>(&to->value)) VALUE();
                    copy_move_helper<VALUE>()(to->Value(), from->Value());
                }
            }
            _items = std::move(newItems);
            _capacity = INDEXER(capacity);
            _tableSize = INDEXER(newTableSize);
            _allocWatermark = PURE_HASH_CHAINS ? _tableSize : 0;
        }
        else  // Rehash by stealing the internals of another map
        {
            // This may occur recursively if the hash function is poor
            this_class_t temp(*this, capacity);
            *this = std::move(temp);
        }
        assert(_tableSize != 0);
    }

    Node *allocate_node(int index, const KEY &key)
    {
        // This function must only be called when an allocation is required
        assert(index >= 0 && index < _tableSize);
        // Try to insert into the hash table, if given index isn't free then find a free node and link onto that
        while ( _items[index].hash_next != HASH_FREE )
        {
            int prev = index;
            index = find_free_index();
            if ( index < 0 )
            {
                // Conservative resize strategy
                resize(_capacity + (_capacity + 1) / 2);
                index = table_index_of(key);
                continue;
            }

            // Find the end of the hash chain
            while ( _items[prev].hash_next != HASH_END )
            {
                prev = _items[prev].hash_next;
            }

            _items[prev].hash_next = INDEXER(index);
        }

        Node *node = &_items[index];
        node->hash_next = HASH_END;  // this node is now used but not linked
        node->key = key;
        if ( _orderBegin == NODE_UNLINKED )
        {
            _orderBegin = _orderLast = INDEXER(index);
            node->order_next = node->order_prev = NODE_UNLINKED;
        }
        else
        {
            _items[_orderLast].order_next = INDEXER(index);
            node->order_next = NODE_UNLINKED;
            node->order_prev = _orderLast;
            _orderLast = INDEXER(index);
        }

        _itemCount++;
        return node;
    }

    void deallocate_node(Node *node, int prevIndex)
    {
        // Remove from ordering chain FIRST:
        unlink_node_order(node);

        // Remove from hash chain: Just unlink node if not in initial bucket
        if ( prevIndex != -1 )
        {
            node->Value().~VALUE();
            _items[prevIndex].hash_next = node->hash_next;
            node->hash_next = HASH_FREE;
        }
        else if ( node->hash_next == HASH_END )
        {
            node->Value().~VALUE();
            node->hash_next = HASH_FREE;
        }
        // If we are at the start of a chain then we need to move the next hashed node into this bucket (painful)
        else
        {
            // If the hash table is used as overflow allocation (i.e. PURE_HASH_CHAINS == false)
            // the next node in this chain may also be the start of another chain, which is not handled here.
            assert(PURE_HASH_CHAINS);  // TODO: Add support for deallocation when PURE_HASH_CHAINS==false

            Node *next = &_items[node->hash_next];
            copy_move_helper<VALUE>()(node->Value(), next->Value());
            node->key = next->key;
            node->hash_next = next->hash_next;
            next->hash_next = HASH_FREE;

            // Relink ordering for the moved node
            node->order_next = next->order_next;
            node->order_prev = next->order_prev;
            update_node_order(node);

            next->order_next = NODE_UNLINKED;
            next->order_prev = NODE_UNLINKED;
        }

        _itemCount--;
    }

    void unlink_node_order(Node *node)
    {
        // If this is not the first node, unlink from previous
        if ( node->order_prev != NODE_UNLINKED )
        {
            _items[node->order_prev].order_next = node->order_next;
        }
        else
        {
            _orderBegin = node->order_next;
        }

        // If this is not the last node, unlink from next
        if ( node->order_next != NODE_UNLINKED )
        {
            _items[node->order_next].order_prev = node->order_prev;
        }
        else
        {
            _orderLast = node->order_prev;
        }

        node->order_next = NODE_UNLINKED;
        node->order_prev = NODE_UNLINKED;
    }

    void update_node_order(Node *node)
    {
        const auto index = INDEXER(node - _items.get());
        assert(size_t(index) < MAX_INDEX_CAPACITY);
        if ( node->order_prev == NODE_UNLINKED )
        {
            _orderBegin = index;
        }
        else
        {
            _items[node->order_prev].order_next = index;
        }
        if ( node->order_next == NODE_UNLINKED )
        {
            _orderLast = index;
        }
        else
        {
            _items[node->order_next].order_prev = index;
        }
    }

    void reorder_node(Node *node, Node *position, bool after)
    {
        assert(node != nullptr);
        assert(position != nullptr);

        if ( node != position )
        {
            unlink_node_order(node);

            if ( after )
            {
                node->order_next = position->order_next;
                node->order_prev = INDEXER(position - _items.get());
                assert(size_t(node->order_prev) < MAX_INDEX_CAPACITY);
            }
            else
            {
                node->order_next = INDEXER(position - _items.get());
                node->order_prev = position->order_prev;
                assert(size_t(node->order_next) < MAX_INDEX_CAPACITY);
            }

            update_node_order(node);
        }
    }

    void swap_node_order(Node *first, Node *second)
    {
        // Given the two nodes, swap the traversal orders
        const INDEXER firstPrev = first->order_prev;
        const INDEXER firstNext = first->order_next;

        first->order_prev = second->order_prev;
        first->order_next = second->order_next;

        second->order_prev = firstPrev;
        second->order_next = firstNext;

        // If the two nodes were adjacent before swapping, then they will each be pointing at themselves now.
        // Point them at each other instead.
        const auto firstIndex = INDEXER(first - _items.get());
        const auto secondIndex = INDEXER(second - _items.get());
        assert(size_t(firstIndex) < MAX_INDEX_CAPACITY);
        assert(size_t(secondIndex) < MAX_INDEX_CAPACITY);

        if ( firstPrev == secondIndex )
        {
            first->order_next = secondIndex;
            second->order_prev = firstIndex;
        }
        else if ( firstNext == secondIndex )
        {
            first->order_prev = secondIndex;
            second->order_next = firstIndex;
        }

        // Inform the affected neighbours of the change in order
        update_node_order(first);
        update_node_order(second);
    }

    const Node *find_node(int index, const KEY &key, int *prev = nullptr) const
    {
        // Initial index must be within the hashtable
        assert(index >= 0 && index < _tableSize);
        assert(_items);

        // Node is unallocated
        if ( _items[index].hash_next == HASH_FREE )
        {
            return nullptr;
        }

        // Look for matching key
        do
        {
            if ( _items[index].key == key )
            {
                assert(_items[index].hash_next != HASH_FREE);  // This node MUST BE allocated
                return &_items[index];
            }
            if ( prev )
            {
                *prev = index;
            }
            index = _items[index].hash_next;
        } while ( index != HASH_END );

        return nullptr;
    }

    Node *find_node(int index, const KEY &key, int *prev = nullptr)
    {
        return const_cast<Node *>(const_cast<const this_class_t *>(this)->find_node(index, key, prev));
    }

    template<typename TYPE, typename ENABLE = void>
    struct copy_move_helper
    {
        void operator()(TYPE &dst, TYPE &src) const
        {
            dst = src;
            src.~TYPE();
        }
    };

    template<typename TYPE>
    struct copy_move_helper<TYPE, typename std::enable_if<std::is_move_assignable<TYPE>::value>::type>
    {
        void operator()(TYPE &dst, TYPE &src) const { dst = std::move(src); }
    };

    template<typename TYPE>
    // clang-format off
    struct copy_move_helper<TYPE, typename std::enable_if<std::is_trivially_copyable<TYPE>::value && !(std::is_move_assignable<TYPE>::value || std::is_arithmetic<TYPE>::value || std::is_pointer<TYPE>::value)>::type >
    // clang-format on
    {
        // coverity[dont_call:FALSE]
        void operator()(TYPE &dst, TYPE &src) const { std::memcpy(&dst, &src, sizeof(TYPE)); }
    };

    int table_index_of(const KEY &key, INDEXER tableSize) const
    {
        assert(tableSize);
        return int(HASH()(key, size_t(tableSize)));
    }
    int table_index_of(const KEY &key) const { return table_index_of(key, _tableSize); }

    int find_free_index()
    {
        const INDEXER MIN_ALLOC = PURE_HASH_CHAINS ? _tableSize : 0;

        // Search for a free slot, starting before the watermark.
        INDEXER i = _allocWatermark;
        do
        {
            i--;
            if ( (i < MIN_ALLOC) || (i >= _capacity) ) i = _capacity - 1;
            if ( _items[i].hash_next == HASH_FREE )
            {
                _allocWatermark = i;  // Watermark should be a valid (but used) slot.
                assert(_allocWatermark >= MIN_ALLOC && _allocWatermark < _capacity);
                return i;
            }
        } while ( i != _allocWatermark );

        return -1;  // Error, no free slots
    }

    size_t hashtable_size(size_t &capacity)
    {
        // Prime table is tuned for rehashing while the capacity is small. Whereas
        // table sizes for larger capacities are sparser, when rehashing is expensive.
        // Note: Not all starting capacities give the same resize pattern!
        // clang-format off
        static constexpr size_t primes[] = { 3, 7, 11, 13, 17, 19, 23, 31, 41, 67, 97, 131,
                                             197, 257, 509, 1021, 2039, 4093, 8191, 16381 };
        // clang-format on
        capacity = std::max<size_t>(capacity, 2);

        // Estimate to expect ~1 collision per element
        size_t estimatedTableSize = (capacity + 1) / 2;
        size_t tableSize = 2;
        // Choose a conservative prime hashtable size for small capacities
        // (stops rehashing after final table size has been seen)
        for ( size_t i : primes )
        {
            if ( i > estimatedTableSize || tableSize >= MAX_INDEX_CAPACITY )
            {
                break;
            }
            tableSize = i;
        }

        // Ensure capacity includes space other than just the table
        capacity = std::min(std::max(capacity, tableSize * 2), MAX_INDEX_CAPACITY);
        assert(tableSize != 0);
        return tableSize;
    }
};

}  // namespace armstd


// WORKAROUND: Pull into regor namespace
namespace regor
{
template<typename KEY, typename VALUE, typename HASH = ::armstd::ordered_map_hash<KEY>, typename INDEXER = int16_t, bool PURE_HASH_CHAINS = true>
using ordered_map = armstd::ordered_map<KEY, VALUE, HASH, INDEXER, PURE_HASH_CHAINS>;

}  // namespace regor
