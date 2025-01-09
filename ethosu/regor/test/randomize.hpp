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

#include "common/common.hpp"

#include <iterator>
#include <numeric>
#include <random>

// The random generator state is initialized from the Catch::rngSeed().
// This happens for each "test" to ensure random number consistency,
// even if only a single section was specified on the command line.
extern std::mt19937 default_rnd_generator;

namespace
{

// Randomizing a struct can easily be done using a friend function.
//
// struct MyStruct
// {
//     int a;
//     bool b;
//
//     [...]
//
//     friend void randomize(MyStruct& value)
//     {
//         randomize(value.a);
//         randomize(value.b);
//     }
// };
//
// MyStruct mine;
//
// randomize(mine);

// Randomize an integer (with an optional min/max value)
template<typename T, std::enable_if_t<std::numeric_limits<T>::is_integer, int> = 0>
void randomize(T &value, T min_value = std::numeric_limits<T>::min(), T max_value = std::numeric_limits<T>::max())
{
    if constexpr ( sizeof(T) == 1 )
    {
        std::uniform_int_distribution<int> dist(min_value, max_value);
        value = T(dist(default_rnd_generator));
    }
    else
    {
        std::uniform_int_distribution<T> dist(min_value, max_value);
        value = dist(default_rnd_generator);
    }
}

// Usage:
// std::vector<int> my_vec;
// my_vec.resize(250);
//
// randomize(my_vec.begin(), my_vec.end());
//
// Note that this handles any type that has a randomize() function defined.
template<typename InputIt>
void randomize(InputIt first, InputIt last)
{
    for ( InputIt it = first; it != last; ++it )
    {
        randomize(*it);
    }
}

// Usage:
// std::vector<int> my_vec;
// my_vec.resize(250);
//
// randomize(my_vec.begin(), my_vec.end(), 5, 10); // Constrain values to [5, 10]
template<typename InputIt, typename T>
void randomize(InputIt first, InputIt last, T min_value, T max_value)
{
    for ( InputIt it = first; it != last; ++it )
    {
        randomize(*it, min_value, max_value);
    }
}

// Helper template to easily randomize a std::vector
template<typename T>
void randomize(std::vector<T> &values)
{
    randomize(values.begin(), values.end());
}

// Create entire randomised vector with bounds
template<typename TYPE>
std::vector<TYPE> random_vector(int length, TYPE min, TYPE max)
{
    std::uniform_int_distribution<int> distribution(min, max);
    std::vector<TYPE> temp(length);
    std::generate(temp.begin(), temp.end(), [&]() { return TYPE(distribution(default_rnd_generator)); });
    return temp;
}

// Helper template to easily randomize a std::array
template<typename T, size_t S>
void randomize(std::array<T, S> &values)
{
    randomize(values.begin(), values.end());
}

// Helper template to easily randomize a fixed size array
template<typename T, size_t N>
void randomize(T (&values)[N])
{
    randomize(std::begin(values), std::end(values));
}

// Helper template to easily randomize a fixed size array
template<typename T, size_t N, typename T2>
void randomize(T (&values)[N], T2 min_value, T2 max_value)
{
    randomize(std::begin(values), std::end(values), min_value, max_value);
}

// Randomly pick one of the arguments. Useful for sparse enums
template<typename T, typename... Ts>
T random_of(T first, Ts... rest)
{
    T arr[] = {first, rest...};
    unsigned index;
    randomize(index, 0U, unsigned(sizeof...(rest)));
    return arr[index];
}

// Helper
inline unsigned urandom_range(unsigned min, unsigned max)
{
    unsigned ret;
    randomize(ret, min, max);
    return ret;
}

inline unsigned urandom()
{
    unsigned ret;
    randomize(ret);
    return ret;
}

}  // namespace
