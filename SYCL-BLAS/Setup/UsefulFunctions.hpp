#pragma once

#include <utility>

/**
 * @brief Calls a function from a function table based on the given function number.
 * 
 * @tparam FuncTable A pointer to an array of function pointers.
 * @tparam Args Variadic template parameters representing the argument types of the function.
 * @param functionNumber The index of the function to call from the function table.
 * @param args Arguments to be forwarded to the selected function.
 * 
 * @note This function assumes that the function table is properly initialized and that 
 *       functionNumber is within valid bounds.
 */
template <auto *FuncTable, typename... Args>
void functionSelector(int functionNumber, Args &&...args)
{
    FuncTable[functionNumber](std::forward<Args>(args)...);
}

template <int N>
int functionIndex(int index)
{
    return (index >= 0 && index < N) ? index : 0;
}
