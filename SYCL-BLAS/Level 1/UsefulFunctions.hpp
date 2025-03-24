#pragma once

#include <utility>

template <auto *FuncTable, typename... Args>
void functionSelector(int functionNumber, Args &&...args)
{
    FuncTable[functionNumber](std::forward<Args>(args)...);
}