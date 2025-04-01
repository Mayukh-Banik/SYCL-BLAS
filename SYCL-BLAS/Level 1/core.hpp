#pragma once
#include <stdlib.h>

class SimpleAxpy
{
    public:
        SimpleAxpy(float a) : a_(a) {}

        void operator()(float* x, float* y, size_t size) const {
            for (size_t i = 0; i < size; ++i) {
                y[i] += a_ * x[i];
            }
        }

    private:
        float a_;
};