#pragma once

namespace syBlast
{
    namespace NumericFunctions
    {
        template <typename T>
        T SQRT(T n, T l = 1e-06)
        {
            if (n <= 0)
            {
                return 0;
            }
            else if (n == 1)
            {
                return 1;
            }
            T x = n;
            T root;
            while (1)
            {
                root = 0.5 * (x + (n / x));
                if (sycl::abs(root - x) < l)
                    break;
                x = root;
            }

            return root;
        }
    }
}