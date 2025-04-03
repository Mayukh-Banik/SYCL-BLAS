#include <iostream>
#include <sycl/sycl.hpp>

int main()
{
        sycl::queue q;
    q.submit(
        [&](sycl::handler &handler)
        {
            handler.parallel_for(sycl::range<1>(100), [=](sycl::id<1> i)
                                 {
            });
        });
    return 0;
}