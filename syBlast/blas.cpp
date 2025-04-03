#include "blas.hpp"

void saxpy()
{
    sycl::queue q;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(10), [=](sycl::id<1> i) {
            int a = i[0];
            a = a * a;
        });
    }).wait();
}