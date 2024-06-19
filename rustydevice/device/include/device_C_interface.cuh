#include "cufft.h"
#include <cmath>

extern "C"
{
    void* memory_allocate_device(int);
    void memory_deallocate_device(void*);
    float bessel_func(float);
}
