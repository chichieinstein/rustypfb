#include <cmath>
#include <complex>
#include <immintrin.h>
using std::cyl_bessel_if;
using std::complex;

extern "C"{
    float bessel_func(float x)
    {
        return cyl_bessel_if(0.0, x);
    }
    void multiply_intrinsic(float* lhs, float* rhs, float* product, size_t ncols)
    {
        for(size_t j=0; j<ncols/16; j++)
        {
            __m512 a = _mm512_loadu_ps((__m512 *)(lhs + j));
            __m512 b = _mm512_loadu_ps((__m512 *)(rhs + j));
            __m512 c = _mm512_mul_ps(a, b);
            _mm512_storeu_ps(product+j, c);
        }
    }
    void filter_apply(complex<float>* lhs, float* rhs, complex<float>* product, size_t ncols)
    {
        multiply_intrinsic(reinterpret_cast<float*>(lhs), rhs, reinterpret_cast<float*>(product), ncols);
    }
}