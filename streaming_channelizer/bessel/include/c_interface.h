#ifndef C_INTERFACE_H
#define C_INTERFACE_H

#include <cmath>
#include <complex>
#include <immintrin.h>
#include <fftw3.h>
using std::cyl_bessel_if;
using std::complex;

extern "C"{
    float bessel_func(float);
    void multiply_intrinsic(float*, float*, float*, int);
    void multiply_complex_intrinsic(complex<float>*, float*, complex<float>*, int);
    // void strided_copy(complex<float>*, complex<float>*, int, int);
    // void multiply_complex_intrinsic(complex<float>*, complex<float>*, complex<float>*, int);
    void strided_copy(complex<float>* lhs, complex<float>* rhs, int nchannels, int nsamples);
    void convolve(complex<float>*, float*, complex<float>*, complex<float>*, int, int);
    void plain_multiply(complex<float>*, complex<float>*, complex<float>*, int);
    void transpose(complex<float>*, complex<float>*, int, int);
}

#endif