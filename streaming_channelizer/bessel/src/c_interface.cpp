
#include "../include/c_interface.h"

extern "C"{
    float bessel_func(float x)
    {
        return cyl_bessel_if(0.0, x);
    }
    void multiply_intrinsic(float* lhs, float* rhs, float* product, int ncols)
    {
        for(int j=0; j<ncols; j += 16)
        {
            __m512 a = _mm512_loadu_ps((__m512 *)(lhs + j));
            __m512 b = _mm512_loadu_ps((__m512 *)(rhs + j));
            __m512 c = _mm512_mul_ps(a, b);
            _mm512_storeu_ps(product+j, c);
        }
    }
    void multiply_complex_intrinsic(complex<float>* lhs, float* rhs, complex<float>* product, int ncols)
    {
        multiply_intrinsic(reinterpret_cast<float*>(lhs), rhs, reinterpret_cast<float*>(product), 2*ncols);
    }

    void strided_copy(complex<float>* lhs, complex<float>* rhs, int nchannels, int nsamples)
    {
        int nslice = 2 * nsamples / nchannels;
        for (int ind=0; ind < nsamples; ind++)
        {
            int tap_id = ind % (nchannels / 2);
            int chann_id = ind / (nchannels / 2);
            rhs[tap_id * nslice + chann_id] = lhs[ind];
        }
    }

    void transpose(complex<float>* lhs, complex<float>* rhs, int rows, int cols)
    {
        for (int ind=0; ind < rows*cols; ind++)
        {
            int row_index = ind % rows;
            int col_index = ind % cols;
            rhs[row_index + col_index * rows] = lhs[col_index + row_index * cols];
        }
    }

    void convolve(complex<float>* lhs, float* filters, complex<float>* output, complex<float>* scratch, int progression, int slice)
    {
        for(int ind=0; ind < progression; ind ++)
        {
            multiply_complex_intrinsic(lhs, filters + ind*slice, scratch, slice);
            for (int scratch_ind=0; scratch_ind < slice; scratch_ind++)
            {
                output[ind] +=  scratch[scratch_ind];
            }
        }
    }

    void plain_multiply(complex<float>* lhs, complex<float>* rhs, complex<float>* prod, int nsamples)
    {
        for (int ind=0; ind < nsamples; ind++)
        {
            prod[ind] = lhs[ind] * rhs[ind];
        }
    }
}