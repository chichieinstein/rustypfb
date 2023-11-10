#include <immintrin.h>
#include <chrono>
#include <iostream>
using std::chrono::steady_clock;
using std::chrono::duration;
using std::milli;
using std::cout;
using std::endl;

void multiply_intrinsic(float* lhs, float* rhs, float* product, size_t nrows, size_t ncols)
{
    for(int i=0; i<nrows; i++)
    {
    for(int j=0; j<ncols/16; j++)
    {
    __m512 a = _mm512_loadu_ps((__m512 *)(lhs + i*ncols + j));
    __m512 b = _mm512_loadu_ps((__m512 *)(rhs + i*ncols + j));
    __m512 c = _mm512_mul_ps(a, b);
    _mm512_storeu_ps(product+i*ncols + j, c);
    }
    }
}
void multiply(float* lhs, float* rhs, float* product, size_t nsamples)
{
    for(int i=0; i<nsamples; i++)
    {
    *(product + i) = lhs[i]*rhs[i];
    }
}
int main()
{
    int NROWS = 1024;
    int NCOLS = 16;
    int Nch = 1024;
    int Nsl = 32;
    float* lhs = new float [NROWS*NCOLS];
    float* rhs = new float [NROWS*NCOLS];
    float* product = new float [NROWS*NCOLS];
    for (int i=0; i<NROWS*NCOLS;i++)
    {
        lhs[i] = static_cast<float>(i*i);
        rhs[i] = static_cast<float>(i+i);
    }
    auto start = steady_clock::now();
    for (int i=0; i<1024*64;i++)
    {
        multiply_intrinsic(lhs, rhs, product, NROWS, NCOLS);
        // naivetransfer(inp, outp, sizeof(float)*Nch*Nsl);
        // transfer_intrinsic(reinterpret_cast<float*>(inp), outp, Nch*Nsl);
        // A_memcpy(inp, outp, sizeof(float)*Nch*Nsl);
    }
    auto end = steady_clock::now();
    // cout << inp[0].real() << " " << inp[0].imag() << " " << outp[0] << " " << outp[1] << " " << inp[1].real() << " " << inp[1].imag() << endl;
    auto dur = duration<double, milli>(end-start).count() / (1024*64); 
    cout << dur << endl;
    cout << "Throughput " << (NROWS*NCOLS*1000)/ dur << endl;

}