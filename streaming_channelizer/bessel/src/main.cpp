#include "../include/c_interface.h"
#include <iostream>
#include <chrono>
using std::cout;
using std::endl;
using std::chrono::steady_clock;
using std::chrono::duration;

int main()
{
    const int NSAMPLES = 8192;
    int CHANNELS = 1024;
    int* n = &CHANNELS;
    const int HOWMANY = 2*NSAMPLES / CHANNELS;
    complex<float>* inp = new complex<float> [2*NSAMPLES];
    complex<float>* outp = new complex<float> [2*NSAMPLES];
    auto plan = fftwf_plan_many_dft(
                1,
                n,
                HOWMANY,
                reinterpret_cast<fftwf_complex*>(inp),
                n,
                1,
                CHANNELS,
                reinterpret_cast<fftwf_complex*>(outp),
                n,
                1,
                CHANNELS,
                1,
                FFTW_MEASURE);

    const int NTIMES=100;
    double total_duration = 0.0;
    for (int i=0; i<NTIMES;i++)
    {  
        auto start = steady_clock::now();
        fftwf_execute(plan);
        auto end = steady_clock::now();
        double f = duration<double, std::milli>(end - start).count();
        total_duration += f;
    }
    cout << "Time taken in milliseconds to process " << NSAMPLES << " is " << (total_duration / NTIMES) << endl;
   
}