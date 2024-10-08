// #include "../include/revert.cuh"
#include "../include/offline_chann_C_interface.cuh"
// #include "/opt/asmlib/asmlib.h"
// #include <string.h>
#include <stdio.h>
#include <cmath>
#include <complex>
#include <chrono>
#include <iostream>

using namespace std::complex_literals;
using std::chrono::high_resolution_clock;
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::cyl_bessel_if;
using std::cout;
using std::endl;
using std::milli;
using std::complex;

float sinc(float x)
{
    return (x == 0.0) ? 1.0 : float(sin(x)/x);
}

void time_test(chann* p_chann, float* input, cufftComplex* output, int ntimes, float &time)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float duration;
    cudaEventRecord(start);
    for (int i=0; i < ntimes; i++)
    {
        chann_process(p_chann, input, output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);
    time += duration;
}

int main()
{
     int Nsamples = 8000000;
     const int Nch   = 1024;
     const int Nslice = 8* 2048;
     int Nproto = 32;
     float kbeta=9.6;
     vector<complex<float>> filter_function;
     for (int j=0; j<Nch*Nproto; j++)
     {
         float arg = Nproto / 2 + static_cast<float>(j + 1) / Nch;
         float darg = static_cast<float>(2 * j) / static_cast<float>(Nch*Nproto) - 1.0;
         float carg = kbeta * sqrt(1-darg*darg);
         try{
         float earg = cyl_bessel_if(0.0, carg) / cyl_bessel_if(0.0, kbeta);
         filter_function.push_back(complex<float>(earg, 0.0));
         }
         catch(int num)
         {
             cout << "Exception occured " << j << endl;
         }
     }
     chann* p_chann = chann_create(&filter_function[0], Nproto, Nch, Nslice);
     float* input = new float [Nch*Nslice];
     // cufftComplex* inp_c = new cufftComplex [Nch * Nslice / 2];
     cufftComplex* output_gpu;
     cufftComplex* output_cpu;
     output_cpu = new cufftComplex [Nch*Nslice];
     cudaMalloc((void **)&output_gpu, sizeof(cufftComplex) * Nch * Nslice);
     cudaHostRegister(input, sizeof(float)*Nch*Nslice, cudaHostRegisterMapped);
     for (int k=0; k<2*Nsamples; k++)
     {
        float inp_arg = static_cast<float>(k / 2);
        if (k%2 == 0)
         {
             input[k] = sin(inp_arg);
         }
        else 
         {
             input[k] = sinc(2.0*inp_arg);
         }
     }
    // cout << "---------------------------------------" << endl;
    float time;
    //for (int i=0; i < 100; i++){
//	cout << input[i] << " " << input[i+1] << endl;
  //  }
    time_test(p_chann, input, output_gpu, 100, time);
    cout << "Channelization of " << Nsamples << " into 1024 channels takes " << time / 100 << " in milliseconds" << endl;
    cudaMemcpy(output_cpu, output_gpu, sizeof(cufftComplex)*Nch*Nslice, cudaMemcpyDeviceToHost); 
    for (int i=0; i < 100; i++){
	cout << output_cpu[i].x << " " << output_cpu[i].y << endl;
    }

    
    chann_destroy(p_chann);
    cudaHostUnregister(input);
    delete [] input;
    delete [] output_cpu;
    cudaFree(output_gpu);

    // int Nproto = 128;
    // float kbeta = 10.2;
    // int Nch = 1024;
    // auto now = steady_clock::now();
    // for (int i=0; i<100; i++)
    // {
    //     float arg = Nproto / 2 + static_cast<float>(i + 1) / Nch;
    //     float darg = static_cast<float>(2 * i) / static_cast<float>(Nch*Nproto) - 1.0;
    //     float carg = kbeta * sqrt(1-darg*darg);
    //     float earg = cyl_bessel_if(0.0, carg) / cyl_bessel_if(0.0, kbeta);
    // }
    // auto end = steady_clock::now();
    // auto elapsed_time = duration<float, milli>(end - now).count();

    // cout << elapsed_time / 100 << endl;

    // auto box_collection = new box [16];

    // box* box_gpu;
    // cudaMalloc((void**)&box_gpu, sizeof(box)*16);

    // cudaMemcpy(box_gpu, box_collection, sizeof(box)*16, cudaMemcpyHostToDevice);

    // test<<<4, 4>>>(box_gpu);
    // cudaMemcpy(box_collection, box_gpu, sizeof(box)*16, cudaMemcpyDeviceToHost);

    // for (int i=0; i< 16; i++)
    // {
    //     cout << box_collection[i].start_time << " " << box_collection[i].start_chann << " " << box_collection[i].stop_time << " " << box_collection[i].stop_chann << endl;
    // }

    // cufftHandle* plans;
    // plans = new cufftHandle [100];
    // auto istrides = new int [100];
    // auto ostrides = new int [100];
    // auto idists   = new int [100];
    // auto odists   = new int [100];
    // auto batches  = new int [100];
    // auto n = new int [100];
    
    // for (int i=0; i < 100; i++)
    // {
    //     istrides[i] = 2*i;
    //     ostrides[i] = 2*i;
    //     idists[i] = 4*i;
    //     odists[i] = 4*i;
    //     batches[i] = 1;
    //     n[i] = 2*(i%10+1);
    // }
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // float my_duration;
    // cudaEventRecord(start);
    // for (int i=0; i<100; i++)
    // {
    //     cufftPlanMany(&plans[i], 1, n + i, n+i, istrides[i], idists[i], n+i, ostrides[i], odists[i], CUFFT_C2C, batches[i]);
    // }

    // for (int i=0; i<100; i++)
    // {
    //     cufftDestroy(plans[i]);
    // }
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&my_duration, start, stop);

    // cout << "Initializing 100 plans and destroying them takes " << my_duration / 100 << " in milliseconds" << endl;
    
    // delete [] plans;
    // delete [] istrides;
    // delete [] ostrides;
    // delete [] idists;
    // delete [] odists;
    // delete [] batches;
    // delete [] n;

   // cufftComplex* input;
   // cudaMalloc((void**)&input, sizeof(cufftComplex)*36);

    //cufftComplex* output;
    //cudaMalloc((void**)&output, sizeof(cufftComplex)*20);

    //cufftComplex* input_cpu = new cufftComplex [36];
    //cufftComplex* output_cpu = new cufftComplex [20];

    //for (int i=0; i<36; i++)
    //{
    //    input_cpu[i] = make_cuComplex(static_cast<float>(i*i), 0.0);
    //}
    //int n = 4;

    //cudaMemcpy(input, input_cpu, sizeof(cufftComplex)*36, cudaMemcpyHostToDevice);
    //cufftHandle plan;
    //cudaMemcpy2D(output, 5*sizeof(cufftComplex), input + 1, 9*sizeof(cufftComplex), 5*sizeof(cufftComplex), 4, cudaMemcpyDeviceToDevice);
    //cufftPlanMany(&plan, 1, &n, &n, 1, 5, &n, 1, 5, CUFFT_C2C, 3);
    //cufftExecC2C(plan, output, output, CUFFT_FORWARD);
    //cudaMemcpy(output_cpu, output, sizeof(cufftComplex)*20, cudaMemcpyDeviceToHost);

    //for (int i=0; i<4; i++)
    //{
      //  for (int j=0; j < 9; j++)
       // {
       //     cout << " " << input_cpu[i*9 + j].x << " " << input_cpu[i*9 + j].y;
       // }
        //cout << endl;
    //}
    //cout << "---------------" << endl;
    //for (int i=0; i<4; i++)
    //{
      //  for (int j=0; j < 5; j++)
      //  {
      //      cout << " " << output_cpu[i*5 + j].x << " " << output_cpu[i*5 + j].y;
      //  }
      //  cout << endl;
    //}

    //int **p = new int* [10];
    //int* q = new int [40];

    //int* r = new int [40];

    //for (int i=0; i<40; i++)
   // {
     //   q[i] = i*i;
    //}

    //for (int i=0; i<10; i++)
   // {
     //   cudaMalloc((void**)&p[i], sizeof(int)*4);
    //}

    //for (int i=0; i<10; i++)
   // {
    //    cudaMemcpy(p[i], q+4*i, sizeof(int)*4, cudaMemcpyHostToDevice);
   // }

   // for (int i=0; i<10; i++)
   // {
   //     cudaMemcpy(r+4*i, p[i], sizeof(int)*4, cudaMemcpyDeviceToHost);
   // }

   // for (int i=0; i<40; i++)
   // {
   //     cout << r[i] << endl;
   // }


   // for (int i=0; i<9; i++)
   // {
   //     cudaFree(p[i]);
   // }
   // delete [] p;
   // delete [] q;
   // delete [] r;
   // delete [] input_cpu;
   // cufftDestroy(plan);
   // cudaFree(input);
   // cudaFree(output);
   // delete [] output_cpu;

    // synth* revert_obj = synth_create(1024, 128, 32);

    // box* boxes = new box [100];

    // int nchannel = 1024;
    // int nslice   = 262144;

    // for (int i=0; i < 100; i++)
    // {
    //     boxes[i] = box(i, i+nslice / 32, 0, 512, i);
    // }


    // delete [] boxes;
    // synth_destroy(revert_obj);

}
