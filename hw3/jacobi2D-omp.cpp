#include <stdio.h>
#include <math.h>
#include <fstream>
#ifdef _OPENMP
    #include <omp.h>
#endif
#include <iostream>
using namespace std;

// Main Function
int main(int argc, char** argv)
{
    long N[8] = {5,10,20,50,100,120,150,200};
    for (long j=0; j<8; j++){
        double h = 1.0/(N[j]+1);
        long n = N[j]*N[j];
        cout<<"Grid Size: "<<N[j]<<endl;
        cout<<"---------------------------------------------------------"<<endl;
        cout<<"#Threads     Time(s)     Speedup"<<endl;
        cout<<"---------------------------------------------------------"<<endl;
        // Dynamically allocate memory using malloc()
        double* f = (double*) malloc(n*sizeof(double));
        double* u = (double*) malloc(n*sizeof(double));
        double* u_n = (double*) malloc(n*sizeof(double));

        for (long n_thread=1; n_thread<=30; n_thread++){
            double stime = 1.0, t;
            #ifdef _OPENMP
                t = omp_get_wtime();
            #endif

            for (long i=0; i < n; i++){
                f[i] = 1.0;
                u[i] = 0.0;
                u_n[i] = 0.0;
            }

            for (long k=0; k<=2000; k++){
                #pragma omp parallel for num_threads(n_thread) shared(u, u_n)
                for (long i=0; i< n; i++){
                    u_n[i] = 0.25*((h*h*f[i])+ u[i+1] + u[i-1] + u[i+N[j]] + u[i-N[j]]);
                    u[i] = u_n[i];
                }
            }

            #ifdef _OPENMP
                t = omp_get_wtime()-t;
            #endif
          
            if(n_thread == 1)
                stime = t;
            printf("%5d       %10f        %10f\n",n_thread,t,stime/t);
        }

        free(f);
        free(u);
        free(u_n);
    }
    return 0;
}
