/*
* (c) 2017 Virginia Polytechnic Institute & State University (Virginia Tech)   
*                                                                              
*   This program is free software: you can redistribute it and/or modify       
*   it under the terms of the GNU Lesser General Public License Version 2.1.                                  
*                                                                              
*   This program is distributed in the hope that it will be useful,            
*   but WITHOUT ANY WARRANTY; without even the implied warranty of             
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              
*   LICENSE in the root of the repository for details.                 
*                                                                              
*/


/* For testing of different SpTrans solutions */
#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <sys/time.h> // timing
#include "matio.h"
#include "sptrans.h"

#ifdef MKL
#include <mkl_spblas.h>
#include <mkl.h>
#include <typeinfo>
#endif

#define valT double

double dtime()  // milliseconds
{
    double tseconds = 0.0;
    struct timeval mytime;
    gettimeofday(&mytime, (struct timezone*)0);
    tseconds = (double)(mytime.tv_sec + mytime.tv_usec*1.0e-6);
    return (tseconds*1.0e3);
}

int main(int argc, char **argv)
{
    char *filename = NULL;
    if(argc > 1)
    {
        filename = argv[1];
    }
    if(filename == NULL)
    {
        std::cout << "Error: No file provided!\n";
        return EXIT_FAILURE;
    }
    std::cout << "matrix: " << filename << std::endl;

    // input
    int m, n, nnzA;
    int *csrRowPtrA;
    int *csrColIdxA;
    valT *csrValA;
    int retCode = read_mtx_mat(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, filename);
    if(retCode != 0)
    {
        std::cout << "Failed to read the matrix from " << filename << "!\n";
        return EXIT_FAILURE;
    }

    double tstart, tstop, ttime; 
    
#ifdef MKL
    int nthread;
#pragma omp parallel
    nthread = omp_get_num_threads();
    mkl_set_num_threads(nthread);

    sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
    sparse_status_t status = SPARSE_STATUS_SUCCESS;

    valT *cscval;
    sparse_matrix_t A, AT;
    MKL_INT cscn, cscm, *pntrb, *pntre, *cscidx;

    if(typeid(valT) == typeid(float))
    {
        status = mkl_sparse_s_create_csr(&A, SPARSE_INDEX_BASE_ZERO, 
                                        (MKL_INT)m, (MKL_INT)n, (MKL_INT *)csrRowPtrA, 
                                        (MKL_INT *)(csrRowPtrA + 1), (MKL_INT *)csrColIdxA, (float*)csrValA);
        tstart = dtime();

        status = mkl_sparse_convert_csr(A, SPARSE_OPERATION_TRANSPOSE, &AT);

        tstop = dtime();
        ttime = tstop - tstart;
        std::cout << "MKL(time): " << ttime << " ms\n";
        status = mkl_sparse_s_export_csr(AT, &indexing, 
                                         &cscn, &cscm, &pntrb, &pntre, &cscidx, (float**)&cscval);
    } else if(typeid(valT) == typeid(double))
    {
        status = mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, 
                                        (MKL_INT)m, (MKL_INT)n, (MKL_INT *)csrRowPtrA, 
                                        (MKL_INT *)(csrRowPtrA + 1), (MKL_INT *)csrColIdxA, (double*)csrValA);
        tstart = dtime();

        status = mkl_sparse_convert_csr(A, SPARSE_OPERATION_TRANSPOSE, &AT);

        tstop = dtime();
        ttime = tstop - tstart;
        std::cout << "MKL(time): " << ttime << " ms\n";
        status = mkl_sparse_d_export_csr(AT, &indexing, 
                                         &cscn, &cscm, &pntrb, &pntre, &cscidx, (double**)&cscval);
    
    }
    if (SPARSE_STATUS_SUCCESS != status)
    {
        std::cout << "Failed to do MKL convert!\n" << std::endl;
        mkl_sparse_destroy(A);
        mkl_sparse_destroy(AT);
        free(csrColIdxA);
        free(csrValA);
        free(csrRowPtrA);
        return EXIT_FAILURE;
    }
#endif

    int *cscRowIdxA = (int *)malloc(nnzA * sizeof(int));
    int *cscColPtrA = (int *)malloc((n + 1) * sizeof(int));
    valT *cscValA = (valT *)malloc(nnzA * sizeof(valT));
    // clear the buffers
    std::fill_n(cscRowIdxA, nnzA, 0);
    std::fill_n(cscValA, nnzA, 0);
    std::fill_n(cscColPtrA, n+1, 0);

    tstart = dtime();

    sptrans_scanTrans<int, valT>(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, cscRowIdxA, cscColPtrA, cscValA);

    tstop = dtime();
    ttime = tstop - tstart;
    std::cout << "scanTrans(time): " << ttime << " ms\n";
#ifdef MKL
    std::cout << "check vals: " << std::boolalpha << std::equal(cscval, cscval+nnzA, cscValA) << std::endl;
    std::cout << "check rowIdx: " << std::boolalpha << std::equal((int*)cscidx, (int*)(cscidx+nnzA), cscRowIdxA) << std::endl;
    std::cout << "check colPtr: " << std::boolalpha << std::equal((int*)pntrb, (int*)(pntrb+n), cscColPtrA) << std::endl;
#endif
    // clear the buffers
    std::fill_n(cscRowIdxA, nnzA, 0);
    std::fill_n(cscValA, nnzA, 0);
    std::fill_n(cscColPtrA, n+1, 0);

    tstart = dtime();

    sptrans_mergeTrans<int, valT>(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, cscRowIdxA, cscColPtrA, cscValA);

    tstop = dtime();
    ttime = tstop - tstart;
    std::cout << "mergeTrans(time): " << ttime << " ms\n";
#ifdef MKL
    std::cout << "check vals: " << std::boolalpha << std::equal(cscval, cscval+nnzA, cscValA) << std::endl;
    std::cout << "check rowIdx: " << std::boolalpha << std::equal((int*)cscidx, (int*)(cscidx+nnzA), cscRowIdxA) << std::endl;
    std::cout << "check colPtr: " << std::boolalpha << std::equal((int*)pntrb, (int*)(pntrb+n), cscColPtrA) << std::endl;
#endif

    free(csrRowPtrA); 
    free(csrColIdxA); 
    free(csrValA);
    free(cscRowIdxA);
    free(cscColPtrA);
    free(cscValA);
#ifdef MKL
    mkl_sparse_destroy(A);
    mkl_sparse_destroy(AT);
#endif
}
