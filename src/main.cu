#include"common.h"
#include"mmio_highlevel.h"
#include"utils.h"
#include"utils_cuda_scan.h"
#include "spgemm_nsparse_kernel.h"
#include "csr2tile.h"
#include "tilespgemm-cuda.h"
#include "spgemm-cpu.h"
#include "tile2csr.h"
#include "spgemm_serialref_spa_new.h"
#include "spgemm_cu.h"
#include "reorder.h"
#include "external_reorder.h"

int main(int argc, char ** argv)
{

	if (argc < 6)
    {
        printf("Run the code by './test -d 0 -aat 0 matrixA.mtx [-b matrixB.mtx] [-reorder 1] [-reorder_algo tasa|degree|rcm|...]'.\n");
        return 0;
    }
	
    printf("--------------------------------!!!!!!!!------------------------------------\n");
    
    int device_id = 0;
    int aat = 0;
    int reorder = 0;
    int reorder_materialized = 0;
    int reorder_shared_csr = 0;
    const char *reorder_algo = "tasa";
    double reorder_ordergen_time_ms = 0.0;

    // "Usage: ``./test -d 0 -aat 0 A.mtx'' for C=AA  on device 0", or
    // "Usage: ``./test -d 0 -aat 1 A.mtx'' for C=AAT on device 0"
    int argi = 1;

    // load device id
    char *devstr;
    if(argc > argi)
    {
        devstr = argv[argi];
        argi++;
    }

    if (strcmp(devstr, "-d") != 0) return 0;

    if(argc > argi)
    {
        device_id = atoi(argv[argi]);
        argi++;
    }
    printf("device_id = %i\n", device_id);
    
    // set device
    cudaSetDevice(device_id);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);

    // Set aside 50% of L2 cache for persisting accesses 
    size_t size = min( int(deviceProp.l2CacheSize * 0.80) , deviceProp.persistingL2CacheMaxSize );
    cudaDeviceSetLimit( cudaLimitPersistingL2CacheSize, size); 

    printf("---------------------------------------------------------------\n");
    printf("Device [ %i ] %s @ %4.2f MHz\n",
           device_id, deviceProp.name, deviceProp.clockRate * 1e-3f);
           
    // load AAT flag
    char *aatstr;
    if(argc > argi)
    {
        aatstr = argv[argi];
        argi++;
    }

    if (strcmp(aatstr, "-aat") != 0) return 0;

    if(argc > argi)
    {
        aat = atoi(argv[argi]);
        argi++;
    }

    char *filename = NULL;
    char *filename_b = NULL;
    while (argi < argc)
    {
        if (strcmp(argv[argi], "-reorder") == 0)
        {
            argi++;
            if (argi >= argc)
            {
                printf("Missing value for -reorder.\n");
                return 0;
            }
            reorder = atoi(argv[argi]);
            argi++;
        }
        else if (strcmp(argv[argi], "-b") == 0)
        {
            argi++;
            if (argi >= argc)
            {
                printf("Missing value for -b.\n");
                return 0;
            }
            filename_b = argv[argi];
            argi++;
        }
        else if (strcmp(argv[argi], "-reorder_algo") == 0)
        {
            argi++;
            if (argi >= argc)
            {
                printf("Missing value for -reorder_algo.\n");
                return 0;
            }
            reorder_algo = argv[argi];
            argi++;
        }
        else
        {
            filename = argv[argi];
            argi++;
        }
    }

    if (filename == NULL)
    {
        printf("Run the code by './test -d 0 -aat 0 matrixA.mtx [-b matrixB.mtx] [-reorder 1] [-reorder_algo tasa|degree|rcm|...]'.\n");
        return 0;
    }

 	struct timeval t1, t2;
	SMatrix *matrixA = (SMatrix *)malloc(sizeof(SMatrix));
    memset(matrixA, 0, sizeof(SMatrix));
	SMatrix *matrixA_original = matrixA;
	SMatrix *matrixA_reordered = NULL;
	SMatrix *matrixB_reordered = NULL;
    SMatrix *matrixB = (SMatrix *)malloc(sizeof(SMatrix));
    memset(matrixB, 0, sizeof(SMatrix));
    SMatrix *matrixB_original = NULL;
    int ab_mode = filename_b != NULL;
    int matrixB_owns_csr = 0;
    ReorderProblemKind reorder_kind = REORDER_PROBLEM_AA;

    printf("MAT A: -------------- %s --------------\n", filename);
    if (ab_mode)
        printf("MAT B: -------------- %s --------------\n", filename_b);

    // load mtx A data to the csr format
    gettimeofday(&t1, NULL);
    mmio_allinone(&matrixA->m, &matrixA->n, &matrixA->nnz, &matrixA->isSymmetric, &matrixA->rowpointer, &matrixA->columnindex, &matrixA->value, filename);
    gettimeofday(&t2, NULL);
    double time_loadmat  = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("input matrix A: ( %i, %i ) nnz = %i\n loadfile time    = %4.5f sec\n", matrixA->m, matrixA->n, matrixA->nnz, time_loadmat/1000.0);

    if (!aat && !ab_mode && matrixA->m != matrixA->n)
    {
        printf("matrix squaring must have rowA == colA. Exit.\n");
        return 0;
    }
    if (ab_mode)
    {
        if (aat)
        {
            printf("explicit A*B mode does not support -aat 1. Exit.\n");
            return 0;
        }

        matrixB_original = matrixB;
        matrixB_owns_csr = 1;
        reorder_kind = REORDER_PROBLEM_AB;
        gettimeofday(&t1, NULL);
        mmio_allinone(&matrixB_original->m, &matrixB_original->n, &matrixB_original->nnz, &matrixB_original->isSymmetric,
                      &matrixB_original->rowpointer, &matrixB_original->columnindex, &matrixB_original->value, filename_b);
        gettimeofday(&t2, NULL);
        double time_loadmat_b = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        printf("input matrix B: ( %i, %i ) nnz = %i\n loadfile time    = %4.5f sec\n",
               matrixB_original->m, matrixB_original->n, matrixB_original->nnz, time_loadmat_b / 1000.0);

        if (matrixA_original->n != matrixB_original->m)
        {
            printf("A*B requires colA == rowB. Exit.\n");
            return 0;
        }
    }

    printf("the tilesize = %d\n",BLOCK_SIZE);

	for (int i = 0; i < matrixA->nnz; i++)
	    matrixA->value[i] = i % 10;
    if (ab_mode)
    {
        for (int i = 0; i < matrixB_original->nnz; i++)
            matrixB_original->value[i] = i % 10;
    }
    else if (aat)
    {
        MAT_PTR_TYPE *cscColPtrA;
        int *cscRowIdxA;
        MAT_VAL_TYPE *cscValA;

        if (matrixA_original->m == matrixA_original->n && matrixA_original->isSymmetric)
        {
           printf("matrix AAT does not do symmetric matrix. Exit.\n");
           return 0;
        }

        matrixB_original = matrixB;
        matrixB_owns_csr = 1;
        reorder_kind = REORDER_PROBLEM_AAT;
        matrixB_original->m = matrixA_original->n;
        matrixB_original->n = matrixA_original->m;
        matrixB_original->nnz = matrixA_original->nnz;
        matrixB_original->isSymmetric = 0;

        cscColPtrA = (MAT_PTR_TYPE *)malloc((matrixA_original->n + 1) * sizeof(MAT_PTR_TYPE));
        cscRowIdxA = (int *)malloc(matrixA_original->nnz * sizeof(int));
        cscValA = (MAT_VAL_TYPE *)malloc(matrixA_original->nnz * sizeof(MAT_VAL_TYPE));

        matrix_transposition(matrixA_original->m, matrixA_original->n, matrixA_original->nnz,
                             matrixA_original->rowpointer, matrixA_original->columnindex, matrixA_original->value,
                             cscRowIdxA, cscColPtrA, cscValA);

        matrixB_original->rowpointer = cscColPtrA;
        matrixB_original->columnindex = cscRowIdxA;
        matrixB_original->value = cscValA;
    }
    else
    {
        reorder_kind = REORDER_PROBLEM_AA;
        matrixB->m = matrixA_original->m;
        matrixB->n = matrixA_original->n;
        matrixB->nnz = matrixA_original->nnz;
        matrixB->isSymmetric = matrixA_original->isSymmetric;

        matrixB->rowpointer = matrixA_original->rowpointer;
        matrixB->columnindex = matrixA_original->columnindex;
        matrixB->value = matrixA_original->value;
    }

    ReorderInfo reorder_info;
    reorder_info_init(&reorder_info);

    if (reorder)
    {
        printf("reorder enabled\n");
        printf("reorder problem = %s\n", reorder_problem_kind_name(reorder_kind));
        printf("reorder algorithm = %s\n", reorder_algo);
        printf("reorder strategy = %s\n", strcmp(reorder_algo, "tasa") == 0 ? "TASA-Fast-Unified" : "External-Order");
        gettimeofday(&t1, NULL);

        int reorder_status = 0;
        if (strcmp(reorder_algo, "tasa") == 0)
        {
            ReorderProblem reorder_problem;
            reorder_problem.kind = reorder_kind;
            reorder_problem.matrixA = matrixA_original;
            reorder_problem.matrixB = (reorder_kind == REORDER_PROBLEM_AA) ? matrixA_original : matrixB_original;

            reorder_status = build_reorder_plan(&reorder_problem, &reorder_info);
            if (reorder_status != 0)
            {
                printf("reorder failed, status = %d. Exit.\n", reorder_status);
                return 0;
            }
        }
        else if (strcmp(reorder_algo, "original") == 0)
        {
            SMatrix *reorder_matrixB = (reorder_kind == REORDER_PROBLEM_AA) ? matrixA_original : matrixB_original;
            reorder_status = build_original_reorder_plan(reorder_kind, matrixA_original, reorder_matrixB, &reorder_info);
            if (reorder_status != 0)
            {
                printf("original reorder plan failed, status = %d. Exit.\n", reorder_status);
                return 0;
            }
        }
        else
        {
            SMatrix *reorder_matrixB = (reorder_kind == REORDER_PROBLEM_AA) ? matrixA_original : matrixB_original;
            reorder_status = build_external_reorder_plan(reorder_kind,
                                                          reorder_algo,
                                                          filename,
                                                          filename_b,
                                                          matrixA_original,
                                                          reorder_matrixB,
                                                          &reorder_info,
                                                          &reorder_ordergen_time_ms);
            if (reorder_status != 0)
            {
                printf("external reorder failed, status = %d. Exit.\n", reorder_status);
                return 0;
            }
        }

        if (reorder_info.guard_degraded_to_identity)
        {
            if (reorder_kind == REORDER_PROBLEM_AA)
            {
                matrixB->m = matrixA_original->m;
                matrixB->n = matrixA_original->n;
                matrixB->nnz = matrixA_original->nnz;
                matrixB->isSymmetric = matrixA_original->isSymmetric;
                matrixB->rowpointer = matrixA_original->rowpointer;
                matrixB->columnindex = matrixA_original->columnindex;
                matrixB->value = matrixA_original->value;
                reorder_shared_csr = 1;
            }
            else
            {
                matrixB = matrixB_original;
                reorder_shared_csr = 0;
            }
            matrixA = matrixA_original;
        }
        else
        {
            struct timeval tp1, tp2;
            gettimeofday(&tp1, NULL);
            matrixA_reordered = (SMatrix *)malloc(sizeof(SMatrix));
            memset(matrixA_reordered, 0, sizeof(SMatrix));
            reorder_status = apply_bipermutation_csr(matrixA_original, &reorder_info.row_perm, &reorder_info.inner_perm, matrixA_reordered);
            if (reorder_status != 0)
            {
                printf("reorder apply A failed, status = %d. Exit.\n", reorder_status);
                return 0;
            }

            int aa_problem = (reorder_kind == REORDER_PROBLEM_AA);
            matrixB_reordered = aa_problem ? matrixB : (SMatrix *)malloc(sizeof(SMatrix));
            memset(matrixB_reordered, 0, sizeof(SMatrix));
            if (aa_problem && reorder_info.shared_symmetric_csr)
            {
                matrixB_reordered->m = matrixA_reordered->m;
                matrixB_reordered->n = matrixA_reordered->n;
                matrixB_reordered->nnz = matrixA_reordered->nnz;
                matrixB_reordered->isSymmetric = 0;
                matrixB_reordered->rowpointer = matrixA_reordered->rowpointer;
                matrixB_reordered->columnindex = matrixA_reordered->columnindex;
                matrixB_reordered->value = matrixA_reordered->value;
                reorder_shared_csr = 1;
            }
            else
            {
                SMatrix *matrixB_source = aa_problem ? matrixA_original : matrixB_original;
                reorder_status = apply_bipermutation_csr(matrixB_source, &reorder_info.inner_perm, &reorder_info.col_perm, matrixB_reordered);
                if (reorder_status != 0)
                {
                    printf("reorder apply B failed, status = %d. Exit.\n", reorder_status);
                    return 0;
                }
            }
            gettimeofday(&tp2, NULL);
            reorder_info.permute_time_ms = (tp2.tv_sec - tp1.tv_sec) * 1000.0 + (tp2.tv_usec - tp1.tv_usec) / 1000.0;

            matrixA = matrixA_reordered;
            matrixB = matrixB_reordered;
            reorder_materialized = 1;
        }

        gettimeofday(&t2, NULL);
        reorder_info.reorder_time_ms = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        printf("reorder time = %.2f ms\n", reorder_info.reorder_time_ms);
        printf("reorder mode = %s\n", reorder_info.mode);
        printf("reorder signature time = %.2f ms\n", reorder_info.signature_time_ms);
        printf("reorder tile profile time = %.2f ms\n", reorder_info.tile_profile_time_ms);
        printf("reorder affinity time = %.2f ms\n", reorder_info.affinity_time_ms);
        printf("reorder sort time = %.2f ms\n", reorder_info.sort_time_ms);
        printf("reorder local refine time = %.2f ms\n", reorder_info.local_refine_time_ms);
        printf("reorder local refine window = %d\n", reorder_info.local_refine_window);
        printf("reorder proxy time = %.2f ms\n", reorder_info.proxy_time_ms);
        printf("reorder proxy samples = %d\n", reorder_info.proxy_sample_rows);
        printf("reorder ordergen time = %.2f ms\n", reorder_ordergen_time_ms);
        printf("reorder permute time = %.2f ms\n", reorder_info.permute_time_ms);
        printf("reorder preprocess time = %.2f ms\n", reorder_ordergen_time_ms + reorder_info.permute_time_ms);
        printf("reorder shared csr = %d\n", reorder_shared_csr);
        printf("risk reason = %s\n", reorder_info.risk_reason);
        printf("risk input tile ratio = %.4f\n", reorder_info.risk_input_tile_ratio);
        printf("risk C tile ratio = %.4f\n", reorder_info.risk_c_tile_ratio);
        if (reorder_info.guard_degraded_to_identity)
            printf("reorder uses conservative identity\n");
        printf("input tiles A before reorder = %lld\n", reorder_info.input_tiles_A_before);
        printf("input tiles A after reorder = %lld\n", reorder_info.input_tiles_A_after);
        printf("input tiles B before reorder = %lld\n", reorder_info.input_tiles_B_before);
        printf("input tiles B after reorder = %lld\n", reorder_info.input_tiles_B_after);
        printf("estimated C tiles before reorder = %lld\n", reorder_info.estimated_c_tiles_before);
        printf("estimated C tiles after reorder = %lld\n", reorder_info.estimated_c_tiles_after);
    }

        // calculate bytes and flops consumed
        unsigned long long int nnzCub = 0;
        for (int i = 0; i < matrixA->nnz; i++)
        {
            int rowidx = matrixA->columnindex[i];
            nnzCub += matrixB->rowpointer[rowidx + 1] - matrixB->rowpointer[rowidx];
        }
    
        printf("SpGEMM nnzCub = %lld\n", nnzCub);

#if TIMING
        gettimeofday(&t1, NULL);
#endif

        csr2tile_row_major(matrixA);
#if TIMING
        gettimeofday(&t2, NULL);
        double time_conversion = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        printf("CSR to Tile conversion uses %.2f ms\n", time_conversion);
#endif

#if SPACE

double tile_bytes = (matrixA->tilem + 1) * sizeof(int) + matrixA->numtile * sizeof(int) + (matrixA->numtile + 1) *sizeof(int) +
                matrixA->nnz * sizeof(MAT_VAL_TYPE) + matrixA->nnz * sizeof(unsigned char) + matrixA->numtile * BLOCK_SIZE * sizeof(unsigned char) +
                matrixA->numtile * BLOCK_SIZE * sizeof(unsigned short);

double mem = tile_bytes/1024/1024;

double CSR_bytes = (matrixA->m +1) * sizeof(int) + (matrixA->nnz) * sizeof(int) + matrixA->nnz * sizeof(MAT_VAL_TYPE);
double csr_mem = CSR_bytes /1024/1024;

printf("tile space overhead = %.2f MB\n", mem);

#endif

        csr2tile_col_major(matrixB);


        int blk_intersec_bitmask_len = ceil((double)matrixA->tilen / 32.0);
        double densityA = (double)matrixA->numtile / ((double)matrixA->tilem*(double)matrixA->tilen);
        double densityB = (double)matrixB->numtile / ((double)matrixB->tilem*(double)matrixB->tilen);


        long long int lengthA = (long long int) (matrixA->tilem) * (long long int)( blk_intersec_bitmask_len) ;

    unsigned int *blk_intersec_bitmask_A = (unsigned int *)malloc(lengthA* sizeof(unsigned int));
    memset(blk_intersec_bitmask_A, 0, lengthA * sizeof(unsigned int));
    for (int i = 0; i < matrixA->tilem; i++)
    {
        for (int j = matrixA->tile_ptr[i]; j < matrixA->tile_ptr[i + 1]; j++)
        {
            int idx = matrixA->tile_columnidx[j];
            unsigned int bitmask = 1;
            bitmask <<=  (31- (idx % 32));
            long long int pos = (long long int)i * (long long int)blk_intersec_bitmask_len + idx / 32;
            blk_intersec_bitmask_A[pos] |= bitmask;
        }
    }

    long long int lengthB = (long long int) (matrixB->tilen) * (long long int)(blk_intersec_bitmask_len) ;

    unsigned int *blk_intersec_bitmask_B = (unsigned int *)malloc(lengthB * sizeof(unsigned int));
    memset(blk_intersec_bitmask_B, 0, lengthB * sizeof(unsigned int));
    for (int i = 0; i < matrixB->tilen; i++)
    {
        for (int j = matrixB->csc_tile_ptr[i]; j < matrixB->csc_tile_ptr[i+1]; j++)
        {
            int idx = matrixB->csc_tile_rowidx[j];
            unsigned int bitmask = 0x1;
            bitmask <<= (31 - (idx % 32));
            long long int pos = (long long int)i * (long long int )blk_intersec_bitmask_len + idx / 32;
            blk_intersec_bitmask_B[pos] |= bitmask;
        }
    }


    // generate rowidx of blockA
    int *tile_rowidx_A = (int *)malloc (matrixA->numtile * sizeof(int ) );
    for (int i = 0; i < matrixA->tilem; i++)
    {
        for (int j = matrixA->tile_ptr[i]; j < matrixA->tile_ptr[i+1]; j++)
        {
            tile_rowidx_A[j] = i;
        }
    }



#ifdef DEBUG
    // --------------------------------------------------------------------------------------------------------
    SMatrix *matrixC = (SMatrix *)malloc(sizeof(SMatrix));
    
    struct timeval tv;
    unsigned long long int nnzC_computed;
    double compression_rate = 0;
    double time_tile = 0;
    double gflops_tile = 0;
    double time_step1 =0,time_step2 =0,time_step3 =0,time_malloc=0; 


    

    tilespgemm(matrixA,
               matrixB,
               matrixC,
               blk_intersec_bitmask_A,
               blk_intersec_bitmask_B,
               blk_intersec_bitmask_len,
               densityA,
               densityB,
               nnzCub,
               &nnzC_computed,
               &compression_rate,
               &time_tile,
               &gflops_tile,
               filename,
               &time_step1,&time_step2,&time_step3,&time_malloc);


    // // write results to text (scv) file
    // FILE *fout = fopen("../data/results_tile.csv", "a");
    // if (fout == NULL)
    //     printf("Writing results fails.\n");
    // fprintf(fout, "%s,%i,%i,%i,%lld,%lld,%f,%f,%f\n",
    //         filename, matrixA->m, matrixA->n, matrixA->nnz, nnzCub, nnzC_computed, compression_rate, time_tile, gflops_tile);
    // fclose(fout);

    // // write runtime of each step to text (scv) file
    // FILE *fout_time = fopen("../data/step_runtime.csv", "a");
    // if (fout_time == NULL)
    //     printf("Writing results fails.\n");
    // fprintf(fout_time, "%s,%i,%i,%i,%lld,%lld,%f,%f,%f,%f,%f\n",
    //             filename, matrixA->m, matrixA->n, matrixA->nnz, nnzCub, nnzC_computed, compression_rate, time_step1, time_step2,time_step3,time_malloc);
    // fclose(fout_time);
    

#if SPACE
    // write memory space of CSR and tile format to text (scv) file
    // FILE *fout_mem = fopen("../data/mem-cost.csv", "a");
    // if (fout_mem == NULL)
    //     printf("Writing results fails.\n");
    // fprintf(fout_mem, "%s,%i,%i,%i,%lld,%lld,%f,%f,%f\n",
    //             filename, matrixA->m, matrixA->n, matrixA->nnz, nnzCub, nnzC_computed, compression_rate, csr_mem,mem);
    // fclose(fout_mem);

#endif

#if TIMING

    // // write preprocessing overhead of CSR and tile format to text (scv) file
    // FILE *fout_pre = fopen("../data/preprocessing.csv", "a");
    // if (fout_pre == NULL)
    //     printf("Writing results fails.\n");
    // fprintf(fout_pre, "%s,%i,%i,%i,%lld,%lld,%f,%f,%f\n",
    //                 filename, matrixA->m, matrixA->n, matrixA->nnz, nnzCub, nnzC_computed, compression_rate, time_conversion,time_tile);
    // fclose(fout_pre);
    
#endif


#endif

#if CHECK_RESULT
printf("-------------------------------check----------------------------------------\n");
tile2csr(matrixC);
        printf("tile to CSR conversion complete!\n");

    unsigned long long int nnzC = 0;
    double compression_rate1 = 0;
    double time_cusparse = 0;
    double gflops_cusparse = 0;
    int flag =0;
    int mC = matrixA->m;
    int nC = matrixB->n;
    int nnzC_golden = matrixC->nnz;
    bool check_result = CHECK_RESULT;

    if (reorder_materialized)
    {
        gettimeofday(&t1, NULL);
        int restore_status = restore_bipermutation_csr(matrixC, &reorder_info.row_perm, &reorder_info.col_perm);
        gettimeofday(&t2, NULL);
        double time_restore = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        if (restore_status != 0)
        {
            printf("restore reorder result failed, status = %d. Exit.\n", restore_status);
            return 0;
        }
        printf("restore reorder result uses %.2f ms\n", time_restore);
    }

    MAT_PTR_TYPE *csrRowPtrC_golden = matrixC->rowpointer;
    int *csrColIdxC_golden = matrixC->columnindex;
    MAT_VAL_TYPE *csrValC_golden = matrixC->value;

    SMatrix *matrixA_check = reorder ? matrixA_original : matrixA;
    SMatrix *matrixB_check = reorder ? ((reorder_kind == REORDER_PROBLEM_AA) ? matrixA_original : matrixB_original) : matrixB;

    spgemm_cu(matrixA_check->m, matrixA_check->n, matrixA_check->nnz, matrixA_check->rowpointer, matrixA_check->columnindex, matrixA_check->value,
              matrixB_check->m, matrixB_check->n, matrixB_check->nnz, matrixB_check->rowpointer, matrixB_check->columnindex, matrixB_check->value,
              mC, nC, nnzC_golden, csrRowPtrC_golden, csrColIdxC_golden, csrValC_golden,
              check_result, nnzCub, &nnzC, &compression_rate1, &time_cusparse, &gflops_cusparse);
    printf("---------------------------------------------------------------\n");

#endif
    matrix_destroy(matrixA);
    matrix_destroy(matrixB);

    if (reorder_materialized)
    {
        free(matrixA->rowpointer);
        free(matrixA->columnindex);
        free(matrixA->value);
        if (!reorder_shared_csr)
        {
            free(matrixB->rowpointer);
            free(matrixB->columnindex);
            free(matrixB->value);
        }
        free(matrixA_original->rowpointer);
        free(matrixA_original->columnindex);
        free(matrixA_original->value);
        if (matrixB_owns_csr && matrixB_original != NULL)
        {
            free(matrixB_original->rowpointer);
            free(matrixB_original->columnindex);
            free(matrixB_original->value);
        }
    }
    else
    {
        free(matrixA_original->rowpointer);
        free(matrixA_original->columnindex);
        free(matrixA_original->value);
        if (matrixB_owns_csr && matrixB_original != NULL)
        {
            free(matrixB_original->rowpointer);
            free(matrixB_original->columnindex);
            free(matrixB_original->value);
        }
    }
    if (reorder)
        reorder_info_destroy(&reorder_info);

    return 0;

}
