#include "../src/common.h"
#include "../src/mmio_highlevel.h"
#include "../src/reorder.h"

static void usage(const char *prog)
{
    printf("Usage: %s -aat 0 matrixA.mtx [-b matrixB.mtx] -o prefix\n", prog);
}

static int write_perm(const char *path, const AxisPermutation *perm)
{
    FILE *f = fopen(path, "w");
    if (f == NULL)
        return -1;
    fprintf(f, "%d\n", perm->n);
    for (int i = 0; i < perm->n; i++)
        fprintf(f, "%d\n", perm->new_id[i]);
    fclose(f);
    return 0;
}

static int write_all_perms(const char *prefix, const ReorderInfo *info)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s_row_old_to_new.txt", prefix);
    if (write_perm(path, &info->row_perm) != 0)
        return -1;
    snprintf(path, sizeof(path), "%s_inner_old_to_new.txt", prefix);
    if (write_perm(path, &info->inner_perm) != 0)
        return -2;
    snprintf(path, sizeof(path), "%s_col_old_to_new.txt", prefix);
    if (write_perm(path, &info->col_perm) != 0)
        return -3;
    return 0;
}

static int write_meta(const char *prefix, ReorderProblemKind kind, const ReorderInfo *info)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s_meta.txt", prefix);
    FILE *f = fopen(path, "w");
    if (f == NULL)
        return -1;
    fprintf(f, "problem=%s\n", reorder_problem_kind_name(kind));
    fprintf(f, "mode=%s\n", info->mode);
    fprintf(f, "reason=%s\n", info->risk_reason);
    fprintf(f, "risk_input_tile_ratio=%.6f\n", info->risk_input_tile_ratio);
    fprintf(f, "risk_c_tile_ratio=%.6f\n", info->risk_c_tile_ratio);
    fprintf(f, "input_tiles_A_before=%lld\n", info->input_tiles_A_before);
    fprintf(f, "input_tiles_A_after=%lld\n", info->input_tiles_A_after);
    fprintf(f, "input_tiles_B_before=%lld\n", info->input_tiles_B_before);
    fprintf(f, "input_tiles_B_after=%lld\n", info->input_tiles_B_after);
    fprintf(f, "estimated_c_tiles_before=%lld\n", info->estimated_c_tiles_before);
    fprintf(f, "estimated_c_tiles_after=%lld\n", info->estimated_c_tiles_after);
    fclose(f);
    return 0;
}

int main(int argc, char **argv)
{
    int aat = 0;
    char *filename_a = NULL;
    char *filename_b = NULL;
    char *prefix = NULL;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-aat") == 0)
        {
            if (++i >= argc)
            {
                usage(argv[0]);
                return 1;
            }
            aat = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "-b") == 0)
        {
            if (++i >= argc)
            {
                usage(argv[0]);
                return 1;
            }
            filename_b = argv[i];
        }
        else if (strcmp(argv[i], "-o") == 0)
        {
            if (++i >= argc)
            {
                usage(argv[0]);
                return 1;
            }
            prefix = argv[i];
        }
        else
        {
            filename_a = argv[i];
        }
    }

    if (filename_a == NULL || prefix == NULL || (aat && filename_b != NULL))
    {
        usage(argv[0]);
        return 1;
    }

    SMatrix *matrixA = (SMatrix *)malloc(sizeof(SMatrix));
    SMatrix *matrixB = (SMatrix *)malloc(sizeof(SMatrix));
    memset(matrixA, 0, sizeof(SMatrix));
    memset(matrixB, 0, sizeof(SMatrix));
    SMatrix *matrixB_original = NULL;
    int matrixB_owns_csr = 0;
    ReorderProblemKind kind = REORDER_PROBLEM_AA;

    mmio_allinone(&matrixA->m, &matrixA->n, &matrixA->nnz, &matrixA->isSymmetric,
                  &matrixA->rowpointer, &matrixA->columnindex, &matrixA->value, filename_a);
    for (int i = 0; i < matrixA->nnz; i++)
        matrixA->value[i] = i % 10;

    if (filename_b != NULL)
    {
        kind = REORDER_PROBLEM_AB;
        matrixB_original = matrixB;
        matrixB_owns_csr = 1;
        mmio_allinone(&matrixB_original->m, &matrixB_original->n, &matrixB_original->nnz, &matrixB_original->isSymmetric,
                      &matrixB_original->rowpointer, &matrixB_original->columnindex, &matrixB_original->value, filename_b);
        for (int i = 0; i < matrixB_original->nnz; i++)
            matrixB_original->value[i] = i % 10;
        if (matrixA->n != matrixB_original->m)
        {
            printf("A*B requires colA == rowB.\n");
            return 2;
        }
    }
    else if (aat)
    {
        kind = REORDER_PROBLEM_AAT;
        matrixB_original = matrixB;
        matrixB_owns_csr = 1;
        matrixB_original->m = matrixA->n;
        matrixB_original->n = matrixA->m;
        matrixB_original->nnz = matrixA->nnz;
        matrixB_original->isSymmetric = 0;
        matrixB_original->rowpointer = (MAT_PTR_TYPE *)malloc((matrixA->n + 1) * sizeof(MAT_PTR_TYPE));
        matrixB_original->columnindex = (int *)malloc(matrixA->nnz * sizeof(int));
        matrixB_original->value = (MAT_VAL_TYPE *)malloc(matrixA->nnz * sizeof(MAT_VAL_TYPE));
        matrix_transposition(matrixA->m, matrixA->n, matrixA->nnz,
                             matrixA->rowpointer, matrixA->columnindex, matrixA->value,
                             matrixB_original->columnindex, matrixB_original->rowpointer, matrixB_original->value);
    }
    else
    {
        kind = REORDER_PROBLEM_AA;
        matrixB_original = matrixA;
    }

    ReorderProblem problem;
    problem.kind = kind;
    problem.matrixA = matrixA;
    problem.matrixB = (kind == REORDER_PROBLEM_AA) ? matrixA : matrixB_original;

    ReorderInfo info;
    reorder_info_init(&info);
    int status = build_reorder_plan(&problem, &info);
    if (status != 0)
    {
        printf("build_reorder_plan failed: %d\n", status);
        return 3;
    }

    printf("problem=%s mode=%s reason=%s\n",
           reorder_problem_kind_name(kind), info.mode, info.risk_reason);
    printf("input_tiles_A %lld -> %lld, input_tiles_B %lld -> %lld, estimated_C %lld -> %lld\n",
           info.input_tiles_A_before, info.input_tiles_A_after,
           info.input_tiles_B_before, info.input_tiles_B_after,
           info.estimated_c_tiles_before, info.estimated_c_tiles_after);

    status = write_all_perms(prefix, &info);
    if (status != 0)
    {
        printf("write permutation failed: %d\n", status);
        return 4;
    }
    if (write_meta(prefix, kind, &info) != 0)
    {
        printf("write meta failed.\n");
        return 5;
    }

    reorder_info_destroy(&info);
    free(matrixA->rowpointer);
    free(matrixA->columnindex);
    free(matrixA->value);
    if (matrixB_owns_csr)
    {
        free(matrixB_original->rowpointer);
        free(matrixB_original->columnindex);
        free(matrixB_original->value);
    }
    free(matrixA);
    free(matrixB);
    return 0;
}
