#ifndef _TILESPGEMM_EXTERNAL_REORDER_
#define _TILESPGEMM_EXTERNAL_REORDER_

#include <errno.h>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

static std::string tasa_shell_quote(const std::string &s)
{
    std::string out = "'";
    for (size_t i = 0; i < s.size(); i++)
    {
        if (s[i] == '\'')
            out += "'\\''";
        else
            out += s[i];
    }
    out += "'";
    return out;
}

static std::string tasa_path_basename(const char *path)
{
    std::string s = path ? path : "";
    size_t slash = s.find_last_of('/');
    if (slash != std::string::npos)
        s = s.substr(slash + 1);
    size_t dot = s.rfind('.');
    if (dot != std::string::npos)
        s = s.substr(0, dot);
    for (size_t i = 0; i < s.size(); i++)
    {
        char c = s[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '_' || c == '-'))
            s[i] = '_';
    }
    return s.empty() ? "matrix" : s;
}

static int tasa_mkdir_if_needed(const std::string &path)
{
    if (mkdir(path.c_str(), 0775) == 0 || errno == EEXIST)
        return 0;
    return -1;
}

static int read_external_order_file(const char *path, int n, AxisPermutation *perm)
{
    int status = axis_permutation_alloc(perm, n);
    if (status != 0)
        return status;

    std::ifstream in(path);
    if (!in.is_open())
    {
        printf("external reorder order file open failed: %s\n", path);
        return -10;
    }

    std::vector<unsigned char> seen(n, 0);
    for (int new_id = 0; new_id < n; new_id++)
    {
        long long old_id_ll = -1;
        if (!(in >> old_id_ll))
        {
            printf("external reorder order file too short: %s at row %d\n", path, new_id);
            return -11;
        }
        if (old_id_ll < 0 || old_id_ll >= n)
        {
            printf("external reorder invalid id %lld at row %d, n = %d\n", old_id_ll, new_id, n);
            return -12;
        }
        int old_id = (int)old_id_ll;
        if (seen[old_id])
        {
            printf("external reorder duplicate id %d in %s\n", old_id, path);
            return -13;
        }
        seen[old_id] = 1;
        perm->old_id[new_id] = old_id;
        perm->new_id[old_id] = new_id;
    }

    long long extra = -1;
    if (in >> extra)
    {
        printf("external reorder order file has extra entries: %s\n", path);
        return -14;
    }
    return 0;
}

static int run_external_order_generator(const char *algorithm,
                                        const char *matrix_path,
                                        int n,
                                        AxisPermutation *perm,
                                        double *ordergen_ms)
{
    const char *root_env = getenv("TASA_REORDER_ALGORITHM_ROOT");
    std::string root = root_env ? root_env : "reorder_algorithm";
    const char *cache_env = getenv("TASA_REORDER_ORDER_DIR");
    std::string cache = cache_env ? cache_env : "/tmp/tasa_reorder_orders";
    tasa_mkdir_if_needed(cache);

    std::string algo = algorithm ? algorithm : "";
    std::string order_path = cache + "/" + algo + "_" + tasa_path_basename(matrix_path) +
                             "_" + std::to_string((long long)getpid()) + ".order";
    std::string order_gen = root + "/" + algo + "/order_gen";
    std::string cmd = tasa_shell_quote(order_gen) + " " +
                      tasa_shell_quote(matrix_path) + " " +
                      tasa_shell_quote(order_path);

    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    int rc = system(cmd.c_str());
    gettimeofday(&t2, NULL);
    *ordergen_ms = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    if (rc != 0)
    {
        int exit_code = WIFEXITED(rc) ? WEXITSTATUS(rc) : rc;
        if (exit_code == 42)
            printf("external reorder missing dependency: %s\n", algorithm);
        else
            printf("external reorder order_gen failed: algorithm = %s, exit = %d\n", algorithm, exit_code);
        return -20;
    }

    int status = read_external_order_file(order_path.c_str(), n, perm);
    if (status != 0)
        return status;
    unlink(order_path.c_str());
    return 0;
}

static void fill_reorder_estimates(ReorderProblemKind kind,
                                   const SMatrix *matrixA,
                                   const SMatrix *matrixB,
                                   ReorderInfo *info)
{
    if (kind == REORDER_PROBLEM_AA)
    {
        info->input_tiles_A_before = estimate_input_tile_count_biperm(matrixA, NULL, NULL);
        info->input_tiles_B_before = info->input_tiles_A_before;
        info->input_tiles_A_after = estimate_input_tile_count_biperm(matrixA, &info->row_perm, &info->inner_perm);
        info->input_tiles_B_after = info->input_tiles_A_after;
        info->estimated_c_tiles_before = estimate_symbolic_c_tile_count(matrixA, NULL, NULL, NULL);
        info->estimated_c_tiles_after = estimate_symbolic_c_tile_count(matrixA, &info->row_perm, &info->inner_perm, &info->col_perm);
    }
    else
    {
        info->input_tiles_A_before = estimate_input_tile_count_biperm(matrixA, NULL, NULL);
        info->input_tiles_B_before = estimate_input_tile_count_biperm(matrixB, NULL, NULL);
        info->input_tiles_A_after = estimate_input_tile_count_biperm(matrixA, &info->row_perm, &info->inner_perm);
        info->input_tiles_B_after = estimate_input_tile_count_biperm(matrixB, &info->inner_perm, &info->col_perm);
        info->estimated_c_tiles_before = estimate_symbolic_c_tile_count_ab(matrixA, matrixB, NULL, NULL, NULL);
        info->estimated_c_tiles_after = estimate_symbolic_c_tile_count_ab(matrixA, matrixB, &info->row_perm, &info->inner_perm, &info->col_perm);
    }

    double input_a_ratio = tasa_safe_ratio(info->input_tiles_A_after, info->input_tiles_A_before);
    double input_b_ratio = tasa_safe_ratio(info->input_tiles_B_after, info->input_tiles_B_before);
    info->risk_input_tile_ratio = input_a_ratio > input_b_ratio ? input_a_ratio : input_b_ratio;
    info->risk_c_tile_ratio = tasa_safe_ratio(info->estimated_c_tiles_after, info->estimated_c_tiles_before);
}

static int build_original_reorder_plan(ReorderProblemKind kind,
                                       const SMatrix *matrixA,
                                       const SMatrix *matrixB,
                                       ReorderInfo *info)
{
    if (matrixA == NULL || info == NULL)
        return -1;
    if (kind != REORDER_PROBLEM_AA && matrixB == NULL)
        return -2;

    reorder_info_destroy(info);
    int status = axis_permutation_identity(&info->row_perm, matrixA->m);
    if (status == 0)
        status = axis_permutation_identity(&info->inner_perm, matrixA->n);
    if (status == 0)
        status = axis_permutation_identity(&info->col_perm,
                                           kind == REORDER_PROBLEM_AA ? matrixA->n : matrixB->n);
    if (status != 0)
        return status;

    info->enabled = 1;
    info->n = matrixA->n;
    info->shared_symmetric_csr = (kind == REORDER_PROBLEM_AA);
    info->guard_degraded_to_identity = 1;
    info->structural_conservative = 0;
    reorder_info_set_text(info->mode, sizeof(info->mode), "original");
    reorder_info_set_text(info->risk_reason, sizeof(info->risk_reason), "original_order");
    fill_reorder_estimates(kind, matrixA, kind == REORDER_PROBLEM_AA ? matrixA : matrixB, info);
    return 0;
}

static int build_external_reorder_plan(ReorderProblemKind kind,
                                       const char *algorithm,
                                       const char *matrix_path_A,
                                       const char *matrix_path_B,
                                       const SMatrix *matrixA,
                                       const SMatrix *matrixB,
                                       ReorderInfo *info,
                                       double *ordergen_ms)
{
    if (matrixA == NULL || info == NULL || ordergen_ms == NULL)
        return -1;
    if (kind != REORDER_PROBLEM_AA && matrixB == NULL)
        return -2;
    if (kind != REORDER_PROBLEM_AA && matrixA->n != matrixB->m)
        return -3;

    reorder_info_destroy(info);
    *ordergen_ms = 0.0;

    double elapsed = 0.0;
    int status = 0;

    if (kind == REORDER_PROBLEM_AA)
    {
        if (matrixA->m != matrixA->n)
        {
            printf("external reorder AA currently requires a square matrix.\n");
            return -4;
        }

        status = run_external_order_generator(algorithm, matrix_path_A, matrixA->m, &info->row_perm, &elapsed);
        *ordergen_ms += elapsed;
        if (status != 0)
            return status;

        status = axis_permutation_copy(&info->inner_perm, &info->row_perm);
        if (status != 0)
            return status;
        status = axis_permutation_copy(&info->col_perm, &info->row_perm);
        if (status != 0)
            return status;

        info->shared_symmetric_csr = 1;
    }
    else if (kind == REORDER_PROBLEM_AAT)
    {
        if (matrixA->m != matrixA->n ||
            matrixB->m != matrixB->n ||
            matrixA->m != matrixB->m)
        {
            printf("external reorder AAT currently requires square A and A^T with matching dimensions.\n");
            return -5;
        }

        status = run_external_order_generator(algorithm, matrix_path_A, matrixA->m, &info->row_perm, &elapsed);
        *ordergen_ms += elapsed;
        if (status != 0)
            return status;
        status = axis_permutation_copy(&info->inner_perm, &info->row_perm);
        if (status != 0)
            return status;
        status = axis_permutation_copy(&info->col_perm, &info->row_perm);
        if (status != 0)
            return status;

        info->shared_symmetric_csr = 0;
    }
    else if (kind == REORDER_PROBLEM_AB)
    {
        if (matrixA->m != matrixA->n)
        {
            printf("external reorder AB currently requires square A so row and inner axes can share A's order.\n");
            return -6;
        }
        if (matrixB->m != matrixB->n)
        {
            printf("external reorder AB currently requires square B so B row and column axes can be ordered by B.\n");
            return -7;
        }
        if (matrixA->n != matrixB->m)
        {
            printf("external reorder AB requires colA == rowB.\n");
            return -8;
        }

        status = run_external_order_generator(algorithm, matrix_path_A, matrixA->m, &info->row_perm, &elapsed);
        *ordergen_ms += elapsed;
        if (status != 0)
            return status;
        status = axis_permutation_copy(&info->inner_perm, &info->row_perm);
        if (status != 0)
            return status;

        const char *col_order_path = matrix_path_B ? matrix_path_B : matrix_path_A;
        status = run_external_order_generator(algorithm, col_order_path, matrixB->n, &info->col_perm, &elapsed);
        *ordergen_ms += elapsed;
        if (status != 0)
            return status;

        info->shared_symmetric_csr = 0;
    }
    else
    {
        return -9;
    }

    info->enabled = 1;
    info->n = matrixA->n;
    info->guard_degraded_to_identity = 0;
    info->structural_conservative = 0;
    reorder_info_set_text(info->mode, sizeof(info->mode), algorithm);
    reorder_info_set_text(info->risk_reason, sizeof(info->risk_reason), "external_order");

    fill_reorder_estimates(kind, matrixA, kind == REORDER_PROBLEM_AA ? matrixA : matrixB, info);
    return 0;
}

#endif
