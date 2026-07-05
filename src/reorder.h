#ifndef _TILESPGEMM_REORDER_
#define _TILESPGEMM_REORDER_

#include "common.h"

#include <algorithm>
#include <limits.h>
#include <numeric>
#include <stdint.h>
#include <utility>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#define GTSP_PROXY_SAMPLE_ROWS 64
#define GTSP_PROXY_GUARD_RATIO 1.10

typedef struct
{
    int n;
    int *old_id; // new id -> original id
    int *new_id; // original id -> new id
} AxisPermutation;

typedef struct
{
    int enabled;
    int n;
    AxisPermutation row_perm;
    AxisPermutation inner_perm;
    AxisPermutation col_perm;
    long long input_tiles_A_before;
    long long input_tiles_A_after;
    long long input_tiles_B_before;
    long long input_tiles_B_after;
    long long estimated_c_tiles_before;
    long long estimated_c_tiles_after;
    double reorder_time_ms;
    double signature_time_ms;
    double tile_profile_time_ms;
    double affinity_time_ms;
    double sort_time_ms;
    double local_refine_time_ms;
    double proxy_time_ms;
    double permute_time_ms;
    int local_refine_window;
    int guard_degraded_to_identity;
} ReorderInfo;

typedef struct
{
    int size;
    thrust::device_vector<int> tile_counts;
    thrust::device_vector<int> min_tiles;
    thrust::device_vector<int> max_tiles;
    thrust::device_vector<int> second_tiles;
    thrust::device_vector<int> third_tiles;
    thrust::device_vector<int> degrees;
    thrust::device_vector<unsigned long long> hash1;
    thrust::device_vector<unsigned long long> hash2;
} GtspSignatureArrays;

typedef struct
{
    int id;
    int band;
    int min_tile;
    int max_tile;
    int second_tile;
    int third_tile;
    int center;
    int tile_bucket;
    int span;
    int degree_bucket;
    unsigned long long hash1;
    unsigned long long hash2;
} GtspSortItem;

typedef struct
{
    const int *row_new_to_old;
    const int *col_old_to_new;
} GtspDevicePermView;

void axis_permutation_init(AxisPermutation *perm)
{
    perm->n = 0;
    perm->old_id = NULL;
    perm->new_id = NULL;
}

void axis_permutation_destroy(AxisPermutation *perm)
{
    if (perm->old_id)
        free(perm->old_id);
    if (perm->new_id)
        free(perm->new_id);
    axis_permutation_init(perm);
}

int axis_permutation_alloc(AxisPermutation *perm, int n)
{
    axis_permutation_destroy(perm);
    perm->n = n;
    perm->old_id = (int *)malloc(n * sizeof(int));
    perm->new_id = (int *)malloc(n * sizeof(int));
    if (perm->old_id == NULL || perm->new_id == NULL)
        return -1;
    return 0;
}

int axis_permutation_identity(AxisPermutation *perm, int n)
{
    int status = axis_permutation_alloc(perm, n);
    if (status != 0)
        return status;
    for (int i = 0; i < n; i++)
    {
        perm->old_id[i] = i;
        perm->new_id[i] = i;
    }
    return 0;
}

int axis_permutation_copy(AxisPermutation *dst, const AxisPermutation *src)
{
    int status = axis_permutation_alloc(dst, src->n);
    if (status != 0)
        return status;
    memcpy(dst->old_id, src->old_id, src->n * sizeof(int));
    memcpy(dst->new_id, src->new_id, src->n * sizeof(int));
    return 0;
}

void reorder_info_init(ReorderInfo *info)
{
    info->enabled = 0;
    info->n = 0;
    axis_permutation_init(&info->row_perm);
    axis_permutation_init(&info->inner_perm);
    axis_permutation_init(&info->col_perm);
    info->input_tiles_A_before = 0;
    info->input_tiles_A_after = 0;
    info->input_tiles_B_before = 0;
    info->input_tiles_B_after = 0;
    info->estimated_c_tiles_before = 0;
    info->estimated_c_tiles_after = 0;
    info->reorder_time_ms = 0.0;
    info->signature_time_ms = 0.0;
    info->tile_profile_time_ms = 0.0;
    info->affinity_time_ms = 0.0;
    info->sort_time_ms = 0.0;
    info->local_refine_time_ms = 0.0;
    info->proxy_time_ms = 0.0;
    info->permute_time_ms = 0.0;
    info->local_refine_window = 0;
    info->guard_degraded_to_identity = 0;
}

void reorder_info_destroy(ReorderInfo *info)
{
    axis_permutation_destroy(&info->row_perm);
    axis_permutation_destroy(&info->inner_perm);
    axis_permutation_destroy(&info->col_perm);
    reorder_info_init(info);
}

static int axis_old_id(const AxisPermutation *perm, int id)
{
    return perm ? perm->old_id[id] : id;
}

static int axis_new_id(const AxisPermutation *perm, int id)
{
    return perm ? perm->new_id[id] : id;
}

static void sort_csr_rows_by_column(int m,
                                    MAT_PTR_TYPE *rowptr,
                                    int *colidx,
                                    MAT_VAL_TYPE *values)
{
#pragma omp parallel
    {
        std::vector<std::pair<int, MAT_VAL_TYPE> > row;
#pragma omp for schedule(dynamic, 1024)
        for (int i = 0; i < m; i++)
        {
            MAT_PTR_TYPE start = rowptr[i];
            MAT_PTR_TYPE end = rowptr[i + 1];
            int len = end - start;
            if (len <= 1)
                continue;

            row.clear();
            row.reserve(len);
            for (MAT_PTR_TYPE j = start; j < end; j++)
                row.push_back(std::make_pair(colidx[j], values[j]));

            std::sort(row.begin(), row.end(),
                      [](const std::pair<int, MAT_VAL_TYPE> &a,
                         const std::pair<int, MAT_VAL_TYPE> &b) {
                          return a.first < b.first;
                      });

            for (int j = 0; j < len; j++)
            {
                colidx[start + j] = row[j].first;
                values[start + j] = row[j].second;
            }
        }
    }
}

__host__ __device__ static inline unsigned long long gtsp_mix64(unsigned long long x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__host__ __device__ static inline int gtsp_bucket_id(int value)
{
    int bucket = 0;
    int v = value > 0 ? value : 0;
    while (v > 1 && bucket < 31)
    {
        v >>= 1;
        bucket++;
    }
    return bucket;
}

__host__ __device__ static inline int gtsp_band_id(int tile_count, int min_tile, int max_tile, int dim_tiles)
{
    if (tile_count <= 0 || min_tile >= dim_tiles)
        return 2147483647;
    int span = max_tile >= min_tile ? max_tile - min_tile + 1 : 0;
    int band_width = span <= 8 ? 4 : (span <= 32 ? 8 : 16);
    return min_tile / band_width;
}

struct GtspSortItemLess
{
    __host__ __device__ bool operator()(const GtspSortItem &a, const GtspSortItem &b) const
    {
        if (a.band != b.band)
            return a.band < b.band;
        if (a.min_tile != b.min_tile)
            return a.min_tile < b.min_tile;
        if (a.second_tile != b.second_tile)
            return a.second_tile < b.second_tile;
        if (a.third_tile != b.third_tile)
            return a.third_tile < b.third_tile;
        if (a.tile_bucket != b.tile_bucket)
            return a.tile_bucket > b.tile_bucket;
        if (a.span != b.span)
            return a.span < b.span;
        if (a.max_tile != b.max_tile)
            return a.max_tile < b.max_tile;
        if (a.degree_bucket != b.degree_bucket)
            return a.degree_bucket > b.degree_bucket;
        if (a.hash1 != b.hash1)
            return a.hash1 < b.hash1;
        if (a.hash2 != b.hash2)
            return a.hash2 < b.hash2;
        return a.id < b.id;
    }
};

__global__ void gtsp_row_tile_signatures_kernel(int rows,
                                                const MAT_PTR_TYPE *rowptr,
                                                const int *colidx,
                                                int *tile_counts,
                                                int *min_tiles,
                                                int *max_tiles,
                                                int *second_tiles,
                                                int *third_tiles,
                                                int *degrees,
                                                unsigned long long *hash1,
                                                unsigned long long *hash2)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
        return;

    MAT_PTR_TYPE begin = rowptr[row];
    MAT_PTR_TYPE end = rowptr[row + 1];
    int count = 0;
    int min_tile = 2147483647;
    int max_tile = -1;
    int second_tile = 2147483647;
    int third_tile = 2147483647;
    int last_tile = -1;
    unsigned long long h1 = 1469598103934665603ULL;
    unsigned long long h2 = 1099511628211ULL ^ (unsigned long long)row;

    for (MAT_PTR_TYPE p = begin; p < end; p++)
    {
        int tile = colidx[p] / BLOCK_SIZE;
        if (tile == last_tile)
            continue;
        if (count == 1)
            second_tile = tile;
        else if (count == 2)
            third_tile = tile;
        count++;
        min_tile = min_tile < tile ? min_tile : tile;
        max_tile = max_tile > tile ? max_tile : tile;
        unsigned long long mixed = gtsp_mix64((unsigned long long)tile);
        h1 ^= mixed;
        h1 *= 1099511628211ULL;
        h2 ^= mixed + (h2 << 6) + (h2 >> 2);
        last_tile = tile;
    }

    tile_counts[row] = count;
    min_tiles[row] = count > 0 ? min_tile : 2147483647;
    max_tiles[row] = max_tile;
    second_tiles[row] = second_tile;
    third_tiles[row] = third_tile;
    degrees[row] = (int)(end - begin);
    hash1[row] = h1;
    hash2[row] = h2;
}

__global__ void gtsp_emit_column_tile_keys_kernel(int rows,
                                                  const MAT_PTR_TYPE *rowptr,
                                                  const int *colidx,
                                                  unsigned long long *tile_keys,
                                                  int *degrees)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
        return;

    int row_tile = row / BLOCK_SIZE;
    for (MAT_PTR_TYPE p = rowptr[row]; p < rowptr[row + 1]; p++)
    {
        int col = colidx[p];
        tile_keys[p] = ((unsigned long long)(unsigned int)col << 32) | (unsigned int)row_tile;
        atomicAdd(degrees + col, 1);
    }
}

__global__ void gtsp_accumulate_column_signatures_kernel(int unique_count,
                                                         const unsigned long long *tile_keys,
                                                         int *tile_counts,
                                                         int *min_tiles,
                                                         int *max_tiles,
                                                         int *second_tiles,
                                                         int *third_tiles,
                                                         unsigned long long *hash1,
                                                         unsigned long long *hash2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= unique_count)
        return;

    unsigned long long key = tile_keys[idx];
    int col = (int)(key >> 32);
    int row_tile = (int)(key & 0xffffffffULL);
    bool first_for_col = idx == 0 || (int)(tile_keys[idx - 1] >> 32) != col;
    bool second_for_col = !first_for_col && (idx < 2 || (int)(tile_keys[idx - 2] >> 32) != col);
    bool third_for_col = !first_for_col && !second_for_col &&
                         (idx < 3 || (int)(tile_keys[idx - 3] >> 32) != col);

    atomicAdd(tile_counts + col, 1);
    atomicMin(min_tiles + col, row_tile);
    atomicMax(max_tiles + col, row_tile);
    if (second_for_col)
        second_tiles[col] = row_tile;
    if (third_for_col)
        third_tiles[col] = row_tile;
    atomicXor(hash1 + col, gtsp_mix64((unsigned long long)row_tile));
    atomicXor(hash2 + col, gtsp_mix64((unsigned long long)row_tile ^ 0x9e3779b97f4a7c15ULL));
}

__global__ void gtsp_make_sort_items_kernel(int n,
                                            int dim_tiles,
                                            const int *tile_counts,
                                            const int *min_tiles,
                                            const int *max_tiles,
                                            const int *second_tiles,
                                            const int *third_tiles,
                                            const int *degrees,
                                            const unsigned long long *hash1,
                                            const unsigned long long *hash2,
                                            GtspSortItem *items)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n)
        return;

    int count = tile_counts[id];
    int min_tile = min_tiles[id];
    int max_tile = max_tiles[id];
    items[id].id = id;
    items[id].band = gtsp_band_id(count, min_tile, max_tile, dim_tiles);
    items[id].min_tile = count > 0 ? min_tile : 2147483647;
    items[id].max_tile = max_tile;
    items[id].second_tile = second_tiles[id];
    items[id].third_tile = third_tiles[id];
    items[id].center = count > 0 && max_tile >= min_tile ? ((min_tile + max_tile) >> 1) : 2147483647;
    items[id].tile_bucket = gtsp_bucket_id(count);
    items[id].span = max_tile >= min_tile ? max_tile - min_tile + 1 : 0;
    items[id].degree_bucket = gtsp_bucket_id(degrees[id]);
    items[id].hash1 = hash1[id];
    items[id].hash2 = hash2[id];
}

__global__ void gtsp_make_combined_sort_items_kernel(int n,
                                                     int dim_tiles,
                                                     const int *left_counts,
                                                     const int *left_min,
                                                     const int *left_max,
                                                     const int *left_second,
                                                     const int *left_third,
                                                     const int *left_degrees,
                                                     const unsigned long long *left_hash1,
                                                     const unsigned long long *left_hash2,
                                                     const int *right_counts,
                                                     const int *right_min,
                                                     const int *right_max,
                                                     const int *right_second,
                                                     const int *right_third,
                                                     const int *right_degrees,
                                                     const unsigned long long *right_hash1,
                                                     const unsigned long long *right_hash2,
                                                     GtspSortItem *items)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n)
        return;

    int count = left_counts[id] + right_counts[id];
    int min_tile = min(left_min[id], right_min[id]);
    int max_tile = max(left_max[id], right_max[id]);
    items[id].id = id;
    items[id].band = gtsp_band_id(count, min_tile, max_tile, dim_tiles);
    items[id].min_tile = count > 0 ? min_tile : 2147483647;
    items[id].max_tile = max_tile;
    items[id].second_tile = min(left_second[id], right_second[id]);
    items[id].third_tile = min(left_third[id], right_third[id]);
    items[id].center = count > 0 && max_tile >= min_tile ? ((min_tile + max_tile) >> 1) : 2147483647;
    items[id].tile_bucket = gtsp_bucket_id(count);
    items[id].span = max_tile >= min_tile ? max_tile - min_tile + 1 : 0;
    items[id].degree_bucket = gtsp_bucket_id(left_degrees[id] + right_degrees[id]);
    items[id].hash1 = left_hash1[id] ^ gtsp_mix64(right_hash1[id]);
    items[id].hash2 = right_hash2[id] ^ gtsp_mix64(left_hash2[id]);
}

__global__ void gtsp_extract_new_to_old_kernel(int n, const GtspSortItem *items, int *new_to_old)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        new_to_old[idx] = items[idx].id;
}

__global__ void gtsp_invert_permutation_kernel(int n, const int *new_to_old, int *old_to_new)
{
    int new_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (new_id < n)
        old_to_new[new_to_old[new_id]] = new_id;
}

__global__ void gtsp_permuted_row_counts_kernel(int rows,
                                                const MAT_PTR_TYPE *rowptr,
                                                const int *new_to_old,
                                                MAT_PTR_TYPE *counts)
{
    int new_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (new_row >= rows)
        return;
    int old_row = new_to_old[new_row];
    counts[new_row] = rowptr[old_row + 1] - rowptr[old_row];
}

__global__ void gtsp_emit_permuted_entries_kernel(int rows,
                                                  const MAT_PTR_TYPE *in_rowptr,
                                                  const int *in_colidx,
                                                  const MAT_VAL_TYPE *in_values,
                                                  const int *row_new_to_old,
                                                  const int *col_old_to_new,
                                                  const MAT_PTR_TYPE *out_rowptr,
                                                  unsigned long long *keys,
                                                  int *out_colidx,
                                                  MAT_VAL_TYPE *out_values)
{
    int new_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (new_row >= rows)
        return;

    int old_row = row_new_to_old[new_row];
    MAT_PTR_TYPE old_begin = in_rowptr[old_row];
    MAT_PTR_TYPE old_end = in_rowptr[old_row + 1];
    MAT_PTR_TYPE out_begin = out_rowptr[new_row];
    for (MAT_PTR_TYPE p = old_begin; p < old_end; p++)
    {
        MAT_PTR_TYPE local = p - old_begin;
        MAT_PTR_TYPE out_pos = out_begin + local;
        int new_col = col_old_to_new[in_colidx[p]];
        keys[out_pos] = ((unsigned long long)(unsigned int)new_row << 32) | (unsigned int)new_col;
        out_colidx[out_pos] = new_col;
        out_values[out_pos] = in_values[p];
    }
}

static void gtsp_cuda_sync(const char *label)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("%s failed: %s\n", label, cudaGetErrorString(err));
    cudaDeviceSynchronize();
}

static GtspSignatureArrays gtsp_make_empty_signatures(int size)
{
    GtspSignatureArrays sig;
    sig.size = size;
    sig.tile_counts = thrust::device_vector<int>(size, 0);
    sig.min_tiles = thrust::device_vector<int>(size, 2147483647);
    sig.max_tiles = thrust::device_vector<int>(size, -1);
    sig.second_tiles = thrust::device_vector<int>(size, 2147483647);
    sig.third_tiles = thrust::device_vector<int>(size, 2147483647);
    sig.degrees = thrust::device_vector<int>(size, 0);
    sig.hash1 = thrust::device_vector<unsigned long long>(size, 0);
    sig.hash2 = thrust::device_vector<unsigned long long>(size, 0);
    return sig;
}

static GtspSignatureArrays gtsp_compute_row_signatures_gpu(const SMatrix *matrix)
{
    GtspSignatureArrays sig = gtsp_make_empty_signatures(matrix->m);
    if (matrix->m <= 0)
        return sig;

    thrust::device_vector<MAT_PTR_TYPE> d_rowptr(matrix->rowpointer, matrix->rowpointer + matrix->m + 1);
    thrust::device_vector<int> d_colidx(matrix->columnindex, matrix->columnindex + matrix->nnz);
    int threads = 256;
    int blocks = (matrix->m + threads - 1) / threads;
    gtsp_row_tile_signatures_kernel<<<blocks, threads>>>(
        matrix->m,
        thrust::raw_pointer_cast(d_rowptr.data()),
        thrust::raw_pointer_cast(d_colidx.data()),
        thrust::raw_pointer_cast(sig.tile_counts.data()),
        thrust::raw_pointer_cast(sig.min_tiles.data()),
        thrust::raw_pointer_cast(sig.max_tiles.data()),
        thrust::raw_pointer_cast(sig.second_tiles.data()),
        thrust::raw_pointer_cast(sig.third_tiles.data()),
        thrust::raw_pointer_cast(sig.degrees.data()),
        thrust::raw_pointer_cast(sig.hash1.data()),
        thrust::raw_pointer_cast(sig.hash2.data()));
    gtsp_cuda_sync("GTSP row signature kernel");
    return sig;
}

static GtspSignatureArrays gtsp_compute_column_signatures_gpu(const SMatrix *matrix)
{
    GtspSignatureArrays sig = gtsp_make_empty_signatures(matrix->n);
    if (matrix->n <= 0 || matrix->m <= 0 || matrix->nnz <= 0)
        return sig;

    thrust::device_vector<MAT_PTR_TYPE> d_rowptr(matrix->rowpointer, matrix->rowpointer + matrix->m + 1);
    thrust::device_vector<int> d_colidx(matrix->columnindex, matrix->columnindex + matrix->nnz);
    thrust::device_vector<unsigned long long> d_tile_keys(matrix->nnz, 0);
    int threads = 256;
    int row_blocks = (matrix->m + threads - 1) / threads;

    gtsp_emit_column_tile_keys_kernel<<<row_blocks, threads>>>(
        matrix->m,
        thrust::raw_pointer_cast(d_rowptr.data()),
        thrust::raw_pointer_cast(d_colidx.data()),
        thrust::raw_pointer_cast(d_tile_keys.data()),
        thrust::raw_pointer_cast(sig.degrees.data()));
    gtsp_cuda_sync("GTSP column key kernel");

    thrust::sort(d_tile_keys.begin(), d_tile_keys.end());
    typename thrust::device_vector<unsigned long long>::iterator unique_end =
        thrust::unique(d_tile_keys.begin(), d_tile_keys.end());
    int unique_count = (int)(unique_end - d_tile_keys.begin());
    int unique_blocks = (unique_count + threads - 1) / threads;
    if (unique_count > 0)
    {
        gtsp_accumulate_column_signatures_kernel<<<unique_blocks, threads>>>(
            unique_count,
            thrust::raw_pointer_cast(d_tile_keys.data()),
            thrust::raw_pointer_cast(sig.tile_counts.data()),
            thrust::raw_pointer_cast(sig.min_tiles.data()),
            thrust::raw_pointer_cast(sig.max_tiles.data()),
            thrust::raw_pointer_cast(sig.second_tiles.data()),
            thrust::raw_pointer_cast(sig.third_tiles.data()),
            thrust::raw_pointer_cast(sig.hash1.data()),
            thrust::raw_pointer_cast(sig.hash2.data()));
        gtsp_cuda_sync("GTSP column signature kernel");
    }
    return sig;
}

static int gtsp_copy_device_order_to_axis(thrust::device_vector<int> &d_new_to_old,
                                          AxisPermutation *perm)
{
    int n = (int)d_new_to_old.size();
    int status = axis_permutation_alloc(perm, n);
    if (status != 0)
        return status;

    thrust::device_vector<int> d_old_to_new(n, -1);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (n > 0)
    {
        gtsp_invert_permutation_kernel<<<blocks, threads>>>(
            n,
            thrust::raw_pointer_cast(d_new_to_old.data()),
            thrust::raw_pointer_cast(d_old_to_new.data()));
        gtsp_cuda_sync("GTSP invert permutation kernel");
    }

    thrust::copy(d_new_to_old.begin(), d_new_to_old.end(), perm->old_id);
    thrust::copy(d_old_to_new.begin(), d_old_to_new.end(), perm->new_id);
    return 0;
}

static int gtsp_make_permutation_from_signatures(const GtspSignatureArrays &sig,
                                                 int dim_tiles,
                                                 AxisPermutation *perm)
{
    thrust::device_vector<GtspSortItem> d_items(sig.size);
    int threads = 256;
    int blocks = (sig.size + threads - 1) / threads;
    if (sig.size > 0)
    {
        gtsp_make_sort_items_kernel<<<blocks, threads>>>(
            sig.size,
            dim_tiles,
            thrust::raw_pointer_cast(sig.tile_counts.data()),
            thrust::raw_pointer_cast(sig.min_tiles.data()),
            thrust::raw_pointer_cast(sig.max_tiles.data()),
            thrust::raw_pointer_cast(sig.second_tiles.data()),
            thrust::raw_pointer_cast(sig.third_tiles.data()),
            thrust::raw_pointer_cast(sig.degrees.data()),
            thrust::raw_pointer_cast(sig.hash1.data()),
            thrust::raw_pointer_cast(sig.hash2.data()),
            thrust::raw_pointer_cast(d_items.data()));
        gtsp_cuda_sync("GTSP sort item kernel");
        thrust::sort(d_items.begin(), d_items.end(), GtspSortItemLess());
    }

    thrust::device_vector<int> d_new_to_old(sig.size);
    if (sig.size > 0)
    {
        gtsp_extract_new_to_old_kernel<<<blocks, threads>>>(
            sig.size,
            thrust::raw_pointer_cast(d_items.data()),
            thrust::raw_pointer_cast(d_new_to_old.data()));
        gtsp_cuda_sync("GTSP extract order kernel");
    }
    return gtsp_copy_device_order_to_axis(d_new_to_old, perm);
}

static int gtsp_make_permutation_from_combined_signatures(const GtspSignatureArrays &left,
                                                          const GtspSignatureArrays &right,
                                                          int dim_tiles,
                                                          AxisPermutation *perm)
{
    if (left.size != right.size)
        return -1;

    thrust::device_vector<GtspSortItem> d_items(left.size);
    int threads = 256;
    int blocks = (left.size + threads - 1) / threads;
    if (left.size > 0)
    {
        gtsp_make_combined_sort_items_kernel<<<blocks, threads>>>(
            left.size,
            dim_tiles,
            thrust::raw_pointer_cast(left.tile_counts.data()),
            thrust::raw_pointer_cast(left.min_tiles.data()),
            thrust::raw_pointer_cast(left.max_tiles.data()),
            thrust::raw_pointer_cast(left.second_tiles.data()),
            thrust::raw_pointer_cast(left.third_tiles.data()),
            thrust::raw_pointer_cast(left.degrees.data()),
            thrust::raw_pointer_cast(left.hash1.data()),
            thrust::raw_pointer_cast(left.hash2.data()),
            thrust::raw_pointer_cast(right.tile_counts.data()),
            thrust::raw_pointer_cast(right.min_tiles.data()),
            thrust::raw_pointer_cast(right.max_tiles.data()),
            thrust::raw_pointer_cast(right.second_tiles.data()),
            thrust::raw_pointer_cast(right.third_tiles.data()),
            thrust::raw_pointer_cast(right.degrees.data()),
            thrust::raw_pointer_cast(right.hash1.data()),
            thrust::raw_pointer_cast(right.hash2.data()),
            thrust::raw_pointer_cast(d_items.data()));
        gtsp_cuda_sync("GTSP combined sort item kernel");
        thrust::sort(d_items.begin(), d_items.end(), GtspSortItemLess());
    }

    thrust::device_vector<int> d_new_to_old(left.size);
    if (left.size > 0)
    {
        gtsp_extract_new_to_old_kernel<<<blocks, threads>>>(
            left.size,
            thrust::raw_pointer_cast(d_items.data()),
            thrust::raw_pointer_cast(d_new_to_old.data()));
        gtsp_cuda_sync("GTSP extract combined order kernel");
    }
    return gtsp_copy_device_order_to_axis(d_new_to_old, perm);
}

long long estimate_input_tile_count_biperm(const SMatrix *matrix,
                                           const AxisPermutation *row_perm,
                                           const AxisPermutation *col_perm)
{
    int m = matrix->m;
    int tilem = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int sample_count = tilem < GTSP_PROXY_SAMPLE_ROWS ? tilem : GTSP_PROXY_SAMPLE_ROWS;
    int stride = sample_count > 0 ? (tilem + sample_count - 1) / sample_count : 1;

    long long total = 0;
    int actual_samples = 0;
#pragma omp parallel reduction(+ : total)
    {
        std::vector<int> tile_cols;
#pragma omp for schedule(dynamic, 64) reduction(+ : actual_samples)
        for (int sample = 0; sample < sample_count; sample++)
        {
            int tile_row = sample * stride;
            if (tile_row >= tilem)
                continue;
            tile_cols.clear();
            int row_start = tile_row * BLOCK_SIZE;
            int row_end = row_start + BLOCK_SIZE;
            row_end = row_end > m ? m : row_end;

            for (int new_row = row_start; new_row < row_end; new_row++)
            {
                int old_row = axis_old_id(row_perm, new_row);
                for (MAT_PTR_TYPE j = matrix->rowpointer[old_row]; j < matrix->rowpointer[old_row + 1]; j++)
                {
                    int new_col = axis_new_id(col_perm, matrix->columnindex[j]);
                    tile_cols.push_back(new_col / BLOCK_SIZE);
                }
            }

            if (!tile_cols.empty())
            {
                std::sort(tile_cols.begin(), tile_cols.end());
                total += std::unique(tile_cols.begin(), tile_cols.end()) - tile_cols.begin();
            }
            actual_samples++;
        }
    }

    if (actual_samples == 0)
        return 0;
    return (long long)((double)total * (double)tilem / (double)actual_samples);
}

static void gtsp_collect_tile_row_cols(const SMatrix *matrix,
                                       int tile_row,
                                       const AxisPermutation *row_perm,
                                       const AxisPermutation *col_perm,
                                       std::vector<int> &cols)
{
    int m = matrix->m;
    cols.clear();
    int row_start = tile_row * BLOCK_SIZE;
    int row_end = row_start + BLOCK_SIZE;
    row_end = row_end > m ? m : row_end;

    for (int new_row = row_start; new_row < row_end; new_row++)
    {
        int old_row = axis_old_id(row_perm, new_row);
        for (MAT_PTR_TYPE j = matrix->rowpointer[old_row]; j < matrix->rowpointer[old_row + 1]; j++)
            cols.push_back(axis_new_id(col_perm, matrix->columnindex[j]) / BLOCK_SIZE);
    }

    if (!cols.empty())
    {
        std::sort(cols.begin(), cols.end());
        cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
    }
}

static int gtsp_mark_tile_row_cols(const SMatrix *matrix,
                                   int tile_row,
                                   const AxisPermutation *row_perm,
                                   const AxisPermutation *col_perm,
                                   std::vector<int> &mark,
                                   int tag)
{
    int m = matrix->m;
    int row_start = tile_row * BLOCK_SIZE;
    int row_end = row_start + BLOCK_SIZE;
    row_end = row_end > m ? m : row_end;
    int added = 0;

    for (int new_row = row_start; new_row < row_end; new_row++)
    {
        int old_row = axis_old_id(row_perm, new_row);
        for (MAT_PTR_TYPE j = matrix->rowpointer[old_row]; j < matrix->rowpointer[old_row + 1]; j++)
        {
            int col_tile = axis_new_id(col_perm, matrix->columnindex[j]) / BLOCK_SIZE;
            if (mark[col_tile] != tag)
            {
                mark[col_tile] = tag;
                added++;
            }
        }
    }

    return added;
}

long long estimate_symbolic_c_tile_count(const SMatrix *matrix,
                                         const AxisPermutation *row_perm,
                                         const AxisPermutation *inner_perm,
                                         const AxisPermutation *col_perm)
{
    int n = matrix->m;
    int tile_count = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int sample_count = tile_count < GTSP_PROXY_SAMPLE_ROWS ? tile_count : GTSP_PROXY_SAMPLE_ROWS;
    int stride = sample_count > 0 ? (tile_count + sample_count - 1) / sample_count : 1;
    long long total = 0;
    int actual_samples = 0;

#pragma omp parallel reduction(+ : total, actual_samples)
    {
        std::vector<int> mark(tile_count, 0);
        std::vector<int> a_inner_tiles;
        int tag = 1;
#pragma omp for schedule(dynamic, 16)
        for (int sample = 0; sample < sample_count; sample++)
        {
            int row_tile = sample * stride;
            if (row_tile >= tile_count)
                continue;
            int row_count = 0;
            tag++;
            gtsp_collect_tile_row_cols(matrix, row_tile, row_perm, inner_perm, a_inner_tiles);
            for (size_t ai = 0; ai < a_inner_tiles.size(); ai++)
            {
                int inner_tile = a_inner_tiles[ai];
                if (inner_tile < 0 || inner_tile >= tile_count)
                    continue;
                row_count += gtsp_mark_tile_row_cols(matrix, inner_tile, inner_perm, col_perm, mark, tag);
            }
            total += row_count;
            actual_samples++;
        }
    }

    if (actual_samples == 0)
        return 0;
    return (long long)((double)total * (double)tile_count / (double)actual_samples);
}

typedef struct
{
    int n;
    std::vector<int> ptr;
    std::vector<int> idx;
    std::vector<int> counts;
    std::vector<int> min_tiles;
    std::vector<int> max_tiles;
    std::vector<int> degrees;
    std::vector<unsigned long long> mh0;
    std::vector<unsigned long long> mh1;
    std::vector<unsigned long long> mh2;
    std::vector<unsigned long long> mh3;
    std::vector<unsigned long long> hash1;
    std::vector<unsigned long long> hash2;
} TasaAxisSets;

typedef struct
{
    int tile_count;
    TasaAxisSets row_sets;
    TasaAxisSets col_sets;
} TasaTileProfile;

typedef struct
{
    int count;
    int min_rank;
    int max_rank;
    int second_rank;
    int third_rank;
    unsigned long long mh0;
    unsigned long long mh1;
    unsigned long long mh2;
    unsigned long long mh3;
    unsigned long long hash1;
    unsigned long long hash2;
} TasaMappedStats;

typedef struct
{
    int id;
    int empty;
    int primary_band;
    int project_band;
    int primary_min;
    int primary_second;
    int primary_third;
    int primary_span;
    int primary_bucket;
    int project_bucket;
    int degree_bucket;
    unsigned long long primary_mh0;
    unsigned long long primary_mh1;
    unsigned long long project_mh0;
    unsigned long long project_mh1;
    unsigned long long hash1;
    unsigned long long hash2;
} TasaOrderItem;

enum TasaRole
{
    TASA_ROLE_ROW = 0,
    TASA_ROLE_INNER = 1,
    TASA_ROLE_COL = 2
};

static inline void tasa_stats_init(TasaMappedStats *stats)
{
    stats->count = 0;
    stats->min_rank = INT_MAX;
    stats->max_rank = -1;
    stats->second_rank = INT_MAX;
    stats->third_rank = INT_MAX;
    stats->mh0 = ULLONG_MAX;
    stats->mh1 = ULLONG_MAX;
    stats->mh2 = ULLONG_MAX;
    stats->mh3 = ULLONG_MAX;
    stats->hash1 = 1469598103934665603ULL;
    stats->hash2 = 1099511628211ULL;
}

static inline void tasa_stats_add_rank(TasaMappedStats *stats, int rank)
{
    if (rank < 0)
        return;
    stats->count++;
    if (rank < stats->min_rank)
    {
        stats->third_rank = stats->second_rank;
        stats->second_rank = stats->min_rank;
        stats->min_rank = rank;
    }
    else if (rank != stats->min_rank && rank < stats->second_rank)
    {
        stats->third_rank = stats->second_rank;
        stats->second_rank = rank;
    }
    else if (rank != stats->min_rank && rank != stats->second_rank && rank < stats->third_rank)
    {
        stats->third_rank = rank;
    }
    stats->max_rank = stats->max_rank > rank ? stats->max_rank : rank;

    unsigned long long r = (unsigned long long)(unsigned int)rank;
    unsigned long long v0 = gtsp_mix64(r ^ 0x243f6a8885a308d3ULL);
    unsigned long long v1 = gtsp_mix64(r ^ 0x13198a2e03707344ULL);
    unsigned long long v2 = gtsp_mix64(r ^ 0xa4093822299f31d0ULL);
    unsigned long long v3 = gtsp_mix64(r ^ 0x082efa98ec4e6c89ULL);
    stats->mh0 = stats->mh0 < v0 ? stats->mh0 : v0;
    stats->mh1 = stats->mh1 < v1 ? stats->mh1 : v1;
    stats->mh2 = stats->mh2 < v2 ? stats->mh2 : v2;
    stats->mh3 = stats->mh3 < v3 ? stats->mh3 : v3;
    stats->hash1 ^= gtsp_mix64(r);
    stats->hash1 *= 1099511628211ULL;
    stats->hash2 ^= gtsp_mix64(r + 0x9e3779b97f4a7c15ULL) + (stats->hash2 << 6) + (stats->hash2 >> 2);
}

static inline void tasa_stats_combine(TasaMappedStats *dst, const TasaMappedStats *src)
{
    if (src->count <= 0)
        return;
    dst->count += src->count;
    tasa_stats_add_rank(dst, src->min_rank);
    if (src->second_rank != INT_MAX)
        tasa_stats_add_rank(dst, src->second_rank);
    if (src->third_rank != INT_MAX)
        tasa_stats_add_rank(dst, src->third_rank);
    if (src->max_rank >= 0)
        tasa_stats_add_rank(dst, src->max_rank);
    dst->mh0 = dst->mh0 < src->mh0 ? dst->mh0 : src->mh0;
    dst->mh1 = dst->mh1 < src->mh1 ? dst->mh1 : src->mh1;
    dst->mh2 = dst->mh2 < src->mh2 ? dst->mh2 : src->mh2;
    dst->mh3 = dst->mh3 < src->mh3 ? dst->mh3 : src->mh3;
    dst->hash1 ^= gtsp_mix64(src->hash1);
    dst->hash2 ^= gtsp_mix64(src->hash2);
}

static inline int tasa_span_bucket(const TasaMappedStats *stats)
{
    if (stats->count <= 0 || stats->max_rank < stats->min_rank)
        return 0;
    return gtsp_bucket_id(stats->max_rank - stats->min_rank + 1);
}

static inline int tasa_band_from_stats(const TasaMappedStats *stats)
{
    if (stats->count <= 0 || stats->min_rank == INT_MAX)
        return INT_MAX;
    int span = stats->max_rank >= stats->min_rank ? stats->max_rank - stats->min_rank + 1 : 1;
    int band_width = span <= 8 ? 4 : (span <= 32 ? 8 : 16);
    return stats->min_rank / band_width;
}

static void tasa_finalize_axis_sets(std::vector<std::vector<int> > &lists,
                                    const std::vector<int> &degrees,
                                    TasaAxisSets *sets)
{
    sets->n = (int)lists.size();
    sets->ptr.assign(sets->n + 1, 0);
    sets->counts.assign(sets->n, 0);
    sets->min_tiles.assign(sets->n, INT_MAX);
    sets->max_tiles.assign(sets->n, -1);
    sets->degrees.assign(sets->n, 0);
    sets->mh0.assign(sets->n, ULLONG_MAX);
    sets->mh1.assign(sets->n, ULLONG_MAX);
    sets->mh2.assign(sets->n, ULLONG_MAX);
    sets->mh3.assign(sets->n, ULLONG_MAX);
    sets->hash1.assign(sets->n, 1469598103934665603ULL);
    sets->hash2.assign(sets->n, 1099511628211ULL);

#pragma omp parallel for schedule(dynamic, 512)
    for (int i = 0; i < sets->n; i++)
    {
        std::vector<int> &v = lists[i];
        if (!v.empty())
        {
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
        }
    }

    for (int i = 0; i < sets->n; i++)
        sets->ptr[i + 1] = sets->ptr[i] + (int)lists[i].size();
    sets->idx.resize(sets->ptr[sets->n]);

#pragma omp parallel for schedule(dynamic, 512)
    for (int i = 0; i < sets->n; i++)
    {
        TasaMappedStats stats;
        tasa_stats_init(&stats);
        int out = sets->ptr[i];
        for (size_t j = 0; j < lists[i].size(); j++)
        {
            int tile = lists[i][j];
            sets->idx[out + (int)j] = tile;
            tasa_stats_add_rank(&stats, tile);
        }
        sets->counts[i] = stats.count;
        sets->min_tiles[i] = stats.min_rank;
        sets->max_tiles[i] = stats.max_rank;
        sets->degrees[i] = i < (int)degrees.size() ? degrees[i] : stats.count;
        sets->mh0[i] = stats.mh0;
        sets->mh1[i] = stats.mh1;
        sets->mh2[i] = stats.mh2;
        sets->mh3[i] = stats.mh3;
        sets->hash1[i] = stats.hash1;
        sets->hash2[i] = stats.hash2;
    }
}

static void tasa_build_row_axis_sets(const SMatrix *matrix, TasaAxisSets *sets)
{
    std::vector<std::vector<int> > lists(matrix->m);
    std::vector<int> degrees(matrix->m, 0);

#pragma omp parallel for schedule(dynamic, 512)
    for (int row = 0; row < matrix->m; row++)
    {
        std::vector<int> tiles;
        MAT_PTR_TYPE begin = matrix->rowpointer[row];
        MAT_PTR_TYPE end = matrix->rowpointer[row + 1];
        degrees[row] = (int)(end - begin);
        tiles.reserve((size_t)(end - begin));
        for (MAT_PTR_TYPE p = begin; p < end; p++)
            tiles.push_back(matrix->columnindex[p] / BLOCK_SIZE);
        lists[row].swap(tiles);
    }

    tasa_finalize_axis_sets(lists, degrees, sets);
}

static void tasa_build_column_axis_sets(const SMatrix *matrix, TasaAxisSets *sets)
{
    std::vector<std::vector<int> > lists(matrix->n);
    std::vector<int> degrees(matrix->n, 0);

    for (int row = 0; row < matrix->m; row++)
    {
        int row_tile = row / BLOCK_SIZE;
        for (MAT_PTR_TYPE p = matrix->rowpointer[row]; p < matrix->rowpointer[row + 1]; p++)
        {
            int col = matrix->columnindex[p];
            lists[col].push_back(row_tile);
            degrees[col]++;
        }
    }

    tasa_finalize_axis_sets(lists, degrees, sets);
}

static void tasa_build_tile_profile(const TasaAxisSets *row_sets, int n, TasaTileProfile *profile)
{
    int tile_count = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    profile->tile_count = tile_count;
    std::vector<std::vector<int> > row_lists(tile_count);
    std::vector<int> row_degrees(tile_count, 0);

#pragma omp parallel for schedule(dynamic, 64)
    for (int tile_row = 0; tile_row < tile_count; tile_row++)
    {
        std::vector<int> cols;
        int row_begin = tile_row * BLOCK_SIZE;
        int row_end = row_begin + BLOCK_SIZE;
        row_end = row_end > n ? n : row_end;
        for (int row = row_begin; row < row_end; row++)
        {
            row_degrees[tile_row] += row_sets->degrees[row];
            for (int p = row_sets->ptr[row]; p < row_sets->ptr[row + 1]; p++)
                cols.push_back(row_sets->idx[p]);
        }
        row_lists[tile_row].swap(cols);
    }

    tasa_finalize_axis_sets(row_lists, row_degrees, &profile->row_sets);

    std::vector<std::vector<int> > col_lists(tile_count);
    std::vector<int> col_degrees(tile_count, 0);
    for (int tile_row = 0; tile_row < tile_count; tile_row++)
    {
        for (int p = profile->row_sets.ptr[tile_row]; p < profile->row_sets.ptr[tile_row + 1]; p++)
        {
            int tile_col = profile->row_sets.idx[p];
            col_lists[tile_col].push_back(tile_row);
            col_degrees[tile_col]++;
        }
    }

    tasa_finalize_axis_sets(col_lists, col_degrees, &profile->col_sets);
}

static long long tasa_count_input_tiles_from_row_sets(const TasaAxisSets *row_sets, int n)
{
    int tile_count = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    long long total = 0;
#pragma omp parallel reduction(+ : total)
    {
        std::vector<int> cols;
#pragma omp for schedule(dynamic, 64)
        for (int tile_row = 0; tile_row < tile_count; tile_row++)
        {
            cols.clear();
            int row_begin = tile_row * BLOCK_SIZE;
            int row_end = row_begin + BLOCK_SIZE;
            row_end = row_end > n ? n : row_end;
            for (int row = row_begin; row < row_end; row++)
            {
                for (int p = row_sets->ptr[row]; p < row_sets->ptr[row + 1]; p++)
                    cols.push_back(row_sets->idx[p]);
            }
            if (!cols.empty())
            {
                std::sort(cols.begin(), cols.end());
                total += std::unique(cols.begin(), cols.end()) - cols.begin();
            }
        }
    }
    return total;
}

static void tasa_stats_from_axis_set(const TasaAxisSets *sets,
                                     int id,
                                     const std::vector<int> &rank,
                                     TasaMappedStats *stats)
{
    tasa_stats_init(stats);
    if (sets == NULL || id < 0 || id >= sets->n)
        return;
    for (int p = sets->ptr[id]; p < sets->ptr[id + 1]; p++)
    {
        int tile = sets->idx[p];
        int mapped = tile >= 0 && tile < (int)rank.size() ? rank[tile] : tile;
        tasa_stats_add_rank(stats, mapped);
    }
}

static void tasa_precompute_project_stats(const TasaAxisSets *tile_sets,
                                          const std::vector<int> &rank,
                                          std::vector<TasaMappedStats> *project_stats)
{
    project_stats->resize(tile_sets->n);
#pragma omp parallel for schedule(dynamic, 64)
    for (int tile = 0; tile < tile_sets->n; tile++)
        tasa_stats_from_axis_set(tile_sets, tile, rank, &(*project_stats)[tile]);
}

static void tasa_project_from_element_set(const TasaAxisSets *sets,
                                          int id,
                                          const std::vector<TasaMappedStats> &per_tile_project,
                                          TasaMappedStats *stats)
{
    tasa_stats_init(stats);
    if (sets == NULL || id < 0 || id >= sets->n)
        return;
    for (int p = sets->ptr[id]; p < sets->ptr[id + 1]; p++)
    {
        int tile = sets->idx[p];
        if (tile >= 0 && tile < (int)per_tile_project.size())
            tasa_stats_combine(stats, &per_tile_project[tile]);
    }
}

static TasaOrderItem tasa_make_item_from_stats(int id,
                                               const TasaMappedStats *primary,
                                               const TasaMappedStats *project,
                                               int degree,
                                               int role)
{
    TasaOrderItem item;
    item.id = id;
    item.empty = primary->count <= 0 ? 1 : 0;
    item.primary_band = tasa_band_from_stats(primary);
    item.project_band = tasa_band_from_stats(project);
    item.primary_min = primary->min_rank;
    item.primary_second = primary->second_rank;
    item.primary_third = primary->third_rank;
    item.primary_span = tasa_span_bucket(primary);
    item.primary_bucket = gtsp_bucket_id(primary->count);
    item.project_bucket = gtsp_bucket_id(project->count);
    item.degree_bucket = gtsp_bucket_id(degree);
    if (role == TASA_ROLE_ROW)
    {
        item.primary_mh0 = primary->mh0;
        item.primary_mh1 = primary->mh1;
        item.project_mh0 = project->mh0;
        item.project_mh1 = project->mh1;
    }
    else if (role == TASA_ROLE_COL)
    {
        item.primary_mh0 = primary->mh2;
        item.primary_mh1 = primary->mh3;
        item.project_mh0 = project->mh2;
        item.project_mh1 = project->mh3;
    }
    else
    {
        item.primary_mh0 = primary->mh0 ^ gtsp_mix64(primary->mh2);
        item.primary_mh1 = primary->mh1 ^ gtsp_mix64(primary->mh3);
        item.project_mh0 = project->mh0 ^ gtsp_mix64(project->mh2);
        item.project_mh1 = project->mh1 ^ gtsp_mix64(project->mh3);
    }
    item.hash1 = primary->hash1 ^ gtsp_mix64(project->hash1) ^ gtsp_mix64((unsigned long long)(unsigned int)role);
    item.hash2 = primary->hash2 ^ gtsp_mix64(project->hash2 + 0x9e3779b97f4a7c15ULL);
    return item;
}

struct TasaOrderItemLess
{
    bool operator()(const TasaOrderItem &a, const TasaOrderItem &b) const
    {
        if (a.empty != b.empty)
            return a.empty < b.empty;
        if (a.project_band != b.project_band)
            return a.project_band < b.project_band;
        if (a.primary_band != b.primary_band)
            return a.primary_band < b.primary_band;
        if (a.primary_mh0 != b.primary_mh0)
            return a.primary_mh0 < b.primary_mh0;
        if (a.project_mh0 != b.project_mh0)
            return a.project_mh0 < b.project_mh0;
        if (a.primary_bucket != b.primary_bucket)
            return a.primary_bucket > b.primary_bucket;
        if (a.project_bucket != b.project_bucket)
            return a.project_bucket > b.project_bucket;
        if (a.primary_second != b.primary_second)
            return a.primary_second < b.primary_second;
        if (a.primary_third != b.primary_third)
            return a.primary_third < b.primary_third;
        if (a.primary_span != b.primary_span)
            return a.primary_span < b.primary_span;
        if (a.degree_bucket != b.degree_bucket)
            return a.degree_bucket > b.degree_bucket;
        if (a.primary_mh1 != b.primary_mh1)
            return a.primary_mh1 < b.primary_mh1;
        if (a.project_mh1 != b.project_mh1)
            return a.project_mh1 < b.project_mh1;
        if (a.hash1 != b.hash1)
            return a.hash1 < b.hash1;
        if (a.hash2 != b.hash2)
            return a.hash2 < b.hash2;
        return a.id < b.id;
    }
};

static int tasa_intersection_size(const TasaAxisSets *sets, int a, int b)
{
    if (sets == NULL || a < 0 || b < 0 || a >= sets->n || b >= sets->n)
        return 0;
    int pa = sets->ptr[a];
    int ea = sets->ptr[a + 1];
    int pb = sets->ptr[b];
    int eb = sets->ptr[b + 1];
    int inter = 0;
    while (pa < ea && pb < eb)
    {
        int va = sets->idx[pa];
        int vb = sets->idx[pb];
        if (va == vb)
        {
            inter++;
            pa++;
            pb++;
        }
        else if (va < vb)
            pa++;
        else
            pb++;
    }
    return inter;
}

static int tasa_pair_score(const TasaAxisSets *primary, const TasaAxisSets *secondary, int a, int b)
{
    int score = 0;
    if (primary != NULL)
    {
        int inter = tasa_intersection_size(primary, a, b);
        int diff = abs(primary->counts[a] - primary->counts[b]);
        score += inter * 16 - diff;
    }
    if (secondary != NULL)
    {
        int inter = tasa_intersection_size(secondary, a, b);
        int diff = abs(secondary->counts[a] - secondary->counts[b]);
        score += inter * 12 - diff;
    }
    return score;
}

static void tasa_local_refine_order(std::vector<int> *order,
                                    const TasaAxisSets *primary,
                                    const TasaAxisSets *secondary,
                                    int window)
{
    if (window <= 1 || order->empty())
        return;
    int n = (int)order->size();
    for (int start = 0; start < n; start += window)
    {
        int end = start + window;
        end = end > n ? n : end;
        for (int pos = start + 1; pos < end; pos++)
        {
            int prev = (*order)[pos - 1];
            int best = pos;
            int best_score = tasa_pair_score(primary, secondary, prev, (*order)[pos]);
            for (int cand = pos + 1; cand < end; cand++)
            {
                int score = tasa_pair_score(primary, secondary, prev, (*order)[cand]);
                if (score > best_score)
                {
                    best_score = score;
                    best = cand;
                }
            }
            if (best != pos)
                std::swap((*order)[pos], (*order)[best]);
        }
    }
}

static void tasa_order_to_rank(const std::vector<int> &order, std::vector<int> *rank)
{
    rank->assign(order.size(), 0);
    for (int i = 0; i < (int)order.size(); i++)
        (*rank)[order[i]] = i;
}

static int tasa_window_for_size(int n)
{
    if (n <= 50000)
        return 128;
    if (n <= 200000)
        return 64;
    return 32;
}

static int tasa_make_tile_order(const TasaAxisSets *primary,
                                const TasaAxisSets *secondary,
                                int role,
                                std::vector<int> *order,
                                std::vector<int> *rank,
                                double *affinity_ms,
                                double *sort_ms,
                                double *refine_ms)
{
    struct timeval t1, t2;
    int n = primary->n;
    std::vector<TasaOrderItem> items(n);

    gettimeofday(&t1, NULL);
#pragma omp parallel for schedule(dynamic, 256)
    for (int id = 0; id < n; id++)
    {
        TasaMappedStats primary_stats;
        TasaMappedStats secondary_stats;
        TasaMappedStats project_stats;
        tasa_stats_init(&primary_stats);
        tasa_stats_init(&secondary_stats);
        tasa_stats_init(&project_stats);
        for (int p = primary->ptr[id]; p < primary->ptr[id + 1]; p++)
            tasa_stats_add_rank(&primary_stats, primary->idx[p]);
        if (secondary != NULL)
        {
            for (int p = secondary->ptr[id]; p < secondary->ptr[id + 1]; p++)
                tasa_stats_add_rank(&secondary_stats, secondary->idx[p]);
            tasa_stats_combine(&primary_stats, &secondary_stats);
        }
        project_stats = primary_stats;
        int degree = primary->degrees[id] + (secondary ? secondary->degrees[id] : 0);
        items[id] = tasa_make_item_from_stats(id, &primary_stats, &project_stats, degree, role);
    }
    gettimeofday(&t2, NULL);
    *affinity_ms += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    gettimeofday(&t1, NULL);
    std::sort(items.begin(), items.end(), TasaOrderItemLess());
    order->resize(n);
    for (int i = 0; i < n; i++)
        (*order)[i] = items[i].id;
    gettimeofday(&t2, NULL);
    *sort_ms += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    gettimeofday(&t1, NULL);
    tasa_local_refine_order(order, primary, secondary, tasa_window_for_size(n));
    gettimeofday(&t2, NULL);
    *refine_ms += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    tasa_order_to_rank(*order, rank);
    return 0;
}

static int tasa_make_axis_permutation(const TasaAxisSets *row_sets,
                                      const TasaAxisSets *col_sets,
                                      const TasaTileProfile *tile_profile,
                                      const std::vector<int> &row_tile_rank,
                                      const std::vector<int> &inner_tile_rank,
                                      const std::vector<int> &col_tile_rank,
                                      const std::vector<TasaMappedStats> &inner_to_col_project,
                                      const std::vector<TasaMappedStats> &inner_to_row_project,
                                      int role,
                                      int window,
                                      AxisPermutation *perm,
                                      double *affinity_ms,
                                      double *sort_ms,
                                      double *refine_ms)
{
    struct timeval t1, t2;
    int n = row_sets->n;
    std::vector<TasaOrderItem> items(n);

    gettimeofday(&t1, NULL);
#pragma omp parallel for schedule(dynamic, 512)
    for (int id = 0; id < n; id++)
    {
        TasaMappedStats primary;
        TasaMappedStats secondary;
        TasaMappedStats project;
        TasaMappedStats tmp;
        tasa_stats_init(&primary);
        tasa_stats_init(&secondary);
        tasa_stats_init(&project);

        if (role == TASA_ROLE_ROW)
        {
            tasa_stats_from_axis_set(row_sets, id, inner_tile_rank, &primary);
            tasa_project_from_element_set(row_sets, id, inner_to_col_project, &project);
        }
        else if (role == TASA_ROLE_COL)
        {
            tasa_stats_from_axis_set(col_sets, id, inner_tile_rank, &primary);
            tasa_project_from_element_set(col_sets, id, inner_to_row_project, &project);
        }
        else
        {
            tasa_stats_from_axis_set(row_sets, id, col_tile_rank, &primary);
            tasa_stats_from_axis_set(col_sets, id, row_tile_rank, &secondary);
            tasa_stats_combine(&primary, &secondary);
            tasa_project_from_element_set(row_sets, id, inner_to_col_project, &project);
            tasa_project_from_element_set(col_sets, id, inner_to_row_project, &tmp);
            tasa_stats_combine(&project, &tmp);
        }

        int degree = row_sets->degrees[id] + col_sets->degrees[id];
        if (role == TASA_ROLE_ROW)
            degree = row_sets->degrees[id];
        else if (role == TASA_ROLE_COL)
            degree = col_sets->degrees[id];
        items[id] = tasa_make_item_from_stats(id, &primary, &project, degree, role);
    }
    gettimeofday(&t2, NULL);
    *affinity_ms += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    gettimeofday(&t1, NULL);
    std::sort(items.begin(), items.end(), TasaOrderItemLess());
    std::vector<int> order(n);
    for (int i = 0; i < n; i++)
        order[i] = items[i].id;
    gettimeofday(&t2, NULL);
    *sort_ms += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    const TasaAxisSets *primary_sets = role == TASA_ROLE_COL ? col_sets : row_sets;
    const TasaAxisSets *secondary_sets = role == TASA_ROLE_INNER ? col_sets : NULL;
    gettimeofday(&t1, NULL);
    tasa_local_refine_order(&order, primary_sets, secondary_sets, window);
    gettimeofday(&t2, NULL);
    *refine_ms += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    int status = axis_permutation_alloc(perm, n);
    if (status != 0)
        return status;
    for (int i = 0; i < n; i++)
    {
        perm->old_id[i] = order[i];
        perm->new_id[order[i]] = i;
    }
    return 0;
}

static int tasa_make_symmetric_rcm_permutation(const SMatrix *matrix,
                                               const TasaAxisSets *row_sets,
                                               AxisPermutation *perm,
                                               int window,
                                               double *affinity_ms,
                                               double *sort_ms,
                                               double *refine_ms)
{
    struct timeval t1, t2;
    int n = matrix->m;
    std::vector<int> seeds(n);
    std::iota(seeds.begin(), seeds.end(), 0);

    gettimeofday(&t1, NULL);
    std::sort(seeds.begin(), seeds.end(),
              [row_sets](int a, int b) {
                  if (row_sets->counts[a] != row_sets->counts[b])
                      return row_sets->counts[a] < row_sets->counts[b];
                  if (row_sets->degrees[a] != row_sets->degrees[b])
                      return row_sets->degrees[a] < row_sets->degrees[b];
                  return a < b;
              });
    gettimeofday(&t2, NULL);
    *sort_ms += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    gettimeofday(&t1, NULL);
    std::vector<unsigned char> visited(n, 0);
    std::vector<int> order;
    std::vector<int> queue;
    std::vector<int> component;
    std::vector<int> neighbors;
    order.reserve(n);
    queue.reserve(4096);
    component.reserve(4096);
    neighbors.reserve(256);

    for (int si = 0; si < n; si++)
    {
        int seed = seeds[si];
        if (visited[seed])
            continue;

        queue.clear();
        component.clear();
        visited[seed] = 1;
        queue.push_back(seed);

        for (size_t head = 0; head < queue.size(); head++)
        {
            int v = queue[head];
            component.push_back(v);
            neighbors.clear();
            for (MAT_PTR_TYPE p = matrix->rowpointer[v]; p < matrix->rowpointer[v + 1]; p++)
            {
                int u = matrix->columnindex[p];
                if (u < 0 || u >= n || visited[u])
                    continue;
                visited[u] = 1;
                neighbors.push_back(u);
            }
            if (neighbors.size() > 1)
            {
                std::sort(neighbors.begin(), neighbors.end(),
                          [row_sets](int a, int b) {
                              if (row_sets->counts[a] != row_sets->counts[b])
                                  return row_sets->counts[a] < row_sets->counts[b];
                              if (row_sets->degrees[a] != row_sets->degrees[b])
                                  return row_sets->degrees[a] < row_sets->degrees[b];
                              return a < b;
                          });
            }
            queue.insert(queue.end(), neighbors.begin(), neighbors.end());
        }

        for (int i = (int)component.size() - 1; i >= 0; i--)
            order.push_back(component[i]);
    }
    gettimeofday(&t2, NULL);
    *affinity_ms += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    gettimeofday(&t1, NULL);
    tasa_local_refine_order(&order, row_sets, NULL, window);
    gettimeofday(&t2, NULL);
    *refine_ms += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    int status = axis_permutation_alloc(perm, n);
    if (status != 0)
        return status;
    for (int i = 0; i < n; i++)
    {
        perm->old_id[i] = order[i];
        perm->new_id[order[i]] = i;
    }
    return 0;
}

static int tasa_symmetric_pack_window_for_size(int n)
{
    if (n <= 50000)
        return 128;
    if (n <= 200000)
        return 128;
    return 64;
}

static int tasa_added_tile_count(const TasaAxisSets *sets,
                                 int id,
                                 std::vector<int> &mark,
                                 int tag,
                                 int *overlap)
{
    int added = 0;
    int ov = 0;
    for (int p = sets->ptr[id]; p < sets->ptr[id + 1]; p++)
    {
        int tile = sets->idx[p];
        if (tile >= 0 && tile < (int)mark.size() && mark[tile] == tag)
            ov++;
        else
            added++;
    }
    *overlap = ov;
    return added;
}

static void tasa_mark_tiles(const TasaAxisSets *sets, int id, std::vector<int> &mark, int tag)
{
    for (int p = sets->ptr[id]; p < sets->ptr[id + 1]; p++)
    {
        int tile = sets->idx[p];
        if (tile >= 0 && tile < (int)mark.size())
            mark[tile] = tag;
    }
}

static void tasa_local_pack_order(std::vector<int> *order,
                                  const TasaAxisSets *sets,
                                  int window,
                                  int tile_dim)
{
    if (window <= BLOCK_SIZE || order->empty())
        return;

    int n = (int)order->size();
    std::vector<int> packed;
    packed.reserve(n);
    std::vector<int> mark(tile_dim, 0);
    int tag = 1;

    for (int start = 0; start < n; start += window)
    {
        int end = start + window;
        end = end > n ? n : end;
        int len = end - start;
        std::vector<unsigned char> used(len, 0);
        int remaining = len;

        while (remaining > 0)
        {
            tag++;
            if (tag == INT_MAX)
            {
                std::fill(mark.begin(), mark.end(), 0);
                tag = 1;
            }

            int seed_pos = -1;
            for (int i = 0; i < len; i++)
            {
                if (!used[i])
                {
                    seed_pos = i;
                    break;
                }
            }
            if (seed_pos < 0)
                break;

            used[seed_pos] = 1;
            remaining--;
            int seed_id = (*order)[start + seed_pos];
            packed.push_back(seed_id);
            tasa_mark_tiles(sets, seed_id, mark, tag);

            int group_size = 1;
            while (group_size < BLOCK_SIZE && remaining > 0)
            {
                int best_pos = -1;
                int best_score = INT_MIN;
                for (int i = 0; i < len; i++)
                {
                    if (used[i])
                        continue;
                    int candidate = (*order)[start + i];
                    int overlap = 0;
                    int added = tasa_added_tile_count(sets, candidate, mark, tag, &overlap);
                    int distance_penalty = abs(candidate - seed_id) / BLOCK_SIZE;
                    int score = overlap * 96 - added * 64 - distance_penalty * 48;
                    if (score > best_score)
                    {
                        best_score = score;
                        best_pos = i;
                    }
                }
                if (best_pos < 0)
                    break;
                used[best_pos] = 1;
                remaining--;
                int chosen = (*order)[start + best_pos];
                packed.push_back(chosen);
                tasa_mark_tiles(sets, chosen, mark, tag);
                group_size++;
            }
        }
    }

    order->swap(packed);
}

static int tasa_make_symmetric_banded_permutation(int n,
                                                  const TasaAxisSets *row_sets,
                                                  AxisPermutation *perm,
                                                  int window,
                                                  double *refine_ms)
{
    struct timeval t1, t2;
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);

    gettimeofday(&t1, NULL);
    int tile_dim = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    tasa_local_pack_order(&order, row_sets, window, tile_dim);
    gettimeofday(&t2, NULL);
    *refine_ms += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    int status = axis_permutation_alloc(perm, n);
    if (status != 0)
        return status;
    for (int i = 0; i < n; i++)
    {
        perm->old_id[i] = order[i];
        perm->new_id[order[i]] = i;
    }
    return 0;
}

static void gtsp_set_identity_permutations(ReorderInfo *info, int n)
{
    axis_permutation_identity(&info->row_perm, n);
    axis_permutation_identity(&info->inner_perm, n);
    axis_permutation_identity(&info->col_perm, n);
}

int build_gtsp_permutation(const SMatrix *matrix, ReorderInfo *info)
{
    if (matrix->m != matrix->n)
        return -1;

    struct timeval t1, t2;
    int n = matrix->m;

    reorder_info_destroy(info);
    info->enabled = 1;
    info->n = n;
    info->guard_degraded_to_identity = 0;

    if (!matrix->isSymmetric)
    {
        int row_tile_dim = (matrix->m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int col_tile_dim = (matrix->n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int combined_dim = row_tile_dim > col_tile_dim ? row_tile_dim : col_tile_dim;

        gettimeofday(&t1, NULL);
        GtspSignatureArrays row_sig = gtsp_compute_row_signatures_gpu(matrix);
        GtspSignatureArrays col_sig = gtsp_compute_column_signatures_gpu(matrix);
        gettimeofday(&t2, NULL);
        info->signature_time_ms = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        gettimeofday(&t1, NULL);
        int status = gtsp_make_permutation_from_signatures(row_sig, col_tile_dim, &info->row_perm);
        if (status != 0)
            return -2;
        status = gtsp_make_permutation_from_signatures(col_sig, row_tile_dim, &info->col_perm);
        if (status != 0)
            return -3;
        status = gtsp_make_permutation_from_combined_signatures(col_sig, row_sig, combined_dim, &info->inner_perm);
        if (status != 0)
            return -4;
        gettimeofday(&t2, NULL);
        info->sort_time_ms = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        gettimeofday(&t1, NULL);
        info->input_tiles_A_before = estimate_input_tile_count_biperm(matrix, NULL, NULL);
        info->input_tiles_B_before = info->input_tiles_A_before;
        info->input_tiles_A_after = estimate_input_tile_count_biperm(matrix, &info->row_perm, &info->inner_perm);
        info->input_tiles_B_after = estimate_input_tile_count_biperm(matrix, &info->inner_perm, &info->col_perm);
        info->estimated_c_tiles_before = estimate_symbolic_c_tile_count(matrix, NULL, NULL, NULL);
        info->estimated_c_tiles_after = estimate_symbolic_c_tile_count(matrix, &info->row_perm, &info->inner_perm, &info->col_perm);
        gettimeofday(&t2, NULL);
        info->proxy_time_ms = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        return 0;
    }

    gettimeofday(&t1, NULL);
    TasaAxisSets row_sets;
    TasaAxisSets col_sets;
    tasa_build_row_axis_sets(matrix, &row_sets);
    if (!matrix->isSymmetric)
        tasa_build_column_axis_sets(matrix, &col_sets);
    gettimeofday(&t2, NULL);
    info->signature_time_ms = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    TasaTileProfile tile_profile;
    if (!matrix->isSymmetric)
    {
        gettimeofday(&t1, NULL);
        tasa_build_tile_profile(&row_sets, n, &tile_profile);
        gettimeofday(&t2, NULL);
        info->tile_profile_time_ms = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    }

    int status = 0;
    std::vector<int> row_tile_order;
    std::vector<int> inner_tile_order;
    std::vector<int> col_tile_order;
    std::vector<int> row_tile_rank;
    std::vector<int> inner_tile_rank;
    std::vector<int> col_tile_rank;

    if (matrix->isSymmetric)
    {
        int tile_dim = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        long long natural_tiles = tasa_count_input_tiles_from_row_sets(&row_sets, n);
        double avg_tiles_per_tile_row = tile_dim > 0 ? (double)natural_tiles / (double)tile_dim : 0.0;
        if (n <= 50000 && avg_tiles_per_tile_row < 32.0)
        {
            info->local_refine_window = BLOCK_SIZE;
            status = axis_permutation_identity(&info->inner_perm, n);
            if (status != 0)
                return -2;
        }
        else if (n <= 50000)
        {
            info->local_refine_window = 128;
            status = tasa_make_symmetric_rcm_permutation(matrix,
                                                         &row_sets,
                                                         &info->inner_perm,
                                                         info->local_refine_window,
                                                         &info->affinity_time_ms,
                                                         &info->sort_time_ms,
                                                         &info->local_refine_time_ms);
            if (status != 0)
                return -2;
        }
        else
        {
            info->local_refine_window = tasa_symmetric_pack_window_for_size(n);
            status = tasa_make_symmetric_banded_permutation(n,
                                                            &row_sets,
                                                            &info->inner_perm,
                                                            info->local_refine_window,
                                                            &info->local_refine_time_ms);
            if (status != 0)
                return -2;
        }
        status = axis_permutation_copy(&info->row_perm, &info->inner_perm);
        if (status != 0)
            return -3;
        status = axis_permutation_copy(&info->col_perm, &info->inner_perm);
        if (status != 0)
            return -4;
    }
    else
    {
        info->local_refine_window = tasa_window_for_size(n);
        status = tasa_make_tile_order(&tile_profile.row_sets,
                                      NULL,
                                      TASA_ROLE_ROW,
                                      &row_tile_order,
                                      &row_tile_rank,
                                      &info->affinity_time_ms,
                                      &info->sort_time_ms,
                                      &info->local_refine_time_ms);
        if (status != 0)
            return -2;

        status = tasa_make_tile_order(&tile_profile.row_sets,
                                      &tile_profile.col_sets,
                                      TASA_ROLE_INNER,
                                      &inner_tile_order,
                                      &inner_tile_rank,
                                      &info->affinity_time_ms,
                                      &info->sort_time_ms,
                                      &info->local_refine_time_ms);
        if (status != 0)
            return -3;

        status = tasa_make_tile_order(&tile_profile.col_sets,
                                      NULL,
                                      TASA_ROLE_COL,
                                      &col_tile_order,
                                      &col_tile_rank,
                                      &info->affinity_time_ms,
                                      &info->sort_time_ms,
                                      &info->local_refine_time_ms);
        if (status != 0)
            return -4;

        std::vector<TasaMappedStats> inner_to_col_project;
        std::vector<TasaMappedStats> inner_to_row_project;
        gettimeofday(&t1, NULL);
        tasa_precompute_project_stats(&tile_profile.row_sets, col_tile_rank, &inner_to_col_project);
        tasa_precompute_project_stats(&tile_profile.col_sets, row_tile_rank, &inner_to_row_project);
        gettimeofday(&t2, NULL);
        info->affinity_time_ms += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        status = tasa_make_axis_permutation(&row_sets,
                                            &col_sets,
                                            &tile_profile,
                                            row_tile_rank,
                                            inner_tile_rank,
                                            col_tile_rank,
                                            inner_to_col_project,
                                            inner_to_row_project,
                                            TASA_ROLE_ROW,
                                            info->local_refine_window,
                                            &info->row_perm,
                                            &info->affinity_time_ms,
                                            &info->sort_time_ms,
                                            &info->local_refine_time_ms);
        if (status != 0)
            return -5;

        status = tasa_make_axis_permutation(&row_sets,
                                            &col_sets,
                                            &tile_profile,
                                            row_tile_rank,
                                            inner_tile_rank,
                                            col_tile_rank,
                                            inner_to_col_project,
                                            inner_to_row_project,
                                            TASA_ROLE_INNER,
                                            info->local_refine_window,
                                            &info->inner_perm,
                                            &info->affinity_time_ms,
                                            &info->sort_time_ms,
                                            &info->local_refine_time_ms);
        if (status != 0)
            return -6;

        status = tasa_make_axis_permutation(&row_sets,
                                            &col_sets,
                                            &tile_profile,
                                            row_tile_rank,
                                            inner_tile_rank,
                                            col_tile_rank,
                                            inner_to_col_project,
                                            inner_to_row_project,
                                            TASA_ROLE_COL,
                                            info->local_refine_window,
                                            &info->col_perm,
                                            &info->affinity_time_ms,
                                            &info->sort_time_ms,
                                            &info->local_refine_time_ms);
        if (status != 0)
            return -7;
    }

    gettimeofday(&t1, NULL);
    info->input_tiles_A_before = estimate_input_tile_count_biperm(matrix, NULL, NULL);
    info->input_tiles_B_before = info->input_tiles_A_before;
    info->input_tiles_A_after = estimate_input_tile_count_biperm(matrix, &info->row_perm, &info->inner_perm);
    info->input_tiles_B_after = estimate_input_tile_count_biperm(matrix, &info->inner_perm, &info->col_perm);
    info->estimated_c_tiles_before = estimate_symbolic_c_tile_count(matrix, NULL, NULL, NULL);
    info->estimated_c_tiles_after = estimate_symbolic_c_tile_count(matrix, &info->row_perm, &info->inner_perm, &info->col_perm);
    gettimeofday(&t2, NULL);
    info->proxy_time_ms = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    return 0;
}

int apply_bipermutation_csr(const SMatrix *input,
                            const AxisPermutation *row_perm,
                            const AxisPermutation *col_perm,
                            SMatrix *output)
{
    int m = input->m;
    if (!row_perm || !col_perm || row_perm->n != input->m || col_perm->n != input->n)
        return -1;

    output->m = input->m;
    output->n = input->n;
    output->nnz = input->nnz;
    output->isSymmetric = 0;
    output->rowpointer = (MAT_PTR_TYPE *)malloc((m + 1) * sizeof(MAT_PTR_TYPE));
    output->columnindex = (int *)malloc(input->nnz * sizeof(int));
    output->value = (MAT_VAL_TYPE *)malloc(input->nnz * sizeof(MAT_VAL_TYPE));
    if (output->rowpointer == NULL || output->columnindex == NULL || output->value == NULL)
        return -2;

    thrust::device_vector<MAT_PTR_TYPE> d_in_rowptr(input->rowpointer, input->rowpointer + input->m + 1);
    thrust::device_vector<int> d_in_colidx(input->columnindex, input->columnindex + input->nnz);
    thrust::device_vector<MAT_VAL_TYPE> d_in_values(input->value, input->value + input->nnz);
    thrust::device_vector<int> d_row_new_to_old(row_perm->old_id, row_perm->old_id + row_perm->n);
    thrust::device_vector<int> d_col_old_to_new(col_perm->new_id, col_perm->new_id + col_perm->n);
    thrust::device_vector<MAT_PTR_TYPE> d_counts(input->m, 0);
    thrust::device_vector<MAT_PTR_TYPE> d_out_rowptr(input->m + 1, 0);

    int threads = 256;
    int blocks = (input->m + threads - 1) / threads;
    if (input->m > 0)
    {
        gtsp_permuted_row_counts_kernel<<<blocks, threads>>>(
            input->m,
            thrust::raw_pointer_cast(d_in_rowptr.data()),
            thrust::raw_pointer_cast(d_row_new_to_old.data()),
            thrust::raw_pointer_cast(d_counts.data()));
        gtsp_cuda_sync("GTSP permuted row count kernel");
        thrust::exclusive_scan(d_counts.begin(), d_counts.end(), d_out_rowptr.begin());
    }
    MAT_PTR_TYPE total_nnz = input->nnz;
    cudaMemcpy(thrust::raw_pointer_cast(d_out_rowptr.data()) + input->m,
               &total_nnz,
               sizeof(MAT_PTR_TYPE),
               cudaMemcpyHostToDevice);

    thrust::device_vector<unsigned long long> d_keys(input->nnz);
    thrust::device_vector<int> d_out_colidx(input->nnz);
    thrust::device_vector<MAT_VAL_TYPE> d_out_values(input->nnz);
    if (input->m > 0)
    {
        gtsp_emit_permuted_entries_kernel<<<blocks, threads>>>(
            input->m,
            thrust::raw_pointer_cast(d_in_rowptr.data()),
            thrust::raw_pointer_cast(d_in_colidx.data()),
            thrust::raw_pointer_cast(d_in_values.data()),
            thrust::raw_pointer_cast(d_row_new_to_old.data()),
            thrust::raw_pointer_cast(d_col_old_to_new.data()),
            thrust::raw_pointer_cast(d_out_rowptr.data()),
            thrust::raw_pointer_cast(d_keys.data()),
            thrust::raw_pointer_cast(d_out_colidx.data()),
            thrust::raw_pointer_cast(d_out_values.data()));
        gtsp_cuda_sync("GTSP emit permuted CSR kernel");
    }

    if (input->nnz > 0)
    {
        thrust::sort_by_key(d_keys.begin(),
                            d_keys.end(),
                            thrust::make_zip_iterator(thrust::make_tuple(d_out_colidx.begin(), d_out_values.begin())));
    }

    thrust::copy(d_out_rowptr.begin(), d_out_rowptr.end(), output->rowpointer);
    thrust::copy(d_out_colidx.begin(), d_out_colidx.end(), output->columnindex);
    thrust::copy(d_out_values.begin(), d_out_values.end(), output->value);
    return 0;
}

int restore_bipermutation_csr(SMatrix *matrix,
                              const AxisPermutation *row_perm,
                              const AxisPermutation *col_perm)
{
    if (!row_perm || !col_perm)
        return 0;

    int m = matrix->m;
    MAT_PTR_TYPE nnz = matrix->nnz;
    MAT_PTR_TYPE *rowptr = (MAT_PTR_TYPE *)malloc((m + 1) * sizeof(MAT_PTR_TYPE));
    int *colidx = (int *)malloc(nnz * sizeof(int));
    MAT_VAL_TYPE *values = (MAT_VAL_TYPE *)malloc(nnz * sizeof(MAT_VAL_TYPE));
    if (rowptr == NULL || colidx == NULL || values == NULL)
        return -1;

    memset(rowptr, 0, (m + 1) * sizeof(MAT_PTR_TYPE));
    for (int new_row = 0; new_row < m; new_row++)
    {
        int old_row = row_perm->old_id[new_row];
        rowptr[old_row] = matrix->rowpointer[new_row + 1] - matrix->rowpointer[new_row];
    }
    exclusive_scan(rowptr, m + 1);

#pragma omp parallel for schedule(dynamic, 1024)
    for (int new_row = 0; new_row < m; new_row++)
    {
        int old_row = row_perm->old_id[new_row];
        MAT_PTR_TYPE out = rowptr[old_row];
        for (MAT_PTR_TYPE j = matrix->rowpointer[new_row]; j < matrix->rowpointer[new_row + 1]; j++, out++)
        {
            colidx[out] = col_perm->old_id[matrix->columnindex[j]];
            values[out] = matrix->value[j];
        }
    }

    sort_csr_rows_by_column(m, rowptr, colidx, values);

    free(matrix->rowpointer);
    free(matrix->columnindex);
    free(matrix->value);
    matrix->rowpointer = rowptr;
    matrix->columnindex = colidx;
    matrix->value = values;

    return 0;
}

#endif
