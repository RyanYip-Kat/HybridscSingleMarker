/* pysingle/_fastseurat.c
 *
 * Seurat FindWeightsC 的 C 实现（对应 seurat/src/integration.cpp）。
 *
 * 输入：
 *   dist_norm    (n_query, k)  float64  归一化距离（1=最近）
 *   neighbor_idx (n_query, k)  int32    每个查询细胞的 k 个最近 anchor 细胞
 *                                       （索引指向 anchor_query_cells）
 *   anchor_offsets (n_anchor_cells+1) intp   anchor 行的前缀和
 *   anchor_rows   (flat)       int32    展平的 anchor 行索引（按 anchor 细胞分组）
 *   anchor_scores (n_anchors,) float64  anchor 得分
 *   sd                       float64   高斯核带宽（sd.weight）
 *
 * 输出：COO 稀疏权重 (rows, cols, vals)，形状 (n_anchors, n_query)。
 *   weight = 1 - exp(-dist_norm * anchor_score / (2/sd)^2)
 *   每查询细胞最多取 k 个 anchor 贡献（与 C++ FindWeightsC 一致）。
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>

static PyObject *find_weights_c(PyObject *self, PyObject *args) {
  PyArrayObject *dist_norm, *neighbor_idx, *anchor_offsets, *anchor_rows,
      *anchor_scores;
  double sd;
  if (!PyArg_ParseTuple(args, "OOOOOd", &dist_norm, &neighbor_idx,
                        &anchor_offsets, &anchor_rows, &anchor_scores, &sd))
    return NULL;
  if (PyArray_TYPE(dist_norm) != NPY_FLOAT64 ||
      PyArray_TYPE(anchor_scores) != NPY_FLOAT64) {
    PyErr_SetString(PyExc_TypeError, "dist_norm/anchor_scores 须为 float64");
    return NULL;
  }
  npy_intp n_query = PyArray_DIM(dist_norm, 0);
  npy_intp k = PyArray_DIM(dist_norm, 1);
  npy_intp n_anchors = PyArray_DIM(anchor_scores, 0);

  double *dn = (double *)PyArray_DATA(dist_norm);
  int *ni = (int *)PyArray_DATA(neighbor_idx);
  npy_intp *off = (npy_intp *)PyArray_DATA(anchor_offsets);
  int *arow = (int *)PyArray_DATA(anchor_rows);
  double *asc = (double *)PyArray_DATA(anchor_scores);

  /* 预分配（上限 = n_query * k） */
  npy_intp cap = n_query * k;
  double *vals = (double *)malloc(cap * sizeof(double));
  int *rows = (int *)malloc(cap * sizeof(int));
  int *cols = (int *)malloc(cap * sizeof(int));
  if (!vals || !rows || !cols) {
    free(vals); free(rows); free(cols);
    PyErr_NoMemory();
    return NULL;
  }

  npy_intp nnz = 0;
  double denom = pow(2.0 / sd, 2.0);
  for (npy_intp q = 0; q < n_query; q++) {
    npy_intp added = 0;
    for (npy_intp j = 0; j < k && added < k; j++) {
      int acell = ni[q * k + j];
      npy_intp start = off[acell], end = off[acell + 1];
      for (npy_intp r = start; r < end && added < k; r++) {
        int a = arow[r];
        double w = 1.0 - exp(-dn[q * k + j] * asc[a] / denom);
        rows[nnz] = a;
        cols[nnz] = (int)q;
        vals[nnz] = w;
        nnz++;
        added++;
      }
    }
  }

  /* 组装返回数组 */
  npy_intp dims[1] = {nnz};
  PyObject *rows_arr = PyArray_SimpleNew(1, dims, NPY_INT32);
  PyObject *cols_arr = PyArray_SimpleNew(1, dims, NPY_INT32);
  PyObject *vals_arr = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
  if (!rows_arr || !cols_arr || !vals_arr) {
    free(vals); free(rows); free(cols);
    Py_XDECREF(rows_arr); Py_XDECREF(cols_arr); Py_XDECREF(vals_arr);
    return NULL;
  }
  memcpy(PyArray_DATA((PyArrayObject *)rows_arr), rows, nnz * sizeof(int));
  memcpy(PyArray_DATA((PyArrayObject *)cols_arr), cols, nnz * sizeof(int));
  memcpy(PyArray_DATA((PyArrayObject *)vals_arr), vals, nnz * sizeof(double));
  free(vals); free(rows); free(cols);

  return Py_BuildValue("NNN", rows_arr, cols_arr, vals_arr);
}

static PyMethodDef _methods[] = {
    {"find_weights_c", (PyCFunction)find_weights_c, METH_VARARGS,
     "Seurat FindWeightsC：计算 anchor 权重 COO (rows, cols, vals)。"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef _module = {
    PyModuleDef_HEAD_INIT, "_fastseurat",
    "pysingle 的 Seurat 标签转移 C 加速扩展。", -1, _methods};

PyMODINIT_FUNC PyInit__fastseurat(void) {
  import_array();
  return PyModule_Create(&_module);
}
