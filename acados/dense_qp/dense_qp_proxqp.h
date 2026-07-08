/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

#ifndef ACADOS_DENSE_QP_DENSE_QP_PROXQP_H_
#define ACADOS_DENSE_QP_DENSE_QP_PROXQP_H_

#ifdef __cplusplus
extern "C" {
#endif

// proxsuite
#include "proxsuite_c/include/proxsuite_c.h"

// acados
#include "acados/dense_qp/dense_qp_common.h"
#include "acados/utils/types.h"

typedef struct dense_qp_proxqp_opts_
{
    proxqp_c_settings settings;
    int warm_start;  // 1: warm start with the result of the previous solve
    int print_level;
} dense_qp_proxqp_opts;


typedef struct dense_qp_proxqp_memory_
{
    // qp with the soft constraints of the original qp expressed
    // as hard constraints on slack variables (only used if ns > 0)
    dense_qp_in *qp_stacked;

    // extracted qp data (column-major matrices)
    double *H;
    double *g;
    double *A;
    double *b;
    double *C;
    double *d_lg;
    double *d_ug;
    double *d_lb0;
    double *d_ub0;
    double *l_box;
    double *u_box;
    int *idxb;
    int *idxb_stacked;
    int *idxs;
    int *idxs_rev;
    double *Zl;
    double *Zu;
    double *zl;
    double *zu;
    double *d_ls;
    double *d_us;

    // solution buffers
    double *prim_sol;
    double *y_sol;
    double *z_sol;

    // the ProxQP object is opaque and cannot be placed in acados-managed memory;
    // it is created at the first call and freed in dense_qp_proxqp_terminate
    void *proxqp;
    int first_run;

    double time_qp_solver_call;
    int iter;
    int status;

} dense_qp_proxqp_memory;

acados_size_t dense_qp_proxqp_opts_calculate_size(void *config, dense_qp_dims *dims);
//
void *dense_qp_proxqp_opts_assign(void *config, dense_qp_dims *dims, void *raw_memory);
//
void dense_qp_proxqp_opts_initialize_default(void *config, dense_qp_dims *dims, void *opts_);
//
void dense_qp_proxqp_opts_update(void *config, dense_qp_dims *dims, void *opts_);
//
acados_size_t dense_qp_proxqp_memory_calculate_size(void *config, dense_qp_dims *dims, void *opts_);
//
void *dense_qp_proxqp_memory_assign(void *config, dense_qp_dims *dims, void *opts_,
                                    void *raw_memory);
//
acados_size_t dense_qp_proxqp_workspace_calculate_size(void *config, dense_qp_dims *dims,
                                                       void *opts_);
//
int dense_qp_proxqp(void *config, void *qp_in_, void *qp_out_, void *opts_, void *mem_,
                    void *work_);
//
void dense_qp_proxqp_config_initialize_default(void *config);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_DENSE_QP_DENSE_QP_PROXQP_H_
