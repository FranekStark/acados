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

#ifndef ACADOS_OCP_QP_OCP_QP_PROXQP_H_
#define ACADOS_OCP_QP_OCP_QP_PROXQP_H_

#ifdef __cplusplus
extern "C" {
#endif

// proxsuite
#include "proxsuite_c/include/proxsuite_c.h"

// acados
#include "acados/ocp_qp/ocp_qp_common.h"
#include "acados/utils/types.h"

typedef struct ocp_qp_proxqp_opts_
{
    proxqp_c_settings settings;
    int warm_start;  // 1: warm start with the result of the previous solve
    int print_level;
} ocp_qp_proxqp_opts;


// The OCP QP is written as one sparse QP over the stage-wise stacked variables
// [u_0, x_0, sl_0, su_0, ..., u_N, x_N, sl_N, su_N]:
// the dynamics form the equality constraints A z = b, all bounds, general and
// soft constraints form the two-sided inequality constraints l <= C z <= u.
typedef struct ocp_qp_proxqp_memory_
{
    int first_run;

    double *g;
    double *b;
    double *l;
    double *u;

    // Hessian, upper triangular part in CSC format
    int P_nnzmax;
    int *P_i;
    int *P_p;
    double *P_x;

    // equality constraint (dynamics) matrix in CSC format
    int A_nnzmax;
    int *A_i;
    int *A_p;
    double *A_x;

    // inequality constraint matrix in CSC format
    int C_nnzmax;
    int *C_i;
    int *C_p;
    double *C_x;

    // solution buffers
    double *prim_sol;
    double *y_sol;
    double *z_sol;

    // the ProxQP object is opaque and cannot be placed in acados-managed memory;
    // it is created at the first call and freed in ocp_qp_proxqp_terminate
    void *proxqp;

    double time_qp_solver_call;
    int iter;
    int status;

} ocp_qp_proxqp_memory;

acados_size_t ocp_qp_proxqp_opts_calculate_size(void *config, void *dims);
//
void *ocp_qp_proxqp_opts_assign(void *config, void *dims, void *raw_memory);
//
void ocp_qp_proxqp_opts_initialize_default(void *config, void *dims, void *opts_);
//
void ocp_qp_proxqp_opts_update(void *config, void *dims, void *opts_);
//
acados_size_t ocp_qp_proxqp_memory_calculate_size(void *config, void *dims, void *opts_);
//
void *ocp_qp_proxqp_memory_assign(void *config, void *dims, void *opts_, void *raw_memory);
//
acados_size_t ocp_qp_proxqp_workspace_calculate_size(void *config, void *dims, void *opts_);
//
int ocp_qp_proxqp(void *config, void *qp_in, void *qp_out, void *opts_, void *mem_, void *work_);
//
void ocp_qp_proxqp_memory_reset(void *config_, void *qp_in_, void *qp_out_, void *opts_,
                                void *mem_, void *work_);
//
void ocp_qp_proxqp_solver_get(void *config_, void *qp_in_, void *qp_out_, void *opts_, void *mem_,
                              const char *field, int stage, void *value, int size1, int size2);
//
void ocp_qp_proxqp_config_initialize_default(void *config);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_OCP_QP_OCP_QP_PROXQP_H_
