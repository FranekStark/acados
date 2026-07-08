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

// Thin C wrapper around the header-only C++ ProxQP solver (proxsuite),
// exposing exactly the subset of the dense and sparse API used by acados.
//
// Problem formulation (see the proxsuite documentation):
//   min_x 1/2 x^T H x + g^T x
//   s.t.  A x = b
//         l <= C x <= u
//         l_box <= x <= u_box   (dense backend only)

#ifndef PROXSUITE_C_H_
#define PROXSUITE_C_H_

#ifdef __cplusplus
extern "C" {
#endif

// values match proxsuite::proxqp::QPSolverOutput,
// PROXQP_C_ERROR is added for exceptions caught at the C boundary
typedef enum
{
    PROXQP_C_SOLVED = 0,
    PROXQP_C_MAX_ITER_REACHED,
    PROXQP_C_PRIMAL_INFEASIBLE,
    PROXQP_C_SOLVED_CLOSEST_PRIMAL_FEASIBLE,
    PROXQP_C_DUAL_INFEASIBLE,
    PROXQP_C_NOT_RUN,
    PROXQP_C_ERROR
} proxqp_c_status;

// values match proxsuite::proxqp::InitialGuessStatus
typedef enum
{
    PROXQP_C_NO_INITIAL_GUESS = 0,
    PROXQP_C_EQUALITY_CONSTRAINED_INITIAL_GUESS,
    PROXQP_C_WARM_START_WITH_PREVIOUS_RESULT,
    PROXQP_C_WARM_START,
    PROXQP_C_COLD_START_WITH_PREVIOUS_RESULT
} proxqp_c_initial_guess;

typedef struct
{
    double eps_abs;
    double eps_rel;
    int max_iter;
    int verbose;
    int initial_guess;  // proxqp_c_initial_guess
} proxqp_c_settings;

// fill with the proxsuite defaults
void proxqp_c_default_settings(proxqp_c_settings *settings);

// value proxsuite uses for absent bounds (sqrt of double max)
double proxqp_c_infinity(void);

/************************************************
 * dense backend
 ************************************************/

// dense matrices are expected in column-major order:
// H (n x n), A (n_eq x n), C (n_in x n); vectors of matching length.
// If box_constraints is nonzero, init/update expect l_box/u_box of length n.
// Passing NULL for a matrix/vector maps to "not present" (dimension 0).

void *proxqp_c_dense_new(int n, int n_eq, int n_in, int box_constraints);

void proxqp_c_dense_free(void *qp);

void proxqp_c_dense_apply_settings(void *qp, const proxqp_c_settings *settings);

// returns 0 on success, nonzero on error
int proxqp_c_dense_init(void *qp, const double *H, const double *g,
                        const double *A, const double *b,
                        const double *C, const double *l, const double *u,
                        const double *l_box, const double *u_box);

// returns 0 on success, nonzero on error
int proxqp_c_dense_update(void *qp, const double *H, const double *g,
                          const double *A, const double *b,
                          const double *C, const double *l, const double *u,
                          const double *l_box, const double *u_box);

// returns a proxqp_c_status
int proxqp_c_dense_solve(void *qp);

// copies the solution; any output pointer may be NULL.
// x: primal (length n), y: equality multipliers (length n_eq),
// z: inequality multipliers (length n_in, followed by n box multipliers
//    if the qp was created with box_constraints)
void proxqp_c_dense_get_solution(void *qp, double *x, double *y, double *z);

int proxqp_c_dense_get_iter(void *qp);

/************************************************
 * sparse backend
 ************************************************/

// sparse matrices are expected in CSC format: column pointers p (size cols+1),
// row indices i, values x; the Hessian may be given as the upper triangular
// part only (proxsuite ignores the lower triangle).

void *proxqp_c_sparse_new(int n, int n_eq, int n_in);

void proxqp_c_sparse_free(void *qp);

void proxqp_c_sparse_apply_settings(void *qp, const proxqp_c_settings *settings);

// returns 0 on success, nonzero on error
int proxqp_c_sparse_init(void *qp,
                         const double *H_x, const int *H_i, const int *H_p,
                         const double *g,
                         const double *A_x, const int *A_i, const int *A_p,
                         const double *b,
                         const double *C_x, const int *C_i, const int *C_p,
                         const double *l, const double *u);

// the sparsity structure must be unchanged wrt the preceding init;
// returns 0 on success, nonzero on error
int proxqp_c_sparse_update(void *qp,
                           const double *H_x, const int *H_i, const int *H_p,
                           const double *g,
                           const double *A_x, const int *A_i, const int *A_p,
                           const double *b,
                           const double *C_x, const int *C_i, const int *C_p,
                           const double *l, const double *u);

// returns a proxqp_c_status
int proxqp_c_sparse_solve(void *qp);

// copies the solution; any output pointer may be NULL.
// x: primal (length n), y: equality multipliers (length n_eq),
// z: inequality multipliers (length n_in)
void proxqp_c_sparse_get_solution(void *qp, double *x, double *y, double *z);

int proxqp_c_sparse_get_iter(void *qp);

#ifdef __cplusplus
}
#endif

#endif  // PROXSUITE_C_H_
