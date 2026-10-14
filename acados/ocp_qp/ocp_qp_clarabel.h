/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * Licensed under the 2-Clause BSD License.
 */


#ifndef ACADOS_OCP_QP_OCP_QP_CLARABEL_H_
#define ACADOS_OCP_QP_OCP_QP_CLARABEL_H_

#ifdef __cplusplus
extern "C" {
#endif

// clarabel
// #include "clarabel/Clarabel.h"
#include "Clarabel.cpp/include/clarabel.h"

// acados
#include "acados/ocp_qp/ocp_qp_common.h"
#include "acados/utils/types.h"

// accuracy preset, set through the field "clarabel_mode"
typedef enum
{
    CLARABEL_MODE_NONE,  // no preset: the requested tolerances are used
    CLARABEL_MODE_REDUCED_ACCURACY,
    CLARABEL_MODE_FULL_ACCURACY,
} ocp_qp_clarabel_mode;


typedef struct ocp_qp_clarabel_opts_
{
    // settings *clarabel_opts;
    ClarabelDefaultSettings *clarabel_opts;
    int print_level;
    int first_run;
    // a constraint side whose bound satisfies |bound| >= inf_bound is treated as absent
    double inf_bound;
    // the clarabel_mode preset sets the stopping tolerances; the requested tolerances below
    // are then not used
    ocp_qp_clarabel_mode mode;
    // tolerances requested through the generic acados fields, <= 0 means not requested.
    // Clarabel has one feasibility tolerance (min of stat/eq/ineq) and one gap tolerance
    // (min of comp/dual_gap). They are used only if no clarabel_mode is set.
    double tol_stat;
    double tol_eq;
    double tol_ineq;
    double tol_comp;
    double tol_dual_gap;

} ocp_qp_clarabel_opts;


typedef struct ocp_qp_clarabel_memory_
{
    ClarabelCscMatrix P; // just upper triangular is enough
    uintptr_t P_nnzmax;
    uintptr_t *P_col_ptr;
    uintptr_t *P_rowval;
    ClarabelFloat *P_nzval;
    uintptr_t P_nnz;

    ClarabelCscMatrix A;
    uintptr_t A_nnzmax;
    uintptr_t *A_col_ptr;
    uintptr_t *A_rowval;
    ClarabelFloat *A_nzval;
    uintptr_t A_nnz;

    ClarabelFloat *q;
    uintptr_t q_nnz;
    ClarabelFloat *b;
    uintptr_t b_nnz;

    ClarabelSupportedConeT cones[2];
    int num_cones;

    // sparsity pattern: union of all nonzeros seen so far, one flag per dense entry
    char *P_mask;   // per stage: triu(RSQ), column-major
    char *BA_mask;  // per stage k<N: [B A]^T, (nu+nx) x nx[k+1], row-major
    char *DC_mask;  // per stage: [D C]^T, (nu+nx) x ng, row-major
    // per stage [lower sides (nb+ng), upper sides (nb+ng)]: bound was finite at least once
    char *d_finite;
    // per stage (nb+ng): constraint is an equality (idxe)
    char *is_eq;
    // per stage (nb+ng): Clarabel row of the lower / upper side, -1 if not emitted.
    // For equalities both point to the single ZeroCone row.
    int *row_lo;
    int *row_up;
    int m;          // number of rows actually emitted
    int m_eq;       // leading rows in the ZeroCone
    int slk_start;  // first row of the slack nonnegativity constraints
    int num_rebuilds;

    // unpack buffers
    double *work_row;
    double *work_d;
    double *work_d_mask;
    double *work_lam;

    ClarabelDefaultSolver *solver;
    ClarabelDefaultSolution solution;

    double time_qp_solver_call;
    double clarabel_solve_time;  // Clarabel's own timer for the last solve
    int iter;
    int status;

} ocp_qp_clarabel_memory;

acados_size_t ocp_qp_clarabel_opts_calculate_size(void *config, void *dims);
//
void *ocp_qp_clarabel_opts_assign(void *config, void *dims, void *raw_memory);
//
void ocp_qp_clarabel_opts_initialize_default(void *config, void *dims, void *opts_);
//
void ocp_qp_clarabel_opts_update(void *config, void *dims, void *opts_);
//
acados_size_t ocp_qp_clarabel_memory_calculate_size(void *config, void *dims, void *opts_);
//
void *ocp_qp_clarabel_memory_assign(void *config, void *dims, void *opts_, void *raw_memory);
//
acados_size_t ocp_qp_clarabel_workspace_calculate_size(void *config, void *dims, void *opts_);
//
int ocp_qp_clarabel(void *config, void *qp_in, void *qp_out, void *opts_, void *mem_, void *work_);
//
void ocp_qp_clarabel_memory_reset(void *config_, void *qp_in_, void *qp_out_, void *opts_, void *mem_, void *work_);
//
void ocp_qp_clarabel_solver_get(void *config_, void *qp_in_, void *qp_out_, void *opts_, void *mem_, const char *field, int stage, void* value, int size1, int size2);
//
void ocp_qp_clarabel_config_initialize_default(void *config);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_OCP_QP_OCP_QP_CLARABEL_H_
