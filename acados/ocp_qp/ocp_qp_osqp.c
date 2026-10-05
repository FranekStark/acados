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


#include <assert.h>
#include <math.h>
#include <string.h>

// blasfeo
#include "blasfeo_d_blasfeo_api.h"

// acados
#include "acados/ocp_qp/ocp_qp_common.h"
#include "acados/ocp_qp/ocp_qp_osqp.h"
#include "acados/utils/mem.h"
#include "acados/utils/math.h"
#include "acados/utils/print.h"
#include "acados/utils/timing.h"
#include "acados/utils/types.h"

// osqp
#include "osqp/include/public/osqp.h"



/************************************************
 * QP layout
 *
 * Variables, per stage: [u; x; s_lower; s_upper], as in ocp_qp_in.
 *
 * Rows of A (l <= A v <= u), in this order:
 *   dynamics of all stages (l = u);
 *   per stage the [box; general] constraints, then the replicated upper sides of the
 *   softened ones (the original row then only keeps the lower side);
 *   the slack bounds of all stages.
 *
 * ocp_qp_in only holds dense blocks, so the sparsity pattern of P and A is detected from the
 * data: the pattern is the union of all nonzeros seen so far. A nonzero outside the pattern
 * grows it and rebuilds the OSQP solver; otherwise the solver data is updated in place.
 * The rows do not depend on the pattern.
 ************************************************/



/************************************************
 * helper functions
 ************************************************/



static int acados_osqp_num_vars(ocp_qp_dims *dims)
{
    int N = dims->N;
    int *nx = dims->nx;
    int *nu = dims->nu;
    int *ns = dims->ns;

    int n = 0;

    for (int ii = 0; ii <= N; ii++)
    {
        n += nx[ii];   // states
        n += nu[ii];   // controls
        n += 2*ns[ii]; // slacks
    }

    return n;
}



static int acados_osqp_num_constr(ocp_qp_dims *dims)
{
    int N = dims->N;
    int *nx = dims->nx;
    int *nb = dims->nb;
    int *ng = dims->ng;
    int *ns = dims->ns;

    int m = 0;

    // inequality constraints
    for (int ii = 0; ii <= N; ii++)
    {
        m += nb[ii];   // box constraints
        m += ng[ii];   // general constraints
        m += ns[ii];   // replicated box/general softed constraint
        m += 2*ns[ii]; // slacks nonnegativity constraints
    }

    // dynamics equality constraints
    for (int ii = 0; ii < N; ii++)
    {
        m += nx[ii + 1];
    }

    return m;
}



static int acados_osqp_nnzmax_P(const ocp_qp_dims *dims)
{
    int N = dims->N;
    int *nx = dims->nx;
    int *nu = dims->nu;
    int *ns = dims->ns;

    int nnz = 0;

    for (int ii = 0; ii <= N; ii++)
    {
        nnz += nx[ii] * nx[ii];      // Q
        nnz += nu[ii] * nu[ii];      // R
        nnz += 2 * nx[ii] * nu[ii];  // S // TODO 1*, likely only the L or U are needed
        nnz += 2 * ns[ii];           // Z
    }

    return nnz;
}



static int acados_osqp_nnzmax_A(const ocp_qp_dims *dims)
{
    int N = dims->N;
    int *nx = dims->nx;
    int *nu = dims->nu;
    int *nb = dims->nb;
    int *ng = dims->ng;
    int *ns = dims->ns;

    int nnz = 0;

    // inequality constraints
    for (int ii = 0; ii <= N; ii++)
    {
        nnz += nb[ii];           // eye of box constraints
        nnz += ng[ii] * nx[ii];  // C
        nnz += ng[ii] * nu[ii];  // D
        nnz += ns[ii] * (nu[ii] + nx[ii]);  // replicated box/general softed constraint at worst case
        nnz += (nb[ii] + ng[ii]) * 2 * ns[ii]; // soft constraints at worst case, when idxs_rev encoding is used. Typically just 2*ns
        nnz += 2 * ns[ii];       // eye of slacks nonnegativity constraints
    }

    // dynamics equality constraints
    for (int ii = 0; ii < N; ii++)
    {
        nnz += nx[ii + 1] * nx[ii];  // A
        nnz += nx[ii + 1] * nu[ii];  // B
        nnz += nx[ii + 1];           // eye
    }

    return nnz;
}



// number of flags in the sparsity masks of triu(RSQ), [B A]^T and [D C]^T
static int acados_osqp_mask_size_P(const ocp_qp_dims *dims)
{
    int size = 0;
    for (int ii = 0; ii <= dims->N; ii++)
    {
        int nv = dims->nu[ii] + dims->nx[ii];
        size += nv*(nv+1)/2;
    }
    return size;
}



static int acados_osqp_mask_size_BA(const ocp_qp_dims *dims)
{
    int size = 0;
    for (int ii = 0; ii < dims->N; ii++)
    {
        size += (dims->nu[ii] + dims->nx[ii]) * dims->nx[ii+1];
    }
    return size;
}



static int acados_osqp_mask_size_DC(const ocp_qp_dims *dims)
{
    int size = 0;
    for (int ii = 0; ii <= dims->N; ii++)
    {
        size += (dims->nu[ii] + dims->nx[ii]) * dims->ng[ii];
    }
    return size;
}



// length of the unpack buffer: one row of RSQ, BAbt or DCt
static int acados_osqp_work_size(const ocp_qp_dims *dims)
{
    int size = 1;
    for (int ii = 0; ii <= dims->N; ii++)
    {
        int nv = dims->nu[ii] + dims->nx[ii];
        size = nv > size ? nv : size;
        size = dims->ng[ii] > size ? dims->ng[ii] : size;
        if (ii < dims->N)
            size = dims->nx[ii+1] > size ? dims->nx[ii+1] : size;
    }
    return size;
}



// index of the box constraint on variable jj of stage kk, -1 if there is none
static int box_of_var(const ocp_qp_in *in, int kk, int jj)
{
    for (int ii = 0; ii < in->dim->nb[kk]; ii++)
    {
        if (in->idxb[kk][ii] == jj)
            return ii;
    }
    return -1;
}



static int is_soft(const ocp_qp_in *in, int kk, int cc)
{
    return in->dim->ns[kk] > 0 && in->idxs_rev[kk][cc] >= 0;
}



// number of softened constraints of stage kk with index below cc, i.e. the position of
// the replicated row of constraint cc among the replicated rows of the stage
static int num_soft_before(const ocp_qp_in *in, int kk, int cc)
{
    int num = 0;
    for (int ii = 0; ii < cc; ii++)
    {
        if (is_soft(in, kk, ii))
            num++;
    }
    return num;
}



static inline void emit(OSQPInt *idx, OSQPFloat *val, OSQPInt *nn, int write_structure,
                        OSQPInt row, double value)
{
    if (write_structure)
        idx[*nn] = row;
    val[*nn] = value;
    (*nn)++;
}



// Writes the nonzeros of P and A in CSC order for the current sparsity masks, and with
// write_structure also their row indices and column pointers. Entries outside the masks are
// not written: if one of them is nonzero, its flag is set and 1 is returned, and the caller
// has to call this again with write_structure to rebuild the pattern.
// Within each column the rows are emitted in increasing order.
static int pack_matrices(const ocp_qp_in *in, ocp_qp_osqp_memory *mem, int write_structure)
{
    ocp_qp_dims *dims = in->dim;

    int N = dims->N;
    int *nx = dims->nx;
    int *nu = dims->nu;
    int *nb = dims->nb;
    int *ng = dims->ng;
    int *ns = dims->ns;

    int ii, jj, kk, cc;
    int grew = 0;
    double *w = mem->work_row;

    // P: triu(RSQ) restricted to the pattern, and the diagonal Z
    OSQPInt nn = 0, col = 0, offset = 0;
    char *mask = mem->P_mask;
    for (kk = 0; kk <= N; kk++)
    {
        int nv = nu[kk] + nx[kk];
        // row jj of the lower triangle of RSQ is column jj of the upper triangle
        for (jj = 0; jj < nv; jj++)
        {
            if (write_structure)
                mem->P_p[col] = nn;
            col++;

            blasfeo_unpack_dmat(1, jj+1, in->RSQrq+kk, jj, 0, w, 1);
            for (ii = 0; ii <= jj; ii++)
            {
                if (mask[ii])
                    emit(mem->P_i, mem->P_x, &nn, write_structure, offset + ii, w[ii]);
                else if (w[ii] != 0.0)
                {
                    mask[ii] = 1;
                    grew = 1;
                }
            }
            mask += jj + 1;
        }
        offset += nv;

        for (jj = 0; jj < 2*ns[kk]; jj++)
        {
            if (write_structure)
                mem->P_p[col] = nn;
            col++;

            emit(mem->P_i, mem->P_x, &nn, write_structure, offset + jj, BLASFEO_DVECEL(in->Z+kk, jj));
        }
        offset += 2*ns[kk];
    }
    if (write_structure)
        mem->P_p[col] = nn;

    // A. Rows: dynamics of all stages; then per stage [box; general] constraints followed by
    // the replicated upper sides of the softened ones; then the slack bounds of all stages.
    OSQPInt con_start = 0;
    OSQPInt slk_start = 0;
    for (kk = 0; kk <= N; kk++)
    {
        con_start += kk < N ? nx[kk+1] : 0;
        slk_start += nb[kk] + ng[kk] + ns[kk];
    }
    slk_start += con_start;

    nn = 0;
    col = 0;
    OSQPInt row_dyn = 0;
    OSQPInt row_con = con_start;
    OSQPInt row_slk = slk_start;
    char *ba_mask = mem->BA_mask;
    char *dc_mask = mem->DC_mask;
    for (kk = 0; kk <= N; kk++)
    {
        int nv = nu[kk] + nx[kk];
        int nc = nb[kk] + ng[kk];
        int nx1 = kk < N ? nx[kk+1] : 0;
        OSQPInt row_rep = row_con + nc;
        int nsb = num_soft_before(in, kk, nb[kk]);

        for (jj = 0; jj < nv; jj++)
        {
            if (write_structure)
                mem->A_p[col] = nn;
            col++;

            // -I of the previous stage's dynamics
            if (kk > 0 && jj >= nu[kk])
                emit(mem->A_i, mem->A_x, &nn, write_structure, row_dyn - nx[kk] + jj - nu[kk], -1.0);

            // column of [B A]
            if (kk < N)
            {
                blasfeo_unpack_dmat(1, nx1, in->BAbt+kk, jj, 0, w, 1);
                for (ii = 0; ii < nx1; ii++)
                {
                    if (ba_mask[ii])
                        emit(mem->A_i, mem->A_x, &nn, write_structure, row_dyn + ii, w[ii]);
                    else if (w[ii] != 0.0)
                    {
                        ba_mask[ii] = 1;
                        grew = 1;
                    }
                }
            }
            ba_mask += nx1;

            // box constraint
            int cb = box_of_var(in, kk, jj);
            if (cb >= 0)
                emit(mem->A_i, mem->A_x, &nn, write_structure, row_con + cb, 1.0);

            // column of [D C]
            blasfeo_unpack_dmat(1, ng[kk], in->DCt+kk, jj, 0, w, 1);
            for (ii = 0; ii < ng[kk]; ii++)
            {
                if (dc_mask[ii])
                    emit(mem->A_i, mem->A_x, &nn, write_structure, row_con + nb[kk] + ii, w[ii]);
            }

            // replicated softened box constraint
            if (cb >= 0 && is_soft(in, kk, cb))
                emit(mem->A_i, mem->A_x, &nn, write_structure, row_rep + num_soft_before(in, kk, cb), 1.0);

            // replicated softened general constraints, same pattern as the originals
            int is = nsb;
            for (ii = 0; ii < ng[kk]; ii++)
            {
                if (is_soft(in, kk, nb[kk]+ii))
                {
                    if (dc_mask[ii])
                        emit(mem->A_i, mem->A_x, &nn, write_structure, row_rep + is, w[ii]);
                    is++;
                }
            }

            // grow the pattern only after both uses of dc_mask in this column
            for (ii = 0; ii < ng[kk]; ii++)
            {
                if (!dc_mask[ii] && w[ii] != 0.0)
                {
                    dc_mask[ii] = 1;
                    grew = 1;
                }
            }
            dc_mask += ng[kk];
        }

        // slack variables on lower inequalities (original rows)
        for (jj = 0; jj < ns[kk]; jj++)
        {
            if (write_structure)
                mem->A_p[col] = nn;
            col++;

            for (cc = 0; cc < nc; cc++)
            {
                // no break, there could possibly be multiple
                if (in->idxs_rev[kk][cc] == jj)
                    emit(mem->A_i, mem->A_x, &nn, write_structure, row_con + cc, 1.0);
            }
            // nonnegativity constraint
            emit(mem->A_i, mem->A_x, &nn, write_structure, row_slk + jj, 1.0);
        }

        // slack variables on upper inequalities (replicated rows)
        for (jj = 0; jj < ns[kk]; jj++)
        {
            if (write_structure)
                mem->A_p[col] = nn;
            col++;

            int is = 0;
            for (cc = 0; cc < nc; cc++)
            {
                if (in->idxs_rev[kk][cc] == jj)
                    emit(mem->A_i, mem->A_x, &nn, write_structure, row_rep + is, -1.0);
                if (in->idxs_rev[kk][cc] >= 0)
                    is++;
            }
            // nonnegativity constraint
            emit(mem->A_i, mem->A_x, &nn, write_structure, row_slk + ns[kk] + jj, 1.0);
        }

        row_dyn += nx1;
        row_con += nc + ns[kk];
        row_slk += 2*ns[kk];
    }
    if (write_structure)
        mem->A_p[col] = nn;

    return grew;
}



static void update_gradient(const ocp_qp_in *in, ocp_qp_osqp_memory *mem)
{
    ocp_qp_dims *dims = in->dim;

    int N = dims->N;
    int *nx = dims->nx;
    int *nu = dims->nu;
    int *ns = dims->ns;

    int kk, nn = 0;
    for (kk = 0; kk <= N; kk++)
    {
        blasfeo_unpack_dvec(nu[kk]+nx[kk]+2*ns[kk], in->rqz + kk, 0, &mem->q[nn], 1);
        nn += nu[kk]+nx[kk]+2*ns[kk];
    }
}



static void update_bounds(const ocp_qp_in *in, ocp_qp_osqp_memory *mem)
{
    ocp_qp_dims *dims = in->dim;

    int N = dims->N;
    int *nx = dims->nx;
    //int *nu = dims->nu;
    int *nb = dims->nb;
    int *ng = dims->ng;
    int *ns = dims->ns;

    int ii, kk, nn = 0;

    // write -b to l and u
    for (kk = 0; kk < N; kk++)
    {
        // unpack b to l
        blasfeo_unpack_dvec(nx[kk + 1], in->b + kk, 0, &mem->l[nn], 1);

        // change sign of l (to get -b) and copy to u
        for (ii = 0; ii < nx[kk + 1]; ii++)
        {
            mem->l[nn + ii] = -mem->l[nn + ii];
            mem->u[nn + ii] = mem->l[nn + ii];
        }

        nn += nx[kk + 1];
    }

    // write lb lg and ub ug
    for (kk = 0; kk <= N; kk++)
    {
        // unpack lb lg to l
        blasfeo_unpack_dvec(nb[kk]+ng[kk], in->d + kk, 0, &mem->l[nn], 1);
        // set replicated to -inf
        for (ii=0; ii<ns[kk]; ii++)
        {
            mem->l[nn+nb[kk]+ng[kk]+ii] = -OSQP_INFTY;
        }

        // unpack ub ug to u and flip signs because in HPIPM the signs are flipped for upper bounds
        // keep in original place if not softed; set to inf and replicated under if softed
        int itmp = 0;
        for (ii = 0; ii < nb[kk] + ng[kk]; ii++)
        {
            if (in->idxs_rev[kk][ii]==-1) // not softed
            {
                mem->u[nn + ii] = -BLASFEO_DVECEL(&in->d[kk], nb[kk]+ng[kk]+ii);
            }
            else
            {
                mem->u[nn + ii] = OSQP_INFTY;
                mem->u[nn + nb[kk]+ng[kk]+itmp] = -BLASFEO_DVECEL(&in->d[kk], nb[kk]+ng[kk]+ii);
                itmp++;
            }
        }

        nn += nb[kk] + ng[kk] + ns[kk];
    }

    // write ls and us
    for (kk = 0; kk <= N; kk++)
    {
        // unpack ls and us to l
        blasfeo_unpack_dvec(2*ns[kk], in->d + kk, 2*nb[kk]+2*ng[kk], &mem->l[nn], 1);

        // OSQP_INFTY at upper bound
        for (ii = 0; ii < 2*ns[kk]; ii++)
        {
            mem->u[nn + ii] = OSQP_INFTY;
        }

        nn += 2*ns[kk];
    }

}



/************************************************
 * opts
 ************************************************/

acados_size_t ocp_qp_osqp_opts_calculate_size(void *config_, void *dims_)
{
    acados_size_t size = 0;
    size += sizeof(ocp_qp_osqp_opts);
    size += sizeof(OSQPSettings);

    return size;
}



void *ocp_qp_osqp_opts_assign(void *config_, void *dims_, void *raw_memory)
{
    ocp_qp_osqp_opts *opts;

    char *c_ptr = (char *) raw_memory;

    opts = (ocp_qp_osqp_opts *) c_ptr;
    c_ptr += sizeof(ocp_qp_osqp_opts);

    opts->osqp_opts = (OSQPSettings *) c_ptr;
    c_ptr += sizeof(OSQPSettings);

    assert((char *) raw_memory + ocp_qp_osqp_opts_calculate_size(config_, dims_) >= c_ptr);

    return (void *) opts;
}



void ocp_qp_osqp_opts_initialize_default(void *config_, void *dims_, void *opts_)
{
    ocp_qp_osqp_opts *opts = opts_;

    osqp_set_default_settings(opts->osqp_opts);
    opts->osqp_opts->verbose = 0;
    opts->osqp_opts->polishing = 1;
    opts->osqp_opts->check_termination = 5;
    opts->osqp_opts->warm_starting = 1;

    opts->print_level = 0;

    opts->tol_stat = -1.0;
    opts->tol_eq = -1.0;
    opts->tol_ineq = -1.0;
    opts->tol_dual_gap = -1.0;

    return;
}



void ocp_qp_osqp_opts_update(void *config_, void *dims_, void *opts_)
{
    // ocp_qp_osqp_opts *opts = (ocp_qp_osqp_opts *)opts_;

    return;
}



static double min_requested_tol(double a, double b)
{
    if (a <= 0.0)
        return b;
    if (b <= 0.0)
        return a;
    return a < b ? a : b;
}



// OSQP stops when the primal residual, the dual residual and the duality gap are each below
// eps_abs + eps_rel * (size of the corresponding terms), so the acados tolerances map onto
// eps_abs = eps_rel. eps_prim_inf and eps_dual_inf only detect infeasibility and keep the
// OSQP defaults.
static void apply_requested_tolerances(ocp_qp_osqp_opts *opts)
{
    double tol = min_requested_tol(opts->tol_stat, min_requested_tol(opts->tol_eq,
                 min_requested_tol(opts->tol_ineq, opts->tol_dual_gap)));

    if (tol > 0.0)
    {
        opts->osqp_opts->eps_abs = tol;
        opts->osqp_opts->eps_rel = tol;
    }
}



void ocp_qp_osqp_opts_set(void *config_, void *opts_, const char *field, void *value)
{
    ocp_qp_osqp_opts *opts = opts_;

    // NOTE: settings are passed to OSQP at every call, via osqp_setup on the first call and
    // osqp_update_settings afterwards; some settings can only be set before the first call,
    // see the documentation of osqp_update_settings.
    if(!strcmp(field, "osqp_linsys_solver")){
        const char *linsys_solver;
        linsys_solver = (const char *) value;
        if (!strcmp(linsys_solver, "direct")){
            opts->osqp_opts->linsys_solver = OSQP_DIRECT_SOLVER;
        }
        else if (!strcmp(linsys_solver, "indirect")){
            opts->osqp_opts->linsys_solver = OSQP_INDIRECT_SOLVER;
        }
        else {
            printf("\nerror: ocp_qp_osqp_opts_set: wrong value: %s\n", (const char *)value);
            exit(1);
        }    
    }
    else if(!strcmp(field, "osqp_polish")){
        int *polish = value; 
        opts->osqp_opts->polishing = *polish;
    }
    else if (!strcmp(field, "iter_max"))
    {
        int *tmp_ptr = value;
        opts->osqp_opts->max_iter = *tmp_ptr;
    }
    else if (!strcmp(field, "print_level"))
    {
        int *tmp_ptr = value;
        opts->print_level = *tmp_ptr;
        if (opts->print_level > 0)
        {
            opts->osqp_opts->verbose = 1;
        }
    }
    else if (!strcmp(field, "tol_stat"))
    {
        opts->tol_stat = *(double *) value;
        apply_requested_tolerances(opts);
    }
    else if (!strcmp(field, "tol_eq"))
    {
        opts->tol_eq = *(double *) value;
        apply_requested_tolerances(opts);
    }
    else if (!strcmp(field, "tol_ineq"))
    {
        opts->tol_ineq = *(double *) value;
        apply_requested_tolerances(opts);
    }
    else if (!strcmp(field, "tol_comp"))
    {
        // "OSQP always satisfies complementary slackness conditions
        //  with machine precision by construction." - Strellato2020
    }
    else if (!strcmp(field, "tol_dual_gap"))
    {
        // OSQP v1 checks the duality gap against eps_abs and eps_rel (check_dualgap)
        opts->tol_dual_gap = *(double *) value;
        apply_requested_tolerances(opts);
    }
    else if (!strcmp(field, "warm_start"))
    {
        int *tmp_ptr = value;
        opts->osqp_opts->warm_starting = *tmp_ptr;
    }
    else
    {
        printf("\nerror: ocp_qp_osqp_opts_set: wrong field: %s\n", field);
        exit(1);
    }

    return;
}


void ocp_qp_osqp_opts_get(void *config_, void *opts_, const char *field, void *value)
{
    // ocp_qp_osqp_opts *opts = opts_;
    printf("\nerror: ocp_qp_osqp_opts_get: not implemented for field %s\n", field);
    exit(1);
}



/************************************************
 * memory
 ************************************************/



acados_size_t ocp_qp_osqp_memory_calculate_size(void *config_, void *dims_, void *opts_)
{
    ocp_qp_dims *dims = dims_;

    size_t n = acados_osqp_num_vars(dims);
    size_t m = acados_osqp_num_constr(dims);

    size_t P_nnzmax = acados_osqp_nnzmax_P(dims);
    size_t A_nnzmax = acados_osqp_nnzmax_A(dims);

    size_t nw = acados_osqp_work_size(dims);

    acados_size_t size = 0;
    size += sizeof(ocp_qp_osqp_memory);

    size += 2 * sizeof(OSQPCscMatrix);  // matrices P and A

    size += nw * sizeof(double);        // work_row

    size += 1 * n * sizeof(OSQPFloat);  // q
    size += 2 * m * sizeof(OSQPFloat);  // l, u
    size += (n + m) * sizeof(OSQPFloat);  // x_prev, y_prev

    size += P_nnzmax * sizeof(OSQPFloat);  // P_x
    size += P_nnzmax * sizeof(OSQPInt);    // P_i
    size += (n + 1) * sizeof(OSQPInt);     // P_p

    size += A_nnzmax * sizeof(OSQPFloat);  // A_x
    size += A_nnzmax * sizeof(OSQPInt);    // A_i
    size += (n + 1) * sizeof(OSQPInt);     // A_p

    size += acados_osqp_mask_size_P(dims);   // P_mask
    size += acados_osqp_mask_size_BA(dims);  // BA_mask
    size += acados_osqp_mask_size_DC(dims);  // DC_mask

    size += 1 * 8;

    return size;
}



void *ocp_qp_osqp_memory_assign(void *config_, void *dims_, void *opts_, void *raw_memory)
{
    UNUSED(opts_);

    ocp_qp_dims *dims = dims_;
    ocp_qp_osqp_memory *mem;

    int n = acados_osqp_num_vars(dims);
    int m = acados_osqp_num_constr(dims);
    int P_nnzmax = acados_osqp_nnzmax_P(dims);
    int A_nnzmax = acados_osqp_nnzmax_A(dims);
    int nw = acados_osqp_work_size(dims);

    // char pointer
    char *c_ptr = (char *) raw_memory;

    mem = (ocp_qp_osqp_memory *) c_ptr;
    c_ptr += sizeof(ocp_qp_osqp_memory);

    mem->P_nnzmax = P_nnzmax;
    mem->A_nnzmax = A_nnzmax;
    mem->num_rebuilds = 0;

    align_char_to(8, &c_ptr);

    mem->P = (OSQPCscMatrix *) c_ptr;
    c_ptr += sizeof(OSQPCscMatrix);

    mem->A = (OSQPCscMatrix *) c_ptr;
    c_ptr += sizeof(OSQPCscMatrix);

    // doubles
    assign_and_advance_double(nw, &mem->work_row, &c_ptr);

    mem->q = (OSQPFloat *) c_ptr;
    c_ptr += n * sizeof(OSQPFloat);

    mem->l = (OSQPFloat *) c_ptr;
    c_ptr += m * sizeof(OSQPFloat);

    mem->u = (OSQPFloat *) c_ptr;
    c_ptr += m * sizeof(OSQPFloat);

    mem->x_prev = (OSQPFloat *) c_ptr;
    c_ptr += n * sizeof(OSQPFloat);

    mem->y_prev = (OSQPFloat *) c_ptr;
    c_ptr += m * sizeof(OSQPFloat);

    mem->P_x = (OSQPFloat *) c_ptr;
    c_ptr += (mem->P_nnzmax) * sizeof(OSQPFloat);

    mem->A_x = (OSQPFloat *) c_ptr;
    c_ptr += (mem->A_nnzmax) * sizeof(OSQPFloat);

    // ints
    mem->P_i = (OSQPInt *) c_ptr;
    c_ptr += (mem->P_nnzmax) * sizeof(OSQPInt);

    mem->P_p = (OSQPInt *) c_ptr;
    c_ptr += (n + 1) * sizeof(OSQPInt);

    mem->A_i = (OSQPInt *) c_ptr;
    c_ptr += (mem->A_nnzmax) * sizeof(OSQPInt);

    mem->A_p = (OSQPInt *) c_ptr;
    c_ptr += (n + 1) * sizeof(OSQPInt);

    // chars: the pattern starts empty and grows with the data
    assign_and_advance_char(acados_osqp_mask_size_P(dims), &mem->P_mask, &c_ptr);
    assign_and_advance_char(acados_osqp_mask_size_BA(dims), &mem->BA_mask, &c_ptr);
    assign_and_advance_char(acados_osqp_mask_size_DC(dims), &mem->DC_mask, &c_ptr);
    memset(mem->P_mask, 0, acados_osqp_mask_size_P(dims));
    memset(mem->BA_mask, 0, acados_osqp_mask_size_BA(dims));
    memset(mem->DC_mask, 0, acados_osqp_mask_size_DC(dims));

    // initialize matrix structs; the array pointers remain acados-owned (owned = 0)
    OSQPCscMatrix_set_data(mem->P, n, n, P_nnzmax, mem->P_x, mem->P_i, mem->P_p);
    OSQPCscMatrix_set_data(mem->A, m, n, A_nnzmax, mem->A_x, mem->A_i, mem->A_p);

    // the OSQPSolver is opaque and cannot be placed in acados-managed memory;
    // it is allocated by osqp_setup at the first call and freed in ocp_qp_osqp_terminate
    mem->osqp_solver = NULL;

    assert((char *) raw_memory + ocp_qp_osqp_memory_calculate_size(config_, dims, opts_) >= c_ptr);

    return mem;
}



void ocp_qp_osqp_memory_get(void *config_, void *mem_, const char *field, void* value)
{
    // qp_solver_config *config = config_;
    ocp_qp_osqp_memory *mem = mem_;

    if (!strcmp(field, "time_qp_solver_call"))
    {
        double *tmp_ptr = value;
        *tmp_ptr = mem->time_qp_solver_call;
    }
    else if (!strcmp(field, "iter"))
    {
        int *tmp_ptr = value;
        *tmp_ptr = mem->iter;
    }
    else if (!strcmp(field, "status"))
    {
        int *tmp_ptr = value;
        *tmp_ptr = mem->status;
    }
    else if (!strcmp(field, "num_rebuilds"))
    {
        int *tmp_ptr = value;
        *tmp_ptr = mem->num_rebuilds;
    }
    else
    {
        printf("\nerror: ocp_qp_osqp_memory_get: field %s not available\n", field);
        exit(1);
    }

    return;

}


void ocp_qp_osqp_memory_reset(void *config_, void *qp_in_, void *qp_out_, void *opts_, void *mem_, void *work_)
{
    // ocp_qp_in *qp_in = qp_in_;
    // reset memory
    printf("acados: reset osqp_mem not implemented.\n");
    exit(1);
}



/************************************************
 * workspace
 ************************************************/

acados_size_t ocp_qp_osqp_workspace_calculate_size(void *config_, void *dims_, void *opts_)
{
    return 0;
}



/************************************************
 * functions
 ************************************************/

static void fill_in_qp_out(const ocp_qp_in *in, ocp_qp_out *out, ocp_qp_osqp_memory *mem)
{
    ocp_qp_dims *dims = in->dim;

    int N = dims->N;
    int *nx = dims->nx;
    int *nu = dims->nu;
    int *nb = dims->nb;
    int *ng = dims->ng;
    int *ns = dims->ns;

    int ii, kk, nn;

    OSQPInt con_start = 0;
    OSQPInt slk_start = 0;
    for (kk = 0; kk <= N; kk++)
    {
        con_start += kk < N ? nx[kk + 1] : 0;
        slk_start += nb[kk] + ng[kk] + ns[kk];
    }

    slk_start += con_start;

    OSQPSolution *sol = mem->osqp_solver->solution;

    // primal variables
    nn = 0;
    for (kk = 0; kk <= N; kk++)
    {
        blasfeo_pack_dvec(nx[kk]+nu[kk]+2*ns[kk], &sol->x[nn], 1, out->ux + kk, 0);
        nn += nx[kk] + nu[kk] + 2*ns[kk];
    }

    // dual variables
    nn = 0;
    for (kk = 0; kk < N; kk++)
    {
        blasfeo_pack_dvec(nx[kk + 1], &sol->y[nn], 1, out->pi + kk, 0);
        nn += nx[kk + 1];
    }

    nn = 0;
    for (kk = 0; kk <= N; kk++)
    {
        //blasfeo_dvecse(2*nb[kk]+2*ng[kk]+2*ns[kk], 0.0, out->lam+kk, 0);
        blasfeo_dvecse(2*nb[kk]+2*ng[kk], 0.0, out->lam+kk, 0);

        int itmp = 0;
        for (ii = 0; ii < nb[kk]+ng[kk]; ii++)
        {
            if (in->idxs_rev[kk][ii]==-1) // not softed: two sided
            {
                double lam = sol->y[con_start+nn+ii];
                if (lam <= 0)
                {
                    BLASFEO_DVECEL(out->lam+kk, ii) = -lam;
                }
                else
                {
                    BLASFEO_DVECEL(out->lam+kk, nb[kk]+ng[kk] + ii) = lam;
                }
            }
            else // softed: replicated one sided
            {
                BLASFEO_DVECEL(out->lam+kk, ii) = -sol->y[con_start+nn+ii];
                BLASFEO_DVECEL(out->lam+kk, nb[kk]+ng[kk] + ii) = sol->y[con_start+nn+nb[kk]+ng[kk]+itmp];
                itmp++;
            }
        }
        nn += nb[kk]+ng[kk]+ns[kk];
    }

    nn = 0;
    for (kk = 0; kk <= N; kk++)
    {
        blasfeo_pack_dvec(2*ns[kk], &sol->y[slk_start+nn], 1, out->lam+kk, 2*nb[kk]+2*ng[kk]);
        blasfeo_dvecsc(2*ns[kk], -1.0, out->lam+kk, 2*nb[kk]+2*ng[kk]);
        nn += 2*ns[kk];
    }
}


void ocp_qp_osqp_terminate(void *config_, void *mem_, void *work_)
{
    ocp_qp_osqp_memory *mem = (ocp_qp_osqp_memory *) mem_;
    if (mem->osqp_solver != NULL)
        osqp_cleanup(mem->osqp_solver);
}



// Sets up an OSQP solver for the structure of the last pack_matrices. A previous solver is
// replaced; the rows of A do not depend on the pattern, so its rho and (with warm starting)
// its last solution carry over, as they would in an update without rebuild.
static void setup_solver(const ocp_qp_in *in, const ocp_qp_osqp_opts *opts, ocp_qp_osqp_memory *mem)
{
    OSQPInt n = acados_osqp_num_vars(in->dim);
    OSQPInt m = acados_osqp_num_constr(in->dim);

    OSQPSettings settings = *opts->osqp_opts;
    int warm_start = 0;

    if (mem->osqp_solver != NULL)
    {
        OSQPSolution *sol = mem->osqp_solver->solution;
        settings.rho = mem->osqp_solver->settings->rho;

        // the solution is NaN after an infeasible solve
        warm_start = opts->osqp_opts->warm_starting;
        for (OSQPInt ii = 0; warm_start && ii < n; ii++)
            warm_start = isfinite(sol->x[ii]);
        for (OSQPInt ii = 0; warm_start && ii < m; ii++)
            warm_start = isfinite(sol->y[ii]);
        if (warm_start)
        {
            memcpy(mem->x_prev, sol->x, n * sizeof(OSQPFloat));
            memcpy(mem->y_prev, sol->y, m * sizeof(OSQPFloat));
        }

        osqp_cleanup(mem->osqp_solver);
        mem->osqp_solver = NULL;
    }

    OSQPCscMatrix_set_data(mem->P, n, n, mem->P_p[n], mem->P_x, mem->P_i, mem->P_p);
    OSQPCscMatrix_set_data(mem->A, m, n, mem->A_p[n], mem->A_x, mem->A_i, mem->A_p);

    if (osqp_setup(&mem->osqp_solver, mem->P, mem->q, mem->A, mem->l, mem->u, m, n, &settings) != 0)
    {
        printf("\nerror: ocp_qp_osqp: osqp_setup failed\n");
        exit(1);
    }

    if (warm_start)
        osqp_warm_start(mem->osqp_solver, mem->x_prev, mem->y_prev);

    mem->num_rebuilds++;
    if (opts->print_level > 0)
    {
        printf("ocp_qp_osqp: built solver (#%d), n %d, m %d, nnz(P) %d, nnz(A) %d\n",
               mem->num_rebuilds, (int) n, (int) m, (int) mem->P_p[n], (int) mem->A_p[n]);
    }
}



int ocp_qp_osqp(void *config_, void *qp_in_, void *qp_out_, void *opts_, void *mem_, void *work_)
{
    ocp_qp_in *qp_in = qp_in_;
    ocp_qp_out *qp_out = qp_out_;

    // print_ocp_qp_dims(qp_in->dim);
    // print_ocp_qp_in(qp_in);

    qp_info *info = (qp_info *) qp_out->misc;
    acados_timer tot_timer, qp_timer, interface_timer, solver_call_timer;
    acados_tic(&tot_timer);

    // cast data structures
    ocp_qp_osqp_opts *opts = (ocp_qp_osqp_opts *) opts_;
    ocp_qp_osqp_memory *mem = (ocp_qp_osqp_memory *) mem_;

    acados_tic(&interface_timer);
    int rebuild = mem->osqp_solver == NULL;
    if (pack_matrices(qp_in, mem, 0) || rebuild)
    {
        // the sparsity pattern grew (or this is the first call)
        pack_matrices(qp_in, mem, 1);
        rebuild = 1;
    }
    update_bounds(qp_in, mem);
    update_gradient(qp_in, mem);
    info->interface_time = acados_toc(&interface_timer);

    acados_tic(&qp_timer);

    if (rebuild)
    {
        setup_solver(qp_in, opts, mem);
    }
    else
    {
        // update osqp solver with new data
        osqp_update_data_vec(mem->osqp_solver, mem->q, mem->l, mem->u);
        osqp_update_data_mat(mem->osqp_solver, mem->P_x, NULL, mem->P->p[mem->P->n],
                             mem->A_x, NULL, mem->A->p[mem->A->n]);
        osqp_update_settings(mem->osqp_solver, opts->osqp_opts);
    }

    // solve OSQP
    acados_tic(&solver_call_timer);
    osqp_solve(mem->osqp_solver);
    mem->time_qp_solver_call = acados_toc(&solver_call_timer);
    mem->iter = mem->osqp_solver->info->iter;

    // fill qp_out
    fill_in_qp_out(qp_in, qp_out, mem);
    ocp_qp_compute_t(qp_in, qp_out);

    // info
    info->solve_QP_time = acados_toc(&qp_timer);
    info->total_time = acados_toc(&tot_timer);
    info->num_iter = mem->osqp_solver->info->iter;
    info->t_computed = 1;

    OSQPInt osqp_status = mem->osqp_solver->info->status_val;
    int acados_status = osqp_status;

    // check exit conditions
    if (osqp_status == OSQP_SOLVED)
    {
        //printf("\nOSQP solved\n");
        acados_status = ACADOS_SUCCESS;
    }
    else if (osqp_status == OSQP_MAX_ITER_REACHED)
    {
        //printf("\nOSQP max iter reached\n");
        acados_status = ACADOS_MAXITER;
    }
    else if (osqp_status == OSQP_PRIMAL_INFEASIBLE)
    {
        //printf("\nOSQP primal infeasible\n");
        acados_status = ACADOS_INFEASIBLE;
    }
    else
    {
        acados_status = ACADOS_UNKNOWN;
    }
    mem->status = acados_status;

    return acados_status;
}



void ocp_qp_osqp_eval_forw_sens(void *config_, void *qp_in, void *seed, void *qp_out, void *opts_, void *mem_, void *work_)
{
    printf("\nerror: ocp_qp_osqp_eval_forw_sens: not implemented yet\n");
    exit(1);
}

void ocp_qp_osqp_eval_adj_sens(void *config_, void *qp_in, void *seed, void *qp_out, void *opts_, void *mem_, void *work_)
{
    printf("\nerror: ocp_qp_osqp_eval_adj_sens: not implemented yet\n");
    exit(1);
}


void ocp_qp_osqp_solver_get(void *config_, void *qp_in_, void *qp_out_, void *opts_, void *mem_, const char *field, int stage, void* value, int size1, int size2)
{
    printf("\nerror: ocp_qp_osqp_solver_get: not implemented yet\n");
    exit(1);
}


void ocp_qp_osqp_config_initialize_default(void *config_)
{
    qp_solver_config *config = config_;

    config->opts_calculate_size = &ocp_qp_osqp_opts_calculate_size;
    config->opts_assign = &ocp_qp_osqp_opts_assign;
    config->opts_initialize_default = &ocp_qp_osqp_opts_initialize_default;
    config->opts_update = &ocp_qp_osqp_opts_update;
    config->opts_set = &ocp_qp_osqp_opts_set;
    config->opts_get = &ocp_qp_osqp_opts_get;
    config->memory_calculate_size = &ocp_qp_osqp_memory_calculate_size;
    config->memory_assign = &ocp_qp_osqp_memory_assign;
    config->memory_get = &ocp_qp_osqp_memory_get;
    config->workspace_calculate_size = &ocp_qp_osqp_workspace_calculate_size;
    config->evaluate = &ocp_qp_osqp;
    config->terminate = &ocp_qp_osqp_terminate;
    config->eval_forw_sens = &ocp_qp_osqp_eval_forw_sens;
    config->eval_adj_sens = &ocp_qp_osqp_eval_adj_sens;
    config->memory_reset = &ocp_qp_osqp_memory_reset;
    config->solver_get = &ocp_qp_osqp_solver_get;

    return;
}
