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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// blasfeo
#include "blasfeo_d_blasfeo_api.h"

// acados
#include "acados/ocp_qp/ocp_qp_common.h"
#include "acados/ocp_qp/ocp_qp_proxqp.h"
#include "acados/utils/math.h"
#include "acados/utils/mem.h"
#include "acados/utils/print.h"
#include "acados/utils/timing.h"
#include "acados/utils/types.h"



/************************************************
 * QP layout
 *
 * Variables, per stage: [u; x; s_lower; s_upper], as in ocp_qp_in.
 *
 * Equality constraints A v = b: the dynamics of all stages.
 *
 * Rows of C (l <= C v <= u), in this order:
 *   per stage the [box; general] constraints, then the replicated upper sides of the
 *   softened ones (the original row then only keeps the lower side);
 *   the slack bounds of all stages.
 *
 * ocp_qp_in only holds dense blocks, so the sparsity pattern of P, A and C is detected from
 * the data: the pattern is the union of all nonzeros seen so far. A nonzero outside the pattern
 * grows it and rebuilds the ProxQP solver; otherwise the solver data is updated in place.
 * The rows do not depend on the pattern.
 *
 * A rebuild has to create a new ProxQP object: ProxQP's sparse update compares the new
 * structure with the stored one and silently keeps the old matrices if they differ.
 ************************************************/



/************************************************
 * helper functions
 ************************************************/

static int acados_proxqp_num_vars(ocp_qp_dims *dims)
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



static int acados_proxqp_num_eq(ocp_qp_dims *dims)
{
    int N = dims->N;
    int *nx = dims->nx;

    int m = 0;

    // dynamics equality constraints
    for (int ii = 0; ii < N; ii++)
    {
        m += nx[ii + 1];
    }

    return m;
}



static int acados_proxqp_num_ineq(ocp_qp_dims *dims)
{
    int N = dims->N;
    int *nb = dims->nb;
    int *ng = dims->ng;
    int *ns = dims->ns;

    int m = 0;

    for (int ii = 0; ii <= N; ii++)
    {
        m += nb[ii];   // box constraints
        m += ng[ii];   // general constraints
        m += ns[ii];   // replicated box/general softed constraint
        m += 2*ns[ii]; // slacks nonnegativity constraints
    }

    return m;
}



static int acados_proxqp_nnzmax_P(const ocp_qp_dims *dims)
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
        nnz += 2 * nx[ii] * nu[ii];  // S
        nnz += 2 * ns[ii];           // Z
    }

    return nnz;
}



static int acados_proxqp_nnzmax_A(const ocp_qp_dims *dims)
{
    int N = dims->N;
    int *nx = dims->nx;
    int *nu = dims->nu;

    int nnz = 0;

    // dynamics equality constraints
    for (int ii = 0; ii < N; ii++)
    {
        nnz += nx[ii + 1] * nx[ii];  // A
        nnz += nx[ii + 1] * nu[ii];  // B
        nnz += nx[ii + 1];           // eye
    }

    return nnz;
}



static int acados_proxqp_nnzmax_C(const ocp_qp_dims *dims)
{
    int N = dims->N;
    int *nx = dims->nx;
    int *nu = dims->nu;
    int *nb = dims->nb;
    int *ng = dims->ng;
    int *ns = dims->ns;

    int nnz = 0;

    for (int ii = 0; ii <= N; ii++)
    {
        nnz += nb[ii];           // eye of box constraints
        nnz += ng[ii] * nx[ii];  // C
        nnz += ng[ii] * nu[ii];  // D
        nnz += ns[ii] * (nu[ii] + nx[ii]);  // replicated box/general softed constraint at worst case
        nnz += (nb[ii] + ng[ii]) * 2 * ns[ii]; // soft constraints at worst case, when idxs_rev encoding is used. Typically just 2*ns
        nnz += 2 * ns[ii];       // eye of slacks nonnegativity constraints
    }

    return nnz;
}



// number of flags in the sparsity masks of triu(RSQ), [B A]^T and [D C]^T
static int acados_proxqp_mask_size_P(const ocp_qp_dims *dims)
{
    int size = 0;
    for (int ii = 0; ii <= dims->N; ii++)
    {
        int nv = dims->nu[ii] + dims->nx[ii];
        size += nv*(nv+1)/2;
    }
    return size;
}



static int acados_proxqp_mask_size_BA(const ocp_qp_dims *dims)
{
    int size = 0;
    for (int ii = 0; ii < dims->N; ii++)
    {
        size += (dims->nu[ii] + dims->nx[ii]) * dims->nx[ii+1];
    }
    return size;
}



static int acados_proxqp_mask_size_DC(const ocp_qp_dims *dims)
{
    int size = 0;
    for (int ii = 0; ii <= dims->N; ii++)
    {
        size += (dims->nu[ii] + dims->nx[ii]) * dims->ng[ii];
    }
    return size;
}



// length of the unpack buffer: one row of RSQ, BAbt or DCt
static int acados_proxqp_work_size(const ocp_qp_dims *dims)
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



static inline void emit(int *idx, double *val, int *nn, int write_structure, int *changed,
                        int row, double value)
{
    if (write_structure)
        idx[*nn] = row;
    if (val[*nn] != value)
    {
        val[*nn] = value;
        *changed = 1;
    }
    (*nn)++;
}



// Writes the nonzeros of P, A and C in CSC order for the current sparsity masks, and with
// write_structure also their row indices and column pointers. Entries outside the masks are
// not written: if one of them is nonzero, its flag is set and 1 is returned, and the caller
// has to call this again with write_structure to rebuild the pattern.
// Within each column the rows are emitted in increasing order.
// mem->{P,A,C}_changed tell whether a value differs from the one written by the previous call.
static int pack_matrices(const ocp_qp_in *in, ocp_qp_proxqp_memory *mem, int write_structure)
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

    mem->P_changed = 0;
    mem->A_changed = 0;
    mem->C_changed = 0;

    // P: triu(RSQ) restricted to the pattern, and the diagonal Z
    int nn = 0, col = 0, offset = 0;
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
                    emit(mem->P_i, mem->P_x, &nn, write_structure, &mem->P_changed, offset + ii, w[ii]);
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

            emit(mem->P_i, mem->P_x, &nn, write_structure, &mem->P_changed, offset + jj,
                 BLASFEO_DVECEL(in->Z+kk, jj));
        }
        offset += 2*ns[kk];
    }
    if (write_structure)
        mem->P_p[col] = nn;

    // A: dynamics; C: per stage [box; general] constraints followed by the replicated upper
    // sides of the softened ones, then the slack bounds of all stages.
    int slk_start = 0;
    for (kk = 0; kk <= N; kk++)
    {
        slk_start += nb[kk] + ng[kk] + ns[kk];
    }

    int na = 0, nc = 0;
    col = 0;
    int row_dyn = 0;
    int row_con = 0;
    int row_slk = slk_start;
    char *ba_mask = mem->BA_mask;
    char *dc_mask = mem->DC_mask;
    for (kk = 0; kk <= N; kk++)
    {
        int nv = nu[kk] + nx[kk];
        int ncon = nb[kk] + ng[kk];
        int nx1 = kk < N ? nx[kk+1] : 0;
        int row_rep = row_con + ncon;
        int nsb = num_soft_before(in, kk, nb[kk]);

        for (jj = 0; jj < nv; jj++)
        {
            if (write_structure)
            {
                mem->A_p[col] = na;
                mem->C_p[col] = nc;
            }
            col++;

            // -I of the previous stage's dynamics
            if (kk > 0 && jj >= nu[kk])
                emit(mem->A_i, mem->A_x, &na, write_structure, &mem->A_changed,
                     row_dyn - nx[kk] + jj - nu[kk], -1.0);

            // column of [B A]
            if (kk < N)
            {
                blasfeo_unpack_dmat(1, nx1, in->BAbt+kk, jj, 0, w, 1);
                for (ii = 0; ii < nx1; ii++)
                {
                    if (ba_mask[ii])
                        emit(mem->A_i, mem->A_x, &na, write_structure, &mem->A_changed, row_dyn + ii, w[ii]);
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
                emit(mem->C_i, mem->C_x, &nc, write_structure, &mem->C_changed, row_con + cb, 1.0);

            // column of [D C]
            blasfeo_unpack_dmat(1, ng[kk], in->DCt+kk, jj, 0, w, 1);
            for (ii = 0; ii < ng[kk]; ii++)
            {
                if (dc_mask[ii])
                    emit(mem->C_i, mem->C_x, &nc, write_structure, &mem->C_changed, row_con + nb[kk] + ii, w[ii]);
            }

            // replicated softened box constraint
            if (cb >= 0 && is_soft(in, kk, cb))
                emit(mem->C_i, mem->C_x, &nc, write_structure, &mem->C_changed,
                     row_rep + num_soft_before(in, kk, cb), 1.0);

            // replicated softened general constraints, same pattern as the originals
            int is = nsb;
            for (ii = 0; ii < ng[kk]; ii++)
            {
                if (is_soft(in, kk, nb[kk]+ii))
                {
                    if (dc_mask[ii])
                        emit(mem->C_i, mem->C_x, &nc, write_structure, &mem->C_changed, row_rep + is, w[ii]);
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
            {
                mem->A_p[col] = na;
                mem->C_p[col] = nc;
            }
            col++;

            for (cc = 0; cc < ncon; cc++)
            {
                // no break, there could possibly be multiple
                if (in->idxs_rev[kk][cc] == jj)
                    emit(mem->C_i, mem->C_x, &nc, write_structure, &mem->C_changed, row_con + cc, 1.0);
            }
            // nonnegativity constraint
            emit(mem->C_i, mem->C_x, &nc, write_structure, &mem->C_changed, row_slk + jj, 1.0);
        }

        // slack variables on upper inequalities (replicated rows)
        for (jj = 0; jj < ns[kk]; jj++)
        {
            if (write_structure)
            {
                mem->A_p[col] = na;
                mem->C_p[col] = nc;
            }
            col++;

            int is = 0;
            for (cc = 0; cc < ncon; cc++)
            {
                if (in->idxs_rev[kk][cc] == jj)
                    emit(mem->C_i, mem->C_x, &nc, write_structure, &mem->C_changed, row_rep + is, -1.0);
                if (in->idxs_rev[kk][cc] >= 0)
                    is++;
            }
            // nonnegativity constraint
            emit(mem->C_i, mem->C_x, &nc, write_structure, &mem->C_changed, row_slk + ns[kk] + jj, 1.0);
        }

        row_dyn += nx1;
        row_con += ncon + ns[kk];
        row_slk += 2*ns[kk];
    }
    if (write_structure)
    {
        mem->A_p[col] = na;
        mem->C_p[col] = nc;
    }

    return grew;
}



static void update_gradient(const ocp_qp_in *in, ocp_qp_proxqp_memory *mem)
{
    ocp_qp_dims *dims = in->dim;

    int N = dims->N;
    int *nx = dims->nx;
    int *nu = dims->nu;
    int *ns = dims->ns;

    int kk, nn = 0;
    for (kk = 0; kk <= N; kk++)
    {
        blasfeo_unpack_dvec(nu[kk]+nx[kk]+2*ns[kk], in->rqz + kk, 0, &mem->g[nn], 1);
        nn += nu[kk]+nx[kk]+2*ns[kk];
    }
}



static void update_bounds(const ocp_qp_in *in, ocp_qp_proxqp_memory *mem)
{
    ocp_qp_dims *dims = in->dim;

    int N = dims->N;
    int *nx = dims->nx;
    int *nb = dims->nb;
    int *ng = dims->ng;
    int *ns = dims->ns;

    int ii, kk, nn = 0;

    double inf = proxqp_c_infinity();

    // write -b to b (dynamics are written as A x + B u - x_next = -b)
    for (kk = 0; kk < N; kk++)
    {
        blasfeo_unpack_dvec(nx[kk + 1], in->b + kk, 0, &mem->b[nn], 1);

        for (ii = 0; ii < nx[kk + 1]; ii++)
        {
            mem->b[nn + ii] = -mem->b[nn + ii];
        }

        nn += nx[kk + 1];
    }

    // write lb lg and ub ug
    nn = 0;
    for (kk = 0; kk <= N; kk++)
    {
        // unpack lb lg to l
        blasfeo_unpack_dvec(nb[kk]+ng[kk], in->d + kk, 0, &mem->l[nn], 1);
        // set masked to -inf
        for (ii = 0; ii < nb[kk]+ng[kk]; ii++)
        {
            if (BLASFEO_DVECEL(in->d_mask+kk, ii) == 0.0)
            {
                mem->l[nn+ii] = -inf;
            }
        }
        // set replicated to -inf
        for (ii=0; ii<ns[kk]; ii++)
        {
            mem->l[nn+nb[kk]+ng[kk]+ii] = -inf;
        }

        // unpack ub ug to u and flip signs because in HPIPM the signs are flipped for upper bounds
        // keep in original place if not softed; set to inf and replicated under if softed
        // set masked to inf
        int itmp = 0;
        for (ii = 0; ii < nb[kk] + ng[kk]; ii++)
        {
            double ub = -BLASFEO_DVECEL(&in->d[kk], nb[kk]+ng[kk]+ii);
            if (BLASFEO_DVECEL(in->d_mask+kk, nb[kk]+ng[kk]+ii) == 0.0)
            {
                ub = inf;
            }

            if (in->idxs_rev[kk][ii]==-1) // not softed
            {
                mem->u[nn + ii] = ub;
            }
            else
            {
                mem->u[nn + ii] = inf;
                mem->u[nn + nb[kk]+ng[kk]+itmp] = ub;
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

        // infinity at upper bound
        for (ii = 0; ii < 2*ns[kk]; ii++)
        {
            mem->u[nn + ii] = inf;
        }

        // -infinity at masked lower bound
        for (ii = 0; ii < 2*ns[kk]; ii++)
        {
            if (BLASFEO_DVECEL(in->d_mask+kk, 2*nb[kk]+2*ng[kk]+ii) == 0.0)
            {
                mem->l[nn + ii] = -inf;
            }
        }

        nn += 2*ns[kk];
    }

}



static int all_finite(int n, const double *v)
{
    for (int ii = 0; ii < n; ii++)
    {
        if (!isfinite(v[ii]))
            return 0;
    }
    return 1;
}



/************************************************
 * opts
 ************************************************/

acados_size_t ocp_qp_proxqp_opts_calculate_size(void *config_, void *dims_)
{
    acados_size_t size = 0;
    size += sizeof(ocp_qp_proxqp_opts);

    return size;
}



void *ocp_qp_proxqp_opts_assign(void *config_, void *dims_, void *raw_memory)
{
    ocp_qp_proxqp_opts *opts;

    char *c_ptr = (char *) raw_memory;

    opts = (ocp_qp_proxqp_opts *) c_ptr;
    c_ptr += sizeof(ocp_qp_proxqp_opts);

    assert((char *) raw_memory + ocp_qp_proxqp_opts_calculate_size(config_, dims_) >= c_ptr);

    return (void *) opts;
}



void ocp_qp_proxqp_opts_initialize_default(void *config_, void *dims_, void *opts_)
{
    ocp_qp_proxqp_opts *opts = opts_;

    proxqp_c_default_settings(&opts->settings);
    // without the duality gap, a warm start from a solution which stays feasible and stationary after a
    // bound change (e.g. relaxed or masked active bound) is accepted after 0 iterations
    opts->settings.check_duality_gap = 1;
    opts->warm_start = 0;
    opts->print_level = 0;

    opts->tol_stat = -1.0;
    opts->tol_eq = -1.0;
    opts->tol_ineq = -1.0;
    opts->tol_comp = -1.0;
    opts->tol_dual_gap = -1.0;

    return;
}



void ocp_qp_proxqp_opts_update(void *config_, void *dims_, void *opts_)
{
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



// ProxQP stops when the primal and dual residuals (infinity norm, unscaled) are below
// eps_abs + eps_rel * (size of the corresponding terms) and, with check_duality_gap, the
// duality gap is below eps_duality_gap_abs + eps_duality_gap_rel * (size). eps_rel and
// eps_duality_gap_rel stay 0. Tolerances which are not requested keep the ProxQP defaults.
static void apply_requested_tolerances(ocp_qp_proxqp_opts *opts)
{
    proxqp_c_settings defaults;
    proxqp_c_default_settings(&defaults);

    double tol = min_requested_tol(opts->tol_stat, min_requested_tol(opts->tol_eq, opts->tol_ineq));
    opts->settings.eps_abs = tol > 0.0 ? tol : defaults.eps_abs;

    double tol_gap = min_requested_tol(opts->tol_comp, opts->tol_dual_gap);
    opts->settings.eps_duality_gap_abs = tol_gap > 0.0 ? tol_gap : defaults.eps_duality_gap_abs;
}



void ocp_qp_proxqp_opts_set(void *config_, void *opts_, const char *field, void *value)
{
    ocp_qp_proxqp_opts *opts = opts_;

    if (!strcmp(field, "tol_stat"))
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
        opts->tol_comp = *(double *) value;
        apply_requested_tolerances(opts);
    }
    else if (!strcmp(field, "tol_dual_gap"))
    {
        opts->tol_dual_gap = *(double *) value;
        apply_requested_tolerances(opts);
    }
    else if (!strcmp(field, "iter_max"))
    {
        int *iter_max = value;
        opts->settings.max_iter = *iter_max;
    }
    else if (!strcmp(field, "warm_start"))
    {
        int *warm_start = value;
        opts->warm_start = *warm_start;
    }
    else if (!strcmp(field, "print_level"))
    {
        int *print_level = value;
        opts->print_level = *print_level;
        opts->settings.verbose = *print_level > 0;
    }
    else if (!strcmp(field, "proxqp_dense_backend"))
    {
        // only used by the dense interface (full condensing), accepted here so that
        // the ProxQP options can be set the same way for both interfaces
    }
    else if (!strncmp(field, "proxqp_", 7))
    {
        // further ProxSuite settings by their ProxSuite name, see proxqp_c_settings_set
        if (proxqp_c_settings_set(&opts->settings, field + 7, value) != 0)
        {
            printf("\nerror: ocp_qp_proxqp_opts_set: wrong field: %s\n", field);
            exit(1);
        }
    }
    else
    {
        printf("\nerror: ocp_qp_proxqp_opts_set: wrong field: %s\n", field);
        exit(1);
    }

    return;
}


void ocp_qp_proxqp_opts_get(void *config_, void *opts_, const char *field, void *value)
{
    printf("\nerror: ocp_qp_proxqp_opts_get: not implemented for field %s\n", field);
    exit(1);
}



/************************************************
 * memory
 ************************************************/

acados_size_t ocp_qp_proxqp_memory_calculate_size(void *config_, void *dims_, void *opts_)
{
    ocp_qp_dims *dims = dims_;

    size_t n = acados_proxqp_num_vars(dims);
    size_t m_eq = acados_proxqp_num_eq(dims);
    size_t m_in = acados_proxqp_num_ineq(dims);

    size_t P_nnzmax = acados_proxqp_nnzmax_P(dims);
    size_t A_nnzmax = acados_proxqp_nnzmax_A(dims);
    size_t C_nnzmax = acados_proxqp_nnzmax_C(dims);

    size_t nw = acados_proxqp_work_size(dims);

    acados_size_t size = 0;
    size += sizeof(ocp_qp_proxqp_memory);

    size += nw * sizeof(double);        // work_row

    size += 1 * n * sizeof(double);     // g
    size += 1 * m_eq * sizeof(double);  // b
    size += 2 * m_in * sizeof(double);  // l, u

    size += P_nnzmax * sizeof(double);  // P_x
    size += P_nnzmax * sizeof(int);     // P_i
    size += (n + 1) * sizeof(int);      // P_p

    size += A_nnzmax * sizeof(double);  // A_x
    size += A_nnzmax * sizeof(int);     // A_i
    size += (n + 1) * sizeof(int);      // A_p

    size += C_nnzmax * sizeof(double);  // C_x
    size += C_nnzmax * sizeof(int);     // C_i
    size += (n + 1) * sizeof(int);      // C_p

    size += 1 * n * sizeof(double);     // prim_sol
    size += 1 * m_eq * sizeof(double);  // y_sol
    size += 1 * m_in * sizeof(double);  // z_sol
    size += 1 * m_in * sizeof(double);  // constraint_value

    size += acados_proxqp_mask_size_P(dims);   // P_mask
    size += acados_proxqp_mask_size_BA(dims);  // BA_mask
    size += acados_proxqp_mask_size_DC(dims);  // DC_mask

    size += 1 * 8;

    return size;
}



void *ocp_qp_proxqp_memory_assign(void *config_, void *dims_, void *opts_, void *raw_memory)
{
    UNUSED(opts_);

    ocp_qp_dims *dims = dims_;
    ocp_qp_proxqp_memory *mem;

    int n = acados_proxqp_num_vars(dims);
    int m_eq = acados_proxqp_num_eq(dims);
    int m_in = acados_proxqp_num_ineq(dims);
    int P_nnzmax = acados_proxqp_nnzmax_P(dims);
    int A_nnzmax = acados_proxqp_nnzmax_A(dims);
    int C_nnzmax = acados_proxqp_nnzmax_C(dims);
    int nw = acados_proxqp_work_size(dims);

    // char pointer
    char *c_ptr = (char *) raw_memory;

    mem = (ocp_qp_proxqp_memory *) c_ptr;
    c_ptr += sizeof(ocp_qp_proxqp_memory);

    mem->P_nnzmax = P_nnzmax;
    mem->A_nnzmax = A_nnzmax;
    mem->C_nnzmax = C_nnzmax;
    mem->num_rebuilds = 0;
    mem->solution_is_valid = 0;

    align_char_to(8, &c_ptr);

    // doubles
    assign_and_advance_double(nw, &mem->work_row, &c_ptr);

    mem->g = (double *) c_ptr;
    c_ptr += n * sizeof(double);

    mem->b = (double *) c_ptr;
    c_ptr += m_eq * sizeof(double);

    mem->l = (double *) c_ptr;
    c_ptr += m_in * sizeof(double);

    mem->u = (double *) c_ptr;
    c_ptr += m_in * sizeof(double);

    mem->P_x = (double *) c_ptr;
    c_ptr += P_nnzmax * sizeof(double);

    mem->A_x = (double *) c_ptr;
    c_ptr += A_nnzmax * sizeof(double);

    mem->C_x = (double *) c_ptr;
    c_ptr += C_nnzmax * sizeof(double);

    mem->prim_sol = (double *) c_ptr;
    c_ptr += n * sizeof(double);

    mem->y_sol = (double *) c_ptr;
    c_ptr += m_eq * sizeof(double);

    mem->z_sol = (double *) c_ptr;
    c_ptr += m_in * sizeof(double);

    mem->constraint_value = (double *) c_ptr;
    c_ptr += m_in * sizeof(double);

    // ints
    mem->P_i = (int *) c_ptr;
    c_ptr += P_nnzmax * sizeof(int);

    mem->P_p = (int *) c_ptr;
    c_ptr += (n + 1) * sizeof(int);

    mem->A_i = (int *) c_ptr;
    c_ptr += A_nnzmax * sizeof(int);

    mem->A_p = (int *) c_ptr;
    c_ptr += (n + 1) * sizeof(int);

    mem->C_i = (int *) c_ptr;
    c_ptr += C_nnzmax * sizeof(int);

    mem->C_p = (int *) c_ptr;
    c_ptr += (n + 1) * sizeof(int);

    // chars: the pattern starts empty and grows with the data
    assign_and_advance_char(acados_proxqp_mask_size_P(dims), &mem->P_mask, &c_ptr);
    assign_and_advance_char(acados_proxqp_mask_size_BA(dims), &mem->BA_mask, &c_ptr);
    assign_and_advance_char(acados_proxqp_mask_size_DC(dims), &mem->DC_mask, &c_ptr);
    memset(mem->P_mask, 0, acados_proxqp_mask_size_P(dims));
    memset(mem->BA_mask, 0, acados_proxqp_mask_size_BA(dims));
    memset(mem->DC_mask, 0, acados_proxqp_mask_size_DC(dims));

    // the ProxQP object is opaque and cannot be placed in acados-managed memory;
    // it is allocated by proxqp_c_sparse_new at the first call and freed in
    // ocp_qp_proxqp_terminate
    mem->proxqp = NULL;

    assert((char *) raw_memory + ocp_qp_proxqp_memory_calculate_size(config_, dims, opts_) >= c_ptr);

    return mem;
}



void ocp_qp_proxqp_memory_get(void *config_, void *mem_, const char *field, void* value)
{
    ocp_qp_proxqp_memory *mem = mem_;

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
        printf("\nerror: ocp_qp_proxqp_memory_get: field %s not available\n", field);
        exit(1);
    }

    return;
}


void ocp_qp_proxqp_memory_reset(void *config_, void *qp_in_, void *qp_out_, void *opts_, void *mem_, void *work_)
{
    ocp_qp_proxqp_memory *mem = mem_;

    // drop the ProxQP object and the warm start, the solver is recreated at the next call;
    // the sparsity pattern is kept
    if (mem->proxqp != NULL)
    {
        proxqp_c_sparse_free(mem->proxqp);
        mem->proxqp = NULL;
    }
    mem->solution_is_valid = 0;

    return;
}



/************************************************
 * workspace
 ************************************************/

acados_size_t ocp_qp_proxqp_workspace_calculate_size(void *config_, void *dims_, void *opts_)
{
    return 0;
}



/************************************************
 * functions
 ************************************************/

static void fill_in_qp_out(const ocp_qp_in *in, ocp_qp_out *out, ocp_qp_proxqp_memory *mem)
{
    ocp_qp_dims *dims = in->dim;

    int N = dims->N;
    int *nx = dims->nx;
    int *nu = dims->nu;
    int *nb = dims->nb;
    int *ng = dims->ng;
    int *ns = dims->ns;

    int ii, kk, nn;

    int slk_start = 0;
    for (kk = 0; kk <= N; kk++)
    {
        slk_start += nb[kk] + ng[kk] + ns[kk];
    }

    // primal variables
    nn = 0;
    for (kk = 0; kk <= N; kk++)
    {
        blasfeo_pack_dvec(nx[kk]+nu[kk]+2*ns[kk], &mem->prim_sol[nn], 1, out->ux + kk, 0);
        nn += nx[kk] + nu[kk] + 2*ns[kk];
    }

    // dual variables of the dynamics
    nn = 0;
    for (kk = 0; kk < N; kk++)
    {
        blasfeo_pack_dvec(nx[kk + 1], &mem->y_sol[nn], 1, out->pi + kk, 0);
        nn += nx[kk + 1];
    }

    // dual variables of the inequality constraints
    // ProxQP convention: z > 0 for active upper, z < 0 for active lower constraints
    nn = 0;
    for (kk = 0; kk <= N; kk++)
    {
        blasfeo_dvecse(2*nb[kk]+2*ng[kk], 0.0, out->lam+kk, 0);

        int itmp = 0;
        for (ii = 0; ii < nb[kk]+ng[kk]; ii++)
        {
            if (in->idxs_rev[kk][ii]==-1) // not softed: two sided
            {
                double z = mem->z_sol[nn+ii];
                if (z <= 0)
                {
                    BLASFEO_DVECEL(out->lam+kk, ii) = -z;
                }
                else
                {
                    BLASFEO_DVECEL(out->lam+kk, nb[kk]+ng[kk] + ii) = z;
                }
            }
            else // softed: replicated one sided
            {
                BLASFEO_DVECEL(out->lam+kk, ii) = -mem->z_sol[nn+ii];
                BLASFEO_DVECEL(out->lam+kk, nb[kk]+ng[kk] + ii) = mem->z_sol[nn+nb[kk]+ng[kk]+itmp];
                itmp++;
            }
        }
        nn += nb[kk]+ng[kk]+ns[kk];
    }

    // dual variables of the slack nonnegativity constraints
    nn = 0;
    for (kk = 0; kk <= N; kk++)
    {
        blasfeo_pack_dvec(2*ns[kk], &mem->z_sol[slk_start+nn], 1, out->lam+kk, 2*nb[kk]+2*ng[kk]);
        blasfeo_dvecsc(2*ns[kk], -1.0, out->lam+kk, 2*nb[kk]+2*ng[kk]);
        nn += 2*ns[kk];
    }

    // multiply with mask to ensure that multipliers associated with masked constraints are zero
    for (kk = 0; kk <= N; kk++)
    {
        blasfeo_dvecmul(2*(nb[kk]+ng[kk]+ns[kk]), in->d_mask+kk, 0, out->lam+kk, 0, out->lam+kk, 0);
    }
}


void ocp_qp_proxqp_terminate(void *config_, void *mem_, void *work_)
{
    ocp_qp_proxqp_memory *mem = (ocp_qp_proxqp_memory *) mem_;
    if (mem->proxqp != NULL)
        proxqp_c_sparse_free(mem->proxqp);
}



// Removes the multipliers of inequalities which are not active anymore at the previous
// solution, e.g. masked or relaxed bounds, from the warm start; ProxQP could accept the
// previous solution otherwise. Uses C, l and u of the current QP.
static void prune_warm_start_multipliers(int n, int m_in, double tol, ocp_qp_proxqp_memory *mem)
{
    for (int ii = 0; ii < m_in; ii++)
    {
        mem->constraint_value[ii] = 0.0;
    }
    for (int jj = 0; jj < n; jj++)
    {
        for (int kk = mem->C_p[jj]; kk < mem->C_p[jj+1]; kk++)
        {
            mem->constraint_value[mem->C_i[kk]] += mem->C_x[kk] * mem->prim_sol[jj];
        }
    }
    for (int ii = 0; ii < m_in; ii++)
    {
        if (mem->z_sol[ii] > 0.0 && mem->constraint_value[ii] < mem->u[ii] - tol)
        {
            mem->z_sol[ii] = 0.0;
        }
        else if (mem->z_sol[ii] < 0.0 && mem->constraint_value[ii] > mem->l[ii] + tol)
        {
            mem->z_sol[ii] = 0.0;
        }
    }
}



// Creates a ProxQP solver for the structure of the last pack_matrices, replacing a previous one.
static void setup_solver(int n, int m_eq, int m_in, const proxqp_c_settings *settings,
                         const ocp_qp_proxqp_opts *opts, ocp_qp_proxqp_memory *mem)
{
    if (mem->proxqp != NULL)
    {
        proxqp_c_sparse_free(mem->proxqp);
        mem->proxqp = NULL;
    }

    mem->proxqp = proxqp_c_sparse_new(n, m_eq, m_in);
    if (mem->proxqp == NULL)
    {
        printf("\nerror: ocp_qp_proxqp: failed to create ProxQP solver\n");
        exit(1);
    }
    proxqp_c_sparse_apply_settings(mem->proxqp, settings);
    if (proxqp_c_sparse_init(mem->proxqp,
                             mem->P_x, mem->P_i, mem->P_p, mem->g,
                             m_eq > 0 ? mem->A_x : NULL, mem->A_i, mem->A_p,
                             m_eq > 0 ? mem->b : NULL,
                             m_in > 0 ? mem->C_x : NULL, mem->C_i, mem->C_p,
                             m_in > 0 ? mem->l : NULL, m_in > 0 ? mem->u : NULL) != 0)
    {
        printf("\nerror: ocp_qp_proxqp: ProxQP init failed\n");
        exit(1);
    }

    mem->num_rebuilds++;
    if (opts->print_level > 0)
    {
        printf("ocp_qp_proxqp: built solver (#%d), n %d, n_eq %d, n_in %d, nnz(H) %d, nnz(A) %d, nnz(C) %d\n",
               mem->num_rebuilds, n, m_eq, m_in, mem->P_p[n], mem->A_p[n], mem->C_p[n]);
    }
}



int ocp_qp_proxqp(void *config_, void *qp_in_, void *qp_out_, void *opts_, void *mem_, void *work_)
{
    ocp_qp_in *qp_in = qp_in_;
    ocp_qp_out *qp_out = qp_out_;

    qp_info *info = (qp_info *) qp_out->misc;
    acados_timer tot_timer, qp_timer, interface_timer, solver_call_timer;
    acados_tic(&tot_timer);

    // cast data structures
    ocp_qp_proxqp_opts *opts = (ocp_qp_proxqp_opts *) opts_;
    ocp_qp_proxqp_memory *mem = (ocp_qp_proxqp_memory *) mem_;

    int n = acados_proxqp_num_vars(qp_in->dim);
    int m_eq = acados_proxqp_num_eq(qp_in->dim);
    int m_in = acados_proxqp_num_ineq(qp_in->dim);

    acados_tic(&interface_timer);
    int rebuild = mem->proxqp == NULL;
    if (pack_matrices(qp_in, mem, 0) || rebuild)
    {
        // the sparsity pattern grew (or this is the first call)
        pack_matrices(qp_in, mem, 1);
        rebuild = 1;
    }
    update_bounds(qp_in, mem);
    update_gradient(qp_in, mem);

    // never warm start from a failed or non-finite solve: the cold initial guess also resets
    // the proximal parameters, which the warm start would carry over
    int warm_start = opts->warm_start > 0 && mem->solution_is_valid;
    if (warm_start && m_in > 0)
        prune_warm_start_multipliers(n, m_in, opts->settings.eps_abs, mem);

    proxqp_c_settings settings = opts->settings;
    if (rebuild)
    {
        // the rows do not depend on the pattern, so the last solution fits the new solver
        if (warm_start)
            settings.initial_guess = PROXQP_C_WARM_START;
        setup_solver(n, m_eq, m_in, &settings, opts, mem);
        if (warm_start)
            proxqp_c_sparse_set_warm_start(mem->proxqp, mem->prim_sol, mem->y_sol, mem->z_sol);
    }
    else
    {
        // warm starting reuses the previous solution and proximal parameters
        if (warm_start)
            settings.initial_guess = PROXQP_C_WARM_START_WITH_PREVIOUS_RESULT;
        proxqp_c_sparse_apply_settings(mem->proxqp, &settings);
        // matrices whose values did not change are not passed, ProxQP keeps them
        if (proxqp_c_sparse_update(mem->proxqp,
                                   mem->P_changed ? mem->P_x : NULL, mem->P_i, mem->P_p, mem->g,
                                   m_eq > 0 && mem->A_changed ? mem->A_x : NULL, mem->A_i, mem->A_p,
                                   m_eq > 0 ? mem->b : NULL,
                                   m_in > 0 && mem->C_changed ? mem->C_x : NULL, mem->C_i, mem->C_p,
                                   m_in > 0 ? mem->l : NULL, m_in > 0 ? mem->u : NULL) != 0)
        {
            printf("\nerror: ocp_qp_proxqp: ProxQP update failed\n");
            exit(1);
        }
        if (warm_start && m_in > 0)
            proxqp_c_sparse_set_z(mem->proxqp, mem->z_sol);
    }
    info->interface_time = acados_toc(&interface_timer);

    acados_tic(&qp_timer);

    // solve ProxQP
    acados_tic(&solver_call_timer);
    int proxqp_status = proxqp_c_sparse_solve(mem->proxqp);
    mem->time_qp_solver_call = acados_toc(&solver_call_timer);
    mem->iter = proxqp_c_sparse_get_iter(mem->proxqp);

    proxqp_c_sparse_get_solution(mem->proxqp, mem->prim_sol, mem->y_sol, mem->z_sol);
    int finite = all_finite(n, mem->prim_sol) && all_finite(m_eq, mem->y_sol) &&
                 all_finite(m_in, mem->z_sol);
    mem->solution_is_valid = finite && proxqp_status == PROXQP_C_SOLVED;

    // fill qp_out; a non-finite solution is not copied
    if (finite)
    {
        fill_in_qp_out(qp_in, qp_out, mem);
        ocp_qp_compute_t(qp_in, qp_out);
    }

    // info
    info->solve_QP_time = acados_toc(&qp_timer);
    info->total_time = acados_toc(&tot_timer);
    info->num_iter = mem->iter;
    info->t_computed = finite;

    // check exit conditions
    int acados_status;
    switch (proxqp_status)
    {
        case PROXQP_C_SOLVED:
            acados_status = ACADOS_SUCCESS;
            break;
        case PROXQP_C_MAX_ITER_REACHED:
            acados_status = ACADOS_MAXITER;
            break;
        case PROXQP_C_PRIMAL_INFEASIBLE:
        case PROXQP_C_SOLVED_CLOSEST_PRIMAL_FEASIBLE:
            acados_status = ACADOS_INFEASIBLE;
            break;
        case PROXQP_C_DUAL_INFEASIBLE:
            acados_status = ACADOS_UNBOUNDED;
            break;
        default:
            acados_status = ACADOS_QP_FAILURE;
            break;
    }
    if (!finite)
        acados_status = ACADOS_NAN_DETECTED;
    mem->status = acados_status;

    return acados_status;
}



void ocp_qp_proxqp_eval_forw_sens(void *config_, void *qp_in, void *seed, void *qp_out, void *opts_, void *mem_, void *work_)
{
    printf("\nerror: ocp_qp_proxqp_eval_forw_sens: not implemented yet\n");
    exit(1);
}

void ocp_qp_proxqp_eval_adj_sens(void *config_, void *qp_in, void *seed, void *qp_out, void *opts_, void *mem_, void *work_)
{
    printf("\nerror: ocp_qp_proxqp_eval_adj_sens: not implemented yet\n");
    exit(1);
}


void ocp_qp_proxqp_solver_get(void *config_, void *qp_in_, void *qp_out_, void *opts_, void *mem_, const char *field, int stage, void* value, int size1, int size2)
{
    printf("\nerror: ocp_qp_proxqp_solver_get: not implemented yet\n");
    exit(1);
}


void ocp_qp_proxqp_config_initialize_default(void *config_)
{
    qp_solver_config *config = config_;

    config->opts_calculate_size = &ocp_qp_proxqp_opts_calculate_size;
    config->opts_assign = &ocp_qp_proxqp_opts_assign;
    config->opts_initialize_default = &ocp_qp_proxqp_opts_initialize_default;
    config->opts_update = &ocp_qp_proxqp_opts_update;
    config->opts_set = &ocp_qp_proxqp_opts_set;
    config->opts_get = &ocp_qp_proxqp_opts_get;
    config->memory_calculate_size = &ocp_qp_proxqp_memory_calculate_size;
    config->memory_assign = &ocp_qp_proxqp_memory_assign;
    config->memory_get = &ocp_qp_proxqp_memory_get;
    config->workspace_calculate_size = &ocp_qp_proxqp_workspace_calculate_size;
    config->evaluate = &ocp_qp_proxqp;
    config->terminate = &ocp_qp_proxqp_terminate;
    config->eval_forw_sens = &ocp_qp_proxqp_eval_forw_sens;
    config->eval_adj_sens = &ocp_qp_proxqp_eval_adj_sens;
    config->memory_reset = &ocp_qp_proxqp_memory_reset;
    config->solver_get = &ocp_qp_proxqp_solver_get;

    return;
}
