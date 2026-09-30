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

// clarabel
#include "Clarabel.cpp/include/clarabel.h"

// acados
#include "acados/ocp_qp/ocp_qp_common.h"
#include "acados/ocp_qp/ocp_qp_clarabel.h"
#include "acados/utils/mem.h"
#include "acados/utils/print.h"
#include "acados/utils/timing.h"
#include "acados/utils/types.h"


/************************************************
 * QP layout
 *
 * Variables, per stage: [u; x; s_lower; s_upper], as in ocp_qp_in.
 *
 * Rows of A (A v + s = b, s in cone), in this order:
 *   ZeroCone:        dynamics of all stages, then equality constraints (idxe)
 *   NonnegativeCone: per stage the emitted lower sides, then the emitted upper sides
 *                    of [box; general] constraints; then the slack bounds of all stages
 *
 * ocp_qp_in only holds dense blocks, so the sparsity pattern is detected from the data:
 * the pattern is the union of all nonzeros seen so far. A nonzero outside the pattern,
 * or a bound that becomes finite where it was infinite so far, grows the pattern and
 * rebuilds the Clarabel solver; otherwise the solver data is updated in place.
 * Constraint sides whose bound is infinite (|bound| >= inf_bound, or masked out in
 * d_mask) are not emitted.
 ************************************************/


/************************************************
 * helper functions
 ************************************************/

static void print_csc_as_dns(ClarabelCscMatrix *M)
{
    int i, j = 0; // Predefine row index and column index
    int idx;

    // Initialize matrix of zeros
    double *A = (double *) calloc(M->m * M->n, sizeof(double));
    for (int ii=0; ii<M->m*M->n; ii++)
        A[ii] = 1e30;

    // Allocate elements
    for (idx = 0; idx < M->colptr[M->n]; idx++)
    {
        // Get row index i (starting from 1)
        i = M->rowval[idx];

        // Get column index j (increase if necessary) (starting from 1)
        while (M->colptr[j + 1] <= idx) j++;

        // Assign values to A
        A[j * (M->m) + i] = M->nzval[idx];
    }

    for (i = 0; i < M->m; i++)
    {
        for (j = 0; j < M->n; j++)
        {
            if (A[j * (M->m) + i]==1e30)
                printf("  *      ");
            else
                printf("%8.4f ", A[j * (M->m) + i]);
        }
        printf("\n");
    }

    free(A);
}


void print_csc_matrix(ClarabelCscMatrix *M, const char *name)
{
    int j, i, row_start, row_stop;
    int k = 0;

    // Print name
    printf("%s :\n", name);

    for (j = 0; j < M->n; j++) {
        row_start = M->colptr[j];
        row_stop  = M->colptr[j + 1];

        if (row_start == row_stop) continue;
        else {
        for (i = row_start; i < row_stop; i++) {
            printf("\t[%3u,%3u] = %.3g\n", (int)M->rowval[i], (int)j, M->nzval[k++]);
        }
        }
    }
}


static int acados_clarabel_num_vars(ocp_qp_dims *dims)
{
    int n = 0;

    for (int ii = 0; ii <= dims->N; ii++)
    {
        n += dims->nx[ii] + dims->nu[ii] + 2*dims->ns[ii];
    }

    return n;
}



// upper bound on the number of rows, the emitted rows are counted in build_structure
static int acados_clarabel_num_constr(ocp_qp_dims *dims)
{
    int m = 0;

    for (int ii = 0; ii <= dims->N; ii++)
    {
        m += 2 * dims->nb[ii];
        m += 2 * dims->ng[ii];
        m += 2 * dims->ns[ii];

        // equalities
        if (ii < dims->N)
        {
            m += dims->nx[ii+1];
        }
    }

    return m;
}



static int acados_clarabel_nnzmax_P(const ocp_qp_dims *dims)
{
    int nnz = 0;

    int *nx = dims->nx;
    int *nu = dims->nu;
    int *ns = dims->ns;

    for (int ii = 0; ii <= dims->N; ii++)
    {
        nnz += (nu[ii]+nx[ii])*(nu[ii]+nx[ii]+1)/2; // triu(RSQ)
        nnz += 2*ns[ii]; // Z
    }

    return nnz;
}



static int acados_clarabel_nnzmax_A(const ocp_qp_dims *dims)
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
        // inequality constraints
        nnz += 2*nb[ii];         // eye of box constraints
        nnz += 2*ng[ii]*nx[ii];  // C
        nnz += 2*ng[ii]*nu[ii];  // D
        nnz += 2*(nb[ii]+ng[ii])*2*ns[ii]; // soft constraints at worst case, when idxs_rev encoding is used. Typically just 2*ns
        nnz += 2*ns[ii]; // eye of slacks nonnegativity constraints

        // dynamics equality constraints
        if (ii < dims->N)
        {
            nnz += nx[ii+1] * nx[ii];  // A
            nnz += nx[ii+1] * nu[ii];  // B
            nnz += nx[ii+1];           // eye
        }
    }

    return nnz;
}



// number of flags in the sparsity masks of triu(RSQ), [B A]^T and [D C]^T
static int acados_clarabel_mask_size_P(const ocp_qp_dims *dims)
{
    int size = 0;
    for (int ii = 0; ii <= dims->N; ii++)
    {
        int nv = dims->nu[ii] + dims->nx[ii];
        size += nv*(nv+1)/2;
    }
    return size;
}



static int acados_clarabel_mask_size_BA(const ocp_qp_dims *dims)
{
    int size = 0;
    for (int ii = 0; ii < dims->N; ii++)
    {
        size += (dims->nu[ii] + dims->nx[ii]) * dims->nx[ii+1];
    }
    return size;
}



static int acados_clarabel_mask_size_DC(const ocp_qp_dims *dims)
{
    int size = 0;
    for (int ii = 0; ii <= dims->N; ii++)
    {
        size += (dims->nu[ii] + dims->nx[ii]) * dims->ng[ii];
    }
    return size;
}



// number of two-sided constraints [box; general] over all stages
static int acados_clarabel_num_two_sided(const ocp_qp_dims *dims)
{
    int size = 0;
    for (int ii = 0; ii <= dims->N; ii++)
    {
        size += dims->nb[ii] + dims->ng[ii];
    }
    return size;
}



// length of the unpack buffers: one row of RSQ, BAbt or DCt, or one stage of d
static int acados_clarabel_work_size(const ocp_qp_dims *dims)
{
    int size = 1;
    for (int ii = 0; ii <= dims->N; ii++)
    {
        int nv = dims->nu[ii] + dims->nx[ii];
        int nd = 2*dims->nb[ii] + 2*dims->ng[ii] + 2*dims->ns[ii];
        size = nv > size ? nv : size;
        size = dims->ng[ii] > size ? dims->ng[ii] : size;
        size = nd > size ? nd : size;
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



// Row numbering, cones and CSC structure of P and A from the current sparsity masks.
static void build_structure(const ocp_qp_in *in, ocp_qp_clarabel_memory *mem)
{
    ocp_qp_dims *dims = in->dim;

    int N = dims->N;
    int *nx = dims->nx;
    int *nu = dims->nu;
    int *nb = dims->nb;
    int *ng = dims->ng;
    int *ns = dims->ns;

    int ii, jj, kk, cc;

    // equality flags from idxe
    int off = 0;
    for (kk = 0; kk <= N; kk++)
    {
        int nc = nb[kk] + ng[kk];
        int ne = dims->nbue[kk] + dims->nbxe[kk] + dims->nge[kk];
        memset(mem->is_eq + off, 0, nc);
        for (ii = 0; ii < ne; ii++)
        {
            cc = in->idxe[kk][ii];
            // a softened equality keeps its two inequality rows, its slacks refer to them
            if (!is_soft(in, kk, cc))
                mem->is_eq[off + cc] = 1;
        }
        off += nc;
    }

    // ZeroCone: dynamics, then equality constraints
    int row = 0;
    for (kk = 0; kk < N; kk++)
        row += nx[kk+1];

    off = 0;
    for (kk = 0; kk <= N; kk++)
    {
        int nc = nb[kk] + ng[kk];
        for (cc = 0; cc < nc; cc++)
        {
            if (mem->is_eq[off + cc])
            {
                mem->row_lo[off + cc] = row;
                mem->row_up[off + cc] = row;
                row++;
            }
        }
        off += nc;
    }
    mem->m_eq = row;

    // NonnegativeCone: per stage emitted lower sides, then emitted upper sides
    off = 0;
    for (kk = 0; kk <= N; kk++)
    {
        int nc = nb[kk] + ng[kk];
        char *finite = mem->d_finite + 2*off;
        for (cc = 0; cc < nc; cc++)
        {
            if (!mem->is_eq[off + cc])
                mem->row_lo[off + cc] = finite[cc] ? row++ : -1;
        }
        for (cc = 0; cc < nc; cc++)
        {
            if (!mem->is_eq[off + cc])
                mem->row_up[off + cc] = finite[nc + cc] ? row++ : -1;
        }
        off += nc;
    }

    // slack nonnegativity constraints
    mem->slk_start = row;
    for (kk = 0; kk <= N; kk++)
        row += 2*ns[kk];
    mem->m = row;

    mem->num_cones = 0;
    if (mem->m_eq > 0)
        mem->cones[mem->num_cones++] = ClarabelZeroConeT(mem->m_eq);
    if (mem->m > mem->m_eq)
        mem->cones[mem->num_cones++] = ClarabelNonnegativeConeT(mem->m - mem->m_eq);

    // P: triu(RSQ) restricted to the pattern, and the diagonal Z
    int nn = 0, col = 0, offset = 0;
    char *mask = mem->P_mask;
    for (kk = 0; kk <= N; kk++)
    {
        int nv = nu[kk] + nx[kk];
        for (jj = 0; jj < nv; jj++)
        {
            mem->P_col_ptr[col++] = nn;
            for (ii = 0; ii <= jj; ii++)
            {
                if (mask[ii])
                    mem->P_rowval[nn++] = offset + ii;
            }
            mask += jj + 1;
        }
        offset += nv;

        for (jj = 0; jj < 2*ns[kk]; jj++)
        {
            mem->P_col_ptr[col++] = nn;
            mem->P_rowval[nn++] = offset + jj;
        }
        offset += 2*ns[kk];
    }
    mem->P_col_ptr[col] = nn;
    mem->P_nnz = nn;

    // A: within each column the rows are emitted in increasing order
    nn = 0;
    col = 0;
    off = 0;
    int row_dyn = 0;
    int row_slk = mem->slk_start;
    char *ba_mask = mem->BA_mask;
    char *dc_mask = mem->DC_mask;
    for (kk = 0; kk <= N; kk++)
    {
        int nv = nu[kk] + nx[kk];
        int nc = nb[kk] + ng[kk];
        int nx1 = kk < N ? nx[kk+1] : 0;
        int *row_lo = mem->row_lo + off;
        int *row_up = mem->row_up + off;
        char *is_eq = mem->is_eq + off;

        for (jj = 0; jj < nv; jj++)
        {
            mem->A_col_ptr[col++] = nn;

            // -I of the previous stage's dynamics
            if (kk > 0 && jj >= nu[kk])
                mem->A_rowval[nn++] = row_dyn - nx[kk] + jj - nu[kk];

            // column of [B A]
            for (ii = 0; ii < nx1; ii++)
            {
                if (ba_mask[ii])
                    mem->A_rowval[nn++] = row_dyn + ii;
            }
            ba_mask += nx1;

            int cb = box_of_var(in, kk, jj);
            // equalities, lower sides, upper sides; box before general in each group
            if (cb >= 0 && is_eq[cb])
                mem->A_rowval[nn++] = row_lo[cb];
            for (ii = 0; ii < ng[kk]; ii++)
            {
                if (dc_mask[ii] && is_eq[nb[kk]+ii])
                    mem->A_rowval[nn++] = row_lo[nb[kk]+ii];
            }
            if (cb >= 0 && !is_eq[cb] && row_lo[cb] >= 0)
                mem->A_rowval[nn++] = row_lo[cb];
            for (ii = 0; ii < ng[kk]; ii++)
            {
                if (dc_mask[ii] && !is_eq[nb[kk]+ii] && row_lo[nb[kk]+ii] >= 0)
                    mem->A_rowval[nn++] = row_lo[nb[kk]+ii];
            }
            if (cb >= 0 && !is_eq[cb] && row_up[cb] >= 0)
                mem->A_rowval[nn++] = row_up[cb];
            for (ii = 0; ii < ng[kk]; ii++)
            {
                if (dc_mask[ii] && !is_eq[nb[kk]+ii] && row_up[nb[kk]+ii] >= 0)
                    mem->A_rowval[nn++] = row_up[nb[kk]+ii];
            }
            dc_mask += ng[kk];
        }

        // slack variables on lower inequalities
        for (jj = 0; jj < ns[kk]; jj++)
        {
            mem->A_col_ptr[col++] = nn;
            for (cc = 0; cc < nc; cc++)
            {
                // no break, there could possibly be multiple
                if (in->idxs_rev[kk][cc] == jj)
                    mem->A_rowval[nn++] = row_lo[cc];
            }
            // nonnegativity constraint
            mem->A_rowval[nn++] = row_slk + jj;
        }

        // slack variables on upper inequalities
        for (jj = 0; jj < ns[kk]; jj++)
        {
            mem->A_col_ptr[col++] = nn;
            for (cc = 0; cc < nc; cc++)
            {
                if (in->idxs_rev[kk][cc] == jj)
                    mem->A_rowval[nn++] = row_up[cc];
            }
            mem->A_rowval[nn++] = row_slk + ns[kk] + jj;
        }

        row_dyn += nx1;
        row_slk += 2*ns[kk];
        off += nc;
    }
    mem->A_col_ptr[col] = nn;
    mem->A_nnz = nn;
}



// Writes the nonzeros of P and A, q and b for the structure of the last build_structure.
// Entries outside the pattern are not written: if one of them is nonzero (or a bound
// became finite), the pattern is grown and 1 is returned, and the caller has to rebuild
// the structure and call this again.
static int fill_values(const ocp_qp_in *in, const ocp_qp_clarabel_opts *opts, ocp_qp_clarabel_memory *mem)
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

    // P
    int nn = 0;
    char *mask = mem->P_mask;
    for (kk = 0; kk <= N; kk++)
    {
        // row ii of the lower triangle of RSQ is column ii of the upper triangle
        for (jj = 0; jj < nu[kk] + nx[kk]; jj++)
        {
            blasfeo_unpack_dmat(1, jj+1, in->RSQrq+kk, jj, 0, w, 1);
            for (ii = 0; ii <= jj; ii++)
            {
                if (mask[ii])
                    mem->P_nzval[nn++] = w[ii];
                else if (w[ii] != 0.0)
                {
                    mask[ii] = 1;
                    grew = 1;
                }
            }
            mask += jj + 1;
        }

        blasfeo_unpack_dvec(2*ns[kk], in->Z+kk, 0, mem->P_nzval+nn, 1);
        nn += 2*ns[kk];
    }

    // q
    nn = 0;
    for (kk = 0; kk <= N; kk++)
    {
        blasfeo_unpack_dvec(nu[kk]+nx[kk]+2*ns[kk], in->rqz + kk, 0, &mem->q[nn], 1);
        nn += nu[kk]+nx[kk]+2*ns[kk];
    }
    mem->q_nnz = nn;

    // A
    nn = 0;
    int off = 0;
    char *ba_mask = mem->BA_mask;
    char *dc_mask = mem->DC_mask;
    for (kk = 0; kk <= N; kk++)
    {
        int nc = nb[kk] + ng[kk];
        int nx1 = kk < N ? nx[kk+1] : 0;
        int *row_lo = mem->row_lo + off;
        int *row_up = mem->row_up + off;
        char *is_eq = mem->is_eq + off;

        for (jj = 0; jj < nu[kk] + nx[kk]; jj++)
        {
            if (kk > 0 && jj >= nu[kk])
                mem->A_nzval[nn++] = -1.0;

            if (kk < N)
            {
                blasfeo_unpack_dmat(1, nx1, in->BAbt+kk, jj, 0, w, 1);
                for (ii = 0; ii < nx1; ii++)
                {
                    if (ba_mask[ii])
                        mem->A_nzval[nn++] = w[ii];
                    else if (w[ii] != 0.0)
                    {
                        ba_mask[ii] = 1;
                        grew = 1;
                    }
                }
            }
            ba_mask += nx1;

            blasfeo_unpack_dmat(1, ng[kk], in->DCt+kk, jj, 0, w, 1);
            int cb = box_of_var(in, kk, jj);
            // same order as in build_structure; lower sides are negated
            if (cb >= 0 && is_eq[cb])
                mem->A_nzval[nn++] = 1.0;
            for (ii = 0; ii < ng[kk]; ii++)
            {
                if (dc_mask[ii] && is_eq[nb[kk]+ii])
                    mem->A_nzval[nn++] = w[ii];
            }
            if (cb >= 0 && !is_eq[cb] && row_lo[cb] >= 0)
                mem->A_nzval[nn++] = -1.0;
            for (ii = 0; ii < ng[kk]; ii++)
            {
                if (dc_mask[ii] && !is_eq[nb[kk]+ii] && row_lo[nb[kk]+ii] >= 0)
                    mem->A_nzval[nn++] = -w[ii];
            }
            if (cb >= 0 && !is_eq[cb] && row_up[cb] >= 0)
                mem->A_nzval[nn++] = 1.0;
            for (ii = 0; ii < ng[kk]; ii++)
            {
                if (dc_mask[ii] && !is_eq[nb[kk]+ii] && row_up[nb[kk]+ii] >= 0)
                    mem->A_nzval[nn++] = w[ii];
            }
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

        // slack variables on lower, then upper inequalities
        for (jj = 0; jj < 2*ns[kk]; jj++)
        {
            for (cc = 0; cc < nc; cc++)
            {
                if (in->idxs_rev[kk][cc] == jj % ns[kk])
                    mem->A_nzval[nn++] = -1.0;
            }
            // nonnegativity constraint
            mem->A_nzval[nn++] = -1.0;
        }

        off += nc;
    }

    // b: dynamics
    nn = 0;
    for (kk = 0; kk < N; kk++)
    {
        blasfeo_unpack_dvec(nx[kk+1], in->b+kk, 0, &mem->b[nn], 1);
        for (ii = 0; ii < nx[kk+1]; ii++)
            mem->b[nn+ii] = -mem->b[nn+ii];
        nn += nx[kk+1];
    }

    // b: constraints. d holds [lb lg -ub -ug ls us], and every row is written as a v <= b
    off = 0;
    int row_slk = mem->slk_start;
    double *d = mem->work_d;
    double *d_mask = mem->work_d_mask;
    for (kk = 0; kk <= N; kk++)
    {
        int nc = nb[kk] + ng[kk];
        int *row_lo = mem->row_lo + off;
        int *row_up = mem->row_up + off;
        char *is_eq = mem->is_eq + off;
        char *finite = mem->d_finite + 2*off;

        blasfeo_unpack_dvec(2*nc+2*ns[kk], in->d+kk, 0, d, 1);
        blasfeo_unpack_dvec(2*nc+2*ns[kk], in->d_mask+kk, 0, d_mask, 1);

        for (cc = 0; cc < nc; cc++)
        {
            if (is_eq[cc])
            {
                if (row_lo[cc] >= 0)
                    mem->b[row_lo[cc]] = 0.5 * (d[cc] - d[nc+cc]);
                continue;
            }

            int soft = is_soft(in, kk, cc);
            int finite_lo = soft || (d_mask[cc] != 0.0 && fabs(d[cc]) < opts->inf_bound);
            int finite_up = soft || (d_mask[nc+cc] != 0.0 && fabs(d[nc+cc]) < opts->inf_bound);
            if (finite_lo && !finite[cc])
            {
                finite[cc] = 1;
                grew = 1;
            }
            if (finite_up && !finite[nc+cc])
            {
                finite[nc+cc] = 1;
                grew = 1;
            }

            if (row_lo[cc] >= 0)
                mem->b[row_lo[cc]] = -d[cc];
            if (row_up[cc] >= 0)
                mem->b[row_up[cc]] = -d[nc+cc];
        }

        // slack bounds
        if (mem->m > 0)
        {
            for (ii = 0; ii < 2*ns[kk]; ii++)
                mem->b[row_slk + ii] = -d[2*nc + ii];
        }
        row_slk += 2*ns[kk];

        off += nc;
    }
    mem->b_nnz = mem->m;

    return grew;
}



static void clarabel_init_data(ocp_qp_clarabel_memory* mem, ocp_qp_in *qp_in)
{
    int n = acados_clarabel_num_vars(qp_in->dim);

    // wraps the csc arrays in memory, the solver copies the data on construction
    clarabel_CscMatrix_init(&mem->A, mem->m, n, mem->A_col_ptr, mem->A_rowval, mem->A_nzval);
    //print_csc_matrix(&mem->A, "A_mat");
    //print_csc_as_dns(&mem->A);

    clarabel_CscMatrix_init(&mem->P, n, n, mem->P_col_ptr, mem->P_rowval, mem->P_nzval);
    //print_csc_matrix(&mem->P, "P_mat");
    //print_csc_as_dns(&mem->P);
}



/************************************************
 * opts
 ************************************************/

acados_size_t ocp_qp_clarabel_opts_calculate_size(void *config_, void *dims_)
{
    acados_size_t size = 0;
    size += sizeof(ocp_qp_clarabel_opts);
    size += sizeof(ClarabelDefaultSettings);

    return size;
}



void *ocp_qp_clarabel_opts_assign(void *config_, void *dims_, void *raw_memory)
{
    ocp_qp_clarabel_opts *opts;

    char *c_ptr = (char *) raw_memory;

    opts = (ocp_qp_clarabel_opts *) c_ptr;
    c_ptr += sizeof(ocp_qp_clarabel_opts);

     opts->clarabel_opts = (ClarabelDefaultSettings *) c_ptr;
     c_ptr += sizeof(ClarabelDefaultSettings);

    assert((char *) raw_memory + ocp_qp_clarabel_opts_calculate_size(config_, dims_) == c_ptr);

    return (void *) opts;
}



void ocp_qp_clarabel_opts_initialize_default(void *config_, void *dims_, void *opts_)
{
    ocp_qp_clarabel_opts *opts = opts_;
    opts->print_level = 0;

    *opts->clarabel_opts = clarabel_DefaultSettings_default();
    opts->clarabel_opts->verbose = false;
    // presolve would forbid the data updates between solves
    opts->clarabel_opts->presolve_enable = false;

    opts->inf_bound = ACADOS_INFTY;

    opts->tol_stat = -1.0;
    opts->tol_eq = -1.0;
    opts->tol_ineq = -1.0;
    opts->tol_comp = -1.0;
    opts->tol_dual_gap = -1.0;

    opts->first_run = 1;

    return;
}



void ocp_qp_clarabel_opts_update(void *config_, void *dims_, void *opts_)
{
    // ocp_qp_clarabel_opts *opts = (ocp_qp_clarabel_opts *)opts_;

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



// Clarabel checks the relative primal and dual residuals against tol_feas and the gap
// against tol_gap_abs or tol_gap_rel, so the acados tolerances map onto these three.
static void apply_requested_tolerances(ocp_qp_clarabel_opts *opts)
{
    double tol_feas = min_requested_tol(opts->tol_stat, min_requested_tol(opts->tol_eq, opts->tol_ineq));
    double tol_gap = min_requested_tol(opts->tol_comp, opts->tol_dual_gap);

    if (tol_feas > 0.0)
        opts->clarabel_opts->tol_feas = tol_feas;
    if (tol_gap > 0.0)
    {
        opts->clarabel_opts->tol_gap_abs = tol_gap;
        opts->clarabel_opts->tol_gap_rel = tol_gap;
    }
}



void ocp_qp_clarabel_opts_set(void *config_, void *opts_, const char *field, void *value)
{
    ocp_qp_clarabel_opts *opts = opts_;

    // Updating options through this function does not work, only before the first call!
    if (!opts->first_run)
    {
#ifndef ACADOS_SILENT
        printf("\nWARNING: ocp_qp_clarabel_opts_set: attempting to set field: %s. However, options cannot be changed after first run, option is NOT updated. \n", field);
#endif
        return;
    }

    if (!strcmp(field, "iter_max"))
    {
        int *tmp_ptr = value;
        opts->clarabel_opts->max_iter = *tmp_ptr;
    }
    else if (!strcmp(field, "print_level"))
    {
        int* print_level = (int *) value;
        opts->print_level = *print_level;
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
        opts->tol_comp = *(double *) value;
        apply_requested_tolerances(opts);
    }
    else if (!strcmp(field, "tol_dual_gap"))
    {
        opts->tol_dual_gap = *(double *) value;
        apply_requested_tolerances(opts);
    }
    else if (!strcmp(field, "warm_start"))
    {
        // do nothing, Clarabel does not warm start
    }
    else if(!strcmp(field, "clarabel_mode"))
    {
        char *mode = value;
        if(!strcmp(mode, "REDUCED_ACCURACY"))
        {
            // Clarabel's "almost solved" tolerances as the regular stopping criterion
            opts->clarabel_opts->tol_gap_abs = 5e-5;
            opts->clarabel_opts->tol_gap_rel = 5e-5;
            opts->clarabel_opts->tol_feas = 1e-4;
            opts->clarabel_opts->tol_ktratio = 1e-4;
        }
        else if(!strcmp(mode, "FULL_ACCURACY"))
        {
            // Clarabel defaults
            opts->clarabel_opts->tol_gap_abs = 1e-8;
            opts->clarabel_opts->tol_gap_rel = 1e-8;
            opts->clarabel_opts->tol_feas = 1e-8;
            opts->clarabel_opts->tol_ktratio = 1e-6;
        }
        else
        {
            printf("\nWARNING: ocp_qp_clarabel_opts_set: clarabel_mode: %s is unknown interfaced yet. Ignoring option and \n", mode);
            exit(1);
        }
        // explicitly requested tolerances take precedence over the preset
        apply_requested_tolerances(opts);
    }
    else if (!strcmp(field, "clarabel_inf_bound"))
    {
        opts->inf_bound = *(double *) value;
    }
    else if (!strcmp(field, "clarabel_iter_ref_enable"))
    {
        opts->clarabel_opts->iterative_refinement_enable = *(int *) value != 0;
    }
    else if (!strcmp(field, "clarabel_iter_ref_max_iter"))
    {
        opts->clarabel_opts->iterative_refinement_max_iter = *(int *) value;
    }
    else if (!strcmp(field, "clarabel_iter_ref_reltol"))
    {
        opts->clarabel_opts->iterative_refinement_reltol = *(double *) value;
    }
    else if (!strcmp(field, "clarabel_iter_ref_abstol"))
    {
        opts->clarabel_opts->iterative_refinement_abstol = *(double *) value;
    }
    else if (!strcmp(field, "clarabel_equilibrate_enable"))
    {
        opts->clarabel_opts->equilibrate_enable = *(int *) value != 0;
    }
    else if (!strcmp(field, "clarabel_max_step_fraction"))
    {
        opts->clarabel_opts->max_step_fraction = *(double *) value;
    }
    else
    {
        printf("\nWARNING: ocp_qp_clarabel_opts_set: field: %s not interfaced yet. Ignoring option and \n", field);
        exit(1);
    }

    return;
}

void ocp_qp_clarabel_opts_get(void *config_, void *opts_, const char *field, void *value)
{
    // ocp_qp_clarabel_opts *opts = opts_;
    printf("\nerror: ocp_qp_clarabel_opts_get: not implemented for field %s\n", field);
    exit(1);
}




/************************************************
 * memory
 ************************************************/

acados_size_t ocp_qp_clarabel_memory_calculate_size(void *config_, void *dims_, void *opts_)
{
    ocp_qp_dims *dims = dims_;

    size_t n = acados_clarabel_num_vars(dims);
    size_t m = acados_clarabel_num_constr(dims);

    size_t A_nnzmax = acados_clarabel_nnzmax_A(dims);
    size_t P_nnzmax = acados_clarabel_nnzmax_P(dims);

    size_t nc = acados_clarabel_num_two_sided(dims);
    size_t nw = acados_clarabel_work_size(dims);

    acados_size_t size = 0;
    size += sizeof(ocp_qp_clarabel_memory);

    size += A_nnzmax * sizeof(ClarabelFloat);  // A_nzval
    size += A_nnzmax * sizeof(uintptr_t);      // A_rowval
    size += (n + 1) * sizeof(uintptr_t);       // A_col_ptr

    size += P_nnzmax * sizeof(ClarabelFloat);  // A_nzval
    size += P_nnzmax * sizeof(uintptr_t);      // P_rowval
    size += (n + 1) * sizeof(uintptr_t);       // P_col_ptr

    size += n * sizeof(ClarabelFloat);  // q
    size += m * sizeof(ClarabelFloat);  // b

    size += 4 * nw * sizeof(double);    // work_row, work_d, work_d_mask, work_lam

    size += 2 * nc * sizeof(int);       // row_lo, row_up

    size += acados_clarabel_mask_size_P(dims);   // P_mask
    size += acados_clarabel_mask_size_BA(dims);  // BA_mask
    size += acados_clarabel_mask_size_DC(dims);  // DC_mask
    size += 2 * nc;                              // d_finite
    size += nc;                                  // is_eq

    size += 2 * 8;

    return size;
}




void *ocp_qp_clarabel_memory_assign(void *config_, void *dims_, void *opts_, void *raw_memory)
{
    ocp_qp_dims *dims = dims_;
    ocp_qp_clarabel_memory *mem;

    int n = acados_clarabel_num_vars(dims);
    int m = acados_clarabel_num_constr(dims);
    int P_nnzmax = acados_clarabel_nnzmax_P(dims);
    int A_nnzmax = acados_clarabel_nnzmax_A(dims);
    int nc = acados_clarabel_num_two_sided(dims);
    int nw = acados_clarabel_work_size(dims);

    // char pointer
    char *c_ptr = (char *) raw_memory;

    mem = (ocp_qp_clarabel_memory *) c_ptr;
    c_ptr += sizeof(ocp_qp_clarabel_memory);

    mem->P_nnzmax = P_nnzmax;
    mem->A_nnzmax = A_nnzmax;

    mem->solver = NULL;
    mem->m = 0;
    mem->m_eq = 0;
    mem->slk_start = 0;
    mem->num_cones = 0;
    mem->num_rebuilds = 0;
    mem->clarabel_solve_time = 0.0;

    align_char_to(8, &c_ptr);

    // doubles
    mem->q = (ClarabelFloat *) c_ptr;
    c_ptr += n * sizeof(ClarabelFloat);

    mem->b = (ClarabelFloat *) c_ptr;
    c_ptr += m * sizeof(ClarabelFloat);

    mem->P_nzval = (ClarabelFloat *) c_ptr;
    c_ptr += (mem->P_nnzmax) * sizeof(ClarabelFloat);

    mem->A_nzval = (ClarabelFloat *) c_ptr;
    c_ptr += (mem->A_nnzmax) * sizeof(ClarabelFloat);

    assign_and_advance_double(nw, &mem->work_row, &c_ptr);
    assign_and_advance_double(nw, &mem->work_d, &c_ptr);
    assign_and_advance_double(nw, &mem->work_d_mask, &c_ptr);
    assign_and_advance_double(nw, &mem->work_lam, &c_ptr);

    // ints
    mem->P_rowval = (uintptr_t *) c_ptr;
    c_ptr += (mem->P_nnzmax) * sizeof(uintptr_t);

    mem->P_col_ptr = (uintptr_t *) c_ptr;
    c_ptr += (n + 1) * sizeof(uintptr_t);

    mem->A_rowval = (uintptr_t *) c_ptr;
    c_ptr += (mem->A_nnzmax) * sizeof(uintptr_t);

    mem->A_col_ptr = (uintptr_t *) c_ptr;
    c_ptr += (n + 1) * sizeof(uintptr_t);

    assign_and_advance_int(nc, &mem->row_lo, &c_ptr);
    assign_and_advance_int(nc, &mem->row_up, &c_ptr);
    for (int ii = 0; ii < nc; ii++)
    {
        mem->row_lo[ii] = -1;
        mem->row_up[ii] = -1;
    }

    // chars: the pattern starts empty and grows with the data
    assign_and_advance_char(acados_clarabel_mask_size_P(dims), &mem->P_mask, &c_ptr);
    assign_and_advance_char(acados_clarabel_mask_size_BA(dims), &mem->BA_mask, &c_ptr);
    assign_and_advance_char(acados_clarabel_mask_size_DC(dims), &mem->DC_mask, &c_ptr);
    assign_and_advance_char(2*nc, &mem->d_finite, &c_ptr);
    assign_and_advance_char(nc, &mem->is_eq, &c_ptr);
    memset(mem->P_mask, 0, acados_clarabel_mask_size_P(dims));
    memset(mem->BA_mask, 0, acados_clarabel_mask_size_BA(dims));
    memset(mem->DC_mask, 0, acados_clarabel_mask_size_DC(dims));
    memset(mem->d_finite, 0, 2*nc);
    memset(mem->is_eq, 0, nc);

    assert((char *) raw_memory + ocp_qp_clarabel_memory_calculate_size(config_, dims, opts_) >= c_ptr);

    return mem;
}



void ocp_qp_clarabel_memory_get(void *config_, void *mem_, const char *field, void* value)
{
    // qp_solver_config *config = config_;
    ocp_qp_clarabel_memory *mem = mem_;

    if (!strcmp(field, "time_qp_solver_call"))
    {
        double *tmp_ptr = value;
        *tmp_ptr = mem->time_qp_solver_call;
    }
    else if (!strcmp(field, "clarabel_solve_time"))
    {
        double *tmp_ptr = value;
        *tmp_ptr = mem->clarabel_solve_time;
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
        printf("\nerror: ocp_qp_clarabel_memory_get: field %s not available\n", field);
        exit(1);
    }

    return;

}


void ocp_qp_clarabel_memory_reset(void *config_, void *qp_in_, void *qp_out_, void *opts_, void *mem_, void *work_)
{
    // ocp_qp_in *qp_in = qp_in_;
    // reset memory
    printf("acados: reset clarabel_mem not implemented.\n");
    exit(1);
}



/************************************************
 * workspace
 ************************************************/

acados_size_t ocp_qp_clarabel_workspace_calculate_size(void *config_, void *dims_, void *opts_)
{
    return 0;
}



/************************************************
 * functions
 ************************************************/

static void fill_in_qp_out(const ocp_qp_in *in, ocp_qp_out *out, ocp_qp_clarabel_memory *mem)
{
    ocp_qp_dims *dims = in->dim;

    int N = dims->N;
    int *nx = dims->nx;
    int *nu = dims->nu;
    int *nb = dims->nb;
    int *ng = dims->ng;
    int *ns = dims->ns;

    int kk, cc, nn;

    ClarabelDefaultSolution *sol = &mem->solution;

    // primal variables
    nn = 0;
    for (kk = 0; kk <= N; kk++)
    {
        blasfeo_pack_dvec(nx[kk]+nu[kk]+2*ns[kk], &sol->x[nn], 1, out->ux + kk, 0);
        nn += nx[kk] + nu[kk] + 2*ns[kk];
    }

    // dual variables of the dynamics
    nn = 0;
    for (kk = 0; kk < N; kk++)
    {
        blasfeo_pack_dvec(nx[kk + 1], &sol->z[nn], 1, out->pi + kk, 0);
        nn += nx[kk + 1];
    }

    // [lam_lb lam_lg lam_ub lam_ug]: sides that were not emitted are inactive, the
    // multiplier of an equality row is split by its sign
    int off = 0;
    double *lam = mem->work_lam;
    for (kk = 0; kk <= N; kk++)
    {
        int nc = nb[kk] + ng[kk];
        int *row_lo = mem->row_lo + off;
        int *row_up = mem->row_up + off;
        char *is_eq = mem->is_eq + off;

        for (cc = 0; cc < nc; cc++)
        {
            if (is_eq[cc])
            {
                double z = sol->z[row_lo[cc]];
                lam[cc] = z < 0.0 ? -z : 0.0;
                lam[nc+cc] = z > 0.0 ? z : 0.0;
            }
            else
            {
                lam[cc] = row_lo[cc] >= 0 ? sol->z[row_lo[cc]] : 0.0;
                lam[nc+cc] = row_up[cc] >= 0 ? sol->z[row_up[cc]] : 0.0;
            }
        }
        blasfeo_pack_dvec(2*nc, lam, 1, out->lam+kk, 0);
        off += nc;
    }

    // [lam_ls lam_us]
    nn = mem->slk_start;
    for (kk = 0; kk <= N; kk++)
    {
        blasfeo_pack_dvec(2*ns[kk], &sol->z[nn], 1, out->lam+kk, 2*nb[kk]+2*ng[kk]);
        nn += 2*ns[kk];
    }
}



// clarabel f64 printing stuff
static void print_array_double(double *array, size_t n)
{
    printf("[");
    for (size_t i = 0; i < n; i++)
    {
        printf("%.10f", array[i]);
        if (i < n - 1)
        {
            printf(", ");
        }
    }
    printf("]\n");
}

static void print_solution(ClarabelDefaultSolution_f64 *solution)
{
    printf("Solution (x)\t = ");
    print_array_double(solution->x, solution->x_length);
    printf("Multipliers (z)\t = ");
    print_array_double(solution->z, solution->z_length);
    printf("Slacks (s)\t = ");
    print_array_double(solution->s, solution->s_length);
}

int ocp_qp_clarabel(void *config_, void *qp_in_, void *qp_out_, void *opts_, void *mem_, void *work_)
{
    ocp_qp_in *qp_in = qp_in_;
    ocp_qp_out *qp_out = qp_out_;

    // print_ocp_qp_in(qp_in);

    qp_info *info = (qp_info *) qp_out->misc;
    acados_timer tot_timer, qp_timer, interface_timer, solver_call_timer;
    acados_tic(&tot_timer);

    // cast data structures
    ocp_qp_clarabel_opts *opts = (ocp_qp_clarabel_opts *) opts_;
    ocp_qp_clarabel_memory *mem = (ocp_qp_clarabel_memory *) mem_;

    acados_tic(&interface_timer);
    int rebuild = mem->solver == NULL;
    if (fill_values(qp_in, opts, mem) || rebuild)
    {
        // the sparsity pattern or the set of finite bounds grew (or this is the first call)
        build_structure(qp_in, mem);
        fill_values(qp_in, opts, mem);
        rebuild = 1;
    }
    info->interface_time = acados_toc(&interface_timer);

    acados_tic(&qp_timer);

    if (rebuild)
    {
        if (mem->solver != NULL)
        {
            clarabel_DefaultSolver_free(mem->solver);
        }
        clarabel_init_data(mem, qp_in);
        mem->solver = clarabel_DefaultSolver_new(&mem->P, mem->q, &mem->A, mem->b,
                                                 mem->num_cones, mem->cones, opts->clarabel_opts);
        mem->num_rebuilds++;
        opts->first_run = 0;
        if (opts->print_level > 0)
        {
            printf("ocp_qp_clarabel: built solver (#%d), n %d, m %d (%d equalities), nnz(P) %d, nnz(A) %d\n",
                   mem->num_rebuilds, acados_clarabel_num_vars(qp_in->dim), mem->m, mem->m_eq,
                   (int) mem->P_nnz, (int) mem->A_nnz);
        }
    }
    else
    {
        clarabel_DefaultSolver_update_P(mem->solver, mem->P_nzval, mem->P_nnz);
        clarabel_DefaultSolver_update_A(mem->solver, mem->A_nzval, mem->A_nnz);
        clarabel_DefaultSolver_update_q(mem->solver, mem->q, mem->q_nnz);
        clarabel_DefaultSolver_update_b(mem->solver, mem->b, mem->b_nnz);
    }

    // solve Clarabel
    acados_tic(&solver_call_timer);
    //clarabel_DefaultSolver_print_to_file(mem->solver, "clarabel_data.txt");
    //clarabel_DefaultSolver_save_to_file(mem->solver, "clarabel_data.txt");
    clarabel_DefaultSolver_solve(mem->solver);
    mem->time_qp_solver_call = acados_toc(&solver_call_timer);


    // Get solution
    mem->solution = clarabel_DefaultSolver_solution(mem->solver);
    if (opts->print_level > 1)
    {
        print_solution(&mem->solution);
    }

    /* fill qp_out */
    fill_in_qp_out(qp_in, qp_out, mem);
    ocp_qp_compute_t(qp_in, qp_out);

    //d_ocp_qp_sol_print(qp_in->dim, qp_out);

    // info
    ClarabelDefaultInfo clarabel_info = clarabel_DefaultSolver_info(mem->solver);
    info->solve_QP_time = acados_toc(&qp_timer);
    info->total_time = acados_toc(&tot_timer);
    info->num_iter = clarabel_info.iterations;
    mem->iter = clarabel_info.iterations;
    mem->clarabel_solve_time = clarabel_info.solve_time;

    // status: "almost solved" satisfies Clarabel's reduced_tol_* tolerances
    int acados_status = ACADOS_QP_FAILURE; // generic QP failure
    ClarabelSolverStatus clarabel_status = mem->solution.status;
    mem->status = (int) clarabel_status;
    if (clarabel_status==ClarabelSolved || clarabel_status==ClarabelAlmostSolved)
        acados_status = ACADOS_SUCCESS;
    else if (clarabel_status==ClarabelMaxIterations)
        acados_status = ACADOS_MAXITER;

    return acados_status;
}



void ocp_qp_clarabel_eval_adj_sens(void *config_, void *param_qp_in_, void *seed, void *sens_qp_out_, void *opts_, void *mem_, void *work_)
{
    printf("\nerror: ocp_qp_clarabel_eval_adj_sens: not implemented yet\n");
    exit(1);
}

void ocp_qp_clarabel_eval_forw_sens(void *config_, void *param_qp_in_, void *seed, void *sens_qp_out_, void *opts_, void *mem_, void *work_)
{
    printf("\nerror: ocp_qp_clarabel_eval_forw_sens: not implemented yet\n");
    exit(1);
}

void ocp_qp_clarabel_solver_get(void *config_, void *qp_in_, void *qp_out_, void *opts_, void *mem_, const char *field, int stage, void* value, int size1, int size2)
{
    printf("\nerror: ocp_qp_clarabel_solver_get: not implemented yet\n");
    exit(1);
}


void ocp_qp_clarabel_terminate(void *config_, void *mem_, void *work_)
{
    ocp_qp_clarabel_memory *mem = (ocp_qp_clarabel_memory *) mem_;
    // Free the matrices and the solver
    if (mem->solver != NULL)
    {
        clarabel_DefaultSolver_free(mem->solver);
        mem->solver = NULL;
    }
}




void ocp_qp_clarabel_config_initialize_default(void *config_)
{
    qp_solver_config *config = config_;

    config->opts_calculate_size = &ocp_qp_clarabel_opts_calculate_size;
    config->opts_assign = &ocp_qp_clarabel_opts_assign;
    config->opts_initialize_default = &ocp_qp_clarabel_opts_initialize_default;
    config->opts_update = &ocp_qp_clarabel_opts_update;
    config->opts_set = &ocp_qp_clarabel_opts_set;
    config->opts_get = &ocp_qp_clarabel_opts_get;
    config->memory_calculate_size = &ocp_qp_clarabel_memory_calculate_size;
    config->memory_assign = &ocp_qp_clarabel_memory_assign;
    config->memory_get = &ocp_qp_clarabel_memory_get;
    config->workspace_calculate_size = &ocp_qp_clarabel_workspace_calculate_size;
    config->evaluate = &ocp_qp_clarabel;
    config->terminate = &ocp_qp_clarabel_terminate;
    config->eval_forw_sens = &ocp_qp_clarabel_eval_forw_sens;
    config->eval_adj_sens = &ocp_qp_clarabel_eval_adj_sens;
    config->memory_reset = &ocp_qp_clarabel_memory_reset;
    config->solver_get = &ocp_qp_clarabel_solver_get;

    return;
}
