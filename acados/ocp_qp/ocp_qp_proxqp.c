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



static void update_hessian_structure(const ocp_qp_in *in, ocp_qp_proxqp_memory *mem)
{
    ocp_qp_dims *dims = in->dim;

    int N = dims->N;
    int *nx = dims->nx;
    int *nu = dims->nu;
    int *ns = dims->ns;

    int ii, jj, kk;

    // CSC format: P_i are row indices and P_p are column pointers
    int nn = 0, offset = 0, col = 0;
    for (kk = 0; kk <= N; kk++)
    {
        // write RSQ[kk]
        for (jj = 0; jj < nx[kk] + nu[kk]; jj++)
        {
            mem->P_p[col] = nn;
            col++;

            for (ii = 0; ii <= jj; ii++)
            {
                // we write only the upper triangular part
                mem->P_i[nn] = offset + ii;
                nn++;
            }
        }
        offset += nx[kk] + nu[kk];

        // write Z[kk]
        for (jj = 0; jj < 2*ns[kk]; jj++)
        {
            mem->P_p[col] = nn;
            col++;

            // diagonal
            mem->P_i[nn] = offset + jj;
            nn++;
        }

        offset += 2*ns[kk];
    }

    mem->P_p[col] = nn;
}



static void update_hessian_data(const ocp_qp_in *in, ocp_qp_proxqp_memory *mem)
{
    ocp_qp_dims *dims = in->dim;

    int N = dims->N;
    int *nx = dims->nx;
    int *nu = dims->nu;
    int *ns = dims->ns;

    int ii, kk;

    // Traversing the matrix in column-major order
    int nn = 0;
    for (kk = 0; kk <= N; kk++)
    {
        // writing RSQ[kk]
        // we write the lower triangular part in row-major order
        // that's the same as writing the upper triangular part in
        // column-major order
        for (ii = 0; ii < nx[kk] + nu[kk]; ii++)
        {
            blasfeo_unpack_dmat(1, ii+1, in->RSQrq+kk, ii, 0, mem->P_x+nn, 1);
            nn += ii+1;
        }

        // write Z[kk]
        blasfeo_unpack_dvec(2*ns[kk], in->Z+kk, 0, mem->P_x+nn, 1);
        nn += 2*ns[kk];
    }
}



static void update_eq_matrix_structure(const ocp_qp_in *in, ocp_qp_proxqp_memory *mem)
{
    ocp_qp_dims *dims = in->dim;

    int N = dims->N;
    int *nx = dims->nx;
    int *nu = dims->nu;
    int *ns = dims->ns;

    int ii, jj, kk;

    int row_offset_dyn = 0;

    // CSC format: A_i are row indices and A_p are column pointers
    int nn = 0, col = 0;
    for (kk = 0; kk <= N; kk++)
    {
        // control variables
        for (jj = 0; jj < nu[kk]; jj++)
        {
            mem->A_p[col] = nn;
            col++;

            if (kk < N)
            {
                // write column from B
                for (ii = 0; ii < nx[kk + 1]; ii++)
                {
                    mem->A_i[nn] = row_offset_dyn + ii;
                    nn++;
                }
            }
        }

        // state variables
        for (jj = 0; jj < nx[kk]; jj++)
        {
            mem->A_p[col] = nn;
            col++;

            if (kk > 0)
            {
                // write column from -I
                mem->A_i[nn] = row_offset_dyn - nx[kk] + jj;
                nn++;
            }

            if (kk < N)
            {
                // write column from A
                for (ii = 0; ii < nx[kk + 1]; ii++)
                {
                    mem->A_i[nn] = row_offset_dyn + ii;
                    nn++;
                }
            }
        }

        // slack variables, no dynamics entries
        for (jj = 0; jj < 2*ns[kk]; jj++)
        {
            mem->A_p[col] = nn;
            col++;
        }

        row_offset_dyn += kk < N ? nx[kk + 1] : 0;
    }

    // end of matrix
    mem->A_p[col] = nn;
}



static void update_eq_matrix_data(const ocp_qp_in *in, ocp_qp_proxqp_memory *mem)
{
    ocp_qp_dims *dims = in->dim;

    int N = dims->N;
    int *nx = dims->nx;
    int *nu = dims->nu;
    int *ns = dims->ns;

    int jj, kk;

    // Traverse matrix in column-major order
    int nn = 0;
    for (kk = 0; kk <= N; kk++)
    {
        // control variables
        for (jj = 0; jj < nu[kk]; jj++)
        {
            if (kk < N)
            {
                // write column from B
                blasfeo_unpack_dmat(1, nx[kk+1], in->BAbt+kk, jj, 0, mem->A_x+nn, 1);
                nn += nx[kk+1];
            }
        }

        // state variables
        for (jj = 0; jj < nx[kk]; jj++)
        {
            if (kk > 0)
            {
                // write column from -I
                mem->A_x[nn] = -1.0;
                nn++;
            }

            if (kk < N)
            {
                // write column from A
                blasfeo_unpack_dmat(1, nx[kk+1], in->BAbt+kk, nu[kk]+jj, 0, mem->A_x+nn, 1);
                nn += nx[kk+1];
            }
        }

        UNUSED(ns);
    }
}



static void update_ineq_matrix_structure(const ocp_qp_in *in, ocp_qp_proxqp_memory *mem)
{
    ocp_qp_dims *dims = in->dim;

    int N = dims->N;
    int *nx = dims->nx;
    int *nu = dims->nu;
    int *nb = dims->nb;
    int *ng = dims->ng;
    int *ns = dims->ns;

    int ii, jj, kk;

    int row_offset_con = 0;
    int row_offset_slk = 0;

    int slk_start = 0;
    for (kk = 0; kk <= N; kk++)
    {
        slk_start += nb[kk]+ng[kk]+ns[kk];
    }

    // CSC format: C_i are row indices and C_p are column pointers
    int nn = 0, col = 0;
    for (kk = 0; kk <= N; kk++)
    {
        // compute number of softed box constraints
        int nsb = 0;
        for (ii=0; ii<nb[kk]; ii++)
        {
            if (in->idxs_rev[kk][ii]>=0)
            {
                nsb++;
            }
        }

        // control variables
        for (jj = 0; jj < nu[kk]; jj++)
        {
            mem->C_p[col] = nn;
            col++;

            // write bound on u
            for (ii = 0; ii < nb[kk]; ii++)
            {
                if (in->idxb[kk][ii] == jj)
                {
                    mem->C_i[nn] = row_offset_con + ii;
                    nn++;
                    break;
                }
            }
            int idxbu = ii;

            // write column from D
            for (ii = 0; ii < ng[kk]; ii++)
            {
                mem->C_i[nn] = row_offset_con + nb[kk] + ii;
                nn++;
            }

            // replicated softed bound on u
            if (idxbu<nb[kk]) // bounded input
            {
                if (in->idxs_rev[kk][idxbu]>=0) // softed bounded input
                {
                    // compute position in "packed" soft constraints, i.e. it is the itmp-th one
                    int itmp = 0;
                    for (ii=0; ii<idxbu; ii++)
                    {
                        if (in->idxs_rev[kk][ii]>=0)
                        {
                            itmp++;
                        }
                    }
                    mem->C_i[nn] = row_offset_con + nb[kk] + ng[kk] + itmp;
                    nn++;
                }
            }

            // replicated softed D
            {
                int itmp = 0;
                for (ii = 0; ii < ng[kk]; ii++)
                {
                    if (in->idxs_rev[kk][nb[kk]+ii]>=0) // softed
                    {
                        mem->C_i[nn] = row_offset_con + nb[kk] + ng[kk] + nsb + itmp;
                        nn++;
                        itmp++;
                    }
                }
            }

        }

        // state variables
        for (jj = 0; jj < nx[kk]; jj++)
        {
            mem->C_p[col] = nn;
            col++;

            // write bound on x
            for (ii = 0; ii < nb[kk]; ii++)
            {
                if (in->idxb[kk][ii] == nu[kk] + jj)
                {
                    mem->C_i[nn] = row_offset_con + ii;
                    nn++;
                    break;
                }
            }
            int idxbx = ii;

            // write column from C
            for (ii = 0; ii < ng[kk]; ii++)
            {
                mem->C_i[nn] = row_offset_con + nb[kk] + ii;
                nn++;
            }

            // replicated softed bound on x
            if (idxbx<nb[kk]) // bounded input
            {
                if (in->idxs_rev[kk][idxbx]>=0) // softed bounded input
                {
                    // compute position in "packed" soft constraints, i.e. it is the itmp-th one
                    int itmp = 0;
                    for (ii=0; ii<idxbx; ii++)
                    {
                        if (in->idxs_rev[kk][ii]>=0)
                        {
                            itmp++;
                        }
                    }
                    mem->C_i[nn] = row_offset_con + nb[kk] + ng[kk] + itmp;
                    nn++;
                }
            }

            // replicated softed C
            {
                int itmp = 0;
                for (ii = 0; ii < ng[kk]; ii++)
                {
                    if (in->idxs_rev[kk][nb[kk]+ii]>=0) // softed
                    {
                        mem->C_i[nn] = row_offset_con + nb[kk] + ng[kk] + nsb + itmp;
                        nn++;
                        itmp++;
                    }
                }
            }

        }

        // slack variables on lower inequalities (original)
        for (jj = 0; jj < ns[kk]; jj++)
        {
            mem->C_p[col] = nn;
            col++;

            // soft constraint
            for (ii=0; ii<nb[kk]+ng[kk]; ii++)
            {
                if (in->idxs_rev[kk][ii]==jj)
                {
                    mem->C_i[nn] = row_offset_con + ii;
                    nn++;
                    // no break, there could possibly be multiple
                }
            }

            // nonnegativity constraint
            mem->C_i[nn] = slk_start + row_offset_slk + jj;
            nn++;
        }

        // slack variables on upper inequalities (replicated)
        for (jj = 0; jj < ns[kk]; jj++)
        {
            mem->C_p[col] = nn;
            col++;

            // soft constraint
            int itmp = 0;
            for (ii=0; ii<nb[kk]+ng[kk]; ii++)
            {
                if (in->idxs_rev[kk][ii]==jj)
                {
                    mem->C_i[nn] = row_offset_con + nb[kk] + ng[kk] + itmp;
                    nn++;
                    // no break, there could possibly be multiple
                }
                if (in->idxs_rev[kk][ii]>=0)
                {
                    itmp++;
                }
            }

            // nonnegativity constraint
            mem->C_i[nn] = slk_start + row_offset_slk + ns[kk] + jj;
            nn++;
        }

        row_offset_con += nb[kk]+ng[kk]+ns[kk];
        row_offset_slk += 2*ns[kk];
    }

    // end of matrix
    mem->C_p[col] = nn;
}



static void update_ineq_matrix_data(const ocp_qp_in *in, ocp_qp_proxqp_memory *mem)
{
    ocp_qp_dims *dims = in->dim;

    int N = dims->N;
    int *nx = dims->nx;
    int *nu = dims->nu;
    int *nb = dims->nb;
    int *ng = dims->ng;
    int *ns = dims->ns;

    int ii, jj, kk;

    // Traverse matrix in column-major order
    int nn = 0;
    for (kk = 0; kk <= N; kk++)
    {
        // control variables
        for (jj = 0; jj < nu[kk]; jj++)
        {
            // write bound on u
            for (ii = 0; ii < nb[kk]; ii++)
            {
                if (in->idxb[kk][ii] == jj)
                {
                    mem->C_x[nn] = 1.0;
                    nn++;
                    break;
                }
            }
            int idxbu = ii;

            // write column from D
            blasfeo_unpack_dmat(1, ng[kk], in->DCt+kk, jj, 0, mem->C_x+nn, 1);
            nn += ng[kk];

            // replicated softed bound on u
            if (idxbu<nb[kk]) // bounded input
            {
                if (in->idxs_rev[kk][idxbu]>=0) // softed bounded input
                {
                    mem->C_x[nn] = 1.0;
                    nn++;
                }
            }

            // replicated softed D
            for (ii = 0; ii < ng[kk]; ii++)
            {
                if (in->idxs_rev[kk][nb[kk]+ii]>=0) // softed
                {
                    mem->C_x[nn] = BLASFEO_DMATEL(in->DCt+kk, jj, ii);
                    nn++;
                }
            }

        }

        // state variables
        for (jj = 0; jj < nx[kk]; jj++)
        {
            // write bound on x
            for (ii = 0; ii < nb[kk]; ii++)
            {
                if (in->idxb[kk][ii] == nu[kk] + jj)
                {
                    mem->C_x[nn] = 1.0;
                    nn++;
                    break;
                }
            }
            int idxbx = ii;

            // write column from C
            blasfeo_unpack_dmat(1, ng[kk], in->DCt+kk, nu[kk]+jj, 0, mem->C_x+nn, 1);
            nn += ng[kk];

            // replicated softed bound on x
            if (idxbx<nb[kk]) // bounded input
            {
                if (in->idxs_rev[kk][idxbx]>=0) // softed bounded input
                {
                    mem->C_x[nn] = 1.0;
                    nn++;
                }
            }

            // replicated softed C
            for (ii = 0; ii < ng[kk]; ii++)
            {
                if (in->idxs_rev[kk][nb[kk]+ii]>=0) // softed
                {
                    mem->C_x[nn] = BLASFEO_DMATEL(in->DCt+kk, nu[kk]+jj, ii);
                    nn++;
                }
            }

        }

        // slack variables on lower inequalities (original)
        for (jj = 0; jj < ns[kk]; jj++)
        {
            // soft constraint
            for (ii=0; ii<nb[kk]+ng[kk]; ii++)
            {
                if (in->idxs_rev[kk][ii]==jj)
                {
                    mem->C_x[nn] = 1.0;
                    nn++;
                    // no break, there could possibly be multiple
                }
            }

            // nonnegativity constraint
            mem->C_x[nn] = 1.0;
            nn++;
        }

        // slack variables on upper inequalities (replicated)
        for (jj = 0; jj < ns[kk]; jj++)
        {
            // soft constraint
            for (ii=0; ii<nb[kk]+ng[kk]; ii++)
            {
                if (in->idxs_rev[kk][ii]==jj)
                {
                    mem->C_x[nn] = -1.0;
                    nn++;
                    // no break, there could possibly be multiple
                }
            }

            // nonnegativity constraint
            mem->C_x[nn] = 1.0;
            nn++;
        }

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
        // set replicated to -inf
        for (ii=0; ii<ns[kk]; ii++)
        {
            mem->l[nn+nb[kk]+ng[kk]+ii] = -inf;
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
                mem->u[nn + ii] = inf;
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

        // infinity at upper bound
        for (ii = 0; ii < 2*ns[kk]; ii++)
        {
            mem->u[nn + ii] = inf;
        }

        nn += 2*ns[kk];
    }

}



static void ocp_qp_proxqp_update_memory(const ocp_qp_in *in, const ocp_qp_proxqp_opts *opts,
                                        ocp_qp_proxqp_memory *mem)
{
    if (mem->first_run)
    {
        update_hessian_structure(in, mem);
        update_eq_matrix_structure(in, mem);
        update_ineq_matrix_structure(in, mem);
    }

    update_bounds(in, mem);
    update_gradient(in, mem);
    update_hessian_data(in, mem);
    update_eq_matrix_data(in, mem);
    update_ineq_matrix_data(in, mem);
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
    opts->warm_start = 0;
    opts->print_level = 0;

    return;
}



void ocp_qp_proxqp_opts_update(void *config_, void *dims_, void *opts_)
{
    return;
}



void ocp_qp_proxqp_opts_set(void *config_, void *opts_, const char *field, void *value)
{
    ocp_qp_proxqp_opts *opts = opts_;

    if (!strcmp(field, "tol_stat"))
    {
        // the single ProxQP tolerance eps_abs bounds primal and dual residuals;
        // complementarity is only controlled implicitly, so add a safety factor
        double *tol = value;
        opts->settings.eps_abs = MAX(1e-3 * (*tol), 1e-12);
    }
    else if (!strcmp(field, "tol_eq"))
    {
        // covered by eps_abs
    }
    else if (!strcmp(field, "tol_ineq"))
    {
        // covered by eps_abs
    }
    else if (!strcmp(field, "tol_comp"))
    {
        // covered by eps_abs
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

    acados_size_t size = 0;
    size += sizeof(ocp_qp_proxqp_memory);

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

    // char pointer
    char *c_ptr = (char *) raw_memory;

    mem = (ocp_qp_proxqp_memory *) c_ptr;
    c_ptr += sizeof(ocp_qp_proxqp_memory);

    mem->P_nnzmax = P_nnzmax;
    mem->A_nnzmax = A_nnzmax;
    mem->C_nnzmax = C_nnzmax;
    mem->first_run = 1;

    align_char_to(8, &c_ptr);

    // doubles
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

    // drop the ProxQP object, it is recreated at the next call
    if (mem->proxqp != NULL)
    {
        proxqp_c_sparse_free(mem->proxqp);
        mem->proxqp = NULL;
    }
    mem->first_run = 1;

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
}


void ocp_qp_proxqp_terminate(void *config_, void *mem_, void *work_)
{
    ocp_qp_proxqp_memory *mem = (ocp_qp_proxqp_memory *) mem_;
    if (mem->proxqp != NULL)
        proxqp_c_sparse_free(mem->proxqp);
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

    acados_tic(&interface_timer);
    int first_run = mem->first_run;
    ocp_qp_proxqp_update_memory(qp_in, opts, mem);
    mem->first_run = 0;

    int n = acados_proxqp_num_vars(qp_in->dim);
    int m_eq = acados_proxqp_num_eq(qp_in->dim);
    int m_in = acados_proxqp_num_ineq(qp_in->dim);

    // per-call settings, warm starting reuses the previous solution
    proxqp_c_settings settings = opts->settings;
    if (!first_run && opts->warm_start > 0)
        settings.initial_guess = PROXQP_C_WARM_START_WITH_PREVIOUS_RESULT;

    // update proxqp solver with new data
    if (first_run)
    {
        mem->proxqp = proxqp_c_sparse_new(n, m_eq, m_in);
        if (mem->proxqp == NULL)
        {
            printf("\nerror: ocp_qp_proxqp: failed to create ProxQP solver\n");
            exit(1);
        }
        proxqp_c_sparse_apply_settings(mem->proxqp, &settings);
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
    }
    else
    {
        proxqp_c_sparse_apply_settings(mem->proxqp, &settings);
        if (proxqp_c_sparse_update(mem->proxqp,
                                   mem->P_x, mem->P_i, mem->P_p, mem->g,
                                   m_eq > 0 ? mem->A_x : NULL, mem->A_i, mem->A_p,
                                   m_eq > 0 ? mem->b : NULL,
                                   m_in > 0 ? mem->C_x : NULL, mem->C_i, mem->C_p,
                                   m_in > 0 ? mem->l : NULL, m_in > 0 ? mem->u : NULL) != 0)
        {
            printf("\nerror: ocp_qp_proxqp: ProxQP update failed\n");
            exit(1);
        }
    }
    info->interface_time = acados_toc(&interface_timer);

    acados_tic(&qp_timer);

    // solve ProxQP
    acados_tic(&solver_call_timer);
    int proxqp_status = proxqp_c_sparse_solve(mem->proxqp);
    mem->time_qp_solver_call = acados_toc(&solver_call_timer);
    mem->iter = proxqp_c_sparse_get_iter(mem->proxqp);

    proxqp_c_sparse_get_solution(mem->proxqp, mem->prim_sol, mem->y_sol, mem->z_sol);

    // fill qp_out
    fill_in_qp_out(qp_in, qp_out, mem);
    ocp_qp_compute_t(qp_in, qp_out);

    // info
    info->solve_QP_time = acados_toc(&qp_timer);
    info->total_time = acados_toc(&tot_timer);
    info->num_iter = mem->iter;
    info->t_computed = 1;

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
