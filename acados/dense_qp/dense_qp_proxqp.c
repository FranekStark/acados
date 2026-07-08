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

// external
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// blasfeo
#include "blasfeo_d_aux.h"
#include "blasfeo_d_blasfeo_api.h"

// acados
#include "acados/dense_qp/dense_qp_proxqp.h"
#include "acados/dense_qp/dense_qp_common.h"
#include "acados/utils/math.h"
#include "acados/utils/mem.h"
#include "acados/utils/print.h"
#include "acados/utils/timing.h"
#include "acados/utils/types.h"

/************************************************
 * opts
 ************************************************/

acados_size_t dense_qp_proxqp_opts_calculate_size(void *config_, dense_qp_dims *dims)
{
    acados_size_t size = 0;
    size += sizeof(dense_qp_proxqp_opts);

    return size;
}



void *dense_qp_proxqp_opts_assign(void *config_, dense_qp_dims *dims, void *raw_memory)
{
    dense_qp_proxqp_opts *opts;

    char *c_ptr = (char *) raw_memory;

    opts = (dense_qp_proxqp_opts *) c_ptr;
    c_ptr += sizeof(dense_qp_proxqp_opts);

    assert((char *) raw_memory + dense_qp_proxqp_opts_calculate_size(config_, dims) >= c_ptr);

    return (void *) opts;
}



void dense_qp_proxqp_opts_initialize_default(void *config_, dense_qp_dims *dims, void *opts_)
{
    dense_qp_proxqp_opts *opts = opts_;

    proxqp_c_default_settings(&opts->settings);
    opts->warm_start = 0;
    opts->print_level = 0;

    return;
}



void dense_qp_proxqp_opts_update(void *config_, dense_qp_dims *dims, void *opts_)
{
    return;
}



void dense_qp_proxqp_opts_set(void *config_, void *opts_, const char *field, void *value)
{
    dense_qp_proxqp_opts *opts = opts_;

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
        printf("\nerror: dense_qp_proxqp_opts_set: wrong field: %s\n", field);
        exit(1);
    }

    return;
}



void dense_qp_proxqp_opts_get(void *config_, void *opts_, const char *field, void *value)
{
    printf("\nerror: dense_qp_proxqp_opts_get: not implemented for field %s\n", field);
    exit(1);
}



/************************************************
 * memory
 ************************************************/

acados_size_t dense_qp_proxqp_memory_calculate_size(void *config_, dense_qp_dims *dims,
                                                    void *opts_)
{
    dense_qp_dims dims_stacked;

    int nv = dims->nv;
    int ne = dims->ne;
    int ng = dims->ng;
    int nb = dims->nb;
    int ns = dims->ns;

    int ng2, nv2, nb2;

    acados_size_t size = sizeof(dense_qp_proxqp_memory);

    if (ns > 0)
    {
        dense_qp_stack_slacks_dims_upperbound(dims, &dims_stacked);
        size += dense_qp_in_calculate_size(&dims_stacked);
        ng2 = dims_stacked.ng;
        nv2 = dims_stacked.nv;
        nb2 = dims_stacked.nb;
    }
    else
    {
        ng2 = ng;
        nv2 = nv;
        nb2 = nb;
    }

    size += 1 * nv2 * nv2 * sizeof(double);    // H
    size += 1 * nv2 * sizeof(double);          // g
    size += 1 * nv2 * ne * sizeof(double);     // A
    size += 1 * ne * sizeof(double);           // b
    size += 1 * nv2 * ng2 * sizeof(double);    // C
    size += 2 * ng2 * sizeof(double);          // d_lg d_ug
    size += 2 * nb2 * sizeof(double);          // d_lb0 d_ub0
    size += 2 * nv2 * sizeof(double);          // l_box u_box
    size += 1 * nb * sizeof(int);              // idxb
    size += 1 * nb2 * sizeof(int);             // idxb_stacked
    size += 1 * ns * sizeof(int);              // idxs
    size += 1 * (nb + ng) * sizeof(int);       // idxs_rev
    size += 6 * ns * sizeof(double);           // Zl, Zu, zl, zu, d_ls, d_us
    size += 1 * nv2 * sizeof(double);          // prim_sol
    size += 1 * ne * sizeof(double);           // y_sol
    size += 1 * (ng2 + nv2) * sizeof(double);  // z_sol

    make_int_multiple_of(8, &size);

    return size;
}



void *dense_qp_proxqp_memory_assign(void *config_, dense_qp_dims *dims, void *opts_,
                                    void *raw_memory)
{
    dense_qp_proxqp_memory *mem;
    dense_qp_dims dims_stacked;

    int nv = dims->nv;
    int ne = dims->ne;
    int ng = dims->ng;
    int nb = dims->nb;
    int ns = dims->ns;

    int ng2, nv2, nb2;

    char *c_ptr = (char *) raw_memory;

    mem = (dense_qp_proxqp_memory *) c_ptr;
    c_ptr += sizeof(dense_qp_proxqp_memory);

    assert((size_t) c_ptr % 8 == 0 && "memory not 8-byte aligned!");

    if (ns > 0)
    {
        dense_qp_stack_slacks_dims_upperbound(dims, &dims_stacked);
        mem->qp_stacked = dense_qp_in_assign(&dims_stacked, c_ptr);
        c_ptr += dense_qp_in_calculate_size(&dims_stacked);
        ng2 = dims_stacked.ng;
        nv2 = dims_stacked.nv;
        nb2 = dims_stacked.nb;
    }
    else
    {
        mem->qp_stacked = NULL;
        ng2 = ng;
        nv2 = nv;
        nb2 = nb;
    }

    assert((size_t) c_ptr % 8 == 0 && "memory not 8-byte aligned!");

    assign_and_advance_double(nv2 * nv2, &mem->H, &c_ptr);
    assign_and_advance_double(nv2, &mem->g, &c_ptr);
    assign_and_advance_double(nv2 * ne, &mem->A, &c_ptr);
    assign_and_advance_double(ne, &mem->b, &c_ptr);
    assign_and_advance_double(nv2 * ng2, &mem->C, &c_ptr);
    assign_and_advance_double(ng2, &mem->d_lg, &c_ptr);
    assign_and_advance_double(ng2, &mem->d_ug, &c_ptr);
    assign_and_advance_double(nb2, &mem->d_lb0, &c_ptr);
    assign_and_advance_double(nb2, &mem->d_ub0, &c_ptr);
    assign_and_advance_double(nv2, &mem->l_box, &c_ptr);
    assign_and_advance_double(nv2, &mem->u_box, &c_ptr);
    assign_and_advance_double(ns, &mem->Zl, &c_ptr);
    assign_and_advance_double(ns, &mem->Zu, &c_ptr);
    assign_and_advance_double(ns, &mem->zl, &c_ptr);
    assign_and_advance_double(ns, &mem->zu, &c_ptr);
    assign_and_advance_double(ns, &mem->d_ls, &c_ptr);
    assign_and_advance_double(ns, &mem->d_us, &c_ptr);
    assign_and_advance_double(nv2, &mem->prim_sol, &c_ptr);
    assign_and_advance_double(ne, &mem->y_sol, &c_ptr);
    assign_and_advance_double(ng2 + nv2, &mem->z_sol, &c_ptr);
    assign_and_advance_int(nb, &mem->idxb, &c_ptr);
    assign_and_advance_int(nb2, &mem->idxb_stacked, &c_ptr);
    assign_and_advance_int(ns, &mem->idxs, &c_ptr);
    assign_and_advance_int(nb + ng, &mem->idxs_rev, &c_ptr);

    // the ProxQP object is created at the first call, once the actual
    // dimensions of the (stacked) qp are known
    mem->proxqp = NULL;
    mem->first_run = 1;

    assert((char *) raw_memory + dense_qp_proxqp_memory_calculate_size(config_, dims, opts_) >=
           c_ptr);

    return mem;
}



void dense_qp_proxqp_memory_get(void *config_, void *mem_, const char *field, void *value)
{
    dense_qp_proxqp_memory *mem = mem_;

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
        printf("\nerror: dense_qp_proxqp_memory_get: field %s not available\n", field);
        exit(1);
    }

    return;
}



void dense_qp_proxqp_memory_reset(void *config_, void *qp_in_, void *qp_out_, void *opts_,
                                  void *mem_, void *work_)
{
    dense_qp_proxqp_memory *mem = mem_;

    // drop the ProxQP object, it is recreated at the next call
    if (mem->proxqp != NULL)
    {
        proxqp_c_dense_free(mem->proxqp);
        mem->proxqp = NULL;
    }
    mem->first_run = 1;

    return;
}



void dense_qp_proxqp_terminate(void *config_, void *mem_, void *work_)
{
    dense_qp_proxqp_memory *mem = mem_;

    if (mem->proxqp != NULL)
        proxqp_c_dense_free(mem->proxqp);
}



/************************************************
 * workspace
 ************************************************/

acados_size_t dense_qp_proxqp_workspace_calculate_size(void *config_, dense_qp_dims *dims,
                                                       void *opts_)
{
    return 0;
}



/************************************************
 * functions
 ************************************************/

int dense_qp_proxqp(void *config_, void *qp_in_, void *qp_out_, void *opts_, void *mem_,
                    void *work_)
{
    dense_qp_in *qp_in = qp_in_;
    dense_qp_out *qp_out = qp_out_;

    qp_info *info = (qp_info *) qp_out->misc;
    acados_timer tot_timer, qp_timer, interface_timer;
    acados_tic(&tot_timer);
    acados_tic(&interface_timer);

    // cast structures
    dense_qp_proxqp_opts *opts = (dense_qp_proxqp_opts *) opts_;
    dense_qp_proxqp_memory *mem = (dense_qp_proxqp_memory *) mem_;
    dense_qp_in *qp_stacked = mem->qp_stacked;

    // extract dense qp size
    int nv = qp_in->dim->nv;
    int ne = qp_in->dim->ne;
    int ng = qp_in->dim->ng;
    int nb = qp_in->dim->nb;
    int ns = qp_in->dim->ns;

    int ng2, nv2, nb2;
    int *idxb2;

    double inf = proxqp_c_infinity();

    // fill in the upper triangular of H in dense_qp
    blasfeo_dtrtr_l(nv, qp_in->Hv, 0, 0, qp_in->Hv, 0, 0);

    if (ns > 0)
    {
        // extract idxs_rev to compute the stacked dims
        d_dense_qp_get_all(qp_in, mem->H, mem->g, mem->A, mem->b, mem->idxb, mem->d_lb0,
                           mem->d_ub0, mem->C, mem->d_lg, mem->d_ug, mem->Zl, mem->Zu, mem->zl,
                           mem->zu, mem->idxs, mem->idxs_rev, mem->d_ls, mem->d_us);

        dense_qp_stack_slacks_dims_from_idxs_rev(qp_in->dim, mem->idxs_rev, qp_stacked->dim);
        ng2 = qp_stacked->dim->ng;
        nv2 = qp_stacked->dim->nv;
        nb2 = qp_stacked->dim->nb;
        idxb2 = mem->idxb_stacked;

        dense_qp_stack_slacks(qp_in, qp_stacked);
        d_dense_qp_get_all(qp_stacked, mem->H, mem->g, mem->A, mem->b, idxb2, mem->d_lb0,
                           mem->d_ub0, mem->C, mem->d_lg, mem->d_ug, NULL, NULL, NULL, NULL,
                           NULL, NULL, NULL, NULL);
    }
    else
    {
        ng2 = ng;
        nv2 = nv;
        nb2 = nb;
        idxb2 = mem->idxb;

        d_dense_qp_get_all(qp_in, mem->H, mem->g, mem->A, mem->b, mem->idxb, mem->d_lb0,
                           mem->d_ub0, mem->C, mem->d_lg, mem->d_ug, mem->Zl, mem->Zu, mem->zl,
                           mem->zu, mem->idxs, mem->idxs_rev, mem->d_ls, mem->d_us);

        // ignore bounds and constraints that are masked out in qp_in via d_mask
        // NOTE: with soft constraints (ns > 0) masks are only applied to the multipliers
        for (int ii = 0; ii < nb; ii++)
        {
            if (BLASFEO_DVECEL(qp_in->d_mask, ii) == 0.0)
                mem->d_lb0[ii] = -inf;
            if (BLASFEO_DVECEL(qp_in->d_mask, nb + ng + ii) == 0.0)
                mem->d_ub0[ii] = +inf;
        }
        for (int ii = 0; ii < ng; ii++)
        {
            if (BLASFEO_DVECEL(qp_in->d_mask, nb + ii) == 0.0)
                mem->d_lg[ii] = -inf;
            if (BLASFEO_DVECEL(qp_in->d_mask, 2 * nb + ng + ii) == 0.0)
                mem->d_ug[ii] = +inf;
        }
    }

    // expand the bounds on the subset idxb2 into full-length box constraints
    for (int ii = 0; ii < nv2; ii++)
    {
        mem->l_box[ii] = -inf;
        mem->u_box[ii] = +inf;
    }
    for (int ii = 0; ii < nb2; ii++)
    {
        mem->l_box[idxb2[ii]] = mem->d_lb0[ii];
        mem->u_box[idxb2[ii]] = mem->d_ub0[ii];
    }

    // per-call settings, warm starting reuses the previous solution
    proxqp_c_settings settings = opts->settings;
    if (!mem->first_run && opts->warm_start > 0)
        settings.initial_guess = PROXQP_C_WARM_START_WITH_PREVIOUS_RESULT;

    if (mem->first_run)
    {
        mem->proxqp = proxqp_c_dense_new(nv2, ne, ng2, 1);
        if (mem->proxqp == NULL)
        {
            printf("\nerror: dense_qp_proxqp: failed to create ProxQP solver\n");
            exit(1);
        }
        proxqp_c_dense_apply_settings(mem->proxqp, &settings);
        if (proxqp_c_dense_init(mem->proxqp, mem->H, mem->g, ne > 0 ? mem->A : NULL,
                                ne > 0 ? mem->b : NULL, ng2 > 0 ? mem->C : NULL,
                                ng2 > 0 ? mem->d_lg : NULL, ng2 > 0 ? mem->d_ug : NULL,
                                mem->l_box, mem->u_box) != 0)
        {
            printf("\nerror: dense_qp_proxqp: ProxQP init failed\n");
            exit(1);
        }
        mem->first_run = 0;
    }
    else
    {
        proxqp_c_dense_apply_settings(mem->proxqp, &settings);
        if (proxqp_c_dense_update(mem->proxqp, mem->H, mem->g, ne > 0 ? mem->A : NULL,
                                  ne > 0 ? mem->b : NULL, ng2 > 0 ? mem->C : NULL,
                                  ng2 > 0 ? mem->d_lg : NULL, ng2 > 0 ? mem->d_ug : NULL,
                                  mem->l_box, mem->u_box) != 0)
        {
            printf("\nerror: dense_qp_proxqp: ProxQP update failed\n");
            exit(1);
        }
    }

    info->interface_time = acados_toc(&interface_timer);
    acados_tic(&qp_timer);

    // solve
    int proxqp_status = proxqp_c_dense_solve(mem->proxqp);

    info->solve_QP_time = acados_toc(&qp_timer);
    mem->time_qp_solver_call = info->solve_QP_time;
    mem->iter = proxqp_c_dense_get_iter(mem->proxqp);

    acados_tic(&interface_timer);

    proxqp_c_dense_get_solution(mem->proxqp, mem->prim_sol, mem->y_sol, mem->z_sol);

    // primal variables (with ns > 0 the stacked qp primal solution [v; sl; su]
    // matches the layout of qp_out->v)
    blasfeo_pack_dvec(nv2, mem->prim_sol, 1, qp_out->v, 0);

    // equality multipliers
    blasfeo_pack_dvec(ne, mem->y_sol, 1, qp_out->pi, 0);

    // inequality multipliers
    // ProxQP convention: z > 0 for active upper, z < 0 for active lower constraints;
    // z = [general constraint multipliers (ng2); box multipliers (nv2)]
    double *z_gen = mem->z_sol;
    double *z_box = mem->z_sol + ng2;

    blasfeo_dvecse(2 * nb + 2 * ng + 2 * ns, 0.0, qp_out->lam, 0);

    // hard box constraints (with ns > 0 softened box constraints are part of
    // the general constraints of the stacked qp and treated below)
    for (int ii = 0; ii < nb2; ii++)
    {
        // slack bounds of the stacked qp are treated below
        if (idxb2[ii] >= nv)
            continue;

        int ib = ns > 0 ? -1 : ii;
        if (ns > 0)
        {
            // recover the position within qp_in bounds
            for (int jj = 0; jj < nb; jj++)
            {
                if (mem->idxb[jj] == idxb2[ii])
                {
                    ib = jj;
                    break;
                }
            }
        }

        double z = z_box[idxb2[ii]];
        if (z >= 0.0)
            BLASFEO_DVECEL(qp_out->lam, nb + ng + ib) = z;
        else
            BLASFEO_DVECEL(qp_out->lam, ib) = -z;
    }

    // general constraints (first ng entries of the stacked general constraints
    // coincide with the general constraints of qp_in)
    for (int ii = 0; ii < ng; ii++)
    {
        double z = z_gen[ii];
        if (z >= 0.0)
            BLASFEO_DVECEL(qp_out->lam, 2 * nb + ng + ii) = z;
        else
            BLASFEO_DVECEL(qp_out->lam, nb + ii) = -z;
    }

    // soft constraints
    if (ns > 0)
    {
        int k = 0;
        for (int ii = 0; ii < ns; ii++)
        {
            int js = mem->idxs[ii];

            double offset_l = 0.0;
            double offset_u = 0.0;

            if (js < nb)
            {
                // softened box constraints are appended to the general constraints
                double z = z_gen[ng + k];
                if (z >= 0.0)
                {
                    BLASFEO_DVECEL(qp_out->lam, nb + ng + js) = z;
                    offset_u = z;
                }
                else
                {
                    BLASFEO_DVECEL(qp_out->lam, js) = -z;
                    offset_l = -z;
                }
                k++;
            }
            else
            {
                offset_l = BLASFEO_DVECEL(qp_out->lam, nb + js - nb);
                offset_u = BLASFEO_DVECEL(qp_out->lam, 2 * nb + ng + js - nb);
            }

            // multipliers of the lower bounds on the slacks sl >= d_ls, su >= d_us
            double z_sl = z_box[nv + ii];
            double z_su = z_box[nv + ns + ii];
            if (z_sl <= 0.0)
                BLASFEO_DVECEL(qp_out->lam, 2 * nb + 2 * ng + ii) = -z_sl - offset_u;
            if (z_su <= 0.0)
                BLASFEO_DVECEL(qp_out->lam, 2 * nb + 2 * ng + ns + ii) = -z_su - offset_l;
        }
    }

    // compute slacks
    dense_qp_compute_t(qp_in, qp_out);
    info->t_computed = 1;

    // multiply with mask to ensure that multipliers associated with masked constraints are zero
    blasfeo_dvecmul(2 * (nb + ng + ns), qp_in->d_mask, 0, qp_out->lam, 0, qp_out->lam, 0);

    info->interface_time += acados_toc(&interface_timer);
    info->total_time = acados_toc(&tot_timer);
    info->num_iter = mem->iter;

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



void dense_qp_proxqp_eval_forw_sens(void *config_, void *qp_in, void *seed, void *qp_out,
                                    void *opts_, void *mem_, void *work_)
{
    printf("\nerror: dense_qp_proxqp_eval_forw_sens: not implemented yet\n");
    exit(1);
}

void dense_qp_proxqp_eval_adj_sens(void *config_, void *qp_in, void *seed, void *qp_out,
                                   void *opts_, void *mem_, void *work_)
{
    printf("\nerror: dense_qp_proxqp_eval_adj_sens: not implemented yet\n");
    exit(1);
}


void dense_qp_proxqp_solver_get(void *config_, void *qp_in_, void *qp_out_, void *opts_,
                                void *mem_, const char *field, int stage, void *value, int size1,
                                int size2)
{
    printf("\nerror: dense_qp_proxqp_solver_get: not implemented yet\n");
    exit(1);
}


void dense_qp_proxqp_config_initialize_default(void *config_)
{
    qp_solver_config *config = config_;

    config->opts_calculate_size = (acados_size_t (*)(void *, void *)) & dense_qp_proxqp_opts_calculate_size;
    config->opts_assign = (void *(*) (void *, void *, void *) ) & dense_qp_proxqp_opts_assign;
    config->opts_initialize_default =
        (void (*)(void *, void *, void *)) & dense_qp_proxqp_opts_initialize_default;
    config->opts_update = (void (*)(void *, void *, void *)) & dense_qp_proxqp_opts_update;
    config->opts_set = &dense_qp_proxqp_opts_set;
    config->opts_get = &dense_qp_proxqp_opts_get;
    config->memory_calculate_size =
        (acados_size_t (*)(void *, void *, void *)) & dense_qp_proxqp_memory_calculate_size;
    config->memory_assign =
        (void *(*) (void *, void *, void *, void *) ) & dense_qp_proxqp_memory_assign;
    config->memory_get = &dense_qp_proxqp_memory_get;
    config->workspace_calculate_size =
        (acados_size_t (*)(void *, void *, void *)) & dense_qp_proxqp_workspace_calculate_size;
    config->eval_forw_sens = &dense_qp_proxqp_eval_forw_sens;
    config->eval_adj_sens = &dense_qp_proxqp_eval_adj_sens;
    config->evaluate = (int (*)(void *, void *, void *, void *, void *, void *)) & dense_qp_proxqp;
    config->memory_reset = &dense_qp_proxqp_memory_reset;
    config->solver_get = &dense_qp_proxqp_solver_get;
    config->terminate = &dense_qp_proxqp_terminate;

    return;
}
