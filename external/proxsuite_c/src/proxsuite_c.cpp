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

#include "proxsuite_c.h"

#include <cstdio>
#include <cstring>
#include <exception>

#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/proxqp/sparse/sparse.hpp>

using proxsuite::nullopt;
using proxsuite::optional;
using proxsuite::proxqp::InitialGuessStatus;

using DenseQP = proxsuite::proxqp::dense::QP<double>;
using SparseQP = proxsuite::proxqp::sparse::QP<double, int>;
using MatMap = Eigen::Map<const Eigen::MatrixXd>;
using VecMap = Eigen::Map<const Eigen::VectorXd>;
using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using SpMatMap = Eigen::Map<const SpMat>;

extern "C" {

void proxqp_c_default_settings(proxqp_c_settings *settings)
{
    proxsuite::proxqp::Settings<double> s;
    settings->eps_abs = s.eps_abs;
    settings->eps_rel = s.eps_rel;
    settings->max_iter = (int) s.max_iter;
    settings->verbose = s.verbose;
    settings->initial_guess = (int) s.initial_guess;
}

double proxqp_c_infinity(void)
{
    return proxsuite::helpers::infinite_bound<double>::value();
}


/************************************************
 * dense backend
 ************************************************/

void *proxqp_c_dense_new(int n, int n_eq, int n_in, int box_constraints)
{
    try
    {
        return new DenseQP(n, n_eq, n_in, box_constraints != 0);
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "proxsuite_c: proxqp_c_dense_new failed: %s\n", e.what());
        return nullptr;
    }
}


void proxqp_c_dense_free(void *qp)
{
    delete static_cast<DenseQP *>(qp);
}


static void apply_settings(proxsuite::proxqp::Settings<double> &s,
                           const proxqp_c_settings *settings)
{
    s.eps_abs = settings->eps_abs;
    s.eps_rel = settings->eps_rel;
    s.max_iter = settings->max_iter;
    s.verbose = settings->verbose != 0;
    s.initial_guess = static_cast<InitialGuessStatus>(settings->initial_guess);
}


void proxqp_c_dense_apply_settings(void *qp_, const proxqp_c_settings *settings)
{
    DenseQP *qp = static_cast<DenseQP *>(qp_);
    apply_settings(qp->settings, settings);
}


static int dense_init_update(void *qp_, const double *H, const double *g,
                             const double *A, const double *b,
                             const double *C, const double *l, const double *u,
                             const double *l_box, const double *u_box, bool init)
{
    DenseQP *qp = static_cast<DenseQP *>(qp_);

    long n = qp->model.dim;
    long n_eq = qp->model.n_eq;
    long n_in = qp->model.n_in;

    optional<MatMap> H_ = H ? optional<MatMap>(MatMap(H, n, n)) : nullopt;
    optional<VecMap> g_ = g ? optional<VecMap>(VecMap(g, n)) : nullopt;
    optional<MatMap> A_ = A ? optional<MatMap>(MatMap(A, n_eq, n)) : nullopt;
    optional<VecMap> b_ = b ? optional<VecMap>(VecMap(b, n_eq)) : nullopt;
    optional<MatMap> C_ = C ? optional<MatMap>(MatMap(C, n_in, n)) : nullopt;
    optional<VecMap> l_ = l ? optional<VecMap>(VecMap(l, n_in)) : nullopt;
    optional<VecMap> u_ = u ? optional<VecMap>(VecMap(u, n_in)) : nullopt;

    try
    {
        if (qp->is_box_constrained())
        {
            optional<VecMap> lb_ = l_box ? optional<VecMap>(VecMap(l_box, n)) : nullopt;
            optional<VecMap> ub_ = u_box ? optional<VecMap>(VecMap(u_box, n)) : nullopt;
            if (init)
                qp->init(H_, g_, A_, b_, C_, l_, u_, lb_, ub_);
            else
                qp->update(H_, g_, A_, b_, C_, l_, u_, lb_, ub_,
                           /* update_preconditioner */ false);
        }
        else
        {
            if (init)
                qp->init(H_, g_, A_, b_, C_, l_, u_);
            else
                qp->update(H_, g_, A_, b_, C_, l_, u_,
                           /* update_preconditioner */ false);
        }
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "proxsuite_c: dense %s failed: %s\n",
                init ? "init" : "update", e.what());
        return 1;
    }
    return 0;
}


int proxqp_c_dense_init(void *qp, const double *H, const double *g,
                        const double *A, const double *b,
                        const double *C, const double *l, const double *u,
                        const double *l_box, const double *u_box)
{
    return dense_init_update(qp, H, g, A, b, C, l, u, l_box, u_box, true);
}


int proxqp_c_dense_update(void *qp, const double *H, const double *g,
                          const double *A, const double *b,
                          const double *C, const double *l, const double *u,
                          const double *l_box, const double *u_box)
{
    return dense_init_update(qp, H, g, A, b, C, l, u, l_box, u_box, false);
}


int proxqp_c_dense_solve(void *qp_)
{
    DenseQP *qp = static_cast<DenseQP *>(qp_);
    try
    {
        qp->solve();
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "proxsuite_c: dense solve failed: %s\n", e.what());
        return PROXQP_C_ERROR;
    }
    return (int) qp->results.info.status;
}


void proxqp_c_dense_get_solution(void *qp_, double *x, double *y, double *z)
{
    DenseQP *qp = static_cast<DenseQP *>(qp_);
    if (x != nullptr)
        memcpy(x, qp->results.x.data(), qp->results.x.size() * sizeof(double));
    if (y != nullptr)
        memcpy(y, qp->results.y.data(), qp->results.y.size() * sizeof(double));
    if (z != nullptr)
        memcpy(z, qp->results.z.data(), qp->results.z.size() * sizeof(double));
}


int proxqp_c_dense_get_iter(void *qp_)
{
    DenseQP *qp = static_cast<DenseQP *>(qp_);
    return (int) qp->results.info.iter;
}


/************************************************
 * sparse backend
 ************************************************/

void *proxqp_c_sparse_new(int n, int n_eq, int n_in)
{
    try
    {
        return new SparseQP(n, n_eq, n_in);
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "proxsuite_c: proxqp_c_sparse_new failed: %s\n", e.what());
        return nullptr;
    }
}


void proxqp_c_sparse_free(void *qp)
{
    delete static_cast<SparseQP *>(qp);
}


void proxqp_c_sparse_apply_settings(void *qp_, const proxqp_c_settings *settings)
{
    SparseQP *qp = static_cast<SparseQP *>(qp_);
    apply_settings(qp->settings, settings);
}


static int sparse_init_update(void *qp_,
                              const double *H_x, const int *H_i, const int *H_p,
                              const double *g,
                              const double *A_x, const int *A_i, const int *A_p,
                              const double *b,
                              const double *C_x, const int *C_i, const int *C_p,
                              const double *l, const double *u, bool init)
{
    SparseQP *qp = static_cast<SparseQP *>(qp_);

    long n = qp->model.dim;
    long n_eq = qp->model.n_eq;
    long n_in = qp->model.n_in;

    // proxsuite takes owning Eigen::SparseMatrix arguments, map + copy
    optional<SpMat> H_ = H_x ? optional<SpMat>(
        SpMatMap(n, n, H_p[n], H_p, H_i, H_x)) : nullopt;
    optional<SpMat> A_ = A_x ? optional<SpMat>(
        SpMatMap(n_eq, n, A_p[n], A_p, A_i, A_x)) : nullopt;
    optional<SpMat> C_ = C_x ? optional<SpMat>(
        SpMatMap(n_in, n, C_p[n], C_p, C_i, C_x)) : nullopt;

    optional<VecMap> g_ = g ? optional<VecMap>(VecMap(g, n)) : nullopt;
    optional<VecMap> b_ = b ? optional<VecMap>(VecMap(b, n_eq)) : nullopt;
    optional<VecMap> l_ = l ? optional<VecMap>(VecMap(l, n_in)) : nullopt;
    optional<VecMap> u_ = u ? optional<VecMap>(VecMap(u, n_in)) : nullopt;

    try
    {
        if (init)
            qp->init(H_, g_, A_, b_, C_, l_, u_);
        else
            qp->update(H_, g_, A_, b_, C_, l_, u_,
                       /* update_preconditioner */ false);
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "proxsuite_c: sparse %s failed: %s\n",
                init ? "init" : "update", e.what());
        return 1;
    }
    return 0;
}


int proxqp_c_sparse_init(void *qp,
                         const double *H_x, const int *H_i, const int *H_p,
                         const double *g,
                         const double *A_x, const int *A_i, const int *A_p,
                         const double *b,
                         const double *C_x, const int *C_i, const int *C_p,
                         const double *l, const double *u)
{
    return sparse_init_update(qp, H_x, H_i, H_p, g, A_x, A_i, A_p, b,
                              C_x, C_i, C_p, l, u, true);
}


int proxqp_c_sparse_update(void *qp,
                           const double *H_x, const int *H_i, const int *H_p,
                           const double *g,
                           const double *A_x, const int *A_i, const int *A_p,
                           const double *b,
                           const double *C_x, const int *C_i, const int *C_p,
                           const double *l, const double *u)
{
    return sparse_init_update(qp, H_x, H_i, H_p, g, A_x, A_i, A_p, b,
                              C_x, C_i, C_p, l, u, false);
}


int proxqp_c_sparse_solve(void *qp_)
{
    SparseQP *qp = static_cast<SparseQP *>(qp_);
    try
    {
        qp->solve();
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "proxsuite_c: sparse solve failed: %s\n", e.what());
        return PROXQP_C_ERROR;
    }
    return (int) qp->results.info.status;
}


void proxqp_c_sparse_get_solution(void *qp_, double *x, double *y, double *z)
{
    SparseQP *qp = static_cast<SparseQP *>(qp_);
    if (x != nullptr)
        memcpy(x, qp->results.x.data(), qp->results.x.size() * sizeof(double));
    if (y != nullptr)
        memcpy(y, qp->results.y.data(), qp->results.y.size() * sizeof(double));
    if (z != nullptr)
        memcpy(z, qp->results.z.data(), qp->results.z.size() * sizeof(double));
}


int proxqp_c_sparse_get_iter(void *qp_)
{
    SparseQP *qp = static_cast<SparseQP *>(qp_);
    return (int) qp->results.info.iter;
}

}  // extern "C"
