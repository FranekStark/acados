/*
 *    This file is part of acados.
 *
 *    acados is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    acados is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with acados; if not, write to the Free Software Foundation,
 *    Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *    
 *    Author: Jonathan Frey, Robin Verschueren
 */

#include <stdlib.h>
#include <vector>
#include <xmmintrin.h>

#include "blasfeo/include/blasfeo_d_aux.h"
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_d_blas.h"

// acados
#include "acados/sim/sim_common.h"
#include "acados/utils/external_function_generic.h"
#include "acados/utils/math.h"
#include "acados/utils/print.h"

#include "acados_c/external_function_interface.h"
#include "acados_c/sim_interface.h"

#include "examples/c/engine_model/engine_impl_dae_fun.h"
#include "examples/c/engine_model/engine_impl_dae_fun_jac_x_xdot_z.h"
#include "examples/c/engine_model/engine_impl_dae_jac_x_xdot_u_z.h"
#include "examples/c/engine_model/engine_impl_dae_fun_jac_x_xdot_u_z.h"

int main()
{
    _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);

    const int n_sim = 100;

    const int nx = 4;
    const int nu = 2;
    const int nz = 2;
    const int num_forw_sens = nx + nu;

    // sampling time (s)
    double T = 0.05;

    // x: u1, u2, xD1, xD2
    auto x0 = std::vector<double> {50, 50, 1.14275, 1.53787};
    // z: xA1, xA2
    auto z0 = std::vector<double> {1.28976, 1.78264};
    // u: u1_r, u2_r
    auto u = std::vector<double> {0, 0};

    // implicit dae
    external_function_casadi impl_dae_fun;
    impl_dae_fun.casadi_fun = &engine_impl_dae_fun;
    impl_dae_fun.casadi_work = &engine_impl_dae_fun_work;
    impl_dae_fun.casadi_sparsity_in = &engine_impl_dae_fun_sparsity_in;
    impl_dae_fun.casadi_sparsity_out = &engine_impl_dae_fun_sparsity_out;
    impl_dae_fun.casadi_n_in = &engine_impl_dae_fun_n_in;
    impl_dae_fun.casadi_n_out = &engine_impl_dae_fun_n_out;
    external_function_casadi_create(&impl_dae_fun);

    external_function_casadi impl_dae_fun_jac_x_xdot_z;
    impl_dae_fun_jac_x_xdot_z.casadi_fun = &engine_impl_dae_fun_jac_x_xdot_z;
    impl_dae_fun_jac_x_xdot_z.casadi_work = &engine_impl_dae_fun_jac_x_xdot_z_work;
    impl_dae_fun_jac_x_xdot_z.casadi_sparsity_in = &engine_impl_dae_fun_jac_x_xdot_z_sparsity_in;
    impl_dae_fun_jac_x_xdot_z.casadi_sparsity_out = &engine_impl_dae_fun_jac_x_xdot_z_sparsity_out;
    impl_dae_fun_jac_x_xdot_z.casadi_n_in = &engine_impl_dae_fun_jac_x_xdot_z_n_in;
    impl_dae_fun_jac_x_xdot_z.casadi_n_out = &engine_impl_dae_fun_jac_x_xdot_z_n_out;
    external_function_casadi_create(&impl_dae_fun_jac_x_xdot_z);

    external_function_casadi impl_dae_jac_x_xdot_u_z;
    impl_dae_jac_x_xdot_u_z.casadi_fun = &engine_impl_dae_jac_x_xdot_u_z;
    impl_dae_jac_x_xdot_u_z.casadi_work = &engine_impl_dae_jac_x_xdot_u_z_work;
    impl_dae_jac_x_xdot_u_z.casadi_sparsity_in = &engine_impl_dae_jac_x_xdot_u_z_sparsity_in;
    impl_dae_jac_x_xdot_u_z.casadi_sparsity_out = &engine_impl_dae_jac_x_xdot_u_z_sparsity_out;
    impl_dae_jac_x_xdot_u_z.casadi_n_in = &engine_impl_dae_jac_x_xdot_u_z_n_in;
    impl_dae_jac_x_xdot_u_z.casadi_n_out = &engine_impl_dae_jac_x_xdot_u_z_n_out;
    external_function_casadi_create(&impl_dae_jac_x_xdot_u_z);

    // Only needed for lifted IRK:

    // external_function_casadi engine_impl_dae_fun_jac_x_xdot_u_z;
    // engine_impl_dae_fun_jac_x_xdot_u_z.casadi_fun = &engine_impl_dae_fun_jac_x_xdot_u_z;
    // engine_impl_dae_fun_jac_x_xdot_u_z.casadi_work = &engine_impl_dae_fun_jac_x_xdot_u_z_work;
    // engine_impl_dae_fun_jac_x_xdot_u_z.casadi_sparsity_in = &engine_impl_dae_fun_jac_x_xdot_u_z_sparsity_in;
    // engine_impl_dae_fun_jac_x_xdot_u_z.casadi_sparsity_out = &engine_impl_dae_fun_jac_x_xdot_u_z_sparsity_out;
    // engine_impl_dae_fun_jac_x_xdot_u_z.casadi_n_in = &engine_impl_dae_fun_jac_x_xdot_u_z_n_in;
    // engine_impl_dae_fun_jac_x_xdot_u_z.casadi_n_out = &engine_impl_dae_fun_jac_x_xdot_u_z_n_out;
    // external_function_casadi_create(&engine_impl_dae_fun_jac_x_xdot_u_z);

    // plan for integrator
    sim_solver_plan plan;
	plan.sim_solver = IRK;
    sim_solver_config *config = sim_config_create(plan);

    // dimensions
    void *dims = sim_dims_create(config);
    config->set_nx(dims, nx);
    config->set_nu(dims, nu);
    config->set_nz(dims, nz);

    // options
    void *opts_ = sim_opts_create(config, dims);
    sim_rk_opts *opts = (sim_rk_opts *) opts_;
    config->opts_initialize_default(config, dims, opts);

    opts->jac_reuse = false;
    opts->newton_iter = 100;         // number of newton iterations per integration step

    opts->ns                = 8;    // number of stages in rk integrator
    opts->num_steps         = 100;  // number of steps
    opts->sens_forw         = false;
    opts->sens_adj          = false;
    opts->output_z          = false;
    opts->sens_algebraic    = false;

    // input, output

    sim_in *in   = sim_in_create(config, dims);
    in->T = T;

    sim_out *out = sim_out_create(config, dims);

    // model
    sim_set_model(config, in, "impl_ode_fun", &impl_dae_fun);
    sim_set_model(config, in, "impl_ode_fun_jac_x_xdot", &impl_dae_fun_jac_x_xdot_z);
    sim_set_model(config, in, "impl_ode_jac_x_xdot_u", &impl_dae_jac_x_xdot_u_z);

    // seeds
    for (int ii = 0; ii < nx * num_forw_sens; ii++)
        in->S_forw[ii] = 0.0;
    for (int ii = 0; ii < nx; ii++)
        in->S_forw[ii * (nx + 1)] = 1.0;

    // seeds adj
    for (int ii = 0; ii < nx; ii++)
        in->S_adj[ii] = 1.0;
    for (int ii = nx; ii < nx + nu; ii++)
        in->S_adj[ii] = 0.0;

    // solver
    sim_solver *sim_solver = sim_create(config, dims, opts);

    acados_timer timer;
    acados_tic(&timer);

    // simulate
    double cpu_time = 0.0;
    double la_time = 0.0;
    double ad_time = 0.0;

    std::vector<std::vector<double>> x_sim = {x0};
    for (int ii = 0; ii < n_sim; ii++)
    {
        in->x = x_sim[ii].data();
        in->u = u.data();
        out->zn = z0.data();

        // execute simulation step with current input and state
        int acados_return = sim_solve(sim_solver, in, out);
        if (acados_return != 0)
        {
            printf("error in sim solver\n");
            return ACADOS_FAILURE;
        }

        cpu_time += out->info->CPUtime;
        la_time += out->info->LAtime;
        ad_time += out->info->ADtime;

        // extract state at next time step
        std::vector<double> tmp;
        for (int jj = 0; jj < nx; jj++)
            tmp.push_back(out->xn[jj]);
        x_sim.push_back(tmp);

    }
    double total_cpu_time = acados_toc(&timer);

    /************************************************
    * printing
    ************************************************/
    printf("\nxn: \n");
    d_print_e_mat(1, nx, x_sim[n_sim].data(), 1);

    if (opts->sens_forw) {
        printf("S_forw_out: \n");
        d_print_e_mat(nx, num_forw_sens, out->S_forw, nx);
    }

    if (opts->sens_adj) {
        printf("S_adj_out: \n");
        d_print_e_mat(1, nx+nu, out->S_adj, 1);
    }

    if (opts->sens_forw) {
        struct blasfeo_dmat S_forw_result;
        struct blasfeo_dvec adjoint_seed;
        struct blasfeo_dvec forw_times_seed;

        blasfeo_allocate_dmat(nx, nx+nu, &S_forw_result);
        blasfeo_allocate_dvec(nx, &adjoint_seed);
        blasfeo_allocate_dvec(nx+nu, &forw_times_seed);

        blasfeo_pack_dmat(nx, nx+nu, out->S_forw, nx, &S_forw_result, 0, 0);
        blasfeo_pack_dvec(nx, in->S_adj, &adjoint_seed, 0);

        blasfeo_dgemv_t(nx, nx+nu, 1.0, &S_forw_result, 0, 0, &adjoint_seed, 0, 0.0, &forw_times_seed, 0, &forw_times_seed, 0);
        printf("S_forw^T * adj_seed = \n");
        blasfeo_print_exp_tran_dvec(nx+nu, &forw_times_seed, 0);

        blasfeo_free_dmat(&S_forw_result);
        blasfeo_free_dvec(&adjoint_seed);
        blasfeo_free_dvec(&forw_times_seed);			
    }

    if (opts->output_z) {
        printf("zn \n");
        d_print_e_mat(1, nz, out->zn, 1);
    }

    if (opts->sens_algebraic) {
        printf("algebraic sensitivities \n");
        d_print_e_mat(nz, num_forw_sens, out->S_algebraic, nz);
    }

    printf("\ntime for %d simulation steps: %f ms (AD time: %f ms (%5.2f%%))\n\n", n_sim, 1e3*total_cpu_time, 1e3*ad_time, 1e2*ad_time/cpu_time);
    printf("time spent in integrator outside of casADi %f \n", 1e3*(total_cpu_time-ad_time));

    // free memory
    free(dims);
    free(sim_solver);
    free(in);
    free(out);
    free(opts);
    free(config);
    
    /* free external function */
    // implicit model
    external_function_casadi_free(&impl_dae_fun);
    external_function_casadi_free(&impl_dae_fun_jac_x_xdot_z);
    external_function_casadi_free(&impl_dae_jac_x_xdot_u_z);
    // external_function_casadi_free(&impl_ode_jac_x_xdot_u);

    return 0;
}
