//
// Created by Niclas Laursen Brok on 2020-02-14.
//

#ifndef SWITCHINGTIMES_SWITCHING_TIMES_HPP
#define SWITCHINGTIMES_SWITCHING_TIMES_HPP

#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include "cppad-eigen.hpp"
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/eigen/eigen.hpp>
#include <coin-or/IpIpoptApplication.hpp>
#include <coin-or/IpTNLP.hpp>
#include <coin-or/IpOptionsList.hpp>

using namespace boost::numeric::odeint;
using namespace Ipopt;

namespace SwitchingTimes {
    /*
     * Internal types
     */
    template <typename scalar>
    using vector = Eigen::Matrix<scalar, Eigen::Dynamic, 1>;
    typedef CppAD::ADFun<double> ad_function;
    typedef CppAD::AD<double> ad_double;
    /*
     * OpenMP sum reduction on scalar sum
     */
    //#pragma omp declare reduction(scalar_sum: double: omp_out += omp_in)
    //#pragma omp declare reduction(scalar_sum: CppAD::AD<double>: omp_out += omp_in)
    /*
     * Internal functions
     */
    double cexp(double x, double cap);
    CppAD::AD<double> cexp(CppAD::AD<double> x, double cap);
    /*
     * Class that defines a PLANT w. switched dynamics
     */
    class Plant: public TNLP {
    public:
        // Plant variables
        vector<double> _p_const;     // Constant parameters
        vector<double> _p_dynamic;   // Dynamical parameters
        vector<double> _p_opt;       // Independent variables           -> to be optimized!
        vector<double> _p_opt_ipopt; // Optimized independent variables -> output from ipopt
        // Independent variable bounds
        vector<double> _lower_bound;
        vector<double> _upper_bound;
        // Regime bounds -> !!could be made more generic!!
        vector<double> _on_bound;
        vector<double> _off_bound;
        // Optimization horizon and discretization used in objective function (+ tape)
        double _t0; // Optimization starting time
        double _tf; // Optimization end time
        double _dt; // ODE-solver discretization
        // ODE initial values
        vector<double> _x0; // Initial state values
        // Tape of objective
        ad_function objective_tape;
        // Bool variables for tape logic
        bool new_tape = true;
        bool new_dynamic = false;
        // IPOPT application status
        int _status_init;
        int _status_solve;
        // Set functions
        void set_p_const(const vector<double> &p_const) {
            if (p_const.size() != _p_const.size() ) { new_tape = true; };
            _p_const = p_const;
        };
        void set_p_dynamic(const vector<double> &p_dynamic) {
            if (p_dynamic.size() != _p_dynamic.size()) { new_tape = true; };
            new_dynamic = true;
            _p_dynamic = p_dynamic;
        };
        void set_p_optimize(const vector<double> &p_opt) {
            if (p_opt.size() != _p_opt.size() ) { new_tape = true; };
            _p_opt = p_opt;
            _p_opt_ipopt = vector<double>::Zero(p_opt.size());
        };
        void set_t0(const double t0) { _t0 = t0; };
        void set_tf(const double tf) { _tf = tf; };
        void set_dt(const double dt) { _dt = dt; };
        void set_lower_bound(const vector<double> &lower_bound) { _lower_bound = lower_bound; };
        void set_upper_bound(const vector<double> &upper_bound) { _upper_bound = upper_bound; };
        void set_on_bound(const vector<double> &on_bound) { _on_bound = on_bound; };
        void set_off_bound(const vector<double> &off_bound) { _off_bound = off_bound; };
        void set_x0(const vector<double> x0) {
            if (x0.size() != _x0.size()) { new_tape = true; };
            _x0 = x0;
        };
        // Get functions
        const vector<double> &get_p_const() const { return _p_const; };
        const vector<double> &get_p_dynamic() const { return _p_dynamic; };
        const vector<double> &get_p_optimize() const { return _p_opt; };
        const vector<double> &get_p_optimize_ipopt() const { return _p_opt_ipopt; };
        const double &get_t0() const { return _t0; };
        const double &get_tf() const { return _tf; };
        const double &get_dt() const { return _dt; };
        const vector<double> &get_x0() const { return _x0; };
        const vector<double> &get_lower_bound() const { return _lower_bound; };
        const vector<double> &get_upper_bound() const { return _lower_bound; };
        const vector<double> &get_on_bound() const { return _on_bound; };
        const vector<double> &get_off_bound() const { return _off_bound; };
        const int &get_init_status() const { return _status_init; };
        const int &get_solve_status() const { return _status_solve; };
        // ODE right-hand-side function template
        template <typename scalar>
        void model(const vector<scalar> &x, vector<scalar> &dxdt,
                   const double t,
                   const vector<scalar> &p_dynamic, const vector<scalar> &p_opt, const vector<double> &p_const);
        // Objective function template (Mayer form -> end-point condition only)
        template <typename scalar>
        scalar objective(const vector<scalar> &x,
                         const vector<scalar> &p_dynamic, const vector<scalar> &p_opt, const vector<double> &p_const);
        // Integrate model from t1 to t2
        vector<double> integrate(const double t1, const double t2, const double dt, const vector<double> x0) {
            //runge_kutta_dopri5<vector<double>, double, vector<double>, double, openmp_range_algebra> rk5_stepper;
            runge_kutta_dopri5<vector<double>> rk5_stepper;
            vector<double> x(x0);
            size_t steps = integrate_const(rk5_stepper,
                                           [&] (const vector<double> &x , vector<double> &dxdt, const double t) {
                                               model(x, dxdt, t, _p_dynamic, _p_opt, _p_const);
                                           }, x, t1, t2, dt);
            return x;
        };
        // Objective function wrapper -> p_dynamic and x0 are include as dynamic parameters in CppAD!
        template <typename scalar>
        scalar objective_wrapper(const vector<scalar> &p_dynamic_x0, const vector<scalar> &p_opt) {
            //runge_kutta_dopri5<vector<scalar>, double, vector<scalar>, double, openmp_range_algebra> rk5_stepper;
            runge_kutta_dopri5<vector<scalar>> rk5_stepper;
            vector<scalar> x = vector<scalar>::Zero(_x0.size());
            // x0 is appended to p_dynamic -> treated as dynamical parameters in CppAD!
            for(int k = 0; k < _x0.size(); ++k) { x(k) = p_dynamic_x0(p_dynamic_x0.size() - _x0.size() + k); };
            size_t steps = integrate_const(rk5_stepper,
                                           [&] (const vector<scalar> &x , vector<scalar> &dxdt , const double t) {
                                               model(x, dxdt, t, p_dynamic_x0, p_opt, _p_const);
                                           }, x, _t0, _tf, _dt);
            return objective(x, p_dynamic_x0, p_opt, _p_const);
        };
        // Overloading -> used in IPOPT function
        double objective_wrapper(const vector<double> &p_opt) {
            runge_kutta_dopri5<vector<double>> rk5_stepper;
            vector<double> x(_x0);
            size_t steps = integrate_const(rk5_stepper,
                                           [&] (const vector<double> &x , vector<double> &dxdt , const double t) {
                                               model(x, dxdt, t, _p_dynamic, p_opt, _p_const);
                                           }, x, _t0, _tf, _dt);
            return objective(x, _p_dynamic, p_opt, _p_const);
        };
        // Jacobian function wrapper
        vector<double> jacobian(const vector<double> &p_opt) {
            if (new_tape) {
                // Fill dynamical parameters
                vector<ad_double> p_dynamic_x0 = vector<ad_double>::Zero(_p_dynamic.size() + _x0.size());
                for(int k = 0; k < _p_dynamic.size(); ++k) { p_dynamic_x0(k) = _p_dynamic(k); };
                for(int k = 0; k < _x0.size(); ++k) { p_dynamic_x0(p_dynamic_x0.size() - _x0.size() + k) = _x0(k); };
                // Fill independent parameters
                vector<ad_double> p_indep = vector<ad_double>::Zero(p_opt.size());
                for(int k = 0; k < p_opt.size(); ++k) { p_indep(k) = p_opt(k); };
                // Make AD tape
                size_t abort_op_index = 0;
                bool record_compare = true;
                CppAD::Independent(p_indep, abort_op_index, record_compare, p_dynamic_x0);
                vector<ad_double> _out = vector<ad_double>::Zero(1);
                _out(0) = objective_wrapper(p_dynamic_x0, p_indep);
                objective_tape = ad_function(p_indep, _out);
                new_tape = false;
            };
            if (new_dynamic) {
                vector<double> p_dynamic_x0 = vector<double>::Zero(_p_dynamic.size() + _x0.size());
                for(int k = 0; k < _p_dynamic.size(); ++k) { p_dynamic_x0(k) = _p_dynamic(k); };
                for(int k = 0; k < _x0.size(); ++k) { p_dynamic_x0(p_dynamic_x0.size() - _x0.size() + k) = _x0(k); };
                objective_tape.new_dynamic(p_dynamic_x0);
                new_dynamic = false;
            };
            return objective_tape.Jacobian(p_opt);
        };
        /*
         * IPOPT functions below
         */
        bool get_nlp_info(
                Index&          n,
                Index&          m,
                Index&          nnz_jac_g,
                Index&          nnz_h_lag,
                IndexStyleEnum& index_style
        ){
            n = _p_opt.size();
            m = _p_opt.size() - 1;
            nnz_jac_g = 2 * m;
            nnz_h_lag = 0;
            index_style = TNLP::C_STYLE;
            return true;
        };
        bool get_bounds_info(
                Index   n,
                Number* x_l,
                Number* x_u,
                Index   m,
                Number* g_l,
                Number* g_u
        ){
            /*
             * We have the bounds:  ON_LOWER  <= OFF_k    - ON_k  <= ON_UPPER  -> ON  period
             *                      OFF_LOWER <= ON_{k+1} - OFF_k <= OFF_LOWER -> OFF period
             */
            for(int k = 0; k < _lower_bound.size(); ++k) { x_l[k] = _lower_bound(k); };
            for(int k = 0; k < _upper_bound.size(); ++k) { x_u[k] = _upper_bound(k); };
            int _tmp = _p_opt.size() / 2;
            for(int k = 0; k < _tmp; ++k) {
                g_l[k] = _on_bound(0);
                g_u[k] = _on_bound(1);
            };
            for(int k = 0; k < _tmp - 1; ++k) {
                g_l[_tmp + k] = _off_bound(0);
                g_u[_tmp + k] = _off_bound(1);
            };
            return true;
        };
        bool get_starting_point(
                Index   n,
                bool    init_x,
                Number* x,
                bool    init_z,
                Number* z_L,
                Number* z_U,
                Index   m,
                bool    init_lambda,
                Number* lambda
        )
        {
            for(int k = 0; k < _p_opt.size(); ++k) { x[k] = _p_opt(k); };
            return true;
        };
        bool eval_f(
                Index         n,
                const Number* x,
                bool          new_x,
                Number&       obj_value
        )
        {
            vector<double> p_opt = vector<double>::Zero(_p_opt.size());
            for(int k = 0; k < _p_opt.size(); ++k) { p_opt(k) = x[k]; };
            obj_value = objective_wrapper(p_opt);
            return true;
        };
        bool eval_grad_f(
                Index         n,
                const Number* x,
                bool          new_x,
                Number*       grad_f
        )
        {
            vector<double> p_opt = vector<double>::Zero(_p_opt.size());
            for(int k = 0; k < p_opt.size(); ++k) { p_opt(k) = x[k]; };
            vector<double> _grad = jacobian(p_opt);
            for(int k = 0; k < p_opt.size(); ++k) { grad_f[k] = _grad(k); };
            return true;
        };
        bool eval_g(
                Index         n,
                const Number* x,
                bool          new_x,
                Index         m,
                Number*       g
        )
        {
            int _tmp = _p_opt.size() / 2;
            for(int k = 0; k < _tmp; ++k) { g[k] = x[_tmp + k] - x[k]; };
            for(int k = 0; k < _tmp - 1; ++k) { g[_tmp + k] = x[k + 1] - x[_tmp + k]; };
            return true;
        };
        bool eval_jac_g(
                Index         n,
                const Number* x,
                bool          new_x,
                Index         m,
                Index         nele_jac,
                Index*        iRow,
                Index*        jCol,
                Number*       values
        )
        {
            if( values == NULL )
            {
                int _tmp = _p_opt.size() / 2;
                int _count = 0;
                for(int k = 0; k < _tmp; ++k) {
                    iRow[_count] = k; jCol[_count] = k;
                    _count += 1;
                    iRow[_count] = k; jCol[_count] = _tmp + k;
                    _count += 1;
                };
                for(int k = 0; k < _tmp - 1; ++k) {
                    iRow[_count] = _tmp + k; jCol[_count] = k + 1;
                    _count += 1;
                    iRow[_count] = _tmp + k; jCol[_count] = _tmp + k;
                    _count += 1;
                };
            }
            else
            {
                int _tmp = _p_opt.size() / 2;
                int _count = 0;
                for(int k = 0; k < _tmp; ++k) {
                    values[_count] = -1.;
                    _count += 1;
                    values[_count] = 1.;
                    _count += 1;
                };
                for(int k = 0; k < _tmp - 1; ++k) {
                    values[_count] = 1;
                    _count += 1;
                    values[_count] = -1.;
                    _count += 1;
                };
            };
            return true;
        };
        bool eval_h(
                Index         n,
                const Number* x,
                bool          new_x,
                Number        obj_factor,
                Index         m,
                const Number* lambda,
                bool          new_lambda,
                Index         nele_hess,
                Index*        iRow,
                Index*        jCol,
                Number*       values
        )
        {
            return true;
        };
        void finalize_solution(
                SolverReturn               status,
                Index                      n,
                const Number*              x,
                const Number*              z_L,
                const Number*              z_U,
                Index                      m,
                const Number*              g,
                const Number*              lambda,
                Number                     obj_value,
                const IpoptData*           ip_data,
                IpoptCalculatedQuantities* ip_cq
        )
        {
            for(int k = 0; k < _p_opt.size(); ++k) {
                _p_opt_ipopt(k) = x[k];
            };
        };
    };
    class NLP {
    public:
        SmartPtr<Plant> plant;
        NLP() { plant = new Plant(); };
        // Set functions
        void set_p_const(const vector<double> &p_const) { (*plant).set_p_const(p_const); };
        void set_p_dynamic(const vector<double> &p_dynamic) { (*plant).set_p_dynamic(p_dynamic); };
        void set_p_optimize(const vector<double> &p_opt) { (*plant).set_p_optimize(p_opt); };
        void set_t0(const double t0) { (*plant).set_t0(t0); };
        void set_tf(const double tf) { (*plant).set_tf(tf); };
        void set_dt(const double dt) { (*plant).set_dt(dt); };
        void set_lower_bound(const vector<double> &lower_bound) { (*plant).set_lower_bound(lower_bound); };
        void set_upper_bound(const vector<double> &upper_bound) { (*plant).set_upper_bound(upper_bound); };
        void set_on_bound(const vector<double> &on_bound) { (*plant).set_on_bound(on_bound); };
        void set_off_bound(const vector<double> &off_bound) { (*plant).set_off_bound(off_bound); };
        void set_x0(const vector<double> x0) { (*plant).set_x0(x0); };
        // Get functions
        const vector<double> &get_p_const() const { return (*plant).get_p_const(); };
        const vector<double> &get_p_dynamic() const { return (*plant).get_p_dynamic(); };
        const vector<double> &get_p_optimize() const { return (*plant).get_p_optimize(); };
        const vector<double> &get_p_optimize_ipopt() const { return (*plant).get_p_optimize_ipopt(); };
        const double &get_t0() const { return (*plant).get_t0(); };
        const double &get_tf() const { return (*plant).get_tf(); };
        const double &get_dt() const { return (*plant).get_dt(); };
        const vector<double> &get_x0() const { return (*plant).get_x0(); };
        const vector<double> &get_lower_bound() const { return (*plant).get_lower_bound(); };
        const vector<double> &get_upper_bound() const { return (*plant).get_upper_bound(); };
        const vector<double> &get_on_bound() const { return (*plant).get_on_bound(); };
        const vector<double> &get_off_bound() const { return (*plant).get_off_bound(); };
        const int &get_init_status() const { return (*plant).get_init_status(); };
        const int &get_solve_status() const { return (*plant).get_solve_status(); };
        // IPOPT wrapper
        void solve() {
            // Define IPOPT application
            SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
            // Set options
            std::string tag = "tol";
            std::string val = "";
            app->Options()->SetNumericValue(tag, 1e-4);
            tag = "hessian_approximation";
            val = "limited-memory";
            app->Options()->SetStringValue(tag, val);
            tag = "print_level";
            app->Options()->SetIntegerValue(tag, 5);
            // Initialize IPOPT application
            (*plant)._status_init = (int) app->Initialize();
            // Solve NLP
            (*plant)._status_solve = (int) app->OptimizeTNLP(plant);
        };
    };
}

#endif //SWITCHINGTIMES_SWITCHING_TIMES_HPP
