#include <iostream>
#include <chrono>
#include "src/switching-times.hpp"

using namespace Ipopt;

int main() {

    SwitchingTimes::NLP nlp;

    SwitchingTimes::vector<double> x0(4); x0(0) = 1.12; x0(1) = 0.87; x0(2) = 0.; x0(3) = 0.; // Initial values
    SwitchingTimes::vector<double> p_const(12);
    p_const(0) = 0.00067; p_const(1) = 36.9; p_const(2) = 0.073;                // Model parameters
    p_const(3) = 0.1; p_const(4) = 2.00; p_const(5) = 0.300; p_const(6) = 7.84; // ... continued
    p_const(7) = 0.5; p_const(8) = 0.;                                          // Tax rates
    p_const(9) = 1.;                                                            // Day-ahead sigmoid parameter
    p_const(10) = 1.; p_const(11) = 1.;                                         // Regime switch sigmoid parameters
    SwitchingTimes::vector<double> p_dynamic(48 + 49);
    Eigen::Map<SwitchingTimes::vector<double>> dap(p_dynamic.data(), 48);       // Day-ahead prices
    Eigen::Map<SwitchingTimes::vector<double>> dat(p_dynamic.data() + 48, 49);  // Day-ahead times
    for(int k = 0; k < 48; ++k) { dap(k) = 10.; };
    for(int k = 0; k < 49; ++k) { dat(k) = k * 60.; };
    dat(0) = dat(0) - 60;
    dat(48) = dat(48) + 60;
    double t0 = 0.;        // Starting time
    double tf = 6. * 60.;  // End time
    double dt = 0.2;       // Discretization

    int n_s = 10;
    SwitchingTimes::vector<double> p_opt = SwitchingTimes::vector<double>::Zero(n_s * 2);
    Eigen::Map<SwitchingTimes::vector<double>> on(p_opt.data(), n_s);        // Regime switch on
    Eigen::Map<SwitchingTimes::vector<double>> off(p_opt.data() + n_s, n_s); // Regime switch off

    SwitchingTimes::vector<double> lower_bound = SwitchingTimes::vector<double>::Zero(p_opt.size());
    SwitchingTimes::vector<double> upper_bound = SwitchingTimes::vector<double>::Zero(p_opt.size());
    for(int k = 0; k < upper_bound.size(); ++k) { upper_bound(k) = tf; };
    SwitchingTimes::vector<double> on_bound(2); on_bound(0) = 6.; on_bound(1) = 60.;
    SwitchingTimes::vector<double> off_bound(2); off_bound(0) = 20.; off_bound(1) = 120.;

    off(0) = on(0) + on_bound(0) + 1.;
    for(int k = 1; k < n_s; ++k) {
        on(k) = off(k - 1) + off_bound(0) + 1.;
        off(k) = on(k) + on_bound(0) + 1.;
    }

    nlp.set_p_const(p_const);
    nlp.set_p_dynamic(p_dynamic);
    nlp.set_p_optimize(p_opt);
    nlp.set_t0(t0);
    nlp.set_tf(tf);
    nlp.set_dt(dt);
    nlp.set_x0(x0);
    nlp.set_lower_bound(lower_bound);
    nlp.set_upper_bound(upper_bound);
    nlp.set_on_bound(on_bound);
    nlp.set_off_bound(off_bound);

    nlp.solve();

    SwitchingTimes::vector<double> p_ipopt = nlp.get_p_optimize_ipopt();
    Eigen::Map<SwitchingTimes::vector<double>> on_ipopt(p_ipopt.data(), n_s);        // Regime switch on
    Eigen::Map<SwitchingTimes::vector<double>> off_ipopt(p_ipopt.data() + n_s, n_s); // Regime switch off
    std::cout << "--" << std::endl;
    for(int k = 0; k < n_s; ++k) {
        std::cout << on_ipopt(k) << " is optimal - started at " << on(k) << std::endl;
    }
    std::cout << "--" << std::endl;
    for(int k = 0; k < n_s; ++k) {
        std::cout << off_ipopt(k) << " is optimal - started at " << off(k) << std::endl;
    }

    /*SwitchingTimes::vector<double> xf(x0);
    // Test integrate function
    auto start = std::chrono::high_resolution_clock::now();
    xf = plant.integrate(t0, tf, dt, xf);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Execution time: " << duration.count() / 1000000. << " sec" << std::endl;
    std::cout << xf(0) << " , " << xf(1) << " , " << xf(2) << " , " << xf(3) << std::endl;
    // Test jacobian function
    start = std::chrono::high_resolution_clock::now();
    SwitchingTimes::vector<double> jac1 = plant.jacobian(p_opt);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Execution time: " << duration.count() / 1000000. << " sec" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    SwitchingTimes::vector<double> jac2 = plant.jacobian(p_opt);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Execution time: " << duration.count() / 1000000. << " sec" << std::endl;*/
    return 0;
}