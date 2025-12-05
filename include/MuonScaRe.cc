#pragma once
#include <cmath>
#include <algorithm>
#include <limits>
#include <random>
#include <iostream>
#include <boost/math/special_functions/erf.hpp>

// Robust Crystal Ball with numerical safeguards
struct CrystalBallScaRe {
    // constants
    static constexpr double pi = 3.14159265358979323846;
    static constexpr double tiny = 1e-300;
    static constexpr double eps_u = 1e-12; // clamp for uniform variates
    static constexpr double erf_eps = 1e-12;
    double sqrtPiOver2 = std::sqrt(pi / 2.0);
    double sqrt2 = std::sqrt(2.0);

    // parameters
    double m;
    double s;
    double a;
    double n;

    // internal derived
    double B{0}, C{0}, D{0}, N{0}, NA{0}, Ns{0}, NC{0}, F{0}, G{0}, k{0};
    double cdfMa{0}, cdfPa{1};
    bool useGaussianOnly{false};

    CrystalBallScaRe() : m(0), s(1), a(1.5), n(3.0) { init(); }
    CrystalBallScaRe(double mean, double sigma, double alpha, double nn) : m(mean), s(sigma), a(alpha), n(nn) { init(); }

    void init() {
        // sanitize parameters
        if (!std::isfinite(m)) m = 0.0;
        if (!std::isfinite(s) || s <= 0.0) s = 1e-6; // force positive sigma
        if (!std::isfinite(a)) a = 1.5;
        if (!std::isfinite(n) || n <= 1.0) n = 10.0; // n must be > 1 for tails; choose safe default

        double fa = std::fabs(a);

        // compute gaussian prefactors
        double ex = std::exp(-fa * fa / 2.0);
        // guard against division by zero
        double safe_fa = std::max(fa, 1e-12);

        double A = std::pow(n / safe_fa, n) * ex;
        double C1 = (n / safe_fa) / (n - 1.0) * ex;
        double D1 = 2.0 * sqrtPiOver2 * boost::math::erf(fa / sqrt2);

        B = n / safe_fa - fa;

        // If C1 is zero or D1 is zero -> fallback to gaussian only
        if (!std::isfinite(C1) || C1 <= 0.0 || !std::isfinite(D1) || D1 <= 0.0) {
            useGaussianOnly = true;
        } else {
            C = (D1 + 2.0 * C1) / C1;
            D = (D1 + 2.0 * C1) / 2.0;
            N = 1.0 / (s * (D1 + 2.0 * C1));
            k = 1.0 / (n - 1.0);
            NA = N * A;
            Ns = N * s;
            NC = Ns * C1;
            F = 1.0 - fa * fa / n;
            G = s * n / safe_fa;

            // if F <= 0 then tail formulas become problematic -> fallback
            if (!std::isfinite(F) || F <= 0.0) {
                useGaussianOnly = true;
            }
        }

        if (useGaussianOnly) {
            // set gaussian-only internal values (simpler)
            N = 1.0 / (s * std::sqrt(2.0 * pi));
            NA = 0.0;
            Ns = N * s;
            NC = 0.0;
            C = 0.0;
            D = std::sqrt(pi / 2.0); // not used in gaussian-only central formula for invcdf
            cdfMa = 0.0;
            cdfPa = 1.0;
            F = 0.0;
            G = s;
            k = 1.0;
        } else {
            // compute safe cdf boundaries for tail/central splits
            // use clamps when calling cdf to avoid recursion pitfalls
            cdfMa = cdf(m - a * s);
            cdfPa = cdf(m + a * s);
            // ensure they are in (0,1)
            cdfMa = std::clamp(cdfMa, 0.0 + eps_u, 1.0 - eps_u);
            cdfPa = std::clamp(cdfPa, 0.0 + eps_u, 1.0 - eps_u);
        }
    }

    // PDF (robust)
    double pdf(double x) const {
        double d = (x - m) / s;
        if (!useGaussianOnly) {
            if (d < -a) {
                double base = B - d;
                if (base <= 0.0) return 0.0;
                double val = NA * std::pow(base, -n);
                return std::isfinite(val) ? val : 0.0;
            }
            if (d > a) {
                double base = B + d;
                if (base <= 0.0) return 0.0;
                double val = NA * std::pow(base, -n);
                return std::isfinite(val) ? val : 0.0;
            }
        }
        double val = N * std::exp(-d * d / 2.0);
        return std::isfinite(val) ? val : 0.0;
    }

    // CDF (robust)
    double cdf(double x) const {
        double d = (x - m) / s;
        if (!useGaussianOnly) {
            if (d < -a) {
                double denom = (F - s * d / G);
                if (denom <= 0.0) {
                    // numeric problem -> return tiny
                    return eps_u;
                }
                double val = NC / std::pow(denom, n - 1.0);
                return std::clamp(val, 0.0 + eps_u, 1.0 - eps_u);
            }
            if (d > a) {
                double term = F + s * d / G;
                // C - term^(1-n)
                if (term <= 0.0) {
                    // numeric problem -> return near 1
                    return 1.0 - eps_u;
                }
                double val = NC * (C - std::pow(term, 1.0 - n));
                return std::clamp(val, 0.0 + eps_u, 1.0 - eps_u);
            }
        }
        // central gaussian part
        double arg = -d / sqrt2;
        double erfval = boost::math::erf(arg);
        double val = Ns * (D - sqrtPiOver2 * erfval);
        return std::clamp(val, 0.0 + eps_u, 1.0 - eps_u);
    }

    // inverse CDF (robust)
    double invcdf(double u_in) const {
        double u = u_in;
        if (!std::isfinite(u)) return m;
        u = std::clamp(u, 0.0 + eps_u, 1.0 - eps_u);

        // gaussian-only fallback
        if (useGaussianOnly) {
            // invert gaussian CDF approximately using erf_inv
            double arg = 1.0 - u / (N * s * std::sqrt(2.0 * pi)); // rough; better invert properly
            // safer approach: treat as standard normal centered in m
            double central_arg = 2.0 * (u - 0.5); // maps [0,1] to [-1,1], only rough
            central_arg = std::clamp(central_arg, -1.0 + erf_eps, 1.0 - erf_eps);
            double rv = m - sqrt2 * s * boost::math::erf_inv(central_arg);
            return std::isfinite(rv) ? rv : m;
        }

        // lower tail
        if (u < cdfMa) {
            if (NC <= 0.0 || !std::isfinite(NC) || !std::isfinite(k)) return m;
            double t = NC / u;
            // avoid too large/power causing overflow
            if (t < 0.0) return m;
            t = std::clamp(t, 0.0, 1e300);
            double p = std::pow(t, k);
            if (!std::isfinite(p)) {
                // fallback to boundary
                double rv = m - a * s; // use boundary value
                return rv;
            }
            double rv = m + G * (F - p);
            return std::isfinite(rv) ? rv : m;
        }

        // upper tail
        if (u > cdfPa) {
            if (NC <= 0.0 || !std::isfinite(NC) || !std::isfinite(k)) return m;
            double t = C - u / NC;
            // base must be positive for pow with negative exponent
            if (t <= 0.0) {
                // fallback to boundary
                double rv = m + a * s;
                return rv;
            }
            t = std::clamp(t, 1e-300, 1e300);
            double p = std::pow(t, -k);
            if (!std::isfinite(p)) {
                double rv = m + a * s;
                return rv;
            }
            double rv = m - G * (F - p);
            return std::isfinite(rv) ? rv : m;
        }

        // central gaussian inversion
        double arg = (D - u / Ns) / sqrtPiOver2;
        arg = std::clamp(arg, -1.0 + erf_eps, 1.0 - erf_eps);
        double inv = boost::math::erf_inv(arg);
        double rv = m - sqrt2 * s * inv;
        return std::isfinite(rv) ? rv : m;
    }
};

// Deterministic get_rndm using a provided seed (recommended for reproducibility)
inline double get_rndm(double eta, float nL, uint64_t seed) {
    double mean  = cset->at("cb_params")->evaluate({std::abs(eta), nL, 0});
    double sigma = cset->at("cb_params")->evaluate({std::abs(eta), nL, 1});
    double n     = cset->at("cb_params")->evaluate({std::abs(eta), nL, 2});
    double alpha = cset->at("cb_params")->evaluate({std::abs(eta), nL, 3});

    CrystalBallScaRe cb(mean, sigma, alpha, n);

    // deterministic RNG
    static thread_local std::mt19937_64 rng;
    rng.seed(seed);
    std::uniform_real_distribution<double> unif(CrystalBallScaRe::eps_u, 1.0 - CrystalBallScaRe::eps_u);
    double u = unif(rng);
    double x = cb.invcdf(u);
    if (!std::isfinite(x)) x = mean;
    return x;
}

// Convenience non-deterministic wrapper (only if you really want it)
inline double get_rndm(double eta, float nL) {
    std::random_device rd;
    uint64_t seed = (uint64_t)rd() ^ (uint64_t)(std::hash<double>{}(eta) << 1) ^ (uint64_t)(uint32_t(nL) << 7);
    return get_rndm(eta, nL, seed);
}


double get_std(double pt, double eta, float nL) {
    // obtain paramters from correctionlib
    double param_0 = cset->at("poly_params")->evaluate({abs(eta), nL, 0});
    double param_1 = cset->at("poly_params")->evaluate({abs(eta), nL, 1});
    double param_2 = cset->at("poly_params")->evaluate({abs(eta), nL, 2});

    // calculate value and return max(0, val)
    double sigma = param_0 + param_1 * pt + param_2 * pt * pt;
    if (sigma < 0)
        sigma = 0;
    return sigma;
}

double get_k(double eta, string var) {
    // obtain parameters from correctionlib
    double k_data = cset->at("k_data")->evaluate({abs(eta), var});
    double k_mc = cset->at("k_mc")->evaluate({abs(eta), var});

    // calculate residual smearing factor
    // return 0 if smearing in MC already larger than in data
    double k = 0;
    if (k_mc < k_data)
        k = sqrt(k_data * k_data - k_mc * k_mc);
    return k;
}

double pt_resol(double pt, double eta, float nL) {
    // load correction values
    double rndm = (double)get_rndm(eta, nL);
    double std = (double)get_std(pt, eta, nL);
    double k = (double)get_k(eta, "nom");

    // calculate corrected value and return original value if a parameter is nan
    double ptc = pt * (1 + k * std * rndm);
    if (isnan(ptc))
        ptc = pt;
    return ptc;
}

double pt_resol_var(double pt_woresol, double pt_wresol, double eta, string updn) {
    double k = (double)get_k(eta, "nom");

    if (k == 0)
        return pt_wresol;

    double k_unc = cset->at("k_mc")->evaluate({abs(eta), "stat"});

    double std_x_rndm = (pt_wresol / pt_woresol - 1) / k;

    double pt_var = pt_wresol;

    if (updn == "up") {
        pt_var = pt_woresol * (1 + (k + k_unc) * std_x_rndm);
    } else if (updn == "dn") {
        pt_var = pt_woresol * (1 + (k - k_unc) * std_x_rndm);
    } else {
        cout << "ERROR: updn must be 'up' or 'dn'" << endl;
    }

    return pt_var;
}

double pt_scale(bool is_data, double pt, double eta, double phi, int charge) {
    // use right correction
    string dtmc = "mc";
    if (is_data)
        dtmc = "data";

    double a = cset->at("a_" + dtmc)->evaluate({eta, phi, "nom"});
    double m = cset->at("m_" + dtmc)->evaluate({eta, phi, "nom"});
    return 1. / (m / pt + charge * a);
}

double pt_scale_var(double pt, double eta, double phi, int charge, string updn) {
    double stat_a = cset->at("a_mc")->evaluate({eta, phi, "stat"});
    double stat_m = cset->at("m_mc")->evaluate({eta, phi, "stat"});
    double stat_rho = cset->at("m_mc")->evaluate({eta, phi, "rho_stat"});

    double unc =
        pt * pt * sqrt(stat_m * stat_m / (pt * pt) + stat_a * stat_a + 2 * charge * stat_rho * stat_m / pt * stat_a);

    double pt_var = pt;

    if (updn == "up") {
        pt_var = pt + unc;
    } else if (updn == "dn") {
        pt_var = pt - unc;
    }

    return pt_var;
}

// #include <boost/math/special_functions/erf.hpp>
// #pragma once
// struct CrystalBallScaRe {
//     double pi = 3.14159;
//     double sqrtPiOver2 = sqrt(pi / 2.0);
//     double sqrt2 = sqrt(2.0);
//     double m;
//     double s;
//     double a;
//     double n;
//     double B;
//     double C;
//     double D;
//     double N;
//     double NA;
//     double Ns;
//     double NC;
//     double F;
//     double G;
//     double k;
//     double cdfMa;
//     double cdfPa;
//     CrystalBallScaRe() : m(0), s(1), a(10), n(10) { init(); }
//     CrystalBallScaRe(double mean, double sigma, double alpha, double n) : m(mean), s(sigma), a(alpha), n(n) { init(); }
//     void init() {
//         double fa = fabs(a);
//         double ex = exp(-fa * fa / 2);
//         double A = pow(n / fa, n) * ex;
//         double C1 = n / fa / (n - 1) * ex;
//         double D1 = 2 * sqrtPiOver2 * erf(fa / sqrt2);
//         B = n / fa - fa;
//         C = (D1 + 2 * C1) / C1;
//         D = (D1 + 2 * C1) / 2;
//         N = 1.0 / s / (D1 + 2 * C1);
//         k = 1.0 / (n - 1);
//         NA = N * A;
//         Ns = N * s;
//         NC = Ns * C1;
//         F = 1 - fa * fa / n;
//         G = s * n / fa;
//         cdfMa = cdf(m - a * s);
//         cdfPa = cdf(m + a * s);
//     }
//     double pdf(double x) const {
//         double d = (x - m) / s;
//         if (d < -a)
//             return NA * pow(B - d, -n);
//         if (d > a)
//             return NA * pow(B + d, -n);
//         return N * exp(-d * d / 2);
//     }
//     double pdf(double x, double ks, double dm) const {
//         double d = (x - m - dm) / (s * ks);
//         if (d < -a)
//             return NA / ks * pow(B - d, -n);
//         if (d > a)
//             return NA / ks * pow(B + d, -n);
//         return N / ks * exp(-d * d / 2);
//     }
//     double cdf(double x) const {
//         double d = (x - m) / s;
//         if (d < -a)
//             return NC / pow(F - s * d / G, n - 1);
//         if (d > a)
//             return NC * (C - pow(F + s * d / G, 1 - n));
//         return Ns * (D - sqrtPiOver2 * erf(-d / sqrt2));
//     }
//     double invcdf(double u) const {
//         if (u < cdfMa){
//             std::cout << "u = " << u <<", cdfMa " << cdfMa << std::endl;
//             std::cout << " m = " << m << std::endl;
//             std::cout << " G = " << G << std::endl;
//             std::cout << " F = " << F << std::endl;
//             std::cout << " NC = " << NC << std::endl;
//             std::cout << " k = " << k << std::endl;
//             std::cout << " m + G * (F - pow(NC / u, k)) << " <<  m + G * (F - pow(NC / u, k)) << std::endl;
//             return m + G * (F - pow(NC / u, k));
//         }
//         if (u > cdfPa){
//             std::cout << "u = " << u <<", cdfPa " << cdfPa << std::endl;
//             std::cout << "m = " << m << std::endl;
//             std::cout << "G = " << G << std::endl;
//             std::cout << "F = " << F << std::endl;
//             std::cout << "NC = " << NC << std::endl;
//             std::cout << "k = " << k << std::endl;
//             std::cout << "C << " << C << std::endl;
//             std::cout << "m - G * (F - pow(C - u / NC, -k)) << " <<   m - G * (F - pow(C - u / NC, -k)) << std::endl;

//             return m - G * (F - pow(C - u / NC, -k));
//         }
//         std::cout << "m = " << m << std::endl;
//         std::cout << "sqrt2 = " << sqrt2 << std::endl;
//         std::cout << "s = " << s << std::endl;
//         std::cout << "D = " << D << std::endl;
//         std::cout << "u = " << u << std::endl;
//         std::cout << "Ns = " << Ns << std::endl;
//         std::cout << "sqrtPiOver2 = " << sqrtPiOver2 << std::endl;

//         std::cout << "boost::math::erf_inv((D - u / Ns) / sqrtPiOver2) = " << boost::math::erf_inv((D - u / Ns) / sqrtPiOver2) << std::endl;
//         std::cout << "returning " << std::endl;
//         return m - sqrt2 * s * boost::math::erf_inv((D - u / Ns) / sqrtPiOver2);
//     }
// };

// double get_rndm(double eta, float nL) {
//     // obtain parameters from correctionlib
//     double mean = cset->at("cb_params")->evaluate({abs(eta), nL, 0});
//     double sigma = cset->at("cb_params")->evaluate({abs(eta), nL, 1});
//     double n = cset->at("cb_params")->evaluate({abs(eta), nL, 2});
//     double alpha = cset->at("cb_params")->evaluate({abs(eta), nL, 3});

//     // instantiate CB and get random number following the CB
//     CrystalBallScaRe cb(mean, sigma, alpha, n);
//     TRandom3 rnd(time(0));
//     double rndm = gRandom->Rndm();
//     std::cout << "rndm = " << rndm << std::endl;
//     std::cout << "cb.invcdf(rndm) " << cb.invcdf(rndm) << std::endl;
//     return cb.invcdf(rndm);
// }

// double get_std(double pt, double eta, float nL) {
//     // obtain paramters from correctionlib
//     double param_0 = cset->at("poly_params")->evaluate({abs(eta), nL, 0});
//     double param_1 = cset->at("poly_params")->evaluate({abs(eta), nL, 1});
//     double param_2 = cset->at("poly_params")->evaluate({abs(eta), nL, 2});

//     // calculate value and return max(0, val)
//     double sigma = param_0 + param_1 * pt + param_2 * pt * pt;
//     if (sigma < 0)
//         sigma = 0;
//     return sigma;
// }

// double get_k(double eta, string var) {
//     // obtain parameters from correctionlib
//     double k_data = cset->at("k_data")->evaluate({abs(eta), var});
//     double k_mc = cset->at("k_mc")->evaluate({abs(eta), var});

//     // calculate residual smearing factor
//     // return 0 if smearing in MC already larger than in data
//     double k = 0;
//     if (k_mc < k_data)
//         k = sqrt(k_data * k_data - k_mc * k_mc);
//     return k;
// }

// double pt_resol(double pt, double eta, float nL) {
//     // load correction values
//     double rndm = (double)get_rndm(eta, nL);
//     double std = (double)get_std(pt, eta, nL);
//     double k = (double)get_k(eta, "nom");

//     // calculate corrected value and return original value if a parameter is nan
//     double ptc = pt * (1 + k * std * rndm);
//     if (isnan(ptc))
//         ptc = pt;
//     return ptc;
// }

// double pt_resol_var(double pt_woresol, double pt_wresol, double eta, string updn) {
//     double k = (double)get_k(eta, "nom");

//     if (k == 0)
//         return pt_wresol;

//     double k_unc = cset->at("k_mc")->evaluate({abs(eta), "stat"});

//     double std_x_rndm = (pt_wresol / pt_woresol - 1) / k;

//     double pt_var = pt_wresol;

//     if (updn == "up") {
//         pt_var = pt_woresol * (1 + (k + k_unc) * std_x_rndm);
//     } else if (updn == "dn") {
//         pt_var = pt_woresol * (1 + (k - k_unc) * std_x_rndm);
//     } else {
//         cout << "ERROR: updn must be 'up' or 'dn'" << endl;
//     }

//     return pt_var;
// }

// double pt_scale(bool is_data, double pt, double eta, double phi, int charge) {
//     // use right correction
//     string dtmc = "mc";
//     if (is_data)
//         dtmc = "data";

//     double a = cset->at("a_" + dtmc)->evaluate({eta, phi, "nom"});
//     double m = cset->at("m_" + dtmc)->evaluate({eta, phi, "nom"});
//     return 1. / (m / pt + charge * a);
// }

// double pt_scale_var(double pt, double eta, double phi, int charge, string updn) {
//     double stat_a = cset->at("a_mc")->evaluate({eta, phi, "stat"});
//     double stat_m = cset->at("m_mc")->evaluate({eta, phi, "stat"});
//     double stat_rho = cset->at("m_mc")->evaluate({eta, phi, "rho_stat"});

//     double unc =
//         pt * pt * sqrt(stat_m * stat_m / (pt * pt) + stat_a * stat_a + 2 * charge * stat_rho * stat_m / pt * stat_a);

//     double pt_var = pt;

//     if (updn == "up") {
//         pt_var = pt + unc;
//     } else if (updn == "dn") {
//         pt_var = pt - unc;
//     }

//     return pt_var;
// }