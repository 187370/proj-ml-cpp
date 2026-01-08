#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using Point = std::vector<double>;
using Points = std::vector<Point>;

struct SearchParams {
    std::size_t dimension = 3;
    std::size_t max_iters = 8000;
    std::size_t restarts = 60;
    std::size_t insert_candidates = 200;
    std::size_t insert_restarts = 20;
    std::size_t basin_hops = 0;
    double step_start = 0.06;
    double step_end = 0.005;
    double repulsion = 0.01;
    double repulsion_power = 3.0;
    double penalty_weight = 2.0;
    double kick_scale = 0.03;
    double accept_temp = 0.002;
    double accept_decay = 0.9;
    double tolerance = 1e-6;
    double jitter = 0.0;
    std::size_t jitter_every = 200;
};

static void normalize(Point &p)
{
    double norm_squared = 0.0;
    for (double v : p)
    {
        norm_squared += v * v;
    }
    if (norm_squared == 0.0)
    {
        p.assign(p.size(), 0.0);
        if (!p.empty())
        {
            p[0] = 1.0;
        }
        return;
    }
    double inv_norm = 1.0 / std::sqrt(norm_squared);
    for (double &v : p)
    {
        v *= inv_norm;
    }
}

static double dot_product(const Point &a, const Point &b)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        sum += a[i] * b[i];
    }
    return sum;
}

static Point random_unit_point(std::size_t dim, std::mt19937 &gen)
{
    std::normal_distribution<double> dist(0.0, 1.0);
    Point p(dim);
    for (std::size_t i = 0; i < dim; ++i)
    {
        p[i] = dist(gen);
    }
    normalize(p);
    return p;
}

static Points random_points(std::size_t count, std::size_t dim, std::mt19937 &gen)
{
    Points points;
    points.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        points.push_back(random_unit_point(dim, gen));
    }
    return points;
}

static std::vector<Point> orthonormalize(const std::vector<Point> &candidates, double tol = 1e-10)
{
    std::vector<Point> basis;
    for (const auto &v : candidates)
    {
        Point w = v;
        for (const auto &b : basis)
        {
            double proj = dot_product(w, b);
            for (std::size_t i = 0; i < w.size(); ++i)
            {
                w[i] -= proj * b[i];
            }
        }
        double norm_sq = dot_product(w, w);
        if (norm_sq > tol * tol)
        {
            double inv_norm = 1.0 / std::sqrt(norm_sq);
            for (double &x : w)
            {
                x *= inv_norm;
            }
            basis.push_back(std::move(w));
        }
    }
    return basis;
}

static std::vector<Point> build_sum_zero_basis(std::size_t dim)
{
    std::vector<Point> candidates;
    candidates.reserve(dim > 0 ? dim - 1 : 0);
    for (std::size_t i = 0; i + 1 < dim; ++i)
    {
        Point v(dim, 0.0);
        v[i] = 1.0;
        v[dim - 1] = -1.0;
        candidates.push_back(std::move(v));
    }
    return orthonormalize(candidates);
}

static Points project_points_to_basis(const Points &points, const std::vector<Point> &basis)
{
    Points projected;
    projected.reserve(points.size());
    for (const auto &p : points)
    {
        Point out(basis.size(), 0.0);
        for (std::size_t i = 0; i < basis.size(); ++i)
        {
            out[i] = dot_product(p, basis[i]);
        }
        normalize(out);
        projected.push_back(std::move(out));
    }
    return projected;
}

static double distance_squared(const Point &a, const Point &b)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

static double min_distance_squared(const Points &points)
{
    if (points.size() < 2)
    {
        return 0.0;
    }

    double min_dist2 = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        for (std::size_t j = i + 1; j < points.size(); ++j)
        {
            min_dist2 = std::min(min_dist2, distance_squared(points[i], points[j]));
        }
    }
    return min_dist2;
}

static Points generate_dn_roots(std::size_t dim)
{
    Points points;
    if (dim < 2)
    {
        return points;
    }

    points.reserve(dim * (dim - 1) * 2);
    static const int signs[2] = {-1, 1};

    for (std::size_t i = 0; i < dim; ++i)
    {
        for (std::size_t j = i + 1; j < dim; ++j)
        {
            for (int s1 : signs)
            {
                for (int s2 : signs)
                {
                    Point p(dim, 0.0);
                    p[i] = static_cast<double>(s1);
                    p[j] = static_cast<double>(s2);
                    normalize(p);
                    points.push_back(std::move(p));
                }
            }
        }
    }
    return points;
}

static Points generate_e8_roots()
{
    constexpr std::size_t dim = 8;
    Points points;
    points.reserve(240);

    static const int signs[2] = {-1, 1};
    for (std::size_t i = 0; i < dim; ++i)
    {
        for (std::size_t j = i + 1; j < dim; ++j)
        {
            for (int s1 : signs)
            {
                for (int s2 : signs)
                {
                    Point p(dim, 0.0);
                    p[i] = static_cast<double>(s1);
                    p[j] = static_cast<double>(s2);
                    normalize(p);
                    points.push_back(std::move(p));
                }
            }
        }
    }

    for (std::size_t mask = 0; mask < (1u << dim); ++mask)
    {
        Point p(dim, 0.0);
        std::size_t neg_count = 0;
        for (std::size_t d = 0; d < dim; ++d)
        {
            int sign = (mask & (1u << d)) ? -1 : 1;
            if (sign < 0)
            {
                ++neg_count;
            }
            p[d] = 0.5 * static_cast<double>(sign);
        }
        if (neg_count % 2 != 0)
        {
            continue;
        }
        normalize(p);
        points.push_back(std::move(p));
    }

    return points;
}

static Points generate_e7_roots_8d()
{
    Points e8 = generate_e8_roots();
    Points e7;
    e7.reserve(126);
    for (const auto &p : e8)
    {
        double sum = std::accumulate(p.begin(), p.end(), 0.0);
        if (std::abs(sum) < 1e-9)
        {
            e7.push_back(p);
        }
    }
    return e7;
}

static Points generate_e7_roots()
{
    Points e7_8d = generate_e7_roots_8d();
    if (e7_8d.empty())
    {
        return {};
    }
    std::vector<Point> basis7 = build_sum_zero_basis(8);
    return project_points_to_basis(e7_8d, basis7);
}

static Points generate_e6_roots()
{
    Points e8 = generate_e8_roots();
    if (e8.empty())
    {
        return {};
    }

    const double tol = 1e-9;
    Point ref1;
    Point ref2;
    bool found = false;

    for (std::size_t i = 0; i < e8.size() && !found; ++i)
    {
        for (std::size_t j = i + 1; j < e8.size() && !found; ++j)
        {
            std::vector<Point> span = orthonormalize({e8[i], e8[j]});
            if (span.size() != 2)
            {
                continue;
            }
            std::size_t count = 0;
            for (const auto &p : e8)
            {
                if (std::abs(dot_product(p, span[0])) < tol &&
                    std::abs(dot_product(p, span[1])) < tol)
                {
                    ++count;
                }
            }
            if (count == 72)
            {
                ref1 = span[0];
                ref2 = span[1];
                found = true;
            }
        }
    }

    if (!found)
    {
        return {};
    }

    std::vector<Point> candidates;
    candidates.reserve(8);
    for (std::size_t k = 0; k < 8; ++k)
    {
        Point v(8, 0.0);
        v[k] = 1.0;
        double proj1 = dot_product(v, ref1);
        double proj2 = dot_product(v, ref2);
        for (std::size_t i = 0; i < v.size(); ++i)
        {
            v[i] -= proj1 * ref1[i] + proj2 * ref2[i];
        }
        candidates.push_back(std::move(v));
    }

    std::vector<Point> basis6 = orthonormalize(candidates);
    if (basis6.size() != 6)
    {
        return {};
    }

    Points e6_8d;
    e6_8d.reserve(72);
    for (const auto &p : e8)
    {
        if (std::abs(dot_product(p, ref1)) < tol &&
            std::abs(dot_product(p, ref2)) < tol)
        {
            e6_8d.push_back(p);
        }
    }
    return project_points_to_basis(e6_8d, basis6);
}

static Points seed_points_for_dimension(std::size_t dim, std::string &label)
{
    if (dim == 8)
    {
        label = "E8 roots";
        return generate_e8_roots();
    }
    if (dim == 7)
    {
        Points e7 = generate_e7_roots();
        if (!e7.empty())
        {
            label = "E7 roots";
            return e7;
        }
    }
    if (dim == 6)
    {
        Points e6 = generate_e6_roots();
        if (!e6.empty())
        {
            label = "E6 roots";
            return e6;
        }
    }
    if (dim >= 4)
    {
        label = "D" + std::to_string(dim) + " roots";
        return generate_dn_roots(dim);
    }
    return {};
}

static bool load_points_from_file(const std::string &path, Points &points, std::size_t &out_dim)
{
    std::ifstream input(path);
    if (!input)
    {
        return false;
    }

    std::string line;
    while (std::getline(input, line))
    {
        auto hash_pos = line.find('#');
        if (hash_pos != std::string::npos)
        {
            line = line.substr(0, hash_pos);
        }
        for (char &c : line)
        {
            if (c == ',' || c == ';')
            {
                c = ' ';
            }
        }
        std::istringstream iss(line);
        Point p;
        double value = 0.0;
        while (iss >> value)
        {
            p.push_back(value);
        }
        if (p.empty())
        {
            continue;
        }
        if (out_dim == 0)
        {
            out_dim = p.size();
        }
        if (p.size() != out_dim)
        {
            return false;
        }
        normalize(p);
        points.push_back(std::move(p));
    }

    return !points.empty();
}

static Points insert_point_best(const Points &base, std::size_t candidates, std::mt19937 &gen)
{
    Points result = base;
    if (base.empty())
    {
        return result;
    }

    double best_min_dist2 = -1.0;
    Point best_point;

    for (std::size_t i = 0; i < candidates; ++i)
    {
        Point candidate = random_unit_point(base[0].size(), gen);
        double min_dist2 = std::numeric_limits<double>::infinity();
        for (const auto &p : base)
        {
            min_dist2 = std::min(min_dist2, distance_squared(candidate, p));
        }
        if (min_dist2 > best_min_dist2)
        {
            best_min_dist2 = min_dist2;
            best_point = candidate;
        }
    }

    if (!best_point.empty())
    {
        result.push_back(best_point);
    }
    return result;
}

static bool optimize_points(Points &points, const SearchParams &params, std::mt19937 &gen, double &out_min_dist2)
{
    const std::size_t n = points.size();
    const std::size_t dim = params.dimension;
    const double target_dist2 = 1.0;
    const double eps = 1e-12;

    double best_min_dist2 = 0.0;

    for (std::size_t iter = 0; iter < params.max_iters; ++iter)
    {
        std::vector<Point> forces(n, Point(dim, 0.0));
        double min_dist2 = std::numeric_limits<double>::infinity();

        for (std::size_t i = 0; i < n; ++i)
        {
            for (std::size_t j = i + 1; j < n; ++j)
            {
                Point diff(dim);
                double dist2 = 0.0;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    double delta = points[i][d] - points[j][d];
                    diff[d] = delta;
                    dist2 += delta * delta;
                }

                min_dist2 = std::min(min_dist2, dist2);
                double dist = std::sqrt(dist2) + eps;

                if (dist2 < target_dist2)
                {
                    double gap = target_dist2 - dist2;
                    double scale = (params.penalty_weight * gap) / dist;
                    for (std::size_t d = 0; d < dim; ++d)
                    {
                        double push = diff[d] * scale;
                        forces[i][d] += push;
                        forces[j][d] -= push;
                    }
                }

                if (params.repulsion > 0.0)
                {
                    double rep = params.repulsion / std::pow(dist + eps, params.repulsion_power);
                    for (std::size_t d = 0; d < dim; ++d)
                    {
                        double push = diff[d] * rep;
                        forces[i][d] += push;
                        forces[j][d] -= push;
                    }
                }
            }
        }

        double t = static_cast<double>(iter) / static_cast<double>(params.max_iters);
        double step = params.step_end + (params.step_start - params.step_end) * (1.0 - t);
        step /= std::sqrt(static_cast<double>(n));

        for (std::size_t i = 0; i < n; ++i)
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                points[i][d] += step * forces[i][d];
            }
            normalize(points[i]);
        }

        best_min_dist2 = std::max(best_min_dist2, min_dist2);

        if (min_dist2 >= target_dist2 - params.tolerance)
        {
            out_min_dist2 = min_dist2;
            return true;
        }

        if (params.jitter > 0.0 && params.jitter_every > 0 && (iter + 1) % params.jitter_every == 0)
        {
            std::normal_distribution<double> noise(0.0, params.jitter * (1.0 - t));
            for (std::size_t i = 0; i < n; ++i)
            {
                for (std::size_t d = 0; d < dim; ++d)
                {
                    points[i][d] += noise(gen);
                }
                normalize(points[i]);
            }
        }
    }

    out_min_dist2 = best_min_dist2;
    return false;
}

static void apply_kick(Points &points, double scale, std::mt19937 &gen)
{
    if (scale <= 0.0)
    {
        return;
    }
    std::normal_distribution<double> noise(0.0, scale);
    for (auto &p : points)
    {
        for (double &v : p)
        {
            v += noise(gen);
        }
        normalize(p);
    }
}

static bool basin_hopping_optimize(Points &points,
                                   const SearchParams &params,
                                   std::mt19937 &gen,
                                   double &out_min_dist2)
{
    Points current = points;
    double current_min = 0.0;
    optimize_points(current, params, gen, current_min);

    Points best = current;
    double best_min = current_min;

    double temperature = params.accept_temp;
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    for (std::size_t hop = 0; hop < params.basin_hops; ++hop)
    {
        Points candidate = current;
        apply_kick(candidate, params.kick_scale, gen);

        double cand_min = 0.0;
        optimize_points(candidate, params, gen, cand_min);

        bool accept = false;
        if (cand_min >= current_min)
        {
            accept = true;
        }
        else if (temperature > 0.0)
        {
            double delta = cand_min - current_min;
            double prob = std::exp(delta / temperature);
            accept = (uni(gen) < prob);
        }

        if (accept)
        {
            current = candidate;
            current_min = cand_min;
        }

        if (cand_min > best_min)
        {
            best = candidate;
            best_min = cand_min;
        }

        temperature *= params.accept_decay;
    }

    points = std::move(best);
    out_min_dist2 = best_min;
    return best_min >= 1.0 - params.tolerance;
}

static std::string format_points(const Points &points)
{
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << std::setprecision(3);
    oss << "[";
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        if (i > 0)
        {
            oss << ", ";
        }
        oss << "(";
        for (std::size_t d = 0; d < points[i].size(); ++d)
        {
            if (d > 0)
            {
                oss << ", ";
            }
            oss << points[i][d];
        }
        oss << ")";
    }
    oss << "]";
    return oss.str();
}

static void run_dimension(std::size_t dim,
                          std::size_t max_points,
                          const SearchParams &base_params,
                          const Points &seed_points,
                          const std::string &seed_label)
{
    SearchParams params = base_params;
    params.dimension = dim;

    std::random_device rd;
    std::mt19937 gen(rd());

    Points best_points = seed_points;
    double best_min_dist2 = 0.0;

    if (!best_points.empty())
    {
        best_min_dist2 = min_distance_squared(best_points);
        std::cout << "d=" << dim
                  << " seed=" << seed_label
                  << " N=" << best_points.size()
                  << " min_dist=" << std::sqrt(best_min_dist2)
                  << std::endl;
    }

    if (best_points.size() >= max_points && !best_points.empty())
    {
        std::cout << "d=" << dim << " result K=" << best_points.size()
                  << " min_dist=" << std::sqrt(best_min_dist2) << std::endl;
        if (best_points.size() <= 60)
        {
            std::cout << format_points(best_points) << std::endl;
        }
        else
        {
            std::cout << "point list omitted (N > 60)" << std::endl;
        }
        return;
    }

    std::size_t start_n = best_points.empty() ? 2 : best_points.size() + 1;
    for (std::size_t n = start_n; n <= max_points; ++n)
    {
        bool success = false;
        double best_attempt_min_dist2 = 0.0;

        if (!best_points.empty())
        {
            for (std::size_t attempt_idx = 0; attempt_idx < params.insert_restarts && !success; ++attempt_idx)
            {
                Points attempt = insert_point_best(best_points, params.insert_candidates, gen);
                if (attempt.size() != n)
                {
                    continue;
                }
                double min_dist2 = 0.0;
                bool ok = params.basin_hops > 0
                              ? basin_hopping_optimize(attempt, params, gen, min_dist2)
                              : optimize_points(attempt, params, gen, min_dist2);
                if (ok)
                {
                    success = true;
                    best_points = attempt;
                    best_min_dist2 = min_dist2;
                }
                else
                {
                    best_attempt_min_dist2 = std::max(best_attempt_min_dist2, min_dist2);
                }
            }
        }

        for (std::size_t restart = 0; !success && restart < params.restarts; ++restart)
        {
            Points attempt = random_points(n, dim, gen);
            double min_dist2 = 0.0;
            bool ok = params.basin_hops > 0
                          ? basin_hopping_optimize(attempt, params, gen, min_dist2)
                          : optimize_points(attempt, params, gen, min_dist2);
            if (ok)
            {
                success = true;
                best_points = attempt;
                best_min_dist2 = min_dist2;
                break;
            }
            best_attempt_min_dist2 = std::max(best_attempt_min_dist2, min_dist2);
        }

        if (success)
        {
            std::cout << "d=" << dim
                      << " N=" << n
                      << " success min_dist=" << std::sqrt(best_min_dist2)
                      << std::endl;
        }
        else
        {
            std::cout << "d=" << dim
                      << " N=" << n
                      << " failed best_min_dist=" << std::sqrt(best_attempt_min_dist2)
                      << std::endl;
            break;
        }
    }

    if (!best_points.empty())
    {
        std::cout << "d=" << dim << " result K=" << best_points.size()
                  << " min_dist=" << std::sqrt(best_min_dist2) << std::endl;
        if (best_points.size() <= 60)
        {
            std::cout << format_points(best_points) << std::endl;
        }
        else
        {
            std::cout << "point list omitted (N > 60)" << std::endl;
        }
    }
}

static SearchParams default_params_for_dimension(std::size_t dim)
{
    SearchParams params;
    if (dim == 2)
    {
        params.max_iters = 4000;
        params.restarts = 30;
        params.insert_candidates = 120;
        params.insert_restarts = 20;
        params.basin_hops = 0;
        params.step_start = 0.05;
        params.step_end = 0.005;
        params.repulsion = 0.01;
        params.repulsion_power = 3.0;
        params.penalty_weight = 2.0;
        params.kick_scale = 0.02;
        params.accept_temp = 0.001;
        params.accept_decay = 0.9;
        params.jitter = 0.005;
        params.jitter_every = 200;
        params.tolerance = 1e-4;
    }
    else if (dim == 3)
    {
        params.max_iters = 20000;
        params.restarts = 120;
        params.insert_candidates = 600;
        params.insert_restarts = 30;
        params.basin_hops = 0;
        params.step_start = 0.08;
        params.step_end = 0.005;
        params.repulsion = 0.02;
        params.repulsion_power = 3.0;
        params.penalty_weight = 4.0;
        params.kick_scale = 0.02;
        params.accept_temp = 0.001;
        params.accept_decay = 0.9;
        params.jitter = 0.01;
        params.jitter_every = 200;
        params.tolerance = 1e-6;
    }
    else if (dim == 4)
    {
        params.max_iters = 6000;
        params.restarts = 20;
        params.insert_candidates = 200;
        params.insert_restarts = 20;
        params.basin_hops = 6;
        params.step_start = 0.05;
        params.step_end = 0.004;
        params.repulsion = 0.01;
        params.repulsion_power = 3.0;
        params.penalty_weight = 2.5;
        params.kick_scale = 0.03;
        params.accept_temp = 0.002;
        params.accept_decay = 0.9;
        params.jitter = 0.006;
        params.jitter_every = 200;
        params.tolerance = 1e-6;
    }
    else if (dim == 8)
    {
        params.max_iters = 4000;
        params.restarts = 0;
        params.insert_candidates = 400;
        params.insert_restarts = 10;
        params.basin_hops = 4;
        params.step_start = 0.05;
        params.step_end = 0.003;
        params.repulsion = 0.01;
        params.repulsion_power = 3.0;
        params.penalty_weight = 2.0;
        params.kick_scale = 0.02;
        params.accept_temp = 0.001;
        params.accept_decay = 0.9;
        params.jitter = 0.004;
        params.jitter_every = 200;
        params.tolerance = 1e-6;
    }
    else if (dim == 13)
    {
        params.max_iters = 5000;
        params.restarts = 0;
        params.insert_candidates = 600;
        params.insert_restarts = 8;
        params.basin_hops = 4;
        params.step_start = 0.04;
        params.step_end = 0.002;
        params.repulsion = 0.01;
        params.repulsion_power = 3.0;
        params.penalty_weight = 3.0;
        params.kick_scale = 0.01;
        params.accept_temp = 0.001;
        params.accept_decay = 0.9;
        params.jitter = 0.003;
        params.jitter_every = 250;
        params.tolerance = 1e-6;
    }
    else
    {
        params.max_iters = 5000;
        params.restarts = 20;
        params.insert_candidates = 200;
        params.insert_restarts = 10;
        params.basin_hops = 8;
        params.step_start = 0.05;
        params.step_end = 0.003;
        params.repulsion = 0.01;
        params.repulsion_power = 3.0;
        params.penalty_weight = 2.0;
        params.kick_scale = 0.03;
        params.accept_temp = 0.002;
        params.accept_decay = 0.9;
        params.jitter = 0.005;
        params.jitter_every = 200;
        params.tolerance = 1e-6;
    }
    return params;
}

static std::size_t default_target_for_dimension(std::size_t dim)
{
    if (dim == 2)
    {
        return 10;
    }
    if (dim == 3)
    {
        return 15;
    }
    if (dim == 4)
    {
        return 30;
    }
    if (dim == 6)
    {
        return 72;
    }
    if (dim == 7)
    {
        return 126;
    }
    if (dim == 8)
    {
        return 245;
    }
    if (dim == 13)
    {
        return 593;
    }
    return dim * 2;
}

int main(int argc, char **argv)
{
    bool no_seed = false;
    int argi = 1;
    if (argi < argc && std::string(argv[argi]) == "--no-seed")
    {
        no_seed = true;
        ++argi;
    }

    if (argi < argc && std::string(argv[argi]) == "--dim")
    {
        if (argi + 1 >= argc)
        {
            std::cerr << "Usage: " << argv[0] << " --dim <D> [target]" << std::endl;
            return 1;
        }
        std::size_t dim = static_cast<std::size_t>(std::stoul(argv[argi + 1]));
        std::size_t target = (argi + 2 < argc) ? static_cast<std::size_t>(std::stoul(argv[argi + 2]))
                                               : default_target_for_dimension(dim);

        std::string seed_label;
        Points seed_points = no_seed ? Points{} : seed_points_for_dimension(dim, seed_label);
        SearchParams params = default_params_for_dimension(dim);

        run_dimension(dim, target, params, seed_points, seed_label);
        return 0;
    }

    if (argi < argc)
    {
        std::string seed_path = argv[argi];
        std::size_t target = 0;
        if (argi + 1 < argc)
        {
            target = static_cast<std::size_t>(std::stoul(argv[argi + 1]));
        }

        Points seed_points;
        std::size_t dim = 0;
        if (!load_points_from_file(seed_path, seed_points, dim))
        {
            std::cerr << "Failed to load seed file: " << seed_path << std::endl;
            return 1;
        }
        if (target == 0)
        {
            target = seed_points.size();
        }

        SearchParams params = default_params_for_dimension(dim);

        run_dimension(dim, target, params, seed_points, "file");
        return 0;
    }

    SearchParams params2 = default_params_for_dimension(2);
    run_dimension(2, default_target_for_dimension(2), params2, {}, "");

    SearchParams params3 = default_params_for_dimension(3);
    run_dimension(3, default_target_for_dimension(3), params3, {}, "");

    std::string seed_label;
    Points seed4 = no_seed ? Points{} : seed_points_for_dimension(4, seed_label);
    SearchParams params4 = default_params_for_dimension(4);
    run_dimension(4, default_target_for_dimension(4), params4, seed4, seed_label);

    seed_label.clear();
    Points seed8 = no_seed ? Points{} : seed_points_for_dimension(8, seed_label);
    SearchParams params8 = default_params_for_dimension(8);
    run_dimension(8, default_target_for_dimension(8), params8, seed8, seed_label);

    return 0;
}
