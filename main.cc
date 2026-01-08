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
    std::size_t insert_softmin_restarts = 0;
    std::size_t insert_softmin_steps = 0;
    double insert_softmin_beta = 12.0;
    double insert_softmin_step_start = 0.05;
    double insert_softmin_step_end = 0.005;
    std::size_t kopt_remove = 1;
    std::size_t kopt_restarts = 6;
    std::size_t kopt_candidates = 400;
    std::size_t kopt_big_remove = 0;
    double kopt_big_chance = 0.0;
    double lns_contact_margin = 0.05;
    std::size_t basin_hops = 0;
    double step_start = 0.06;
    double step_end = 0.005;
    double repulsion = 0.01;
    double repulsion_power = 3.0;
    double penalty_weight = 2.0;
    double target_start = 0.9;
    double target_end = 1.0;
    double penalty_start = 0.4;
    double penalty_end = 1.0;
    double repulsion_start = 0.4;
    double repulsion_end = 1.0;
    double repulsion_power_start = 2.0;
    double repulsion_power_end = 6.0;
    double kick_scale = 0.03;
    double kick_start = 0.0;
    double kick_end = 0.0;
    double accept_temp = 0.002;
    double accept_decay = 0.9;
    double accept_temp_start = 0.0;
    double accept_temp_end = 0.0;
    double kopt_kick_scale = 0.04;
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

static double min_distance_squared_to_set(const Point &p, const Points &points)
{
    if (points.empty())
    {
        return std::numeric_limits<double>::infinity();
    }
    double min_dist2 = std::numeric_limits<double>::infinity();
    for (const auto &q : points)
    {
        min_dist2 = std::min(min_dist2, distance_squared(p, q));
    }
    return min_dist2;
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

static bool load_seed_for_dimension(std::size_t dim, Points &points, std::string &label);
static Points seed_points_for_dimension(std::size_t dim, std::string &label);
static Point find_best_layer_direction(const Points &current,
                                       double h,
                                       std::size_t restarts,
                                       std::size_t steps,
                                       double beta,
                                       double step_start,
                                       double step_end,
                                       std::mt19937 &gen);

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

static Point lift_with_height(const Point &u, double h)
{
    double h2 = h * h;
    double scale = std::sqrt(std::max(0.0, 1.0 - h2));
    Point p(u.size() + 1, 0.0);
    for (std::size_t i = 0; i < u.size(); ++i)
    {
        p[i] = u[i] * scale;
    }
    p[u.size()] = h;
    return p;
}

static Points embed_equator(const Points &base, std::size_t dim)
{
    Points out;
    out.reserve(base.size());
    for (const auto &p : base)
    {
        Point q(dim, 0.0);
        for (std::size_t i = 0; i + 1 < dim && i < p.size(); ++i)
        {
            q[i] = p[i];
        }
        out.push_back(std::move(q));
    }
    return out;
}

static Points generate_laminated_seed(std::size_t dim, std::string &label)
{
    if (dim <= 8)
    {
        return {};
    }

    std::string base_label;
    Points base = seed_points_for_dimension(dim - 1, base_label);
    if (base.empty() || base[0].size() != dim - 1)
    {
        return {};
    }

    Points equator = embed_equator(base, dim);
    Points best = equator;
    std::string best_mode;
    double best_h = 0.0;

    std::vector<double> heights = {0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85};
    std::size_t layer_attempts = 10 + dim * 2;
    std::size_t layer_restarts = 6 + dim / 4;
    std::size_t layer_steps = 80 + dim;
    double layer_beta = 12.0;
    double layer_step_start = 0.08;
    double layer_step_end = 0.01;

    std::mt19937 gen(static_cast<std::mt19937::result_type>(0x9E3779B9u ^ (dim * 2654435761u)));

    for (double h : heights)
    {
        if (h <= 0.0 || h >= 1.0)
        {
            continue;
        }
        Points current = equator;
        for (std::size_t i = 0; i < layer_attempts; ++i)
        {
            Point u = find_best_layer_direction(current,
                                                h,
                                                layer_restarts,
                                                layer_steps,
                                                layer_beta,
                                                layer_step_start,
                                                layer_step_end,
                                                gen);
            if (u.empty())
            {
                continue;
            }
            Point p_plus = lift_with_height(u, h);
            Point p_minus = lift_with_height(u, -h);

            double min_plus = min_distance_squared_to_set(p_plus, current);
            double min_minus = min_distance_squared_to_set(p_minus, current);
            bool ok_plus = min_plus >= 1.0;
            bool ok_minus = min_minus >= 1.0;
            bool allow_pair = (4.0 * h * h) >= 1.0 - 1e-12;

            if (ok_plus && ok_minus && allow_pair)
            {
                current.push_back(std::move(p_plus));
                current.push_back(std::move(p_minus));
            }
            else if (ok_plus || ok_minus)
            {
                if (ok_plus && (!ok_minus || min_plus >= min_minus))
                {
                    current.push_back(std::move(p_plus));
                }
                else
                {
                    current.push_back(std::move(p_minus));
                }
            }
        }
        if (current.size() > best.size())
        {
            best = current;
            best_h = h;
            best_mode = "single";
        }
    }

    Points stacked = equator;
    for (double h : heights)
    {
        if (h <= 0.0 || h >= 1.0)
        {
            continue;
        }
        for (std::size_t i = 0; i < layer_attempts; ++i)
        {
            Point u = find_best_layer_direction(stacked,
                                                h,
                                                layer_restarts,
                                                layer_steps,
                                                layer_beta,
                                                layer_step_start,
                                                layer_step_end,
                                                gen);
            if (u.empty())
            {
                continue;
            }
            Point p_plus = lift_with_height(u, h);
            Point p_minus = lift_with_height(u, -h);

            double min_plus = min_distance_squared_to_set(p_plus, stacked);
            double min_minus = min_distance_squared_to_set(p_minus, stacked);
            bool ok_plus = min_plus >= 1.0;
            bool ok_minus = min_minus >= 1.0;
            bool allow_pair = (4.0 * h * h) >= 1.0 - 1e-12;

            if (ok_plus && ok_minus && allow_pair)
            {
                stacked.push_back(std::move(p_plus));
                stacked.push_back(std::move(p_minus));
            }
            else if (ok_plus || ok_minus)
            {
                if (ok_plus && (!ok_minus || min_plus >= min_minus))
                {
                    stacked.push_back(std::move(p_plus));
                }
                else
                {
                    stacked.push_back(std::move(p_minus));
                }
            }
        }
    }

    if (stacked.size() > best.size())
    {
        best = stacked;
        best_mode = "stacked";
        best_h = 0.0;
    }

    if (best.size() == equator.size())
    {
        return {};
    }

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << std::setprecision(2);
    if (best_mode == "stacked")
    {
        oss << "Laminated(" << base_label << ") stacked";
    }
    else
    {
        oss << "Laminated(" << base_label << ") h=" << best_h;
    }
    label = oss.str();
    return best;
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
    if (dim > 8)
    {
        Points file_seed;
        if (load_seed_for_dimension(dim, file_seed, label))
        {
            return file_seed;
        }
    }
    if (dim > 8)
    {
        Points laminated = generate_laminated_seed(dim, label);
        if (!laminated.empty())
        {
            return laminated;
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

static bool load_seed_for_dimension(std::size_t dim, Points &points, std::string &label)
{
    std::vector<std::string> candidates = {
        "seeds/seed_d" + std::to_string(dim) + ".txt",
        "seeds/d" + std::to_string(dim) + ".txt",
        "seeds/" + std::to_string(dim) + ".txt",
        "seeds/seed_d" + std::to_string(dim) + ".dat",
        "seeds/d" + std::to_string(dim) + ".dat",
        "seeds/" + std::to_string(dim) + ".dat"};

    for (const auto &path : candidates)
    {
        Points tmp;
        std::size_t out_dim = 0;
        if (load_points_from_file(path, tmp, out_dim) && out_dim == dim)
        {
            points = std::move(tmp);
            label = "seed:" + path;
            return true;
        }
    }
    return false;
}

static Point optimize_softmin_point(const Points &points,
                                    std::size_t steps,
                                    double beta,
                                    double step_start,
                                    double step_end,
                                    std::mt19937 &gen)
{
    std::size_t dim = points.empty() ? 0 : points[0].size();
    Point x = random_unit_point(dim, gen);
    if (points.empty() || steps == 0)
    {
        return x;
    }

    for (std::size_t iter = 0; iter < steps; ++iter)
    {
        double max_arg = -std::numeric_limits<double>::infinity();
        for (const auto &p : points)
        {
            double d2 = distance_squared(x, p);
            double arg = -beta * d2;
            if (arg > max_arg)
            {
                max_arg = arg;
            }
        }

        double weight_sum = 0.0;
        Point weighted_sum(dim, 0.0);
        for (const auto &p : points)
        {
            double d2 = distance_squared(x, p);
            double w = std::exp(-beta * d2 - max_arg);
            weight_sum += w;
            for (std::size_t d = 0; d < dim; ++d)
            {
                weighted_sum[d] += w * p[d];
            }
        }
        if (weight_sum <= 0.0)
        {
            break;
        }

        for (double &v : weighted_sum)
        {
            v /= weight_sum;
        }

        Point grad(dim, 0.0);
        for (std::size_t d = 0; d < dim; ++d)
        {
            grad[d] = 2.0 * (x[d] - weighted_sum[d]);
        }
        double dot = dot_product(grad, x);
        for (std::size_t d = 0; d < dim; ++d)
        {
            grad[d] -= dot * x[d];
        }

        double t = (steps > 1)
                       ? static_cast<double>(iter) / static_cast<double>(steps - 1)
                       : 1.0;
        double step = step_end + (step_start - step_end) * (1.0 - t);
        for (std::size_t d = 0; d < dim; ++d)
        {
            x[d] += step * grad[d];
        }
        normalize(x);
    }
    return x;
}

static Point find_best_insert_point(const Points &base, const SearchParams &params, std::mt19937 &gen)
{
    if (base.empty())
    {
        return {};
    }

    double best_min_dist2 = -1.0;
    Point best_point;

    for (std::size_t i = 0; i < params.insert_softmin_restarts; ++i)
    {
        Point candidate = optimize_softmin_point(base,
                                                params.insert_softmin_steps,
                                                params.insert_softmin_beta,
                                                params.insert_softmin_step_start,
                                                params.insert_softmin_step_end,
                                                gen);
        double min_dist2 = min_distance_squared_to_set(candidate, base);
        if (min_dist2 > best_min_dist2)
        {
            best_min_dist2 = min_dist2;
            best_point = candidate;
        }
    }

    for (std::size_t i = 0; i < params.insert_candidates; ++i)
    {
        Point candidate = random_unit_point(base[0].size(), gen);
        double min_dist2 = min_distance_squared_to_set(candidate, base);
        if (min_dist2 > best_min_dist2)
        {
            best_min_dist2 = min_dist2;
            best_point = candidate;
        }
    }

    return best_point;
}

static Point optimize_layer_direction(const Points &current,
                                      double h,
                                      std::size_t steps,
                                      double beta,
                                      double step_start,
                                      double step_end,
                                      std::mt19937 &gen)
{
    if (current.empty())
    {
        return {};
    }

    std::size_t dim = current[0].size();
    if (dim < 2)
    {
        return {};
    }
    std::size_t sub_dim = dim - 1;
    double scale = std::sqrt(std::max(0.0, 1.0 - h * h));

    Point u = random_unit_point(sub_dim, gen);
    for (std::size_t iter = 0; iter < steps; ++iter)
    {
        double max_arg = -std::numeric_limits<double>::infinity();
        for (const auto &p : current)
        {
            double dot = 0.0;
            for (std::size_t d = 0; d < sub_dim; ++d)
            {
                dot += u[d] * p[d];
            }
            dot = scale * dot + h * p[sub_dim];
            double d2 = 2.0 - 2.0 * dot;
            double arg = -beta * d2;
            if (arg > max_arg)
            {
                max_arg = arg;
            }
        }

        double weight_sum = 0.0;
        Point weighted_sum(sub_dim, 0.0);
        for (const auto &p : current)
        {
            double dot = 0.0;
            for (std::size_t d = 0; d < sub_dim; ++d)
            {
                dot += u[d] * p[d];
            }
            dot = scale * dot + h * p[sub_dim];
            double d2 = 2.0 - 2.0 * dot;
            double w = std::exp(-beta * d2 - max_arg);
            weight_sum += w;
            for (std::size_t d = 0; d < sub_dim; ++d)
            {
                weighted_sum[d] += w * p[d];
            }
        }

        if (weight_sum <= 0.0)
        {
            break;
        }
        for (double &v : weighted_sum)
        {
            v /= weight_sum;
        }

        Point grad(sub_dim, 0.0);
        for (std::size_t d = 0; d < sub_dim; ++d)
        {
            grad[d] = -2.0 * scale * weighted_sum[d];
        }
        double dot = dot_product(grad, u);
        for (std::size_t d = 0; d < sub_dim; ++d)
        {
            grad[d] -= dot * u[d];
        }

        double t = (steps > 1)
                       ? static_cast<double>(iter) / static_cast<double>(steps - 1)
                       : 1.0;
        double step = step_end + (step_start - step_end) * (1.0 - t);
        for (std::size_t d = 0; d < sub_dim; ++d)
        {
            u[d] += step * grad[d];
        }
        normalize(u);
    }
    return u;
}

static Point find_best_layer_direction(const Points &current,
                                       double h,
                                       std::size_t restarts,
                                       std::size_t steps,
                                       double beta,
                                       double step_start,
                                       double step_end,
                                       std::mt19937 &gen)
{
    Point best_u;
    double best_min_dist2 = -1.0;
    for (std::size_t i = 0; i < restarts; ++i)
    {
        Point u = optimize_layer_direction(current, h, steps, beta, step_start, step_end, gen);
        if (u.empty())
        {
            continue;
        }
        Point p = lift_with_height(u, h);
        double min_dist2 = min_distance_squared_to_set(p, current);
        if (min_dist2 > best_min_dist2)
        {
            best_min_dist2 = min_dist2;
            best_u = std::move(u);
        }
    }
    return best_u;
}

static Points insert_point_best(const Points &base, const SearchParams &params, std::mt19937 &gen)
{
    Points result = base;
    if (base.empty())
    {
        return result;
    }

    Point best_point = find_best_insert_point(base, params, gen);
    if (!best_point.empty())
    {
        result.push_back(best_point);
    }
    return result;
}

static std::vector<std::size_t> worst_points_by_min_distance(const Points &points, std::size_t count)
{
    std::size_t n = points.size();
    std::vector<double> min_d2(n, std::numeric_limits<double>::infinity());
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = i + 1; j < n; ++j)
        {
            double d2 = distance_squared(points[i], points[j]);
            if (d2 < min_d2[i])
            {
                min_d2[i] = d2;
            }
            if (d2 < min_d2[j])
            {
                min_d2[j] = d2;
            }
        }
    }
    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](std::size_t a, std::size_t b)
              { return min_d2[a] < min_d2[b]; });
    if (count < indices.size())
    {
        indices.resize(count);
    }
    return indices;
}

static std::vector<std::size_t> random_indices(std::size_t n, std::size_t count, std::mt19937 &gen)
{
    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);
    if (count < indices.size())
    {
        indices.resize(count);
    }
    return indices;
}

static std::vector<std::size_t> contact_degree_indices(const Points &points, std::size_t count, double threshold)
{
    std::size_t n = points.size();
    std::vector<std::size_t> degree(n, 0);
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = i + 1; j < n; ++j)
        {
            if (distance_squared(points[i], points[j]) < threshold)
            {
                ++degree[i];
                ++degree[j];
            }
        }
    }
    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](std::size_t a, std::size_t b)
              { return degree[a] > degree[b]; });
    if (count < indices.size())
    {
        indices.resize(count);
    }
    return indices;
}

static std::vector<std::size_t> cluster_indices(const Points &points, std::size_t count, std::mt19937 &gen)
{
    std::size_t n = points.size();
    if (n == 0)
    {
        return {};
    }
    std::uniform_int_distribution<std::size_t> uni(0, n - 1);
    std::size_t pivot = uni(gen);
    std::vector<std::pair<double, std::size_t>> dist_idx;
    dist_idx.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        dist_idx.emplace_back(distance_squared(points[pivot], points[i]), i);
    }
    std::sort(dist_idx.begin(), dist_idx.end(),
              [](const auto &a, const auto &b)
              { return a.first < b.first; });
    std::vector<std::size_t> indices;
    for (std::size_t i = 0; i < dist_idx.size() && indices.size() < count; ++i)
    {
        indices.push_back(dist_idx[i].second);
    }
    return indices;
}

static std::vector<std::size_t> choose_removal_indices(const Points &points,
                                                       std::size_t count,
                                                       const SearchParams &params,
                                                       std::mt19937 &gen)
{
    if (points.empty() || count == 0)
    {
        return {};
    }

    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double r = uni(gen);
    if (r < 0.4)
    {
        return worst_points_by_min_distance(points, count);
    }
    if (r < 0.6)
    {
        return random_indices(points.size(), count, gen);
    }
    if (r < 0.8)
    {
        double threshold = std::max(1.0, params.target_end + params.lns_contact_margin);
        return contact_degree_indices(points, count, threshold);
    }
    return cluster_indices(points, count, gen);
}

static Points remove_points_by_index(const Points &points, const std::vector<std::size_t> &remove_indices)
{
    std::vector<char> remove_mask(points.size(), 0);
    for (std::size_t idx : remove_indices)
    {
        if (idx < remove_mask.size())
        {
            remove_mask[idx] = 1;
        }
    }

    Points result;
    result.reserve(points.size() - remove_indices.size());
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        if (!remove_mask[i])
        {
            result.push_back(points[i]);
        }
    }
    return result;
}

static Points kopt_repair(const Points &base,
                          std::size_t remove_k,
                          std::size_t insert_k,
                          const SearchParams &params,
                          std::mt19937 &gen)
{
    if (base.size() <= remove_k)
    {
        return base;
    }
    SearchParams local_params = params;
    local_params.insert_candidates = params.kopt_candidates;
    std::vector<std::size_t> remove_indices = choose_removal_indices(base, remove_k, params, gen);
    Points current = remove_points_by_index(base, remove_indices);
    for (std::size_t i = 0; i < insert_k; ++i)
    {
        Points next = insert_point_best(current, local_params, gen);
        if (next.size() == current.size())
        {
            break;
        }
        current = std::move(next);
    }
    return current;
}

static bool optimize_points(Points &points, const SearchParams &params, std::mt19937 &gen, double &out_min_dist2)
{
    const std::size_t n = points.size();
    const std::size_t dim = params.dimension;
    const double eps = 1e-12;

    double best_min_dist2 = 0.0;

    for (std::size_t iter = 0; iter < params.max_iters; ++iter)
    {
        double t = (params.max_iters > 1)
                       ? static_cast<double>(iter) / static_cast<double>(params.max_iters - 1)
                       : 1.0;
        double target_dist2 = params.target_start + (params.target_end - params.target_start) * t;
        double penalty_weight = params.penalty_weight *
                                (params.penalty_start + (params.penalty_end - params.penalty_start) * t);
        double repulsion = params.repulsion * (params.repulsion_start + (params.repulsion_end - params.repulsion_start) * t);
        double repulsion_power =
            params.repulsion_power_start + (params.repulsion_power_end - params.repulsion_power_start) * t;

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
                    double scale = (penalty_weight * gap) / dist;
                    for (std::size_t d = 0; d < dim; ++d)
                    {
                        double push = diff[d] * scale;
                        forces[i][d] += push;
                        forces[j][d] -= push;
                    }
                }

                if (repulsion > 0.0)
                {
                    double rep = repulsion / std::pow(dist + eps, repulsion_power);
                    for (std::size_t d = 0; d < dim; ++d)
                    {
                        double push = diff[d] * rep;
                        forces[i][d] += push;
                        forces[j][d] -= push;
                    }
                }
            }
        }

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

        if (min_dist2 >= params.target_end - params.tolerance)
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

    double kick_start = (params.kick_start > 0.0) ? params.kick_start : params.kick_scale;
    double kick_end = (params.kick_end > 0.0) ? params.kick_end : params.kick_scale * 0.5;
    double temp_start = (params.accept_temp_start > 0.0) ? params.accept_temp_start : params.accept_temp;
    double temp_end = params.accept_temp_end;
    if (temp_end <= 0.0)
    {
        std::size_t decay_steps = params.basin_hops > 0 ? params.basin_hops : 1;
        temp_end = temp_start * std::pow(params.accept_decay, static_cast<double>(decay_steps));
    }
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    for (std::size_t hop = 0; hop < params.basin_hops; ++hop)
    {
        double hop_t = (params.basin_hops > 1)
                           ? static_cast<double>(hop) / static_cast<double>(params.basin_hops - 1)
                           : 1.0;
        double temperature = temp_start + (temp_end - temp_start) * hop_t;
        double kick_scale = kick_start + (kick_end - kick_start) * hop_t;

        Points candidate = current;
        apply_kick(candidate, kick_scale, gen);

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
    }

    points = std::move(best);
    out_min_dist2 = best_min;
    return best_min >= params.target_end - params.tolerance;
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
                Points attempt = insert_point_best(best_points, params, gen);
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

        if (!best_points.empty() && !success && params.kopt_restarts > 0 && params.kopt_remove > 0)
        {
            std::size_t max_remove = std::min(params.kopt_remove, best_points.size() - 1);
            if (max_remove > 0)
            {
                std::uniform_real_distribution<double> uni(0.0, 1.0);
                for (std::size_t attempt_idx = 0; attempt_idx < params.kopt_restarts && !success; ++attempt_idx)
                {
                    std::size_t remove_k = 1 + (attempt_idx % max_remove);
                    if (params.kopt_big_remove > 0 && uni(gen) < params.kopt_big_chance)
                    {
                        remove_k = std::min(params.kopt_big_remove, best_points.size() - 1);
                    }
                    Points attempt = kopt_repair(best_points,
                                                 remove_k,
                                                 remove_k + 1,
                                                 params,
                                                 gen);
                    if (attempt.size() != n)
                    {
                        continue;
                    }
                    if (params.kopt_kick_scale > 0.0)
                    {
                        apply_kick(attempt, params.kopt_kick_scale, gen);
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
        params.insert_softmin_restarts = 0;
        params.insert_softmin_steps = 0;
        params.kopt_remove = 1;
        params.kopt_restarts = 2;
        params.kopt_candidates = 200;
        params.kopt_big_remove = 0;
        params.kopt_big_chance = 0.0;
        params.lns_contact_margin = 0.05;
        params.basin_hops = 0;
        params.step_start = 0.05;
        params.step_end = 0.005;
        params.repulsion = 0.01;
        params.repulsion_power = 3.0;
        params.penalty_weight = 2.0;
        params.target_start = 0.95;
        params.penalty_start = 0.6;
        params.repulsion_start = 0.6;
        params.repulsion_power_start = 2.0;
        params.repulsion_power_end = 6.0;
        params.kick_scale = 0.02;
        params.accept_temp = 0.001;
        params.accept_decay = 0.9;
        params.kopt_kick_scale = 0.02;
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
        params.insert_softmin_restarts = 0;
        params.insert_softmin_steps = 0;
        params.kopt_remove = 1;
        params.kopt_restarts = 3;
        params.kopt_candidates = 300;
        params.kopt_big_remove = 0;
        params.kopt_big_chance = 0.0;
        params.lns_contact_margin = 0.05;
        params.basin_hops = 0;
        params.step_start = 0.08;
        params.step_end = 0.005;
        params.repulsion = 0.02;
        params.repulsion_power = 3.0;
        params.penalty_weight = 4.0;
        params.target_start = 0.93;
        params.penalty_start = 0.5;
        params.repulsion_start = 0.5;
        params.repulsion_power_start = 2.0;
        params.repulsion_power_end = 6.0;
        params.kick_scale = 0.02;
        params.accept_temp = 0.001;
        params.accept_decay = 0.9;
        params.kopt_kick_scale = 0.03;
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
        params.insert_softmin_restarts = 4;
        params.insert_softmin_steps = 60;
        params.insert_softmin_beta = 10.0;
        params.kopt_remove = 1;
        params.kopt_restarts = 4;
        params.kopt_candidates = 300;
        params.kopt_big_remove = 0;
        params.kopt_big_chance = 0.0;
        params.lns_contact_margin = 0.06;
        params.basin_hops = 6;
        params.step_start = 0.05;
        params.step_end = 0.004;
        params.repulsion = 0.01;
        params.repulsion_power = 3.0;
        params.penalty_weight = 2.5;
        params.target_start = 0.9;
        params.penalty_start = 0.45;
        params.repulsion_start = 0.4;
        params.repulsion_power_start = 2.0;
        params.repulsion_power_end = 6.0;
        params.kick_scale = 0.03;
        params.kick_start = 0.05;
        params.kick_end = 0.02;
        params.accept_temp = 0.002;
        params.accept_decay = 0.9;
        params.accept_temp_start = 0.003;
        params.accept_temp_end = 0.0005;
        params.kopt_kick_scale = 0.04;
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
        params.insert_softmin_restarts = 12;
        params.insert_softmin_steps = 80;
        params.insert_softmin_beta = 14.0;
        params.kopt_remove = 2;
        params.kopt_restarts = 6;
        params.kopt_candidates = 600;
        params.kopt_big_remove = 6;
        params.kopt_big_chance = 0.15;
        params.lns_contact_margin = 0.08;
        params.basin_hops = 6;
        params.step_start = 0.05;
        params.step_end = 0.003;
        params.repulsion = 0.01;
        params.repulsion_power = 3.0;
        params.penalty_weight = 2.0;
        params.target_start = 0.85;
        params.penalty_start = 0.3;
        params.repulsion_start = 0.3;
        params.repulsion_power_start = 2.0;
        params.repulsion_power_end = 8.0;
        params.kick_scale = 0.02;
        params.kick_start = 0.06;
        params.kick_end = 0.02;
        params.accept_temp = 0.001;
        params.accept_decay = 0.9;
        params.accept_temp_start = 0.003;
        params.accept_temp_end = 0.0005;
        params.kopt_kick_scale = 0.05;
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
        params.insert_softmin_restarts = 24;
        params.insert_softmin_steps = 120;
        params.insert_softmin_beta = 16.0;
        params.insert_softmin_step_start = 0.06;
        params.insert_softmin_step_end = 0.008;
        params.kopt_remove = 2;
        params.kopt_restarts = 8;
        params.kopt_candidates = 900;
        params.kopt_big_remove = 10;
        params.kopt_big_chance = 0.2;
        params.lns_contact_margin = 0.1;
        params.basin_hops = 10;
        params.step_start = 0.04;
        params.step_end = 0.002;
        params.repulsion = 0.01;
        params.repulsion_power = 3.0;
        params.penalty_weight = 3.0;
        params.target_start = 0.82;
        params.penalty_start = 0.25;
        params.repulsion_start = 0.25;
        params.repulsion_power_start = 2.0;
        params.repulsion_power_end = 10.0;
        params.kick_scale = 0.01;
        params.kick_start = 0.08;
        params.kick_end = 0.02;
        params.accept_temp = 0.001;
        params.accept_decay = 0.9;
        params.accept_temp_start = 0.004;
        params.accept_temp_end = 0.0005;
        params.kopt_kick_scale = 0.06;
        params.jitter = 0.003;
        params.jitter_every = 250;
        params.tolerance = 1e-6;
    }
    else if (dim >= 9 && dim <= 20)
    {
        params.max_iters = 6000;
        params.restarts = 2;
        params.insert_candidates = 800;
        params.insert_restarts = 10;
        params.insert_softmin_restarts = 20;
        params.insert_softmin_steps = 100;
        params.insert_softmin_beta = 15.0;
        params.insert_softmin_step_start = 0.06;
        params.insert_softmin_step_end = 0.008;
        params.kopt_remove = 3;
        params.kopt_restarts = 10;
        params.kopt_candidates = 1000;
        params.kopt_big_remove = 12;
        params.kopt_big_chance = 0.2;
        params.lns_contact_margin = 0.1;
        params.basin_hops = 12;
        params.step_start = 0.04;
        params.step_end = 0.002;
        params.repulsion = 0.01;
        params.repulsion_power = 3.0;
        params.penalty_weight = 3.0;
        params.target_start = 0.82;
        params.penalty_start = 0.25;
        params.repulsion_start = 0.25;
        params.repulsion_power_start = 2.0;
        params.repulsion_power_end = 10.0;
        params.kick_scale = 0.01;
        params.kick_start = 0.08;
        params.kick_end = 0.02;
        params.accept_temp = 0.001;
        params.accept_decay = 0.9;
        params.accept_temp_start = 0.004;
        params.accept_temp_end = 0.0005;
        params.kopt_kick_scale = 0.06;
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
        params.insert_softmin_restarts = 8;
        params.insert_softmin_steps = 80;
        params.insert_softmin_beta = 12.0;
        params.kopt_remove = 2;
        params.kopt_restarts = 6;
        params.kopt_candidates = 600;
        params.kopt_big_remove = 8;
        params.kopt_big_chance = 0.15;
        params.lns_contact_margin = 0.08;
        params.basin_hops = 10;
        params.step_start = 0.05;
        params.step_end = 0.003;
        params.repulsion = 0.01;
        params.repulsion_power = 3.0;
        params.penalty_weight = 2.0;
        params.target_start = 0.84;
        params.penalty_start = 0.3;
        params.repulsion_start = 0.3;
        params.repulsion_power_start = 2.0;
        params.repulsion_power_end = 8.0;
        params.kick_scale = 0.03;
        params.accept_temp = 0.002;
        params.accept_decay = 0.9;
        params.kick_start = 0.07;
        params.kick_end = 0.02;
        params.accept_temp_start = 0.003;
        params.accept_temp_end = 0.0005;
        params.kopt_kick_scale = 0.05;
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
    if (dim == 5)
    {
        return 40;
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
    if (dim == 9)
    {
        return 306;
    }
    if (dim == 10)
    {
        return 510;
    }
    if (dim == 11)
    {
        return 593;
    }
    if (dim == 12)
    {
        return 840;
    }
    if (dim == 13)
    {
        return 1154;
    }
    if (dim == 14)
    {
        return 1932;
    }
    if (dim == 15)
    {
        return 2564;
    }
    if (dim == 16)
    {
        return 4320;
    }
    if (dim == 17)
    {
        return 5730;
    }
    if (dim == 18)
    {
        return 7654;
    }
    if (dim == 19)
    {
        return 11692;
    }
    if (dim == 20)
    {
        return 19448;
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
