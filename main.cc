// #include <iostream>
#include <print>
#include <string>
#include "PIP.hh" // 引入刚刚定义的三个遗传算法用的类
#include <format>
#include <iostream>

static constexpr struct parameters{ // 可以用于调整的参数
    std::size_t DIMENSION = 3; // 三维空间
    std::size_t POINT_COUNT = 10; // 每个个体包含10个点
    std::size_t POPULATION_SIZE = 50; // 种群规模
    std::size_t GENERATIONS = 1000; // 进化代数，这个不用于掐断，仅用于输出日志的频次
    double MUTATION_RATE = 0.1; // 变异率，对于这种问题，肯定是要调大的
} params; // 这些参数在编译时就确定了

int main() { // 遗传算法本体
    Population<params.DIMENSION> population(
        params.POPULATION_SIZE
    ); // 初始化种群

    // 先输出一个初始最优解的信息
    auto bestOverall = population.getBestIndividual();
    std::println("Initial best fitness: {}, point count: {}",
                 bestOverall.fitness(),
                 bestOverall.getPointCount()
    );

    int generation = 1;
    while (true){ // 一直进化，等人按Ctrl+C停止
        population.evolve(params.GENERATIONS, params.MUTATION_RATE); // 进化若干代

        // 为了让人知道确实在“进化”，输出当前最优个体的信息
        auto best = population.getBestIndividual();
        std::println("Current generation: {}, Current best fitness: {}, point count: {}",
                     params.GENERATIONS * generation,
                     best.fitness(),
                     best.getPointCount()
        );
        std::cout << best.toString() << std::endl; // 输出当前最优个体的一串点
        generation++;
    }
}