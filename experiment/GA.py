import numpy as np
import random

# 权重将被用来对这些基础预测进行加权平均，得到最终预测 p_i
# 在您的实际应用中，请替换这部分为您自己的数据和模型
NUM_SAMPLES = 100
NUM_BASE_MODELS = 5 # 这将是每个个体的权重数量，即染色体长度
y_true_classification = np.random.randint(0, 2, size=NUM_SAMPLES) # 用于CE
y_true_regression = np.random.rand(NUM_SAMPLES) * 10 # 用于MAE和RMSE
base_model_preds_classification = np.random.rand(NUM_SAMPLES, NUM_BASE_MODELS)
base_model_preds_regression = np.random.rand(NUM_SAMPLES, NUM_BASE_MODELS) * 10

def dummy_model_predict(weights, base_predictions):
    """
    模拟一个使用权重来生成最终预测的模型。
    这里使用加权平均。
    weights: 一个个体的权重数组 (chromosome)
    base_predictions: 基础模型们的预测结果
    """
    # 确保权重和为1
    weights = weights / np.sum(weights)
    # 计算加权平均预测
    final_prediction = np.dot(base_predictions, weights)
    return final_prediction

# --- 2. 适应度函数定义 ---
# 根据论文中的公式 (3-8), (3-9), (3-10)
# 注意：我们计算的是误差，后续会转换为适应度

def calculate_ce(weights):
    """计算交叉熵 (Cross-Entropy)"""
    p_i = dummy_model_predict(weights, base_model_preds_classification)
    # 防止log(0)
    p_i = np.clip(p_i, 1e-10, 1 - 1e-10)
    ce = -np.mean(y_true_classification * np.log(p_i) + (1 - y_true_classification) * np.log(1 - p_i))
    return ce

def calculate_mae(weights):
    """计算平均绝对误差 (Mean Absolute Error)"""
    p_i = dummy_model_predict(weights, base_model_preds_regression)
    mae = np.mean(np.abs(y_true_regression - p_i))
    return mae

def calculate_rmse(weights):
    """计算均方根误差 (Root Mean Squared Error)"""
    p_i = dummy_model_predict(weights, base_model_preds_regression)
    rmse = np.sqrt(np.mean((y_true_regression - p_i)**2))
    return rmse

# --- 3. 核心遗传算法实现 ---
# 对应伪代码中的 genetic_algorithm 函数
class GeneticAlgorithm:
    def __init__(self, fitness_func, population_size, chromosome_length,
                 num_generations, crossover_rate, mutation_rate, elite_size):
        self.fitness_func = fitness_func
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

    def _initialize_population(self, seed_individual=None):
        """初始化种群，可以包含一个种子个体"""
        population = [np.random.rand(self.chromosome_length) for _ in range(self.population_size)]
        if seed_individual is not None:
            population[0] = seed_individual # 用种子替换第一个
        return population

    def _evaluate_fitness(self, population):
        """评估种群中每个个体的适应度"""
        fitnesses = []
        for individual in population:
            error = self.fitness_func(individual)
            # 转换误差为适应度，误差越小，适应度越高
            fitness = 1.0 / (error + 1e-10)
            fitnesses.append(fitness)
        return np.array(fitnesses)

    def _selection(self, population, fitnesses):
        """轮盘赌选择"""
        total_fitness = np.sum(fitnesses)
        selection_probs = fitnesses / total_fitness
        selected_indices = np.random.choice(
            len(population), size=len(population), p=selection_probs
        )
        return [population[i] for i in selected_indices]

    def _crossover(self, parent1, parent2):
        """单点交叉"""
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.chromosome_length - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1, parent2

    def _mutation(self, individual):
        """高斯变异"""
        for i in range(self.chromosome_length):
            if random.random() < self.mutation_rate:
                # 添加一个小的随机扰动
                individual[i] += np.random.normal(0, 0.1)
                # 确保权重非负
                individual[i] = max(0, individual[i])
        return individual

    def run(self, seed_individual=None):
        """执行遗传算法主循环"""
        population = self._initialize_population(seed_individual)

        for generation in range(self.num_generations):
            fitnesses = self._evaluate_fitness(population)

            # --- 精英主义 ---
            elite_indices = np.argsort(fitnesses)[-self.elite_size:]
            elites = [population[i] for i in elite_indices]

            # --- 选择、交叉、变异 ---
            parents = self._selection(population, fitnesses)
            offspring = []
            for i in range(0, self.population_size - self.elite_size, 2):
                parent1, parent2 = parents[i], parents[i+1]
                child1, child2 = self._crossover(parent1, parent2)
                offspring.append(self._mutation(child1))
                offspring.append(self._mutation(child2))
            
            # --- 形成新种群 ---
            population = elites + offspring[:self.population_size - self.elite_size]

            if (generation + 1) % 10 == 0:
                best_fitness = np.max(fitnesses)
                print(f"Generation {generation+1}: Best Fitness = {best_fitness:.4f}, Error = {1/best_fitness:.4f}")

        # 返回最终种群中的最佳个体
        final_fitnesses = self._evaluate_fitness(population)
        best_index = np.argmax(final_fitnesses)
        return population[best_index]

# --- 4. 多阶段优化器 ---
# 对应伪代码的主体逻辑 (lines 11-15)
class MultiStageGAOptimizer:
    def __init__(self, population_size, num_generations, crossover_rate, mutation_rate, elite_size, chromosome_length):
        self.params = {
            "population_size": population_size,
            "num_generations": num_generations,
            "crossover_rate": crossover_rate,
            "mutation_rate": mutation_rate,
            "elite_size": elite_size,
            "chromosome_length": chromosome_length
        }
        print("多阶段遗传算法优化器已创建。")

    def run(self):
        # --- 阶段 1: 优化 CE ---
        print("\n--- [阶段 1] 开始: 优化交叉熵 (CE) ---")
        ga_ce = GeneticAlgorithm(fitness_func=calculate_ce, **self.params)
        elite_ce = ga_ce.run()
        print(f"阶段 1 完成。最优CE权重: {elite_ce / np.sum(elite_ce)}")
        print(f"对应的CE误差: {calculate_ce(elite_ce):.4f}")

        # --- 阶段 2: 优化 MAE，以 elite_ce 为种子 ---
        print("\n--- [阶段 2] 开始: 优化平均绝对误差 (MAE) ---")
        ga_mae = GeneticAlgorithm(fitness_func=calculate_mae, **self.params)
        elite_mae = ga_mae.run(seed_individual=elite_ce)
        print(f"阶段 2 完成。最优MAE权重: {elite_mae / np.sum(elite_mae)}")
        print(f"对应的MAE误差: {calculate_mae(elite_mae):.4f}")

        # --- 阶段 3: 优化 RMSE，以 elite_mae 为种子 ---
        print("\n--- [阶段 3] 开始: 优化均方根误差 (RMSE) ---")
        ga_rmse = GeneticAlgorithm(fitness_func=calculate_rmse, **self.params)
        elite_rmse = ga_rmse.run(seed_individual=elite_mae)
        print(f"阶段 3 完成。最优RMSE权重: {elite_rmse / np.sum(elite_rmse)}")
        print(f"对应的RMSE误差: {calculate_rmse(elite_rmse):.4f}")
        
        # --- 返回最终的最优权重配置 ---
        print("\n--- 多阶段优化完成 ---")
        final_weights = elite_rmse / np.sum(elite_rmse)
        return final_weights

# --- 主程序 ---
if __name__ == "__main__":
    # 定义超参数
    POPULATION_SIZE = 50
    NUM_GENERATIONS = 50
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.1
    ELITE_SIZE = 2
    CHROMOSOME_LENGTH = NUM_BASE_MODELS

    # 创建并运行优化器
    optimizer = MultiStageGAOptimizer(
        population_size=POPULATION_SIZE,
        num_generations=NUM_GENERATIONS,
        crossover_rate=CROSSOVER_RATE,
        mutation_rate=MUTATION_RATE,
        elite_size=ELITE_SIZE,
        chromosome_length=CHROMOSOME_LENGTH
    )
    
    final_optimal_weights = optimizer.run()
    
    print(f"\n最终得到的最优权重配置为: \n{final_optimal_weights}")

