import json
import math
import random
import numpy as np

# 从JSON文件中读取服务依赖图和资源使用情况
def read_service_graph(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data['services'], data['dependencies']

# 分组算法
def group_services(SGraph, rho, services):
    nodes = set()
    edges = []
    for edge in SGraph:
        nodes.add(edge['c'])
        nodes.add(edge['p'])
        edges.append((edge['c'], edge['p'], edge['intensity']))

    G = []
    grouped = set()

    while edges:
        max_intensity = 0
        max_edge = None
        for c, p, intensity in edges:
            if intensity > max_intensity:
                max_intensity = intensity
                max_edge = (c, p, intensity)
        
        if max_edge:
            c, p, intensity = max_edge
            edges.remove(max_edge)
            
            added_to_group = False
            for group in G:
                if c in group or p in group:
                    if len(group) < rho:
                        if c not in grouped:
                            group.add(c)
                            grouped.add(c)
                        if p not in grouped:
                            group.add(p)
                            grouped.add(p)
                        added_to_group = True
                        break
                    else:
                        added_to_group = True
                        break
            
            if not added_to_group:
                new_group = {c, p}
                G.append(new_group)
                grouped.update({c, p})

    for node in nodes - grouped:
        if node not in grouped:
            G.append({node})

    return G

# 计算每个组的CPU和内存使用率总和
def calculate_group_usage(G, services):
    group_usage = []
    for group in G:
        group_cpu = 0
        group_memory = 0
        for service_id in group:
            for service in services:
                if service['id'] == service_id:
                    group_cpu += service['cpu_usage']
                    group_memory += service['memory_usage']
        group_usage.append({'cpu': group_cpu, 'memory': group_memory})
    return group_usage

# 计算标准差
def calculate_std_dev(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    return math.sqrt(variance)

# 适应度函数：计算CPU和内存使用率的标准差之和
def fitness(G, services):
    group_usage = calculate_group_usage(G, services)
    cpu_std_dev = calculate_std_dev([usage['cpu'] for usage in group_usage])
    memory_std_dev = calculate_std_dev([usage['memory'] for usage in group_usage])
    return cpu_std_dev + memory_std_dev

# 初始化种群位置
def initialize_population(pop_size, num_services):
    return [random.randint(2, num_services) for _ in range(pop_size)]

# 更新海鸥位置（迁徙）
def update_position(seagulls, best_seagull, num_services, iteration, max_iteration):
    # A = 2 - (iteration * 2) / max_iteration  # 控制A变化频率的函数
    A = 2 * math.exp(-6 * (iteration / max_iteration)**3)
    for i in range(len(seagulls)):
        Cs = seagulls[i] * A
        B = 2 * (A ** 2) * random.random()
        Ms = B * (best_seagull - seagulls[i])
        Ds = abs(Cs + Ms)
        seagulls[i] = max(2, min(int(Ds), num_services))
    return seagulls

# 局部搜索（攻击猎物）
def local_search(seagulls, best_seagull, SGraph, services, num_services):
    for i in range(len(seagulls)):
        r = random.uniform(0, 1)  # 螺旋半径
        theta = random.uniform(0, 2 * np.pi)  # 随机角度
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = r * theta
        new_position = (x * y * z + best_seagull)
        new_position = max(2, min(int(new_position), num_services))
        # 评估新位置的适应度
        current_fitness = fitness(group_services(SGraph, seagulls[i], services), services)
        new_fitness = fitness(group_services(SGraph, new_position, services), services)
        if new_fitness < current_fitness:
            seagulls[i] = new_position
    return seagulls

# 动态反向学习
def dynamic_opposite_learning(best_seagull, num_services, w=0.5):
    # 随机生成两个随机数
    rand1 = random.random()
    rand2 = random.random()
    
    # 计算动态相反点
    dynamic_opposite = best_seagull + w * rand1 * (rand2 * num_services - best_seagull)
    
    # 确保动态相反点在有效范围内
    dynamic_opposite = max(2, min(int(dynamic_opposite), num_services))
    
    return dynamic_opposite

# 更新海鸥的位置（迁徙）之前，执行动态反向学习
dynamic_opposite_seagull = dynamic_opposite_learning(best_seagull, num_services)
dynamic_opposite_fitness = fitness(group_services(SGraph, dynamic_opposite_seagull, services), services)

# 如果动态相反海鸥的适应度小于当前最优适应度，则更新最优海鸥
if dynamic_opposite_fitness < best_fitness:
    best_fitness = dynamic_opposite_fitness
    best_seagull = dynamic_opposite_seagull

# 更新海鸥的位置（迁徙）
seagulls = update_position(seagulls, best_seagull, num_services, iteration, max_iteration)
# 海鸥优化算法主函数
def seagull_optimization(filename, pop_size, max_iteration):
    services, SGraph = read_service_graph(filename)  # 读取服务依赖图和资源使用情况
    num_services = len(services)
    
    # 初始化种群位置
    seagulls = initialize_population(pop_size, num_services)
    
    # 计算初始适应度值
    fitness_values = [fitness(group_services(SGraph, seagull, services), services) for seagull in seagulls]
    
    # 找到初始全局最优解
    best_fitness = min(fitness_values)
    best_seagull_index = fitness_values.index(best_fitness)
    best_seagull = seagulls[best_seagull_index]
    
    # 开始迭代
    for iteration in range(max_iteration):
        
        # 动态反向学习
        dynamic_opposite_seagull = dynamic_opposite_learning(best_seagull, num_services)
        dynamic_opposite_fitness = fitness(group_services(SGraph, dynamic_opposite_seagull, services), services)
        
        # 如果动态相反海鸥的适应度小于当前最优适应度，则更新最优海鸥
        if dynamic_opposite_fitness < best_fitness:
            best_fitness = dynamic_opposite_fitness
            best_seagull = dynamic_opposite_seagull

        # 更新海鸥的位置（迁徙）
        seagulls = update_position(seagulls, best_seagull, num_services, iteration, max_iteration)
        
        # 计算适应度值
        fitness_values = [fitness(group_services(SGraph, seagull, services), services) for seagull in seagulls]
        
        # 更新全局最优解
        if min(fitness_values) < best_fitness:
            best_fitness = min(fitness_values)
            best_seagull_index = fitness_values.index(best_fitness)
            best_seagull = seagulls[best_seagull_index]
        
        # 局部搜索（攻击猎物）
        seagulls = local_search(seagulls, best_seagull, SGraph, services, num_services)
    
    # 最终最优分组
    optimal_groups = group_services(SGraph, best_seagull, services)
    
    return best_seagull, best_fitness, optimal_groups

# 参数
filename = 'intensity_3.json'  # JSON文件名
pop_size = 3                 # 种群大小
max_iteration = 100           # 最大迭代次数

# 运行海鸥优化算法
optimal_rho, optimal_fitness, optimal_groups = seagull_optimization(filename, pop_size, max_iteration)
print(f"Optimal group scale (rho): {optimal_rho}")
print(f"minimum total standard deviation: {optimal_fitness}")
print("Optimal group distribution:")
for i, group in enumerate(optimal_groups):
    members = ', '.join(group)
    print(f"Group {i+1}: Members: {members}")

# 计算并打印各组的资源使用率
# group_usage = calculate_group_usage(optimal_groups, services)
# for i, (group, usage) in enumerate(zip(optimal_groups, group_usage)):
#     print(f"Group {i+1}:")
#     print(f"  - CPU Usage: {usage['cpu']:.2f}")
#     print(f"  - Memory Usage: {usage['memory']:.2f}")
#     print()