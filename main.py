import numpy as np
from sklearn.cluster import KMeans
from itertools import combinations
import random
from collections import defaultdict
import csv
import json

def precompute_allocation_schemes(orders, location_map, config):
    """
    为每个订单的明细预计算可能的托盘分配方案，并记录缺货信息。
    返回：(allocation_schemes, shortage_report)
    """
    allocation_schemes = {}
    shortage_report = []
    allow_partial = config.get('allow_partial', False)
    max_schemes = config.get('max_schemes', 10)

    for order in orders:
        order_id = order['order_id']
        order_schemes = []

        # 为每个明细生成分配方案
        line_schemes = defaultdict(list)
        for detail in order['details']:
            line_id = detail['line_id']
            sku_id = detail['sku_id']
            required_qty = detail['required_qty']
            inventory = detail['available_inventory']
            total_qty = sum(t['qty'] for t in inventory)

            # 检查是否缺货
            if total_qty < required_qty:
                shortage_report.append({
                    'order_id': order_id,
                    'line_id': line_id,
                    'sku_id': sku_id,
                    'required_qty': required_qty,
                    'available_qty': total_qty,
                    'shortage_qty': required_qty - total_qty
                })

            # 生成可能的库存组合
            possible_combinations = []
            if allow_partial or total_qty >= required_qty:
                for r in range(1, len(inventory) + 1):
                    for combo in combinations(range(len(inventory)), r):
                        combo_qty = sum(inventory[i]['qty'] for i in combo)
                        if allow_partial or combo_qty >= required_qty:
                            alloc = []
                            remaining = min(required_qty, combo_qty) if allow_partial else required_qty
                            for i in combo:
                                qty = min(inventory[i]['qty'], remaining)
                                location_id = str(inventory[i]['location']).strip()
                                coords = location_map.get(location_id, (0, 0))
                                alloc.append({
                                    'tote_id': inventory[i]['tote_id'],
                                    'qty': qty,
                                    'coords': coords
                                })
                                remaining -= qty
                            if remaining == 0 or allow_partial:
                                possible_combinations.append(alloc)

            # 限制组合数量
            if possible_combinations:
                if len(possible_combinations) > max_schemes:
                    possible_combinations = random.sample(possible_combinations, max_schemes)
                line_schemes[line_id] = possible_combinations
            else:
                line_schemes[line_id] = []  # 无可用方案

        # 组合所有明细的方案
        if all(line_schemes[lid] for lid in line_schemes):
            import itertools
            line_ids = list(line_schemes.keys())
            scheme_combinations = list(itertools.product(*[line_schemes[lid] for lid in line_ids]))
            for combo in scheme_combinations[:max_schemes]:
                scheme = {line_ids[i]: combo[i] for i in range(len(line_ids))}
                order_schemes.append(scheme)
        allocation_schemes[order_id] = order_schemes or [{}]  # 空方案表示不选择

    return allocation_schemes, shortage_report

class GeneticAlgorithm:
    def __init__(self, orders, task_target, location_map, config, pop_size=50, max_gen=100, mutation_rate=0.01):
        """
        初始化遗传算法。
        """
        self.orders = orders
        self.task_target = task_target
        self.location_map = location_map
        self.config = config
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.mutation_rate = mutation_rate
        self.order_ids = [o['order_id'] for o in orders]
        self.allocation_schemes, self.initial_shortage = precompute_allocation_schemes(orders, location_map, config)
        self.population = self.initialize_population()

    def initialize_population(self):
        """初始化种群，确保生成有效染色体"""
        population = []
        while len(population) < self.pop_size:
            chromosome = []
            task_count = 0
            for order_id in self.order_ids:
                schemes = self.allocation_schemes[order_id]
                if schemes == [{}] or not schemes:
                    chromosome.append(-1)
                    continue
                if random.random() < 0.7 and task_count < self.task_target:
                    scheme_idx = random.randint(0, len(schemes) - 1)
                    order = next(o for o in self.orders if o['order_id'] == order_id)
                    tasks = sum(len(scheme[detail['line_id']]) 
                               for detail in order['details'] 
                               for scheme in [schemes[scheme_idx]])
                    if task_count + tasks <= self.task_target:
                        chromosome.append(scheme_idx)
                        task_count += tasks
                    else:
                        chromosome.append(-1)
                else:
                    chromosome.append(-1)
            if self.is_valid(chromosome):
                population.append(chromosome)
        return population

    def generate_tasks(self, chromosome):
        """根据染色体生成托盘任务，并记录部分分配的缺货"""
        tasks = []
        tote_usage = defaultdict(float)
        partial_shortages = []
        allow_partial = self.config.get('allow_partial', False)

        for i, order_id in enumerate(self.order_ids):
            scheme_idx = chromosome[i]
            if scheme_idx == -1:
                continue
            scheme = self.allocation_schemes[order_id][scheme_idx]
            order = next(o for o in self.orders if o['order_id'] == order_id)
            for detail in order['details']:
                line_id = detail['line_id']
                required_qty = detail['required_qty']
                if line_id not in scheme:
                    return [], partial_shortages
                allocated_qty = 0
                for alloc in scheme[line_id]:
                    tote_id = alloc['tote_id']
                    qty = alloc['qty']
                    tote_usage[tote_id] += qty
                    allocated_qty += qty
                    max_qty = next(
                        (t['qty'] for d in order['details']
                         for t in d['available_inventory']
                         if t['tote_id'] == tote_id), 0)
                    if tote_usage[tote_id] > max_qty:
                        return [], partial_shortages
                    tasks.append({
                        'tote_id': tote_id,
                        'order_id': order_id,
                        'line_id': line_id,
                        'assigned_qty': qty,
                        'coords': alloc['coords']
                    })
                if allow_partial and allocated_qty < required_qty:
                    partial_shortages.append({
                        'order_id': order_id,
                        'line_id': line_id,
                        'sku_id': detail['sku_id'],
                        'required_qty': required_qty,
                        'allocated_qty': allocated_qty,
                        'shortage_qty': required_qty - allocated_qty
                    })
        return tasks, partial_shortages

    def is_valid(self, chromosome):
        """检查染色体是否有效"""
        tasks, _ = self.generate_tasks(chromosome)
        return len(tasks) <= self.task_target and tasks != []

    def fitness(self, chromosome):
        """计算适应度"""
        tasks, partial_shortages = self.generate_tasks(chromosome)
        if not tasks:
            return 0.0

        # 任务数量适应度
        task_count = len(tasks)
        task_fitness = np.exp(-((task_count - self.task_target) / self.task_target) ** 2)

        # 空间聚集性适应度
        coords = np.array([task['coords'] for task in tasks])
        unique_coords = np.unique(coords, axis=0)
        if len(unique_coords) < 1:
            clustering_fitness = 1.0
        else:
            n_clusters = min(len(unique_coords), self.config.get('max_clusters', 5))
            if n_clusters == 1:
                clustering_fitness = 1.0
            else:
                kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                kmeans.fit(unique_coords)
                clustering_fitness = 1 / (1 + kmeans.inertia_)

        # 部分分配惩罚（可选）
        if partial_shortages and not self.config.get('allow_partial', False):
            return 0.0  # 如果不允许部分分配，惩罚无效染色体
        return task_fitness * clustering_fitness

    def select(self):
        """轮盘赌选择"""
        fitnesses = [self.fitness(chrom) for chrom in self.population]
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            return random.sample(self.population, 2)
        probs = [f / total_fitness for f in fitnesses]
        return random.choices(self.population, weights=probs, k=2)

    def crossover(self, parent1, parent2):
        """单点交叉"""
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def mutate(self, chromosome):
        """变异"""
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                order_id = self.order_ids[i]
                schemes = self.allocation_schemes[order_id]
                if schemes == [{}]:
                    chromosome[i] = -1
                else:
                    if random.random() < 0.5:
                        chromosome[i] = -1
                    else:
                        chromosome[i] = random.randint(0, len(schemes) - 1)
        return chromosome

    def evolve(self):
        """进化一代"""
        new_population = []
        while len(new_population) < self.pop_size:
            parent1, parent2 = self.select()
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            if self.is_valid(child1):
                new_population.append(child1)
            if len(new_population) < self.pop_size and self.is_valid(child2):
                new_population.append(child2)
        self.population = new_population

    def run(self):
        """运行遗传算法，返回任务和缺货报告"""
        best_fitness = 0
        best_tasks = []
        best_shortages = []
        for gen in range(self.max_gen):
            self.evolve()
            best_chrom = max(self.population, key=self.fitness)
            fitness = self.fitness(best_chrom)
            tasks, partial_shortages = self.generate_tasks(best_chrom)
            if fitness > best_fitness:
                best_fitness = fitness
                best_tasks = tasks
                best_shortages = partial_shortages
            print(f"Generation {gen + 1}, Best Fitness: {best_fitness:.4f}, Tasks: {len(best_tasks)}, Shortages: {len(best_shortages)}")
        
        # 合并初始缺货和部分分配缺货
        shortage_report = self.initial_shortage + best_shortages
        return best_tasks, shortage_report

def read_location_map(file_path):
    """读取 location_map.csv 文件"""
    location_map = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            locationid, x, y = row
            location_map[locationid] = (float(x), float(y))
    return location_map

def read_config(file_path):
    """读取 config.json 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

if __name__ == "__main__":
    # 读取输入
    location_map_file = 'location_map.csv'
    config_file = 'config.json'
    orders_file = 'test_orders.json'

    try:
        location_map = read_location_map(location_map_file)
        config = read_config(config_file)
        with open(orders_file, 'r', encoding='utf-8') as f:
            orders = json.load(f)
    except Exception as e:
        print(f"读取文件错误：{e}")
        exit(1)

    # 运行遗传算法
    task_target = 50
    ga = GeneticAlgorithm(orders, task_target, location_map, config, pop_size=50, max_gen=100)
    tasks, shortage_report = ga.run()

    # 输出结果
    print("\n最佳托盘任务：")
    for task in tasks:
        print(task)
    print("\n缺货报告：")
    for report in shortage_report:
        print(report)