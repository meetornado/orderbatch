import numpy as np
from sklearn.cluster import KMeans
from itertools import combinations
import random
from collections import defaultdict
import csv
import json
from joblib import Parallel, delayed

def precompute_allocation_schemes(orders, location_map, config):
    """
    为每个订单的明细预计算可能的托盘分配方案，并记录缺货信息。
    返回：(allocation_schemes, shortage_report)
    """
    allocation_schemes = {}
    shortage_report = []
    allow_partial = config.get('allow_partial', False)
    max_schemes = config.get('max_schemes', 20)  # 增加方案数量

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

            # 生成可能的库存组合，优先选择高库存托盘
            possible_combinations = []
            if allow_partial or total_qty >= required_qty:
                sorted_inventory = sorted(inventory, key=lambda x: x['qty'], reverse=True)
                for r in range(1, len(sorted_inventory) + 1):
                    for combo in combinations(range(len(sorted_inventory)), r):
                        combo_qty = sum(sorted_inventory[i]['qty'] for i in combo)
                        if allow_partial or combo_qty >= required_qty:
                            alloc = []
                            remaining = required_qty
                            for i in combo:
                                qty = min(sorted_inventory[i]['qty'], remaining)
                                if qty == 0:
                                    continue
                                location_id = str(sorted_inventory[i]['location']).strip()
                                coords = location_map.get(location_id, (0, 0))
                                alloc.append({
                                    'tote_id': sorted_inventory[i]['tote_id'],
                                    'qty': qty,
                                    'coords': coords
                                })
                                remaining -= qty
                            if (remaining == 0 or allow_partial) and alloc:
                                possible_combinations.append(alloc)

            # 限制组合数量，确保包含高库存组合
            if possible_combinations:
                if len(possible_combinations) > max_schemes:
                    # 优先保留满足需求量大的方案
                    possible_combinations.sort(key=lambda x: sum(a['qty'] for a in x), reverse=True)
                    possible_combinations = possible_combinations[:max_schemes]
                line_schemes[line_id] = possible_combinations
            else:
                line_schemes[line_id] = []

        # 组合所有明细的方案
        if all(line_schemes[lid] for lid in line_schemes):
            import itertools
            line_ids = list(line_schemes.keys())
            scheme_combinations = list(itertools.product(*[line_schemes[lid] for lid in line_ids]))
            for combo in scheme_combinations[:max_schemes]:
                scheme = {line_ids[i]: combo[i] for i in range(len(line_ids))}
                order_schemes.append(scheme)
        allocation_schemes[order_id] = order_schemes or [{}]

    return allocation_schemes, shortage_report

class GeneticAlgorithm:
    def __init__(self, orders, task_target, location_map, config, pop_size=50, max_gen=100, mutation_rate=0.1):
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
        self.order_map = {o['order_id']: o for o in orders}
        self.allocation_schemes, self.initial_shortage = precompute_allocation_schemes(orders, location_map, config)
        self.scheme_task_counts = self.precompute_scheme_task_counts()
        # 预筛选可行订单
        self.viable_order_indices = [i for i, oid in enumerate(self.order_ids)
                                     if self.allocation_schemes[oid] != [{}] and
                                     any(self.scheme_task_counts[oid])]
        self.population = self.initialize_population()

    def precompute_scheme_task_counts(self):
        """预计算每个订单的分配方案的任务数"""
        scheme_task_counts = {}
        for order_id in self.order_ids:
            schemes = self.allocation_schemes[order_id]
            if schemes == [{}] or not schemes:
                scheme_task_counts[order_id] = [0]
                continue
            counts = []
            order = self.order_map[order_id]
            for scheme in schemes:
                task_count = sum(len(scheme.get(detail['line_id'], []))
                                for detail in order['details'])
                counts.append(task_count)
            scheme_task_counts[order_id] = counts
        return scheme_task_counts

    def initialize_population(self):
        """初始化种群，优化速度和任务数"""
        population = []
        n_orders = len(self.order_ids)
        scheme_lengths = np.array([len(self.allocation_schemes[oid]) if self.allocation_schemes[oid] != [{}] else 0
                                  for oid in self.order_ids])
        max_schemes = max(len(counts) for counts in self.scheme_task_counts.values())
        task_counts_matrix = np.zeros((n_orders, max_schemes), dtype=int)
        for i, oid in enumerate(self.order_ids):
            counts = self.scheme_task_counts[oid]
            task_counts_matrix[i, :len(counts)] = counts

        # 动态选择概率，优先任务数适中的订单
        avg_tasks = np.array([np.mean(self.scheme_task_counts[oid]) if self.scheme_task_counts[oid] != [0] else 0
                              for oid in self.order_ids])
        select_prob = np.clip(0.95 - avg_tasks / (self.task_target + 1e-6) * 0.3, 0.7, 0.95)

        while len(population) < self.pop_size:
            batch_size = min(self.pop_size - len(population), 200)  # 增大批量
            select_probs = np.random.random((batch_size, n_orders)) < select_prob
            chromosomes = np.where(
                select_probs & (scheme_lengths > 0)[:, np.newaxis].T,
                np.random.randint(0, scheme_lengths, size=(batch_size, n_orders)),
                -1
            )

            # 向量化计算任务数
            task_counts = np.zeros(batch_size, dtype=int)
            for i in range(batch_size):
                valid_idx = chromosomes[i] != -1
                task_counts[i] = np.sum(task_counts_matrix[np.arange(n_orders)[valid_idx], chromosomes[i][valid_idx]])

            # 筛选有效染色体，优先接近 task_target
            valid_mask = (task_counts <= self.task_target * 1.5) & (task_counts >= self.task_target * 0.5)
            valid_chroms = chromosomes[valid_mask]
            valid_counts = task_counts[valid_mask]

            if valid_counts.size > 0:
                scores = np.exp(-((valid_counts - self.task_target) / self.task_target) ** 2)
                top_indices = np.argsort(scores)[-min(len(scores), self.pop_size - len(population)):]
                for idx in top_indices:
                    chrom = valid_chroms[idx].tolist()
                    if self.is_valid(chrom):
                        population.append(chrom)

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
            order = self.order_map[order_id]
            for detail in order['details']:
                line_id = detail['line_id']
                required_qty = detail['required_qty']
                if line_id not in scheme:
                    return [], partial_shortages
                allocated_qty = 0
                # 优先使用方案中的分配
                for alloc in scheme[line_id]:
                    tote_id = alloc['tote_id']
                    qty = alloc['qty']
                    if qty == 0:
                        continue
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

                # 动态补充剩余需求
                if allow_partial and allocated_qty < required_qty:
                    remaining = required_qty - allocated_qty
                    available_totes = [
                        t for t in detail['available_inventory']
                        if t['tote_id'] not in [a['tote_id'] for a in scheme[line_id]]
                    ]
                    sorted_totes = sorted(available_totes, key=lambda x: x['qty'], reverse=True)
                    for t in sorted_totes:
                        qty = min(t['qty'] - tote_usage[t['tote_id']], remaining)
                        if qty <= 0:
                            continue
                        location_id = str(t['location']).strip()
                        coords = self.location_map.get(location_id, (0, 0))
                        tote_usage[t['tote_id']] += qty
                        allocated_qty += qty
                        remaining -= qty
                        tasks.append({
                            'tote_id': t['tote_id'],
                            'order_id': order_id,
                            'line_id': line_id,
                            'assigned_qty': qty,
                            'coords': coords
                        })
                        if remaining <= 0:
                            break

                if allocated_qty < required_qty:
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
        return len(tasks) <= self.task_target * 1.5 and tasks != []

    def fitness(self, chromosome):
        """计算适应度"""
        tasks, partial_shortages = self.generate_tasks(chromosome)
        if not tasks:
            return 0.0

        task_count = len(tasks)
        task_fitness = np.exp(-((task_count - self.task_target) / self.task_target) ** 2) * 2.0  # 增加权重

        coords = np.array([task['coords'] for task in tasks])
        unique_coords = np.unique(coords, axis=0)
        if len(unique_coords) < 1:
            clustering_fitness = 1.0
        else:
            n_clusters = min(len(unique_coords), self.config.get('max_clusters', 5))
            if n_clusters == 1:
                clustering_fitness = 1.0
            else:
                kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0)
                kmeans.fit(unique_coords)
                clustering_fitness = 0.5 / (1 + kmeans.inertia_)  # 降低聚集性权重

        if partial_shortages and not self.config.get('allow_partial', False):
            return 0.0
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
        """两点交叉"""
        point1 = random.randint(1, len(parent1) - 2)
        point2 = random.randint(point1 + 1, len(parent1) - 1)
        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        return child1, child2

    def mutate(self, chromosome):
        """变异，优先调整有效方案"""
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                order_id = self.order_ids[i]
                schemes = self.allocation_schemes[order_id]
                if schemes == [{}]:
                    chromosome[i] = -1
                else:
                    if chromosome[i] == -1 or random.random() < 0.3:  # 减少变异为 -1
                        chromosome[i] = random.randint(0, len(schemes) - 1)
                    else:
                        chromosome[i] = random.randint(0, len(schemes) - 1)
        return chromosome

    def evolve(self):
        """进化一代，保留精英"""
        new_population = []
        fitnesses = [self.fitness(chrom) for chrom in self.population]
        elite_indices = np.argsort(fitnesses)[-max(1, self.pop_size // 10):]  # 保留前 10%
        for idx in elite_indices:
            new_population.append(self.population[idx])

        while len(new_population) < self.pop_size:
            parent1, parent2 = self.select()
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            if self.is_valid(child1):
                new_population.append(child1)
            if len(new_population) < self.pop_size and self.is_valid(child2):
                new_population.append(child2)
        self.population = new_population[:self.pop_size]

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

    task_target = 15
    ga = GeneticAlgorithm(orders, task_target, location_map, config, pop_size=50, max_gen=20)
    tasks, shortage_report = ga.run()

    print("\n最佳托盘任务：")
    for task in tasks:
        print(task)
    print("\n缺货报告：")
    for report in shortage_report:
        print(report)