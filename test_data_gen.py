import random
import csv
import json
from collections import defaultdict

# 1. 从CSV读取所有货位ID
def load_locations():
    with open('location_map.csv', 'r', encoding='utf-8') as f:
        return [row['locationid'] for row in csv.DictReader(f)]

# 2. 生成全局托盘池
def generate_tote_pool(locations, total_totes=180):
    return [
        {
            'tote_id': f"T{tote_num}",
            'location': random.choice(locations),
            'stock': defaultdict(int)  # 存放 {SKU_ID: 数量}
        }
        for tote_num in range(1, total_totes + 1)
    ]

# 3. 分配SKU库存到托盘
def assign_sku_inventory(tote_pool):
    sku_inventory = defaultdict(list)
    
    for sku_num in range(1, 61):
        sku_id = f"SKU{sku_num}"
        
        # 随机选择2-3个托盘（允许重复选择）
        selected_totes = random.sample(tote_pool, k=random.randint(2, 3))
        
        for tote in selected_totes:
            # 生成当前SKU在这个托盘中的库存量
            qty = random.randint(10, 30)
            tote['stock'][sku_id] = qty
            
            # 记录到SKU库存信息（保持原结构）
            sku_inventory[sku_id].append({
                'tote_id': tote['tote_id'],
                'location': tote['location'],
                'qty': qty
            })
    
    return sku_inventory

# 4. 生成订单数据
def generate_orders(sku_inventory, num_orders=100):
    orders = []
    sku_list = list(sku_inventory.keys())
    
    for order_num in range(1, num_orders + 1):
        order_id = f"O{order_num}"
        
        # 随机选择两个不同的SKU
        selected_skus = random.sample(sku_list, 2)
        
        details = []
        for line_num, sku_id in enumerate(selected_skus, 1):
            details.append({
                'line_id': f"L{line_num}",
                'sku_id': sku_id,
                'required_qty': random.randint(5, 20),
                'available_inventory': sku_inventory[sku_id]
            })
        
        orders.append({
            'order_id': order_id,
            'details': details
        })
    
    return orders

# 主程序流程
if __name__ == "__main__":
    # 步骤1：加载货位数据
    locations = load_locations()
    
    # 步骤2：生成全局托盘池
    tote_pool = generate_tote_pool(locations)
    
    # 步骤3：分配SKU库存
    sku_inventory = assign_sku_inventory(tote_pool)
    
    # 步骤4：生成订单
    orders = generate_orders(sku_inventory)
    
    # 保存结果文件
    with open('test_orders.json', 'w', encoding='utf-8') as f:
        json.dump(orders, f, indent=2)
    
    # 验证输出结构
    print("示例订单数据：")
    print(json.dumps(orders[:2], indent=2))
    
    # 检查托盘共享情况（调试用）
    shared_totes = defaultdict(list)
    for tote in tote_pool:
        if len(tote['stock']) > 1:
            shared_totes[tote['tote_id']] = list(tote['stock'].keys())
    print(f"\n共享托盘数量：{len(shared_totes)} 个")
    print("示例共享托盘：")
    for tote_id, skus in list(shared_totes.items())[:3]:
        print(f"{tote_id} 存放：{skus}")