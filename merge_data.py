import sys
import os
import re
import csv

def main():
    # 1. 解析命令行参数获取目标路径
    if len(sys.argv) != 2:
        print("用法: python merge_data.py <目标文件夹路径>")
        sys.exit(1)

    target_dir = sys.argv[1]
    
    if not os.path.isdir(target_dir):
        print(f"错误: 路径 '{target_dir}' 不存在或不是一个文件夹。")
        sys.exit(1)

    # 2. 定义文件命名规则：saved_col_n (n为1-5位数字)
    csv_pattern = re.compile(r'^saved_col_(\d{1,5})\.csv$')
    txt_pattern = re.compile(r'^saved_col_(\d{1,5})\.txt$')
    
    all_files = os.listdir(target_dir)
    csv_files = []
    txt_files = []
    
    for f in all_files:
        if csv_pattern.match(f):
            csv_files.append(f)
        elif txt_pattern.match(f):
            txt_files.append(f)
            
    # 3. 判断处理格式 (CSV优先级大于TXT)
    if csv_files:
        target_files = csv_files
        pattern = csv_pattern
        print(f"检测到 {len(csv_files)} 个CSV文件，将只处理CSV。")
    elif txt_files:
        target_files = txt_files
        pattern = txt_pattern
        print(f"未检测到CSV文件，但检测到 {len(txt_files)} 个TXT文件，将处理TXT。")
    else:
        print("目标路径下未找到符合 'saved_col_n.csv' 或 'saved_col_n.txt' 规则的文件。")
        sys.exit(0)
        
    # 按照 n 的数值大小进行升序排序，确保列的顺序符合直觉（如 1, 2, 10 而不是 1, 10, 2）
    target_files.sort(key=lambda x: int(pattern.match(x).group(1)))
    
    merged_data = []
    
    # 4. 遍历文件并提取数据
    for file_idx, filename in enumerate(target_files):
        filepath = os.path.join(target_dir, filename)
        file_no_ext = os.path.splitext(filename)[0]
        
        # 假设文件编码为utf-8，并且以逗号分隔（哪怕是txt也兼容标准的csv逗号读取模式）
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            for row_idx, row in enumerate(reader):
                if len(row) < 2:
                    continue # 容错：忽略列数不足2的空行或异常行
                    
                col1_val = row[0]
                col2_val = row[1]
                
                # 第一个文件：负责初始化整个二维列表（包含第一列和自己的第二列）
                if file_idx == 0:
                    if row_idx == 0:
                        merged_data.append([col1_val, file_no_ext])
                    else:
                        merged_data.append([col1_val, col2_val])
                # 后续文件：只追加第二列的数据到现有行中
                else:
                    if row_idx < len(merged_data):
                        if row_idx == 0:
                            merged_data[row_idx].append(file_no_ext)
                        else:
                            merged_data[row_idx].append(col2_val)

    # 5. 输出合并结果
    out_path = os.path.join(target_dir, 'merged.csv')
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(merged_data)
        
    print(f"合并完成！共合并 {len(target_files)} 个文件。")
    print(f"输出文件路径: {out_path}")

if __name__ == '__main__':
    main()