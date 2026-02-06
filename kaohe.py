import csv
# 打开CSV文件，变量名用f（合法且常用）
with open('task2.csv', 'r', encoding='utf-8') as f:
    # 创建CSV读取器
    reader = csv.reader(f)
    # 遍历读取器，逐行打印
    for row in reader:
        print(row)  # row为列表，每个元素是CSV的一列数据
