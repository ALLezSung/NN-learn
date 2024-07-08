import csv
import os
from PIL import Image, ImageDraw

# 生成器函数，逐行读取CSV文件并处理
def read_csv_rows(csv_file):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            yield list(map(int, row))  # 将每行数据转换为整数列表

# 处理单行数据并绘制图像
def process_row_and_draw_image(row):
    n = int(len(row) ** 0.5)  # 计算边长 n
    pixel_grid = [row[i:i+n] for i in range(0, len(row), n)]  # 将一维列表转换为二维像素网格

    # 创建新图像
    img = Image.new('L', (n, n))
    draw = ImageDraw.Draw(img)

    # 绘制图像
    for y in range(n):
        for x in range(n):
            gray_value = pixel_grid[y][x]
            draw.point((x, y), fill=gray_value)

    return img

# 主函数，处理CSV文件中的每一行数据
def main(csv_file, img_folder_path):
    rows = read_csv_rows(csv_file)
    os.makedirs(img_folder_path, exist_ok=True)

    size = 0
    for idx, row in enumerate(rows):
        try:
            print(f"Draw img-{idx}", end="")
            img = process_row_and_draw_image(row[1:])
            img.save(f'{img_folder_path}\\{idx}-{row[0]}.png')  #JPEG格式为有损压缩会导致数据出现噪点
            print(" Done.")
            size += 1
            if size == 100:
                break
        except Exception as e:
            print(f"Error processing row {idx}: {e}")

# 测试示例
if __name__ == '__main__':
    csv_file = r'database\MNIST\mnist_test.csv'
    img_folder_path = r'database\MNIST\imgs'

    main(csv_file, img_folder_path)
