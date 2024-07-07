import csv
import pathlib
from PIL import Image

def img_reader(img_path):
    label = pathlib.Path(img_path).stem.split('-')[-1]
    print(f"正在读取图像: {pathlib.Path(img_path).stem}", end="")
    img = Image.open(img_path).convert('L')  # 转换图像为灰度图像
    img_array = list(img.getdata())  # 获取扁平化的像素值列表
    # width, height = img.size
    # img_array = [img_array[i * width:(i + 1) * width] for i in range(height)]  # 将列表重新整形为二维列表/数组
    return label, img_array

def img_to_csv(img_path, csv_path):
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        label, img_list = img_reader(img_path)
        writer.writerow([label] + img_list)
        print(" Done.")


if __name__ == "__main__":
    img_folder_path = r"E:\.app\GitsDepository\.works\neural-network\database\MNIST\imgs"
    csv_path = r"E:\.app\GitsDepository\.works\neural-network\database\MNIST\test.csv"

    img_folder = pathlib.Path(img_folder_path)
    
    for entry in img_folder.iterdir():
        if entry.is_file() and entry.suffix.lower() == '.png':
            img_to_csv(entry, csv_path)
        elif entry.is_dir():
            pass