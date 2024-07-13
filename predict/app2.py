'''
使用MNIST_zh_CNN模型进行预测
'''
import torch
import torchvision.transforms as transforms
import numpy as np
import PIL.Image as Image
import sys

sys.path.append('E:/\.app/GitsDepository/.works/neural-network/')
from src.modules import nets

dict = {
    0: "零", 1: "一",
    2: "二", 3: "三",
    4: "四", 5: "五",
    6: "六", 7: "七",
    8: "八", 9: "九",
    10: "十", 11: "百",
    12: "千", 13: "万",
    14: "亿"
}

def preprocess_image(imagepath):
    image = Image.open(imagepath)
    image = image.convert('L')
    if np.mean(np.array(image)) > 160:
        image = Image.eval(image, lambda x: 255 - x)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    image = transform(image)
    image = image.unsqueeze(0)

    return image.to('cuda:0')

def output(prediction):
    predictions = torch.softmax(prediction, dim=1)
    _, predicted = torch.max(predictions.data, dim=1)
    print(f"Prediction: {dict[predicted.item()]}")
    print(f"Confidence: {predictions[0][predicted.item()].item()}")
    predictions = predictions.cpu().detach().numpy()
    predictions = predictions.round(3)
    print(f"Probabilities: {predictions}")

def main():
    # Load the model
    model = nets.MNIST_zh_CNN()
    model.load_state_dict(torch.load(r"models\MNIST_zh_DNN.pth"))
    # Load the image
    image = preprocess_image(r"predict\board.png")

    # Make predictions
    prediction = model.forward(image)
    output(prediction)

if __name__ == "__main__":
    main()
