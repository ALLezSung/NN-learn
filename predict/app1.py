import torch
import torchvision.transforms as transforms
import numpy as np
import PIL.Image as Image
import sys

sys.path.append(r"E:\.app\GitsDepository\.works\neural-network")
from src.modules import nets

def preprocess_image(imagepath):
    image = Image.open(imagepath)
    image = image.convert('L')
    if np.mean(np.array(image)) > 160:
        image = Image.eval(image, lambda x: 255 - x)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    image = transform(image)
    image = image.unsqueeze(0)

    return image.to('cuda:0')

def main():
    # Load the model
    model = nets.MNIST_DNN().to('cuda:0')
    model.load_state_dict(torch.load(r"models\MNIST_DNN_best.pth"))
    # Load the image
    image = preprocess_image(r"predict\board.png")

    # Make predictions
    prediction = model.forward(image)
    prediction = torch.softmax(prediction, dim=1)
    answer = torch.argmax(prediction, dim=1)
    print(f"probability: {prediction.tolist()}")
    print(f"Prediction: {answer.item()}")

if __name__ == "__main__":
    main()
