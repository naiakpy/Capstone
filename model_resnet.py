import torch
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

# Load the input image
InputImg = Image.open("/content/drive/MyDrive/capstone-master/Data/Pandar Island/download (1).jpg")

# Display the input image
plt.imshow(InputImg)

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),  # Resize the image to 256 pixels
    transforms.CenterCrop(224),  # Crop the center 224x224 pixels
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize the image
                         std=[0.229, 0.224, 0.225])
])

# Apply the transformations to the input image
Transformed_InputImg = transform(InputImg)
print(Transformed_InputImg.shape)

# Add a batch dimension to the image tensor
InputImg_Batched = torch.unsqueeze(Transformed_InputImg, 0)
print(InputImg_Batched.shape)

# Load the pre-trained ResNet101 model
resnet = models.resnet101(pretrained=True)

# Modify the final fully connected layer to have 2 output neurons
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 2)

# Set the model to evaluation mode
resnet.eval()

# Make a prediction using the ResNet model
output = resnet(InputImg_Batched)

# Load the ImageNet class labels
with open('/content/drive/MyDrive/capstone-master/Data/ImageNet1000Classes.txt') as classesfile:
    ImageNetClasses = [line.strip() for line in classesfile.readlines()]

# Get the predicted class
_, predicted = torch.max(output, 1)
percentage = torch.softmax(output, dim=1)[0] * 100
print(ImageNetClasses[predicted[0]], percentage[predicted[0]].item())

# Load the pre-trained AlexNet model
alexnet = models.alexnet(pretrained=True)
alexnet.eval()

# Make a prediction using the AlexNet model
out = alexnet(InputImg_Batched)

# Load the ImageNet class labels again
with open('/content/drive/MyDrive/capstone-master/Data/ImageNet1000Classes.txt') as classesfile:
    ImageNetClasses = [line.strip() for line in classesfile.readlines()]

# Get the predicted class for AlexNet
_, predicted = torch.max(out, 1)
percentage = torch.softmax(out, dim=1)[0] * 100
print(ImageNetClasses[predicted[0]], percentage[predicted[0]].item())

# Repeat the process with ResNet152

# Load the input image
InputImg_ = Image.open("/content/drive/MyDrive/capstone-master/Data/Pandar Island/download (1).jpg")

# Display the input image
plt.imshow(InputImg_)

# Apply the transformations to the input image
Transformed_InputImg_ = transform(InputImg_)
print(Transformed_InputImg_.shape)

# Add a batch dimension to the image tensor
InputImg_Batched_ = torch.unsqueeze(Transformed_InputImg_, 0)
print(InputImg_Batched_.shape)

# Load the pre-trained ResNet152 model
resnet = models.resnet152(pretrained=True)
resnet.eval()

# Make a prediction using the ResNet152 model
Output_ = resnet(InputImg_Batched_)

# Load the ImageNet class labels again
with open('/content/drive/MyDrive/capstone-master/Data/ImageNet1000Classes.txt') as classesfile:
    ImageNetClasses = [line.strip() for line in classesfile.readlines()]

# Get the predicted class for ResNet152
_, predicted = torch.max(Output_, 1)
percentage = torch.softmax(Output_, dim=1)[0] * 100
print(ImageNetClasses[predicted[0]], percentage[predicted[0]].item())
