import torch
from torchvision import models
from torch import nn


if __name__ == '__main__':
	# Load Resnet18
	torch.manual_seed(42)
	model_ft = models.resnet18(weights=None)
	# Finetune Final few layers to adjust for tiny imagenet input
	model_ft.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
	model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
	num_features = model_ft.fc.in_features
	model_ft.fc = nn.Linear(num_features, 200)
