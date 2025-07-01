import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from engien import train, val
from utils import save_to_csv


if __name__ == '__main__':
	device = "cuda:0"

	data_dir = '/media/xdl/lxd/datasets'
	dataset = 'tiny_imagenet'
	batch_size = 256
	test_batch_size = 256
	epochs = 3

	dataset_dir = os.path.join(data_dir, dataset)
	train_dir = os.path.join(dataset_dir, 'train')
	val_dir = os.path.join(dataset_dir, 'val', 'images')

	print('Preparing dataset ...')
	norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])

	train_trans = [
		transforms.RandomHorizontalFlip(),
		transforms.RandomResizedCrop(224),
		transforms.ToTensor(),
	]

	val_trans = [
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		norm
	]

	train_data = datasets.ImageFolder(train_dir,
									 transform=transforms.Compose(train_trans + [norm]))

	val_data = datasets.ImageFolder(val_dir,
									transform=transforms.Compose(val_trans))

	print('Preparing data loaders ...')
	train_data_loader = torch.utils.data.DataLoader(train_data,
													batch_size=batch_size,
													shuffle=True)

	val_data_loader = torch.utils.data.DataLoader(val_data,
													batch_size=test_batch_size,
													shuffle=False)
	dataloaders = {"train": train_data_loader,
				   "val": val_data_loader}

	# Load Resnet18
	torch.manual_seed(42)
	model_ft = models.resnet18(weights=None)
	# Finetune Final few layers to adjust for tiny imagenet input
	model_ft.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
	model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
	num_features = model_ft.fc.in_features
	model_ft.fc = nn.Linear(num_features, 200)
	model_ft = model_ft.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

	train_losses = []
	train_acces = []
	val_acces = []
	best_acc = 0
	save_dir = "./weights/resnet18"

	for epoch in range(epochs):
		train_loss, train_acc = train(
			model=model_ft,
			dataloaders=dataloaders["train"],
			criterion=criterion,
			optimizer=optimizer_ft,
			device=device,
			epoch=epoch
		)
		torch.save(model_ft.state_dict(), f"{save_dir}/last.pt")

		val_acc = val(
			model=model_ft,
			dataloaders=dataloaders["val"],
			device=device,
			epoch=epoch
		)

		train_losses.append(train_loss)
		train_acces.append(train_acc)
		val_acces.append(val_acc)

		if val_acc > best_acc:
			val_acc = best_acc
			torch.save(model_ft.state_dict(), f"{save_dir}/best.pt")

	save_to_csv(train_losses, train_acces, val_acces, save_dir=save_dir)
