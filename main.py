import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import argparse

from engien import train, val
from utils import save_to_csv


def get_args():
	parser = argparse.ArgumentParser(description="")
	# ---------------- hyper params ------------------
	parser.add_argument('--img_size', type=int, default=64, help='image resize')
	parser.add_argument('--bs', type=int, default=128)
	parser.add_argument('--topk', type=int, default=1, help='topk channel score')
	parser.add_argument('--dataset', type=str, default='tiny_imagenet', help='')
	parser.add_argument('--data_dir', type=str, default='/media/xdl/lxd/datasets')
	parser.add_argument('--lr', type=float, default=0.01)
	parser.add_argument('--model', type=str, default='resnet18')
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--device', type=str, default='cuda:0')

	args = parser.parse_args()
	args.device = torch.device(args.device)

	return args


if __name__ == '__main__':
	args = get_args()

	test_batch_size = 256

	# ================================= dataset =============================================
	if args.dataset == 'tiny_imagenet':
		dataset_dir = os.path.join(args.data_dir, args.dataset)
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
														batch_size=args.bs,
														shuffle=True)

		val_data_loader = torch.utils.data.DataLoader(val_data,
														batch_size=test_batch_size,
														shuffle=False)
		dataloaders = {"train": train_data_loader,
					   "val": val_data_loader}
	else:
		raise ValueError("invalid dataset name")

	# ========================================== model =============================================
	if args.model == 'resnet18':
		# Load Resnet18
		torch.manual_seed(42)
		model_ft = models.resnet18(weights=None)
		# Finetune Final few layers to adjust for tiny imagenet input
		model_ft.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
		model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
		num_features = model_ft.fc.in_features
		model_ft.fc = nn.Linear(num_features, 200)
		model_ft = model_ft.to(args.device)

		criterion = nn.CrossEntropyLoss()
		optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

		train_losses = []
		train_acces = []
		val_acces = []
		best_acc = 0
		save_dir = "./weights/resnet18"
	else:
		raise ValueError("invalid dataset name")

	# ========== train =============================================
	for epoch in range(args.epochs):
		train_loss, train_acc = train(
			model=model_ft,
			dataloaders=dataloaders["train"],
			criterion=criterion,
			optimizer=optimizer_ft,
			device=args.device,
			epoch=epoch
		)
		torch.save(model_ft.state_dict(), f"{save_dir}/last.pt")

		val_acc = val(
			model=model_ft,
			dataloaders=dataloaders["val"],
			device=args.device,
			epoch=epoch
		)

		train_losses.append(train_loss)
		train_acces.append(train_acc)
		val_acces.append(val_acc)

		# ========================== 保存结果数据 =======================
		if val_acc > best_acc:
			val_acc = best_acc
			torch.save(model_ft.state_dict(), f"{save_dir}/best.pt")

		save_to_csv(train_losses, train_acces, val_acces, save_dir=save_dir)
