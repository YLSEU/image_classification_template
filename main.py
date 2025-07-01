import torch
import torch.nn as nn
import torch.optim as optim
from datasets.tiny_imagenet import get_tiny_imagenet
import torchvision.models as models
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

	# ================================= dataset =============================================
	if args.dataset == 'tiny_imagenet':
		train_dataloader, val_dataloader = get_tiny_imagenet(data_dir=args.data_dir,
		                                dataset=args.dataset,
		                                bs=args.bs,
		                                img_size=args.img_size)
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
			dataloaders=train_dataloader,
			criterion=criterion,
			optimizer=optimizer_ft,
			device=args.device,
			epoch=epoch
		)
		torch.save(model_ft.state_dict(), f"{save_dir}/last.pt")

		val_acc = val(
			model=model_ft,
			dataloaders=val_dataloader,
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
