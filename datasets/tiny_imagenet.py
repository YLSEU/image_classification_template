import os
from torchvision import transforms, datasets
import torch


def create_val_img_folder(data_dir, dataset):
	dataset_dir = os.path.join(data_dir, dataset)
	val_dir = os.path.join(dataset_dir, 'val')
	img_dir = os.path.join(val_dir, 'images')

	fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
	data = fp.readlines()
	val_img_dict = {}

	for line in data:
		words = line.split('\t')
		val_img_dict[words[0]] = words[1]

	fp.close()

	# Create folder if not present and move images into proper folders
	for img, folder in val_img_dict.items():
		newpath = (os.path.join(img_dir, folder))

		if not os.path.exists(newpath):
			os.makedirs(newpath)

		if os.path.exists(os.path.join(img_dir, img)):
			os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))


def prepare_imagenet(data_dir, dataset, batch_size, test_batch_size):
	dataset_dir = os.path.join(data_dir, dataset)
	train_dir = os.path.join(dataset_dir, 'train')
	val_dir = os.path.join(dataset_dir, 'val', 'images')

	print('Preparing dataset ...')
	# Normalization
	norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                            std=[0.229, 0.224, 0.225])

	train_trans = [
		transforms.RandomHorizontalFlip(),
		transforms.RandomResizedCrop(224),
		transforms.ToTensor()
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

	return train_data_loader, val_data_loader, train_data, val_data


if __name__ == '__main__':
	# create_val_img_folder(data_dir='/media/xdl/lxd/datasets', dataset='tiny_imagenet')
	train_data_loader, val_data_loader, train_data, val_data = prepare_imagenet(data_dir='/media/xdl/lxd/datasets',
	                                                                            dataset='tiny_imagenet',
	                                                                            batch_size=128,
	                                                                            test_batch_size=128)
	print(train_data.shape, val_data.shape)

