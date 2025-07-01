import pandas as pd
import os


def save_to_csv(train_losses, train_acces, val_acces, save_dir):
	# 保存结果数据
	PlotData = {
			'train_losses': train_losses,
			'train_acces': train_acces,
			'val_acces': val_acces,
		}

	df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in PlotData.items()]))
	df.to_csv(os.path.join(save_dir, 'res.csv'), index=True)


def plot_result():
	pass


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

