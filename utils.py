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
	df.to_csv(os.path.join(save_dir, 'res.csv'), index=False)


def plot_result():
	pass
