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


if __name__ == '__main__':
    ################### PRINT COLOR CLASS ############################
    class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

    bc = bcolors
    print(bc.HEADER + 'Setting DataLoader... ' + bc.ENDC)
    print(bc.OKBLUE + 'Setting DataLoader... ' + bc.ENDC)
    print(bc.OKGREEN + 'Setting DataLoader... ' + bc.ENDC)
    print(bc.WARNING + 'Setting DataLoader... ' + bc.ENDC)
    print(bc.BOLD + bc.WARNING + 'Starting Jigsaw Network Training!\n' + bc.ENDC + bc.ENDC)


    ################## WRITE TO CSV FILE #####################
    import csv
    
    class CSV_Writer():
        def __init__(self, save_path, columns):
            self.save_path = save_path
            self.columns   = columns

            with open(self.save_path, "a") as csv_file:
                writer = csv.writer(csv_file, delimiter=",")
                writer.writerow(self.columns)

        def log(self, inputs):
            with open(self.save_path, "a") as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow(inputs)
