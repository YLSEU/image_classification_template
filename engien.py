import torch
from utils import AverageMeter


def train(model, dataloaders, criterion, optimizer, device, epoch, scheduler=None, print_feq=50):
    print("-" * 10)

    if scheduler is not None:
        scheduler.step()

    model.train()
    loss_meter = AverageMeter(name='train_loss')
    acc_meter = AverageMeter(name='train_acc')


    # Iterate over data.
    for i, (inputs, labels) in enumerate(dataloaders):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        _, preds = torch.max(outputs, 1)
        correct = (preds == labels).sum().item()
        batch_size = labels.size(0)

        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(correct / batch_size, batch_size)

        if i % print_feq == 0:
            print(f'Epoch: {epoch} train loss {loss_meter.avg:.3f} train acc {acc_meter.avg * 100.0:.3f}%')

    return loss_meter.avg, acc_meter.avg


def val(model, dataloaders, device, epoch):
    model.eval()  # Set model to evaluate mode

    acc_meter = AverageMeter(name='val_acc')

    for i, (inputs, labels) in enumerate(dataloaders):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        # statistics
        _, preds = torch.max(outputs, 1)

        correct = (preds == labels).sum().item()
        batch_size = labels.size(0)

        acc_meter.update(correct / batch_size, batch_size)

    print(f'Epoch: {epoch} val acc {acc_meter.avg * 100.0:.3f}%')

    return acc_meter.avg
