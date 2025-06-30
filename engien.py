import torch


def train(model, dataloaders, criterion, optimizer, device, epoch, scheduler=None):
    print("-" * 10)

    if scheduler is not None:
        scheduler.step()

    model.train()

    running_loss = 0.0
    running_corrects = 0

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
        running_loss += loss.detach() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloaders.dataset)
    epoch_acc = running_corrects.float() / len(dataloaders.dataset)
    print(f'Epoch: {epoch} train loss {epoch_loss:.3f} train acc {epoch_acc * 100.0:.3f}%')

    return epoch_loss, epoch_acc


def val(model, dataloaders, device, epoch):
    model.eval()  # Set model to evaluate mode

    running_corrects = 0

    for i, (inputs, labels) in enumerate(dataloaders):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        # statistics
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

    epoch_acc = running_corrects.float() / len(dataloaders.dataset)
    print(f'Epoch: {epoch} val acc {epoch_acc * 100.0:.3f}%')

    return epoch_acc
