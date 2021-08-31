import os

import torch
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def compute_metrics(labels, preds):
    """Compute the statistics for a set of labels and predictions"""
    # Compute stats
    _, y_hat = torch.max(preds, 1)
    tp = torch.sum((y_hat == 1) * (labels.data == 1))
    fp = torch.sum((y_hat == 1) * (labels.data == 0))
    tn = torch.sum((y_hat == 0) * (labels.data == 0))
    fn = torch.sum((y_hat == 0) * (labels.data == 1))

    # Compute metrics
    stats = {}
    stats['acc'] = float((tp + tn).double() / (tp + fp + tn + fn).double())
    try:
        stats['sens'] = float(tp.double() / (tp + fn).double())
        stats['spec'] = float(tn.double() / (tn + fp).double())
        stats['prec'] = float(tp.double() / (tp + fp).double())
        stats['f1'] = 2 * (stats['prec'] * stats['sens']
                           / (stats['prec'] + stats['sens']))
        stats['auc-roc'] = roc_auc_score(labels.data,
                                         preds[:, 1].detach().numpy())
        precision, recall, thresholds = precision_recall_curve(
            labels.data, preds[:, 1].detach().numpy())
        stats['auc-pr'] = auc(recall, precision)
    except:
        stats['sens'] = 0
        stats['spec'] = 0
        stats['prec'] = 0
        stats['f1'] = 0
        stats['auc-roc'] = 0
        stats['auc-pr'] = 0
    return stats


def train_model(model, dataloaders, criterion, optimizer, scheduler, device,
                num_epochs=25, save_folder=None):
    """Train the model"""

    # Initialize the history
    history = {}
    for phase in ['train', 'val']:
        history[phase] = {}
        history[phase]['loss'] = []
        history[phase]['acc'] = []
        history[phase]['sens'] = []
        history[phase]['spec'] = []
        history[phase]['prec'] = []
        history[phase]['f1'] = []
        history[phase]['auc-roc'] = []
        history[phase]['auc-pr'] = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1), flush=True)
        print('-' * 10, flush=True)
        for param_group in optimizer.param_groups:
            print('Current Learning rate:', param_group['lr'])

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            n_samples = 0
            all_preds = []
            all_labels = []
            total_windows = 0
            for samples in dataloaders[phase]:
                n_samples += 1
                if isinstance(samples['buffers'], list):
                    inputs = [s.to(device) for s in samples['buffers']]
                    labels = [s.to(device) for s in samples['labels']]
                else:
                    inputs = samples['buffers'].to(device)
                    labels = samples['labels'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if isinstance(inputs, list):
                        outputs = [model(input) for input in inputs]
                        loss = torch.tensor(0.0).to(device)
                        for output, label in zip(outputs, labels):
                            loss += criterion(output, label)
                    else:
                        outputs = model(inputs)
                        if outputs.dim() == 3:  # LSTM output
                            b, t, d = outputs.shape
                            loss = criterion(outputs.view((b * t, d)),
                                             labels.view((b * t, 1)))
                        else:
                            loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                if isinstance(outputs, list):
                    for output, label in zip(outputs, labels):
                        all_preds.append(torch.softmax(output, dim=1))
                        all_labels.append(label.data)
                elif outputs.dim() == 3:  # channelwise outputs
                    _, C, classes = outputs.size()
                    all_preds.append(
                        torch.softmax(outputs.view(-1, classes), dim=1))
                    # all_labels.append(labels.data.repeat_interleave(C))
                    # BAD FOR CHANNELWISE MODELS
                    all_labels.append(labels.view(-1))

                else:
                    all_preds.append(torch.softmax(outputs, dim=1))
                    all_labels.append(labels.data)

                # statistics
                nwindows = 0
                if isinstance(labels, list):
                    for label in labels:
                        nwindows += label.size(0)
                else:
                    nwindows = labels.size(0)
                running_loss += loss.item() * nwindows
                total_windows += nwindows

            # Average the statistics for the epoch
            epoch_loss = float(running_loss / total_windows)
            stats = compute_metrics(
                torch.cat(all_labels).cpu(), torch.cat(all_preds).cpu())

            print('{} Loss: {:.4f} Acc: {:.4f} Sens: {:.4f} Spec: {:.4f} Prec: {:.4f} f1: {:.4f} AUC-ROC: {:.4f} AUC-PR: {:.4f}'.format(
                phase, epoch_loss, stats['acc'],
                stats['sens'], stats['spec'], stats['prec'], stats['f1'], stats['auc-roc'], stats['auc-pr']), flush=True)

            # Save the history
            history[phase]['loss'].append(epoch_loss)
            history[phase]['acc'].append(stats['acc'])
            history[phase]['sens'].append(stats['sens'])
            history[phase]['spec'].append(stats['spec'])
            history[phase]['prec'].append(stats['prec'])
            history[phase]['f1'].append(stats['f1'])
            history[phase]['auc-roc'].append(stats['auc-roc'])
            history[phase]['auc-pr'].append(stats['auc-pr'])

        # Post epoch
        if save_folder:
            fn = os.path.join(save_folder,
                              'model_epoch{}.pt'.format(epoch))
            torch.save(model.state_dict, fn)
        scheduler.step()
    return model, history
