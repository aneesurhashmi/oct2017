
from __future__ import print_function, division

import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import seaborn as sn
import pandas as pd

# import torch.utils.tensorboard
def train_epoch(model, optimizer, data_loader, loss_history, ep, device, criterion=None, writer=None):
    total_samples = len(data_loader.dataset)
    model.train()
    total_loss = 0
    # for cnn models
    if criterion:
        for i, (data, target) in tqdm(enumerate(data_loader)):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            # output = output.to(device)
            # loss = F.nll_loss(output, target)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if i % 1000 == 0:
                print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                    ' (' + '{:3.0f}'.format(100 * i * len(data) / total_samples) + '%)]  Loss: ' +
                    '{:6.4f}'.format(loss.item()))
                # loss_history.append(loss.item())
        if writer:
            writer.add_scalar("Train Loss", loss.item(), ep)
        # average loss per epoch
        avg_loss = total_loss / total_samples
        loss_history.append(avg_loss)
    else:
        # total_samples = len(data_loader.dataset)
        # model.train()
        # total_loss = 0
        for i, (data, target) in tqdm(enumerate(data_loader)):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = F.log_softmax(model(data), dim=1)
            # output = output.to(device)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if i % 1000 == 0:
                print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                    ' (' + '{:3.0f}'.format(100 * i * len(data) / total_samples) + '%)]  Loss: ' +
                    '{:6.4f}'.format(loss.item()))
                # loss_history.append(loss.item())
        writer.add_scalar("Vit Loss", loss.item(), ep)
        # average loss per epoch
        avg_loss = total_loss / total_samples
        loss_history.append(avg_loss)


# ======================================== #
# ======================================== #
# ============ PERFORMANCE =============== #
# ======================================== #
# ======================================== #

def show_confusion_matrix(y_pred, y_true, model_name):
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    classes = ["CNV", "DME", "DRUSEN", "NORMAL"]
    
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    report = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose()
    
    df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes], columns = [i for i in classes])
    
    print(f"F1 score micro: {f1_score(y_true, y_pred, average='micro')}")
    print(f"F1 score macro: {f1_score(y_true, y_pred, average='macro')}")
    print(f"F1 score weighted: {f1_score(y_true, y_pred, average='weighted')}")

    # print("confusion matrix: ", cf_matrix)
    print("classification report: ",report)
    # save classification report
    report.to_csv(f'./analysis/{model_name}.csv')

    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(f'./analysis/{model_name}.png')


def evaluate(model, data_loader, loss_history="na", criterion=None, device = None, model_name="na"):
    model.eval()
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0
    # for confusion matrix
    all_targets = torch.tensor([]).to(device)
    all_pred = torch.tensor([]).to(device)
    
    # for vit
    if criterion==None:
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(device)
                target = target.to(device)
                output = F.log_softmax(model(data), dim=1)
                loss = F.nll_loss(output, target, reduction='sum')
                _, pred = torch.max(output, dim=1)
                total_loss += loss.item()
                correct_samples += pred.eq(target).sum()

                # for testing only (not for eval)
                all_pred = torch.cat((all_pred,pred))
                all_targets = torch.cat((all_targets,target))

    # for cnn models
    else:
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(device)
                target = target.to(device)
                # output = F.log_softmax(model(data), dim=1)
                output = model(data)
                # loss = F.nll_loss(output, target, reduction='sum')
                loss = criterion(output, target)
                _, pred = torch.max(output, dim=1)
                total_loss += loss.item()
                correct_samples += pred.eq(target).sum()

                # for testing only (not for eval)
                all_pred = torch.cat((all_pred,pred))
                all_targets = torch.cat((all_targets,target))

    # same for all models
    avg_loss = total_loss / total_samples
    # test mode
    if loss_history == 'na':
        show_confusion_matrix(all_pred, all_targets, model_name)
        print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
            '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
            '{:5}'.format(total_samples) + ' (' +
            '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')
    # eval mode
    else:
        loss_history.append(avg_loss)
        print('\nAverage tFest loss: ' + '{:.4f}'.format(avg_loss) +
            '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
            '{:5}'.format(total_samples) + ' (' +
            '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')




# ======================================== #
