from __future__ import print_function, division
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import seaborn as sn
import pandas as pd
import os
from typing import Iterable
import torch
from torch.optim._multi_tensor import SGD

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
            # writer.add_scalar("Eval Loss", loss.item(), ep)
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
        writer.add_scalar("Train Loss", loss.item(), ep)
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
    report_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose()
    df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes], columns = [i for i in classes])
    precision, recall, fscore, support = precision_recall_fscore_support(y_true,y_pred,average='macro')
    
    # metrics
    # fscore = f1_score(y_true, y_pred, average='micro')
    print(f"F1 score micro: {fscore}")
    # print(f"F1 score macro: {f1_score(y_true, y_pred, average='macro')}")
    # print(f"F1 score weighted: {f1_score(y_true, y_pred, average='weighted')}")

    # print("confusion matrix: ", cf_matrix)
    print("classification report: ",report_df)
    # save classification report
    report_df.to_csv(f'./analysis/{model_name}.csv')

    data = {
        # "model":model_name,
        'f1_score': [fscore],
        'precision': [precision],
        'recall': [recall],
        'support': [support],
        }

    # save confusion matrix heatmap
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(f'./analysis/{model_name}.png')

    return data


results_file_path = "./analysis/results.csv"
# to save all results together
def save_test_results(results, results_file_path = results_file_path):
    # Check whether the specified
    # file exists or not
    file_exist = os.path.exists(results_file_path)
    if file_exist:
        df = pd.read_csv(results_file_path)
        df_con = pd.concat([df, pd.DataFrame(results)])
        df_con.to_csv(results_file_path, index = False)
    else: 
        # Convert the dictionary into DataFrame
        df = pd.DataFrame(results)
        # df.save_csv(results_file_path)
        df.to_csv(results_file_path, index=False)


def evaluate(model, data_loader, loss_history="na", criterion=None, device = None, model_name="na", writer = None, ep=None):
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
            for metrics, target in data_loader:
                metrics = metrics.to(device)
                target = target.to(device)
                output = F.log_softmax(model(metrics), dim=1)
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
            for metrics, target in data_loader:
                metrics = metrics.to(device)
                target = target.to(device)
                # output = F.log_softmax(model(data), dim=1)
                output = model(metrics)
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
        metrics = show_confusion_matrix(all_pred, all_targets, model_name)
        # add all results to csv
        metrics["accuracy"] = float(100.0 * correct_samples / total_samples)
        metrics["model"] = model_name
        metrics["epoch"] = model_name.split('-')[-1]
        metrics["params"] = sum(p.numel() for p in model.parameters())

        save_test_results(metrics)

        # print results
        print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
            '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
            '{:5}'.format(total_samples) + ' (' +
            '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')

    # eval mode
    else:
        loss_history.append(avg_loss)
        print('\nAverage val loss: ' + '{:.4f}'.format(avg_loss) +
            '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
            '{:5}'.format(total_samples) + ' (' +
            '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')
        if writer:
            writer.add_scalar("Eval ", avg_loss, ep)




# =================================================================================================================================  #
# =================================================================================================================================  #
# ======================================================= SAM OPTIMIZER ===========================================================  #
# =================================================================================================================================  #
# =================================================================================================================================  #


__all__ = ["SAMSGD"]


class SAMSGD(SGD):
    """ SGD wrapped with Sharp-Aware Minimization
    Args:
        params: tensors to be optimized
        lr: learning rate
        momentum: momentum factor
        dampening: damping factor
        weight_decay: weight decay factor
        nesterov: enables Nesterov momentum
        rho: neighborhood size
    """

    def __init__(self,
                 params: Iterable[torch.Tensor],
                 lr: float,
                 momentum: float = 0,
                 dampening: float = 0,
                 weight_decay: float = 0,
                 nesterov: bool = False,
                 rho: float = 0.05,
                 ):
        if rho <= 0:
            raise ValueError(f"Invalid neighborhood size: {rho}")
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        # todo: generalize this
        if len(self.param_groups) > 1:
            raise ValueError("Not supported")
        self.param_groups[0]["rho"] = rho

    @torch.no_grad()
    def step(self,
             closure
             ) -> torch.Tensor:
        """
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        Returns: the loss value evaluated on the original point
        """
        closure = torch.enable_grad()(closure)
        loss = closure().detach()

        for group in self.param_groups:
            grads = []
            params_with_grads = []

            rho = group['rho']
            # update internal_optim's learning rate

            for p in group['params']:
                if p.grad is not None:
                    # without clone().detach(), p.grad will be zeroed by closure()
                    grads.append(p.grad.clone().detach())
                    params_with_grads.append(p)
            device = grads[0].device

            # compute \hat{\epsilon}=\rho/\norm{g}\|g\|
            grad_norm = torch.stack([g.detach().norm(2).to(device) for g in grads]).norm(2)
            epsilon = grads  # alias for readability
            torch._foreach_mul_(epsilon, rho / grad_norm)

            # virtual step toward \epsilon
            torch._foreach_add_(params_with_grads, epsilon)
            # compute g=\nabla_w L_B(w)|_{w+\hat{\epsilon}}
            closure()
            # virtual step back to the original point
            torch._foreach_sub_(params_with_grads, epsilon)

        super().step()
        return loss
