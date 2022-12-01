import torch
import matplotlib.pyplot as plt
from dataset import test_loader, val_loader
from resnet50 import device
from models import Network2
import torchvision.models as models
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sn
import pandas as pd
import numpy as np

# from resnet18 import MODEL_PATH, USE_GPU, device
# from tensorboardX import SummaryWriter
# # import torch.utils.tensorboard
# writer = SummaryWriter()


MODEL_PATH = "./models/resnet50_full.pt"
print("Loading... ", MODEL_PATH)
# MODEL_PATH = "./saved_models/resnet18_01.pt"
# trained from scratch
# MODEL_PATH = "./saved_models/resnet50_full.pt"
USE_GPU = False


# if "cnn" in MODEL_PATH.lower():
#     loaded_model = Network2()
# elif "resnet" in MODEL_PATH.lower():
#     loaded_model = models.resnet50()

loaded_model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))

def show_confusion_matrix(y_pred, y_true, model_name = "cnn"):

    # iterate over test data
    # for inputs, labels in testloader:
    #         output = net(inputs) # Feed Network

    #         output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
    #         y_pred.extend(output) # Save Prediction
            
    #         labels = labels.data.cpu().numpy()
    #         y_true.extend(labels) # Save Truth

    # constant for classes
    # classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    #         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
    classes = ["CNV", "DME", "DRUSEN", "NORMAL"]
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    # print(cf_matrix)
    df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes], columns = [i for i in classes])
    print(df_cm)
    
    
    print(f"F1 score micro: {f1_score(y_true, y_pred, average='micro')}")
    print(f"F1 score macro: {f1_score(y_true, y_pred, average='macro')}")
    print(f"F1 score weighted: {f1_score(y_true, y_pred, average='weighted')}")
    print(f"F1 score: {f1_score(y_true, y_pred, average='weighted')}")

    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(f'./analysis/{model_name}.png')



with torch.no_grad():
    n_correct = 0
    n_samples = 0
    test_acc_vec = []
    test_acc_dict = {}
    y_predict = torch.tensor([])
    y_true = torch.tensor([])

    for idx, (images, labels) in enumerate(val_loader):
        print(images)
        print(labels)
        if USE_GPU:
            images = images.to(device)
            labels = labels.to(device)

        out = loaded_model(images)
        _, predictions = torch.max(out, 1)
        n_samples += labels.shape[0]
        # print(predictions == labels)
        y_predict = torch.cat((y_predict,predictions))
        y_true = torch.cat((y_true,labels))
        n_correct += (predictions == labels).sum().item()
        # print(n_samples, n_correct)
        # print(n_correct/n_samples)
        # test_acc_dict[]
        test_acc_vec.append(n_correct/n_samples)
# from torchmetrics import ConfusionMatrix
# target = torch.tensor([1, 1, 0, 0])
#  preds = torch.tensor([0, 1, 0, 0])
#  confmat = ConfusionMatrix(num_classes=2)
#  confmat(preds, target)

    # print(y_predict, y_predict)
    acc = 100*(n_correct/n_samples)
    print(f'Accuracy on the test dataset: {acc}')
    # plt.plot(list(range(len(test_acc_vec))), test_acc_vec)
    # # plt.xlabel("")
    # plt.ylabel("Accuract")
    # plt.savefig("test.png")
    show_confusion_matrix(y_predict, y_true, model_name = "resnet18_e100")






