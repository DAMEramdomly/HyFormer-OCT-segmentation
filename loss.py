import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
from loss.dataset import LinearLesion
from pytorch_grad_cam import GradCAM

torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True
CUDA_LAUNCH_BLOCKING = 1


class Semantic_segmentation_target:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


class cam_loss(nn.Module):
    def __init__(self, class_num=5):
        super(cam_loss, self).__init__()
        self.class_num = class_num
        self.bce = nn.BCELoss(reduction="mean")
        self.sig = nn.Sigmoid()

    def forward(self, input, output, target):
        class_category = [
            "__background__", "scar1", "scar2", "scar3", "scar4"
        ]
        sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(class_category)}
        target = one_hot(target, self.class_num)
        RMSE = Variable(torch.Tensor([0]).float()).cuda()
        num = {0: "__background__", 1: "scar1" , 2: "scar2", 3: "scar3", 4: "scar4"}

        for i in range(0, self.class_num):
            class_cate = sem_class_to_idx[num[i]]

            from model.HybridFormer import HybridFormer
            model = HybridFormer()
            target_layer = [model.layer4]

            normalized_mask = F.softmax(output.detach(), 1)
            soft_labels = (normalized_mask + target) / 2.0
            layer1_mask_float = np.float32(normalized_mask.cpu() == class_cate)
            targets = [Semantic_segmentation_target(class_cate, layer1_mask_float)]

            with GradCAM(model=model, target_layers=target_layer, use_cuda=torch.cuda.is_available()) as cam:
                grayscale_cam = cam(input_tensor=input, targets=targets)

            cam_i = torch.from_numpy(grayscale_cam).cuda()
            soft_labels_i = soft_labels[:, i, :, :].float()
            #target_i = target[:, i, :, :].float()

            class_RMSE = self.bce(self.sig(cam_i), soft_labels_i)
            RMSE += class_RMSE

        cam_loss = torch.sqrt(RMSE / (self.class_num - 1))
        return cam_loss
