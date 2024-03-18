import torch
import torch.nn as nn
import torch.nn.functional as F

class ADWT(nn.Module):
  def __init__(self, kernel_size = 4, stride = 2, initialization = True, trainable = False):
    super().__init__()

    self.stride = stride
    self.kernel_size = kernel_size
    self.trainable = trainable

    if initialization:

      if kernel_size == 2:
        LL_init = torch.ones(2, 2)/2
        HH_init = torch.tensor([[1, -1], [-1, 1]])/2
        HL_init = torch.tensor([[1, -1], [1, -1]])/2

        self.LL = nn.Parameter(LL_init)
        self.HL = nn.Parameter(HL_init)
        self.HH = nn.Parameter(HH_init)

      else:
        LPF = torch.ones(kernel_size)
        HPF = torch.ones(kernel_size)
        i = 1
        while i <= len(HPF)//2:
          HPF[i] = -1
          HPF[len(HPF)-i-1] = -1
          i += 2

        LL_init = torch.ger(LPF, LPF)/kernel_size
        HH_init = torch.ger(HPF, HPF)/kernel_size
        HL_init = torch.ger(LPF, HPF)/kernel_size

        self.LL = nn.Parameter(LL_init)
        self.HL = nn.Parameter(HL_init)
        self.HH = nn.Parameter(HH_init)

    else:
      self.LL = nn.Parameter(torch.ones(kernel_size, kernel_size)/kernel_size)
      self.HL = nn.Parameter(torch.zeros(kernel_size, kernel_size))
      self.HH = nn.Parameter(torch.ones(kernel_size, kernel_size)/kernel_size)

    for param in self.parameters():
      param.requires_grad = trainable

  def forward(self, x):

    x = x.float()

    B, C, H, W = x.shape
    LL_kernel = self.LL.unsqueeze(0).unsqueeze(0).expand(C, C, -1, -1)
    HH_kernel = self.HH.unsqueeze(0).unsqueeze(0).expand(C, C, -1, -1)
    HL_kernel = self.HL.unsqueeze(0).unsqueeze(0).expand(C, C, -1, -1)
    LH_kernel = self.HL.transpose(0, 1).unsqueeze(0).unsqueeze(0).expand(C, C, -1, -1)

    xLL = F.conv2d(x, LL_kernel, stride=self.stride)
    xHH = F.conv2d(x, HH_kernel, stride=self.stride)
    xLH = F.conv2d(x, LH_kernel, stride=self.stride)
    xHL = F.conv2d(x, HL_kernel, stride=self.stride)

    return xLL, xLH, xHL, xHH, self.LL, self.HH, self.HL
