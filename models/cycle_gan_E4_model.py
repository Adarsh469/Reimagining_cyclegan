import torch
from .cycle_gan_model import CycleGANModel as BaseCycleGANModel

class CycleGANE4Model(BaseCycleGANModel):

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        pred_fake = netD(fake.detach())

        loss_real = torch.mean(torch.relu(1 - pred_real))
        loss_fake = torch.mean(torch.relu(1 + pred_fake))

        self.loss_D = loss_real + loss_fake
        self.loss_D.backward()
        return self.loss_D

    def backward_G(self):
        super().backward_G()

        pred_fake = self.netD_A(self.fake_B)
        self.loss_G_A = -torch.mean(pred_fake)

        pred_fake = self.netD_B(self.fake_A)
        self.loss_G_B = -torch.mean(pred_fake)

        self.loss_G = self.loss_G_A + self.loss_G_B + \
                      self.loss_cycle_A + self.loss_cycle_B + \
                      self.loss_idt_A + self.loss_idt_B

        self.loss_G.backward()