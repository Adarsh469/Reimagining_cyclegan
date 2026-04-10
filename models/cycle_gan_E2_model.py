import torch
from .cycle_gan_model import CycleGANModel as BaseCycleGANModel

class CycleGANE2Model(BaseCycleGANModel):

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = torch.mean((pred_real - 1) ** 2)

        pred_fake = netD(fake.detach())
        loss_D_fake = torch.mean(pred_fake ** 2)

        loss_D = 0.5 * (loss_D_real + loss_D_fake)
        loss_D.backward()
        return loss_D

    def backward_G(self):
        # run full base pipeline first
        super().backward_G()

        # overwrite GAN loss ONLY (keep cycle + identity same)
        pred_fake = self.netD_A(self.fake_B)
        self.loss_G_A = 0.5 * torch.mean((pred_fake - 1) ** 2)

        pred_fake = self.netD_B(self.fake_A)
        self.loss_G_B = 0.5 * torch.mean((pred_fake - 1) ** 2)

        # recompute total loss properly
        self.loss_G = self.loss_G_A + self.loss_G_B + \
                      self.loss_cycle_A + self.loss_cycle_B + \
                      self.loss_idt_A + self.loss_idt_B

        self.loss_G.backward()