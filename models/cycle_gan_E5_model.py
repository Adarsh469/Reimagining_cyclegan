import torch
import torch.nn.functional as F
from .cycle_gan_model import CycleGANModel as BaseCycleGANModel

class CycleGANE5Model(BaseCycleGANModel):

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        pred_fake = netD(fake.detach())

        real_mean = torch.mean(pred_fake)
        fake_mean = torch.mean(pred_real)

        loss_real = F.binary_cross_entropy_with_logits(
            pred_real - real_mean,
            torch.ones_like(pred_real)
        )

        loss_fake = F.binary_cross_entropy_with_logits(
            pred_fake - fake_mean,
            torch.zeros_like(pred_fake)
        )

        self.loss_D = (loss_real + loss_fake) / 2
        self.loss_D.backward()
        return self.loss_D

    def backward_G(self):
        super().backward_G()

        pred_real = self.netD_A(self.real_B)
        pred_fake = self.netD_A(self.fake_B)

        real_mean = torch.mean(pred_fake)
        fake_mean = torch.mean(pred_real)

        loss_real = F.binary_cross_entropy_with_logits(
            pred_real - real_mean,
            torch.zeros_like(pred_real)
        )

        loss_fake = F.binary_cross_entropy_with_logits(
            pred_fake - fake_mean,
            torch.ones_like(pred_fake)
        )

        self.loss_G_A = (loss_real + loss_fake) / 2
        self.loss_G_B = self.loss_G_A  # symmetric

        self.loss_G = self.loss_G_A + self.loss_G_B + \
                      self.loss_cycle_A + self.loss_cycle_B + \
                      self.loss_idt_A + self.loss_idt_B

        self.loss_G.backward()