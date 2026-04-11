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
        # GAN LOSS (LSGAN)
        pred_fake_B = self.netD_A(self.fake_B)
        self.loss_G_A = 0.5 * torch.mean((pred_fake_B - 1) ** 2)

        pred_fake_A = self.netD_B(self.fake_A)
        self.loss_G_B = 0.5 * torch.mean((pred_fake_A - 1) ** 2)

        # CYCLE LOSS
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.opt.lambda_B

        if self.opt.lambda_identity > 0:

            self.idt_A = self.netG_A(self.real_B)
            self.idt_B = self.netG_B(self.real_A)

            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * self.opt.lambda_B * self.opt.lambda_identity
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * self.opt.lambda_A * self.opt.lambda_identity

        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # TOTAL LOSS
        self.loss_G = (
            self.loss_G_A +
            self.loss_G_B +
            self.loss_cycle_A +
            self.loss_cycle_B +
            self.loss_idt_A +
            self.loss_idt_B
        )

        self.loss_G.backward()