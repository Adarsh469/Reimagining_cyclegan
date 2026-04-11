import torch
import torch.autograd as autograd
from .cycle_gan_model import CycleGANModel as BaseCycleGANModel

# Gradient Penalty
def compute_gp(D, real, fake):
    alpha = torch.rand(real.size(0), 1, 1, 1).to(real.device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    d_interpolates = D(interpolates)

    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


class CycleGANE3Model(BaseCycleGANModel):

    # Critic (Discriminator)
    def backward_D_basic(self, netD, real, fake):

        pred_real = netD(real)
        pred_fake = netD(fake.detach())

        # WGAN loss
        loss = torch.mean(pred_fake) - torch.mean(pred_real)

        # Gradient penalty
        gp = compute_gp(netD, real, fake.detach())

        loss_D = loss + 10 * gp
        loss_D.backward()

        return loss_D


    # Generator
    def backward_G(self):

        # GAN LOSS (WGAN)
        pred_fake_B = self.netD_A(self.fake_B)
        self.loss_G_A = -torch.mean(pred_fake_B)

        pred_fake_A = self.netD_B(self.fake_A)
        self.loss_G_B = -torch.mean(pred_fake_A)

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

        self.loss_G = (
            self.loss_G_A +
            self.loss_G_B +
            self.loss_cycle_A +
            self.loss_cycle_B +
            self.loss_idt_A +
            self.loss_idt_B
        )

        self.loss_G.backward()
