import torch
import torch.autograd as autograd
from .cycle_gan_model import CycleGANModel as BaseCycleGANModel


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

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        pred_fake = netD(fake.detach())

        loss = -torch.mean(pred_real) + torch.mean(pred_fake)
        gp = compute_gp(netD, real, fake.detach())

        self.loss_D = loss + 10 * gp
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