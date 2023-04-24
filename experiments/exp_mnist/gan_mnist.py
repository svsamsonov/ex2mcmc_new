from torch import nn


class GeneratorMNIST(nn.Module):
    def __init__(self, lat_size):
        super(GeneratorMNIST, self).__init__()

        img_size = 28

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(
            nn.Linear(lat_size, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128 * self.init_size**2),
        )

        self.conv_blocks = nn.Sequential(
            # nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            # nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            # nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, foo=None):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class DiscriminatorMNIST(nn.Module):
    def __init__(self, wgan_cp=False):
        super(DiscriminatorMNIST, self).__init__()

        img_size = 28
        self.wgan_cp = wgan_cp

        def discriminator_block(in_filters, out_filters, bn=False):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(0.0),
            ]
            # if bn:
            #    block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        # ds_size = img_size // 2**4
        self.adv_layer = nn.Sequential(nn.Linear(512, 1, bias=not wgan_cp))

    def forward(self, img, foo=None):
        if self.wgan_cp:
            for p in self.parameters():
                p.data.clamp_(-0.06, 0.06)

        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
