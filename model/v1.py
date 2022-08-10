'''
    Unet encoder, differentiable fft decoder
'''
import torch
import torch.nn as nn

M = 512  # input and output image width

class NormRelu(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.process = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.process(x)


class ResDown(nn.Module):

    def __init__(self, n_in, n_out):
        super().__init__()
        self.line1 = nn.Sequential(
            NormRelu(n_in),
            nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=5, stride=2, padding=2),
            NormRelu(n_out),
            nn.Conv2d(in_channels=n_out, out_channels=n_out, kernel_size=1, stride=1, padding='same'),
        )
        self.line2 = nn.Sequential(
            nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=2),
        )
        self.line3 = nn.Sequential(
            NormRelu(n_out),
            nn.Conv2d(in_channels=n_out, out_channels=n_out, kernel_size=3, stride=1, padding='same'),
            NormRelu(n_out),
            nn.Conv2d(in_channels=n_out, out_channels=n_out, kernel_size=3, stride=1, padding='same'),
        )

    def forward(self, input):
        x1 = self.line1(input)
        x2 = self.line2(input)
        x3 = torch.add(x1, x2)
        x4 = self.line3(x3)
        output = torch.add(x3, x4)
        return output


class ResUp(nn.Module):

    def __init__(self, n_in, n_out):
        super().__init__()
        self.line1 = nn.Sequential(
            NormRelu(n_in),
            nn.ConvTranspose2d(in_channels=n_in, out_channels=n_out, kernel_size=2, stride=2),
            NormRelu(n_out),
            nn.Conv2d(in_channels=n_out, out_channels=n_out, kernel_size=3, stride=1, padding=1),
        )
        self.line2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=n_in, out_channels=n_out, kernel_size=2, stride=2),
        )
        self.line3 = nn.Sequential(
            NormRelu(n_out),
            nn.Conv2d(in_channels=n_out, out_channels=n_out, kernel_size=3, stride=1, padding='same'),
            NormRelu(n_out),
            nn.Conv2d(in_channels=n_out, out_channels=n_out, kernel_size=3, stride=1, padding='same'),
        )

    def forward(self, input):
        x1 = self.line1(input)
        x2 = self.line2(input)
        x3 = torch.add(x1, x2)
        x4 = self.line3(x3)
        output = torch.add(x3, x4)
        return output


class UNet(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.down1 = ResDown(n_in=n_in, n_out=16)
        self.down2 = ResDown(n_in=16, n_out=32)
        self.down3 = ResDown(n_in=32, n_out=64)
        self.down4 = ResDown(n_in=64, n_out=96)
        self.up4 = ResUp(n_in=96, n_out=64)
        self.up3 = ResUp(n_in=64, n_out=32)
        self.up2 = ResUp(n_in=32, n_out=16)
        self.up1 = ResUp(n_in=16, n_out=1)
        self.norm = nn.BatchNorm2d(num_features=1)

    def forward(self, input):
        d1 = self.down1(input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u4 = self.up4(d4)
        u3 = self.up3(torch.add(u4, d3))
        u2 = self.up2(torch.add(u3, d2))
        u1 = self.up1(torch.add(u2, d1))
        output = self.norm(u1)
        return output


class Decoder(nn.Module):
    '''
        h: lambda (nm)
        z: distance (mm)
        pix: SLM pix width (mm)
    '''
    def __init__(self, h, z):
        super().__init__()
        h = h * 1e-6
        pix = 0.008
        lm = M * pix
        m = torch.linspace(0, M - 1, M)
        x = -lm / 2 + lm / M * m
        y = x
        xx, yy = torch.meshgrid([x, y], indexing='xy')
        self.spherical = torch.remainder(torch.pi / z * (xx ** 2 + yy ** 2) / h, 2 * torch.pi).cuda()
        
    def forward(self, phase):
        phase = phase - self.spherical
        real = torch.cos(phase)
        imag = torch.sin(phase)
        real_fft = torch.fft.fft2(real)
        real_r = real_fft.clone().real
        real_i = real_fft.clone().imag
        imag_fft = torch.fft.fft2(imag)
        imag_r = imag_fft.clone().real
        imag_i = imag_fft.clone().imag
        ur = real_r - imag_i
        ui = real_i + imag_r
        u = ur.square() + ui.square()
        u = u.sqrt()
        return u


class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()
        self.norm = nn.LayerNorm([M, M], elementwise_affine=False)
        self.encoder = UNet(n_in=1)
        self.decoder = Decoder(h=532, z=800)
        
    def forward(self, image):
        image_norm = self.norm(image)
        phase = torch.pi * self.tanh(self.encoder(image_norm))
        reconstruction = self.decoder(phase)
        return phase, reconstruction
    
    

if __name__ == '__main__':
    
    net = Model().cuda()
    net.eval()
    
    input = torch.randn([1, 1, M, M]).cuda()
    phase, image_norm, image_fft = net(input)
    print(phase.shape)
    print(image_fft.shape)
