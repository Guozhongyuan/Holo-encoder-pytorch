'''
    Unet encoder, differentiable fft decoder
'''
import torch
import torch.nn as nn

N = 512
M = 512

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
        self.down4 = ResDown(n_in=64, n_out=128)
        self.up4 = ResUp(n_in=128, n_out=64)
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
        LM = M * pix
        LN = N * pix
        L0 = h * z / pix
        n = torch.linspace(0, N - 1, N)
        m = torch.linspace(0, M - 1, M)
        x0 = -L0 / 2 + L0 / M * m
        y0 = -L0 / 2 + L0 / N * n
        x_img, y_img = torch.meshgrid([x0, y0], indexing='xy')
        x = -LM / 2 + LM / M * m
        y = -LN / 2 + LN / N * n
        x_slm, y_slm = torch.meshgrid([x, y], indexing='xy')
        L_img = (x_img ** 2 + y_img ** 2) / z
        L_slm = (x_slm ** 2 + y_slm ** 2) / z
        self.theta_img = (torch.pi / h * L_img).cuda()
        self.theta_slm = (torch.pi / h * L_slm).cuda()
    
    def complex_multipy(self, x_real, x_imag, y_real, y_imag):
        real = x_real*y_real - x_imag*y_imag
        imag = x_real*y_imag + x_imag*y_real
        return real, imag

    def diffraction(self, img, theta_rand):
        theta_real, theta_imag = self.complex_multipy(
            torch.cos(theta_rand),
            torch.sin(theta_rand),
            torch.cos(self.theta_img),
            torch.sin(self.theta_img)
        )
        U_real = img * theta_real
        U_imag = img * theta_imag
        real_fft = torch.fft.fft2(U_real)
        real_r = real_fft.clone().real
        real_i = real_fft.clone().imag
        imag_fft = torch.fft.fft2(U_imag)
        imag_r = imag_fft.clone().real
        imag_i = imag_fft.clone().imag
        Uf_real = real_r - imag_i
        Uf_imag = real_i + imag_r
        Uf_real = torch.fft.fftshift(Uf_real)
        Uf_imag = torch.fft.fftshift(Uf_imag)
        Uf_real, Uf_imag = self.complex_multipy(
            Uf_real,
            Uf_imag,
            torch.cos(self.theta_slm),
            torch.sin(self.theta_slm)
        )
        phase = torch.atan2(Uf_imag, Uf_real)
        return phase

    def reconstruction(self, phase):
        phase = phase - self.theta_slm
        
        real = torch.cos(phase)
        imag = torch.sin(phase)
        real_pad = torch.zeros([phase.shape[0], 1, N*2, M*2], device=phase.device)
        real_pad[:, :, (N-N//2):(N+N//2), (M-M//2):(M+M//2)] = real
        imag_pad = torch.zeros([phase.shape[0], 1, N*2, M*2], device=phase.device)
        imag_pad[:, :, (N-N//2):(N+N//2), (M-M//2):(M+M//2)] = imag

        real_fft = torch.fft.ifft2(real)
        real_r = real_fft.clone().real
        real_i = real_fft.clone().imag
        imag_fft = torch.fft.ifft2(imag)
        imag_r = imag_fft.clone().real
        imag_i = imag_fft.clone().imag
        ur = real_r - imag_i
        ui = real_i + imag_r
        u = (ur.square() + ui.square()).sqrt()
        u_center = u[:, :, (N-N//2):(N+N//2), (M-M//2):(M+M//2)]
        return u_center

    def forward(self, image, phase):
        phase = self.diffraction(image, phase)
        reconstruction = self.reconstruction(phase)
        return phase, reconstruction


class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()
        self.encoder = UNet(n_in=1)
        self.decoder = Decoder(h=532, z=800)  # TODO Z
        
    def forward(self, image):
        image = image / 255.0
        random_phase = torch.pi * self.tanh(self.encoder(image))
        phase, reconstruction = self.decoder(image, random_phase)
        return phase, reconstruction
    
    

if __name__ == '__main__':
    
    net = Model().cuda()
    net.eval()
    
    input = torch.randn([1, 1, N, M]).cuda()
    phase, reconstruction = net(input)
    print(phase.shape)
    print(reconstruction.shape)
