import cv2 as cv
import numpy as np

# GS迭代计算相息图

M = 1920
N = 1080
h = 0.532 * 0.001
pix = 0.008
z0 = 200
iter = 20

LM = M * pix
LN = N * pix
L0 = h * z0 / pix
n = np.linspace(0, N - 1, N)
m = np.linspace(0, M - 1, M)
x0 = -L0 / 2 + L0 / M * m
y0 = -L0 / 2 + L0 / N * n
xx0, yy0 = np.meshgrid(x0, y0)
x = -LM / 2 + LM / M * m
y = -LN / 2 + LN / N * n
xx, yy = np.meshgrid(x, y)

Fresnel_0 = np.exp(1j * np.pi / h * (xx0 ** 2 + yy0 ** 2) / z0)
Fresnel_1 = np.exp(1j * 2 * np.pi / h * z0) / (1j * h * z0) * np.exp(1j * np.pi / h * (xx ** 2 + yy ** 2) / z0)
Fresnel_2 = np.exp(-1j * np.pi / h * (xx ** 2 + yy ** 2) / z0)
Fresnel_3 = np.exp(-1j * 2 * np.pi / h * z0) / (-1j * h * z0) * np.exp(-1j * np.pi / h * (xx0 ** 2 + yy0 ** 2) / z0)

img = cv.imread('./data/1.jpg')
img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
img = cv.resize(img, (M, N))
img = np.array(img)

U00 = img.astype(np.float)
U0 = U00

for i in range(iter):
    print('iter: ', i)
    Uf = np.fft.fftshift(np.fft.fft2(U0 * Fresnel_0))
    Uf = Uf * Fresnel_1
    Phase = np.angle(Uf)
    U_slm = np.exp(1j * Phase)
    U_img = np.fft.ifft2(U_slm * Fresnel_2)
    U_show = abs(U_img)
    U_img = U_img * Fresnel_3
    U0 = U00 * np.exp(1j * np.angle(U_img))

save_img = np.uint8(U_show/np.max(U_show)*255)
save_Ih = np.uint8((Phase+np.pi)*255/2/np.pi)
cv.imwrite('save/simulate.bmp', save_img)
cv.imwrite('save/CGH.bmp', save_Ih)

print('end')
