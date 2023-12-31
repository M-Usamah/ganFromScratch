import torch
import torch.nn as nn


class Criti(nn.Module):
    def __init__(self, channel_img, features_d):
        super(Criti, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(in_channels=channel_img,
                               out_channels=features_d,
                               kernel_size=4,
                               stride=2,
                               padding=1
                               ), # 64x64
            nn.LeakyReLU(0.2),
            self._block(in_chanels=features_d,
                        out_channels=features_d*2,
                        kernel_size=4,
                        stride=2,
                        padding=1
                        ),#16x16
            self._block(in_chanels=features_d*2,
                        out_channels=features_d*4,
                        kernel_size=4,
                        stride=2,
                        padding=1
                        ),#8x8
            self._block(in_chanels=features_d*4,
                        out_channels=features_d*8,
                        kernel_size=4,
                        stride=2,
                        padding=1
                        ),#4x4
            nn.Conv2d(
                in_channels=features_d*8,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=0
            ),#1x1
        )

    def _block(self,
               in_chanels,
               out_channels,
               kernel_size,
               stride,
               padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_chanels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )
    def forward(self,x):
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, z_din,channel_img,features_g):
        super(Generator,self).__init__()
        self.net = nn.Sequential(
            self._block(
                in_chanels=z_din,
                out_channels=features_g*16,
                kernel_size=4,
                stride=1,
                padding=0
            ),#N x f_g*16 x 4 x 4
            self._block(
                in_chanels=features_g*16,
                out_channels=features_g*8,
                kernel_size=4,
                stride=2,
                padding=1
            ),#8x8
            self._block(
                in_chanels=features_g*8,
                out_channels=features_g*4,
                kernel_size=4,
                stride=2,
                padding=1
            ),#16x16
            self._block(
                in_chanels=features_g*4,
                out_channels=features_g*2,
                kernel_size=4,
                stride=2,
                padding=1
            ),#32x32
            nn.ConvTranspose2d(
                in_channels=features_g*2,
                out_channels=channel_img,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )
        
    def _block(self,
               in_chanels,
               out_channels,
               kernel_size,
               stride,
               padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_chanels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(0.2),
        )
    def forward(self,x):
        return self.net(x)
        
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m,
                      (
                          nn.Conv2d,
                          nn.ConvTranspose2d,
                          nn.BatchNorm2d
                        )):
            nn.init.normal_(m.weight.data,0.0,0.02)
        
def test():
    N,in_channels,H,W = 8,3,64,64
    z_dim = 100
    x = torch.randn((N,in_channels,H,W))
    
    disc = Criti(in_channels,8)
    initialize_weights(disc)
    assert disc(x).shape == (N,1,1,1)
    
    gen = Generator(z_dim, in_channels, 8)
    initialize_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)
    
    print("Success")
    
if __name__ == '__main__':
    test()
    