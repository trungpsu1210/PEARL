import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import trunc_normal_

class ConvLayer(nn.Module):
	def __init__(self, net_depth, dim, kernel_size=3, gate_act=nn.Sigmoid):
		super().__init__()
		self.dim = dim

		self.net_depth = net_depth
		self.kernel_size = kernel_size

		self.Wv = nn.Sequential(
			nn.Conv2d(dim, dim, 1),
			nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, padding_mode='reflect')
		)
        

		self.Wg = nn.Sequential(
			nn.Conv2d(dim, dim, 1),
			gate_act() if gate_act in [nn.Sigmoid, nn.Tanh] else gate_act(inplace=True)
		)
        
		self.sca = nn.Sequential(
			nn.AdaptiveAvgPool2d(1), 
			nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
		)
		
		self.proj = nn.Conv2d(dim, dim, 1)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Conv2d):
			gain = (8 * self.net_depth) ** (-1/4)    # self.net_depth ** (-1/2), the deviation seems to be too small, a bigger one may be better
			fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
			std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
			trunc_normal_(m.weight, std=std)

			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def forward(self, X):
		out = self.Wv(X) * self.Wg(X)
		out = out * self.sca(out)		
		out = self.proj(out)
		return out


class BasicBlock(nn.Module):
	def __init__(self, net_depth, dim, kernel_size=3, conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid):
		super().__init__()
		self.norm = norm_layer(dim)
		self.conv = conv_layer(net_depth, dim, kernel_size, gate_act)
	def forward(self, x):
		identity = x
		x = self.norm(x)
		x = self.conv(x)
		x = identity + x
		return x


class BasicLayer(nn.Module):
	def __init__(self, net_depth, dim, depth, kernel_size=3, conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid):

		super().__init__()
		self.dim = dim
		self.depth = depth

		# build blocks
		self.blocks = nn.ModuleList([
			BasicBlock(net_depth, dim, kernel_size, conv_layer, norm_layer, gate_act)
			for i in range(depth)])

	def forward(self, x):
		for blk in self.blocks:
			x = blk(x)
		return x


class PatchEmbed(nn.Module):
	def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
		super().__init__()
		self.in_chans = in_chans
		self.embed_dim = embed_dim

		if kernel_size is None:
			kernel_size = patch_size

		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
							  padding=(kernel_size-patch_size+1)//2, padding_mode='reflect')

	def forward(self, x):
		x = self.proj(x)
		return x


class PatchUnEmbed(nn.Module):
	def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
		super().__init__()
		self.out_chans = out_chans
		self.embed_dim = embed_dim

		if kernel_size is None:
			kernel_size = 1

		self.proj = nn.Sequential(
			nn.Conv2d(embed_dim, out_chans*patch_size**2, kernel_size=kernel_size,
					  padding=kernel_size//2, padding_mode='reflect'),
			nn.PixelShuffle(patch_size)
		)

	def forward(self, x):
		x = self.proj(x)
		return x


class SKFusion(nn.Module):
	def __init__(self, dim, height=2, reduction=8):
		super(SKFusion, self).__init__()

		self.height = height
		d = max(int(dim/reduction), 4)

		self.mlp = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(dim, d, 1, bias=False),
			nn.ReLU(True),
			nn.Conv2d(d, dim*height, 1, bias=False)
		)

		self.softmax = nn.Softmax(dim=1)

	def forward(self, in_feats):
		B, C, H, W = in_feats[0].shape

		in_feats = torch.cat(in_feats, dim=1)
		in_feats = in_feats.view(B, self.height, C, H, W)

		feats_sum = torch.sum(in_feats, dim=1)
		attn = self.mlp(feats_sum)
		attn = self.softmax(attn.view(B, self.height, C, 1, 1))

		out = torch.sum(in_feats*attn, dim=1)
		return out


class gUNet(nn.Module):
	def __init__(self, kernel_size=5, base_dim=24, depths=[3, 3, 3, 6, 3, 3, 3], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion):
		super(gUNet, self).__init__()
		# setting
		assert len(depths) % 2 == 1
		stage_num = len(depths)
		half_num = stage_num // 2
		net_depth = sum(depths)
		embed_dims = [2**i*base_dim for i in range(half_num)]
		embed_dims = embed_dims + [2**half_num*base_dim] + embed_dims[::-1]

		self.patch_size = 2 ** (stage_num // 2)
		self.stage_num = stage_num
		self.half_num = half_num

		# input convolution
		self.inconv = PatchEmbed(patch_size=1, in_chans=3, embed_dim=embed_dims[0], kernel_size=3)

		# backbone
		self.layers = nn.ModuleList()
		self.downs = nn.ModuleList()
		self.ups = nn.ModuleList()
		self.skips = nn.ModuleList()
		self.fusions = nn.ModuleList()

		for i in range(self.stage_num):
			self.layers.append(BasicLayer(dim=embed_dims[i], depth=depths[i], net_depth=net_depth, kernel_size=kernel_size, 
										  conv_layer=conv_layer, norm_layer=norm_layer, gate_act=gate_act))

		for i in range(self.half_num):
			self.downs.append(PatchEmbed(patch_size=2, in_chans=embed_dims[i], embed_dim=embed_dims[i+1]))
			self.ups.append(PatchUnEmbed(patch_size=2, out_chans=embed_dims[i], embed_dim=embed_dims[i+1]))
			self.skips.append(nn.Conv2d(embed_dims[i], embed_dims[i], 1))
			self.fusions.append(fusion_layer(embed_dims[i]))

		# output convolution
		self.outconv = PatchUnEmbed(patch_size=1, out_chans=4, embed_dim=embed_dims[-1], kernel_size=3)


	def forward(self, x):
		feat = self.inconv(x)

		skips = []

		for i in range(self.half_num):
			feat = self.layers[i](feat)
			skips.append(self.skips[i](feat))
			feat = self.downs[i](feat)

		x_mid = feat

		feat = self.layers[self.half_num](feat)

		x_decoder = feat

		for i in range(self.half_num-1, -1, -1):
			feat = self.ups[i](feat)
			feat = self.fusions[i]([feat, skips[i]])
			feat = self.layers[self.stage_num-i-1](feat)

		feat = self.outconv(feat)

		K, B = torch.split(feat, (1, 3), dim=1)

		x = K * x + B + x

		return x, x_mid, x_decoder
	

class gUNet_Decoder(nn.Module):
	def __init__(self, kernel_size=5, base_dim=24, depths=[3, 3, 3], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion):
		super(gUNet_Decoder, self).__init__()
		# setting
		assert len(depths) % 2 == 1
		stage_num = len(depths)
		half_num = stage_num
		net_depth = sum(depths)
		embed_dims = [2**i*base_dim for i in range(half_num+1)][::-1]

		self.patch_size = 2 ** (stage_num // 2)
		self.stage_num = stage_num
		self.half_num = half_num

		# backbone
		self.layers = nn.ModuleList()
		self.ups = nn.ModuleList()

		for i in range(self.stage_num):
			self.layers.append(BasicLayer(dim=embed_dims[i+1], depth=depths[i], net_depth=net_depth, kernel_size=kernel_size, 
										  conv_layer=conv_layer, norm_layer=norm_layer, gate_act=gate_act))
			self.ups.append(PatchUnEmbed(patch_size=2, out_chans=embed_dims[i+1], embed_dim=embed_dims[i]))


		# output convolution
		self.outconv = PatchUnEmbed(patch_size=1, out_chans=3, embed_dim=embed_dims[-1], kernel_size=3)


	def forward(self, x):

		for i in range(self.half_num):
			x = self.ups[i](x)
			x = self.layers[i](x)

		x = self.outconv(x)

		return x


class DecomNet(nn.Module):
    def __init__(self, channel=192, kernel_size=3):
        super(DecomNet, self).__init__()
        
        # Activated layers!

        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, 1),
										nn.Conv2d(channel, channel, kernel_size=kernel_size, padding=kernel_size//2, groups=channel, padding_mode='reflect'),
                                        nn.ReLU(),
					
                                        nn.Conv2d(channel, channel, 1),
										nn.Conv2d(channel, channel, kernel_size=kernel_size, padding=kernel_size//2, groups=channel, padding_mode='reflect'),
                                        nn.ReLU(),
					
										nn.Conv2d(channel, channel, 1),
										nn.Conv2d(channel, channel, kernel_size=kernel_size, padding=kernel_size//2, groups=channel, padding_mode='reflect'),
                                        nn.ReLU(),
					
                                        nn.Conv2d(channel, channel, 1),
										nn.Conv2d(channel, channel, kernel_size=kernel_size, padding=kernel_size//2, groups=channel, padding_mode='reflect'),
                                        nn.ReLU(),
					
										nn.Conv2d(channel, channel, 1),
										nn.Conv2d(channel, channel, kernel_size=kernel_size, padding=kernel_size//2, groups=channel, padding_mode='reflect'),
                                        nn.ReLU()
										)
        # Final recon layer
        self.net1_recon = nn.Conv2d(channel, 4, kernel_size,
                                    padding=1, padding_mode='reflect')

    def forward(self, input_im):
        featss   = self.net1_convs(input_im)
        outs     = self.net1_recon(featss)
        R        = outs[:, 0:3, :, :]
        I        = torch.sigmoid(outs[:, 3:4, :, :])
        return R, I


class gUNet_Teacher_Encoder(nn.Module):
	def __init__(self, kernel_size=5, base_dim=24, depths=[3, 3, 3], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion):
		super(gUNet_Teacher_Encoder, self).__init__()

		stage_num = len(depths)
		half_num = stage_num
		net_depth = sum(depths)
		embed_dims = [2**i*base_dim for i in range(half_num)]
		embed_dims = embed_dims + [2**half_num*base_dim] + embed_dims[::-1]

		self.patch_size = 2 ** (stage_num)
		self.stage_num = stage_num
		self.half_num = half_num

		# input convolution
		self.inconv = PatchEmbed(patch_size=1, in_chans=3, embed_dim=embed_dims[0], kernel_size=3)

		# backbone
		self.layers = nn.ModuleList()
		self.downs = nn.ModuleList()

		for i in range(self.stage_num):
			self.layers.append(BasicLayer(dim=embed_dims[i], depth=depths[i], net_depth=net_depth, kernel_size=kernel_size, 
										  conv_layer=conv_layer, norm_layer=norm_layer, gate_act=gate_act))

		for i in range(self.half_num):
			self.downs.append(PatchEmbed(patch_size=2, in_chans=embed_dims[i], embed_dim=embed_dims[i+1]))


	def forward(self, x):
		feat = self.inconv(x)

		for i in range(self.half_num):
			feat = self.layers[i](feat)
			feat = self.downs[i](feat)

		return feat
	
class RelightNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(RelightNet, self).__init__()

        self.relu         = nn.ReLU()
        self.net2_conv0_1 = nn.Conv2d(4, channel, kernel_size,
                                      padding=1, padding_mode='reflect')

        self.net2_conv1_1 = nn.Conv2d(channel, channel, kernel_size, stride=2,
                                      padding=1, padding_mode='reflect')
        self.net2_conv1_2 = nn.Conv2d(channel, channel, kernel_size, stride=2,
                                      padding=1, padding_mode='reflect')
        self.net2_conv1_3 = nn.Conv2d(channel, channel, kernel_size, stride=2,
                                      padding=1, padding_mode='reflect')

        self.net2_deconv1_1= nn.Conv2d(channel*2, channel, kernel_size,
                                       padding=1, padding_mode='reflect')
        self.net2_deconv1_2= nn.Conv2d(channel*2, channel, kernel_size,
                                       padding=1, padding_mode='reflect')
        self.net2_deconv1_3= nn.Conv2d(channel*2, channel, kernel_size,
                                       padding=1, padding_mode='reflect')

        self.net2_fusion = nn.Conv2d(channel*3, channel, kernel_size=1,
                                     padding=1, padding_mode='reflect')
        self.net2_output = nn.Conv2d(channel, 1, kernel_size=3, padding=0)

    def forward(self, input_L, input_R):
        input_img = torch.cat((input_R, input_L), dim=1)
        out0      = self.net2_conv0_1(input_img)
        out1      = self.relu(self.net2_conv1_1(out0))
        out2      = self.relu(self.net2_conv1_2(out1))
        out3      = self.relu(self.net2_conv1_3(out2))

        out3_up   = F.interpolate(out3, size=(out2.size()[2], out2.size()[3]))
        deconv1   = self.relu(self.net2_deconv1_1(torch.cat((out3_up, out2), dim=1)))
        deconv1_up= F.interpolate(deconv1, size=(out1.size()[2], out1.size()[3]))
        deconv2   = self.relu(self.net2_deconv1_2(torch.cat((deconv1_up, out1), dim=1)))
        deconv2_up= F.interpolate(deconv2, size=(out0.size()[2], out0.size()[3]))
        deconv3   = self.relu(self.net2_deconv1_3(torch.cat((deconv2_up, out0), dim=1)))

        deconv1_rs= F.interpolate(deconv1, size=(input_R.size()[2], input_R.size()[3]))
        deconv2_rs= F.interpolate(deconv2, size=(input_R.size()[2], input_R.size()[3]))
        feats_all = torch.cat((deconv1_rs, deconv2_rs, deconv3), dim=1)
        feats_fus = self.net2_fusion(feats_all)
        output    = self.net2_output(feats_fus)
        return output


class LRNet(nn.Module):
	def __init__(self, depths):
		super(LRNet, self).__init__()

		self.main_LRNet = gUNet(kernel_size=5, base_dim=24, depths=[depths[0], depths[1], depths[2], depths[3], depths[4], depths[5], depths[6]], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)
		self.teacher = gUNet_Teacher_Encoder(kernel_size=5, base_dim=24, depths=[depths[0], depths[1], depths[2]], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)
		self.enhance = RelightNet()
		self.decom_1 = DecomNet()
		self.decom_2 = DecomNet()

	def forward(self, input, lle_input):

		output, x_mid, x_decoder = self.main_LRNet(input)
		R_low, I_low = self.decom_1(x_mid)
		R_high, I_high = self.decom_2(self.teacher(lle_input))
		I_delta = self.enhance(I_low, R_low)

		I_high_3 = torch.cat((I_high, I_high, I_high), dim=1)
		I_delta_3= torch.cat((I_delta, I_delta, I_delta), dim=1)


		lle_gt = F.interpolate(lle_input, size=(input.shape[2]//8, input.shape[3]//8), mode='bicubic', align_corners=False)

		return output, R_low, R_high, I_high_3, I_delta_3, lle_gt, x_decoder


class Translation2D(nn.Module):
    def __init__(self, channel):
        super(Translation2D, self).__init__()
        self.trans = nn.Sequential(
                nn.Conv2d(channel, channel * 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel * 8, 3, 1, padding=0, bias=True),
        )
    def forward(self, x):
        y = self.trans(x)
        return y

class LRNet_main(nn.Module):
	def __init__(self, depths=[1, 1, 1, 2, 1, 1, 1]):
		super(LRNet_main, self).__init__()

		self.LRNet = LRNet(depths=[depths[0], depths[1], depths[2], depths[3], depths[4], depths[5], depths[6]])	
		self.es_T = gUNet_Decoder(kernel_size=5, base_dim=24, depths=[depths[0], depths[1], depths[2]], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)
	
		self.trans = Translation2D(3)

	def forward(self, input, lle_input):

		output, R_low, R_high, I_high_3, I_delta_3, lle_gt, x_decoder = self.LRNet(input, lle_input)

		T = self.es_T(x_decoder)

		noise = self.trans(input)

		I_re = (T*output) + noise

		return output, R_low, R_high, I_high_3, I_delta_3, lle_gt, I_re

def trainPEARL_s():
	return LRNet_main(depths=[1, 1, 1, 2, 1, 1, 1])

def trainPEARL_m():
	return LRNet_main(depths=[3, 3, 3, 6, 3, 3, 3])

def trainPEARL_b():
	return LRNet_main(depths=[9, 9, 9, 18, 9, 9, 9])

def testPEARL_s():
	return gUNet(kernel_size=5, base_dim=24, depths=[1, 1, 1, 2, 1, 1, 1], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def testPEARL_m():
	return gUNet(kernel_size=5, base_dim=24, depths=[3, 3, 3, 6, 3, 3, 3], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def testPEARL_b():
	return gUNet(kernel_size=5, base_dim=24, depths=[9, 9, 9, 18, 9, 9, 9], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

