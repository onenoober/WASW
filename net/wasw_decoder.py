import torch
import torch.nn as nn
from utils.model_components import TransformerBlock, Upsample
from utils.fusion_modules import DASM


class WASWDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_blocks,
        heads,
        ffn_expansion_factor: float,
        bias: bool,
        LayerNorm_type: str,
        enable_dasm: bool,
        wavelet_levels_fre1: int,
        wavelet_levels_fre2: int,
        wavelet_levels_fre3: int,
        single_level: bool,
        use_adaptive_weighting: bool,
        asw_spatial_attention: bool,
        asw_frequency_aware: bool,
    ):
        super().__init__()

        self.enable_dasm = enable_dasm

        if self.enable_dasm:
            self.dasm1 = DASM(
                dim=dim * 2**3,
                num_heads=heads[2],
                bias=bias,
                levels=wavelet_levels_fre1,
                single_level=single_level,
                use_adaptive_weighting=use_adaptive_weighting,
                asw_spatial_attention=asw_spatial_attention,
                asw_frequency_aware=asw_frequency_aware,
            )
            self.dasm2 = DASM(
                dim=dim * 2**2,
                num_heads=heads[2],
                bias=bias,
                levels=wavelet_levels_fre2,
                single_level=single_level,
                use_adaptive_weighting=use_adaptive_weighting,
                asw_spatial_attention=asw_spatial_attention,
                asw_frequency_aware=asw_frequency_aware,
            )
            self.dasm3 = DASM(
                dim=dim * 2**1,
                num_heads=heads[2],
                bias=bias,
                levels=wavelet_levels_fre3,
                single_level=single_level,
                use_adaptive_weighting=use_adaptive_weighting,
                asw_spatial_attention=asw_spatial_attention,
                asw_frequency_aware=asw_frequency_aware,
            )

        self.up4_3 = Upsample(int(dim * 2**3))
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias
        )
        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(int(dim * 2**1))
        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[0])
            ]
        )

    def forward(self, enc_feats, inp_img):
        latent = enc_feats["latent"]

        if self.enable_dasm:
            latent = self.dasm1(inp_img, latent)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, enc_feats["out_enc_level3"]], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        if self.enable_dasm:
            out_dec_level3 = self.dasm2(inp_img, out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, enc_feats["out_enc_level2"]], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        if self.enable_dasm:
            out_dec_level2 = self.dasm3(inp_img, out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, enc_feats["out_enc_level1"]], 1)

        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        return out_dec_level1

