import torch.nn as nn
from net.wasw_encoder import WASWEncoder
from net.wasw_decoder import WASWDecoder


class WASW(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=(4, 6, 6, 8),
        num_refinement_blocks=4,
        heads=(1, 2, 4, 8),
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        decoder=True,
        finetune_fmim=False,
        wavelet_levels=3,
        single_level=False,
        use_adaptive_weighting=False,
        asw_spatial_attention=True,
        asw_frequency_aware=True,
        wavelet_levels_fre1=3,
        wavelet_levels_fre2=2,
        wavelet_levels_fre3=1,
    ):
        super().__init__()

        self.finetune_fmim = finetune_fmim
        self.wavelet_levels = wavelet_levels
        self.single_level = single_level
        self.use_adaptive_weighting = use_adaptive_weighting

        self.encoder = WASWEncoder(
            inp_channels=inp_channels,
            dim=dim,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )

        self.decoder = WASWDecoder(
            dim=dim,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            enable_dasm=decoder,
            wavelet_levels_fre1=wavelet_levels_fre1,
            wavelet_levels_fre2=wavelet_levels_fre2,
            wavelet_levels_fre3=wavelet_levels_fre3,
            single_level=single_level,
            use_adaptive_weighting=use_adaptive_weighting,
            asw_spatial_attention=asw_spatial_attention,
            asw_frequency_aware=asw_frequency_aware,
        )

        if num_refinement_blocks > 0:
            self.refinement = nn.Sequential(
                *[
                    self._make_refinement_block(
                        dim=int(dim * 2**1),
                        num_heads=heads[0],
                        ffn_expansion_factor=ffn_expansion_factor,
                        bias=bias,
                        LayerNorm_type=LayerNorm_type,
                    )
                    for _ in range(num_refinement_blocks)
                ]
            )
        else:
            self.refinement = nn.Identity()

        self.output = nn.Conv2d(
            int(dim * 2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def _make_refinement_block(
        self,
        dim,
        num_heads,
        ffn_expansion_factor,
        bias,
        LayerNorm_type,
    ):
        from utils.model_components import TransformerBlock

        return TransformerBlock(
            dim=dim,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )

    def forward(self, inp_img, noise_emb=None):
        enc_feats = self.encoder(inp_img)
        out_dec_level1 = self.decoder(enc_feats, inp_img)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        return out_dec_level1

