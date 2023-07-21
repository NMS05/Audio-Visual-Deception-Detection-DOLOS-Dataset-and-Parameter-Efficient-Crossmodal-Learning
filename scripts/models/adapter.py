import torch
import torch.nn as nn


################# Non-Linear Adapters ######################

# The adapter is placed in-between MHSA and FFN with skip connection, instead of a parallel configuration

class w2v2_adapter_nlp(nn.Module):
    def __init__(self, transformer_encoder):
        super(w2v2_adapter_nlp, self).__init__()

        self.attention = nn.Sequential(
            transformer_encoder.attention,
            transformer_encoder.dropout,
            transformer_encoder.layer_norm,
        )
        self.adapter = nn.Sequential(
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=768, out_features=256, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=768, bias=True),
            nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True),
        )
        self.feed_forward = nn.Sequential(
            transformer_encoder.feed_forward,
            transformer_encoder.final_layer_norm,
        )

    def forward(self, x):
        mhsa = self.attention(x) + x
        adapter_seq = self.adapter(mhsa) + mhsa
        ffn = self.feed_forward(adapter_seq) + adapter_seq
        return ffn


class vit_adapter_nlp(nn.Module):
    def __init__(self, transformer_encoder):
        super(vit_adapter_nlp, self).__init__()

        self.ln1 = transformer_encoder.ln_1
        self.attention = transformer_encoder.self_attention
        self.drop = transformer_encoder.dropout

        self.adapter = nn.Sequential(
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=768, out_features=256, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=768, bias=True),
            nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True),
        )

        # Feed Forward Layers
        self.feed_forward = nn.Sequential(
            transformer_encoder.ln_2,
            transformer_encoder.mlp,
        )

    def forward(self, x):
        norm_x = self.ln1(x)
        attention, _ = self.attention(query=norm_x, key=norm_x, value=norm_x, need_weights=False)
        mhsa = self.drop(attention) + x

        adapter_seq = self.adapter(mhsa) + mhsa
        ffn = self.feed_forward(adapter_seq) + adapter_seq
        return ffn


################# Convolutional Adapters ####################

# Conv pass runs parallel to MHSA and/or FFN.
# We use 1d cnn for both audio and visiual modalities

class Efficient_Conv_Pass(nn.Module):
    def __init__(self,):
        super(Efficient_Conv_Pass, self).__init__()

        # More efficient 1d conv - 492k params per encoder layer
        self.adapter_down = nn.Linear(768, 32)  # equivalent to 1 * 1 Conv
        self.adapter_gelu = nn.GELU()
        self.adapter_1d_cnn = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same', groups=1, bias=True, padding_mode='zeros')
        self.adapter_up = nn.Linear(32, 768)  # equivalent to 1 * 1 Conv

        self.adapter_norm = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

    def forward(self, x):
        down = self.adapter_gelu(self.adapter_down(x))  # shape = [batch_size, 64, 128]
        down = down.permute(0, 2, 1)  # shape = [batch_size, 128, 64]
        conv = self.adapter_gelu(self.adapter_1d_cnn(down))  # shape = [batch_size, 128, 64]
        conv = conv.permute(0, 2, 1)  # shape = [batch_size, 64, 128]
        up = self.adapter_gelu(self.adapter_up(conv))  # shape = [batch_size, 64, 768]

        out = self.adapter_norm(up + x)  # shape = [batch_size, 64, 768]
        return out


class w2v2_adapter_conv(nn.Module):
    def __init__(self, transformer_encoder):
        super(w2v2_adapter_conv, self).__init__()

        # Attention Layers
        # note that for W2V2 encoder layer_norm is after MHSA while for ViT encoder layer_norm is before
        self.attention = nn.Sequential(
            transformer_encoder.attention,
            transformer_encoder.dropout,
            transformer_encoder.layer_norm,
        )

        self.mhsa_conv_pass = Efficient_Conv_Pass()
        self.ffn_conv_pass = Efficient_Conv_Pass()

        # norm required after conv pass
        self.adapter_norm1 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.adapter_norm2 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

        # Feed Forward Layers
        self.feed_forward = nn.Sequential(
            transformer_encoder.feed_forward,
            transformer_encoder.final_layer_norm,
        )

    def forward(self, x):

        # shape of x = [batch_size, 64, 768]
        mhsa = self.attention(x) + x + self.mhsa_conv_pass(x)
        mhsa = self.adapter_norm1(mhsa)

        ffn = self.feed_forward(mhsa) + mhsa + self.ffn_conv_pass(mhsa)
        ffn = self.adapter_norm2(ffn)

        return ffn


class vit_adapter_conv(nn.Module):
    def __init__(self, transformer_encoder):
        super(vit_adapter_conv, self).__init__()

        # Attention Layers. refer EncoderBlock() class in https://pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html
        self.ln1 = transformer_encoder.ln_1
        self.attention = transformer_encoder.self_attention
        self.drop = transformer_encoder.dropout

        # conv layer in adapters
        self.mhsa_conv_pass = Efficient_Conv_Pass()
        self.ffn_conv_pass = Efficient_Conv_Pass()

        # norm required after conv pass
        self.adapter_norm1 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.adapter_norm2 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

        # Feed Forward Layers
        self.feed_forward = nn.Sequential(
            transformer_encoder.ln_2,
            transformer_encoder.mlp,
        )

    def forward(self, x):
        # shape of x = [batch_size, 64, 768]

        conv_pass = self.mhsa_conv_pass(x)

        # Attention need to be performed individually. Refer EncoderBlock() class in https://pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html
        norm_x = self.ln1(x)
        attention, _ = self.attention(query=norm_x, key=norm_x, value=norm_x, need_weights=False)
        attention = self.drop(attention)

        mhsa = attention + x + conv_pass
        mhsa = self.adapter_norm1(mhsa)

        ffn = self.feed_forward(mhsa) + mhsa + self.ffn_conv_pass(mhsa)
        ffn = self.adapter_norm2(ffn)

        return ffn