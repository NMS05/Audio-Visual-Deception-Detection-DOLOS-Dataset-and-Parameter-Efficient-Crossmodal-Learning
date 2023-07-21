
import torch
import torch.nn as nn
import torchaudio
import torchvision
from models.adapter import w2v2_adapter_nlp, w2v2_adapter_conv, vit_adapter_nlp, vit_adapter_conv
from models.visual_model import cnn_face
from torch.nn import functional as F


class CrossFusionModule(nn.Module):
    """
    crossmodal fusion module: to get crossmodal attention and to return the fused feature.
    The cross attention is calculated by the output embedding from each encoder layers of audio and visual modalities.
    """
    def __init__(self, dim=256):
        super(CrossFusionModule, self).__init__()

        # linear project + norm + corr + concat + conv_layer + tanh
        self.project_audio = nn.Linear(768, dim)  # linear projection
        self.project_vision = nn.Linear(768, dim)
        self.corr_weights = torch.nn.Parameter(torch.empty(
            dim, dim, requires_grad=True).type(torch.cuda.FloatTensor))
        nn.init.xavier_normal_(self.corr_weights)
        self.project_bottleneck = nn.Sequential(nn.Linear(dim * 2, 64),
                                                nn.LayerNorm((64,), eps=1e-05, elementwise_affine=True),
                                                nn.ReLU())

    def forward(self, audio_feat, visual_feat):
        """

        :param audio_feat: [batchsize 64 768]
        :param visual_feat:[batchsize 64 768]
        :return: fused feature
        """
        audio_feat = self.project_audio(audio_feat)
        visual_feat = self.project_vision(visual_feat)

        visual_feat = visual_feat.transpose(1, 2)  # 768, 64

        a1 = torch.matmul(audio_feat, self.corr_weights)  # 768, 768
        cc_mat = torch.bmm(a1, visual_feat)  # 64*64

        audio_att = F.softmax(cc_mat, dim=1)
        visual_att = F.softmax(cc_mat.transpose(1, 2), dim=1)
        atten_audiofeatures = torch.bmm(audio_feat.transpose(1, 2), audio_att)
        atten_visualfeatures = torch.bmm(visual_feat, visual_att)
        atten_audiofeatures = atten_audiofeatures + audio_feat.transpose(1, 2)
        atten_visualfeatures = atten_visualfeatures + visual_feat  # 256, 64

        fused_features = self.project_bottleneck(torch.cat((atten_audiofeatures,
                                                            atten_visualfeatures), dim=1).transpose(1, 2))

        return fused_features


class Fusion(nn.Module):
    """
    including types of encoder w/ or w/o adapters and fusion methods
    """
    def __init__(self, fusion_type, num_encoders, adapter, adapter_type, multi=False):
        super(Fusion, self).__init__()
        self.fusion_type = fusion_type  # to indicate fusion type: "concat" or "cross2"
        self.num_encoders = num_encoders  # number of encoders used for audio and visual modalities
        self.adapter = adapter   # to indicate using adapter or not
        self.adapter_type = adapter_type  # type of adapr "nlp" or  "efficient_conv"
        self.multi = multi  # multitask learning with multiple losses for audio and visual

        # audio model
        model = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
        for p in model.parameters():
            p.requires_grad = False

        # pretrained CNN feature extracor
        self.FEATURE_EXTRACTOR = model.feature_extractor

        # pretrained feature projection + pos encoding
        self.FEATURE_PROJECTOR = nn.Sequential(
            model.encoder.feature_projection,
            model.encoder.transformer.pos_conv_embed,
            model.encoder.transformer.layer_norm,
            model.encoder.transformer.dropout,
        )

        # build w2V2 encoder with desired number of encoder layers
        audio_layer_list = []

        for i in range(self.num_encoders):

            if self.adapter:
                if self.adapter_type == 'nlp':
                    audio_layer_list.append(w2v2_adapter_nlp(transformer_encoder=model.encoder.transformer.layers[i]))
                else:
                    audio_layer_list.append(w2v2_adapter_conv(transformer_encoder=model.encoder.transformer.layers[i]))

            else:
                # fine_tune enoder in case we donot use adapters
                for p in model.encoder.transformer.layers[i].parameters(): p.requires_grad = True
                audio_layer_list.append(model.encoder.transformer.layers[i])

        self.TRANSFORMER = nn.Sequential(*audio_layer_list)

        #  visual model

        self.projection = nn.Sequential(
            nn.Linear(256, 768),
            nn.ReLU(),
        )

        # Load imagenet pretrained ViT Base 16 and freeze all parameters first
        vit_b_16 = torchvision.models.vit_b_16(pretrained=True)
        for p in vit_b_16.parameters():
            p.requires_grad = False

        # extract encoder alone and discard CNN (patchify + linear projection) feature extractor, classifer head
        # Refer Encoder() class in https://pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html
        vit = vit_b_16.encoder

        # add learnable positional embedding for 64 tokens (dim=768). Original ViT uses 196+1 tokens for position embedding
        self.pos_embedding = nn.Parameter(torch.empty(1, 64, 768).normal_(std=0.02))

        # start building ViT encoder layers
        face_layer_list = []
        for i in range(self.num_encoders):
            if self.adapter:
                if self.adapter_type == 'nlp':
                    face_layer_list.append(vit_adapter_nlp(transformer_encoder=vit.layers[i]))
                else:
                    face_layer_list.append(
                        vit_adapter_conv(transformer_encoder=vit.layers[i]))
            else:
                # fine_tune enoder in case we do not use adapters
                for p in vit.layers[i].parameters(): p.requires_grad = True
                face_layer_list.append(vit.layers[i])

        # assign models for forward pass
        self.cnn_feature_extractor = cnn_face()
        self.ViT_Encoder = nn.Sequential(*face_layer_list)

        if self.fusion_type == "concat":
            self.classifier = nn.Sequential(nn.Linear(768 * 2, 2))
        elif self.fusion_type == "cross2":
            cross_conv_layer = []
            for i in range(self.num_encoders):
                cross_conv_layer.append(CrossFusionModule(dim=256))
            self.cross_conv_layer = nn.Sequential(*cross_conv_layer)
            self.classifier = nn.Sequential(nn.Linear(64 * self.num_encoders, 64),
                                            nn.Dropout(p=0.5),
                                            nn.Linear(64, 2))
            if self.multi:
                self.audio_classifier = nn.Linear(768, 2)
                self.vision_classifier = nn.Linear(768, 2)
        else:
            self.classifier = nn.Sequential(nn.Linear(768 * 2, 2))

    def forward(self, x, y):
        x, _ = self.FEATURE_EXTRACTOR(x, None)
        audios = self.FEATURE_PROJECTOR(x)

        # Feature extraction by resnet 18 for all 64 images
        b_s, no_of_frames, C, H, W = y.shape
        y = torch.reshape(y, (b_s * no_of_frames, C, H, W))
        faces = self.cnn_feature_extractor(y)
        faces = torch.reshape(faces, (b_s, no_of_frames, 256))
        # projection for vit encoder + position embedding
        faces = self.projection(faces) + self.pos_embedding

        feat_ls = []
        if self.fusion_type == "concat":
            # simple concatenation
            audios = self.TRANSFORMER(audios)
            faces = self.ViT_Encoder(faces)
            fused_output = torch.cat((audios, faces), dim=-1)
        elif self.fusion_type in ["cross2", ]:
            # attention+conv1d+tanh
            assert len(self.TRANSFORMER) == len(self.ViT_Encoder), "unmatched encoders between audio and face"
            for audio_net, visual_net, cross_conv in zip(self.TRANSFORMER, self.ViT_Encoder,
                                                         self.cross_conv_layer):
                audios = audio_net(audios)
                faces = visual_net(faces)
                fused_features = cross_conv(audios, faces)
                feat_ls.append(fused_features)
            fused_output = torch.cat(feat_ls, dim=-1)
        else:
            raise Exception("undefined fusion type")

        logits = self.classifier(fused_output)

        if self.multi:
            a_logits = self.audio_classifier(audios)
            v_logits = self.vision_classifier(faces)
            return torch.mean(logits, 1), torch.mean(a_logits, 1), torch.mean(v_logits, 1)
        else:
            return torch.mean(logits, 1), None, None


if __name__ == '__main__':

    # for testing :
    model = Fusion("concat", num_encoders=2, adapter=True, adapter_type="efficient_conv").cuda()
    print(model)
    inp = torch.rand(8, 20601).cuda()  # batch_size = 8
    vis = torch.rand(8, 64, 3, 160, 160).cuda()
    out = model(inp, vis)
    print(out.shape)

"""
fusion types:

1- simple concatenation of the final outputs from audio and face models
2- concatenation between each encoders and final concatenation 
"""
