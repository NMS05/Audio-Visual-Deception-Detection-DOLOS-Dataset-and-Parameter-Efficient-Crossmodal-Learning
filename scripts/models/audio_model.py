import torch
import torch.nn as nn
import torchaudio
from models.adapter import w2v2_adapter_nlp, w2v2_adapter_conv


class W2V2_Model(nn.Module):
    def __init__(self, num_encoders, adapter, adapter_type):  # adapter_conv_params as a tuple (kernel_size, stride)
        super(W2V2_Model, self).__init__()

        self.num_encoders = num_encoders
        self.adapter = adapter
        self.adapter_type = adapter_type

        model = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
        for p in model.parameters(): p.requires_grad = False

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
        layer_list = []

        for i in range(self.num_encoders):

            if self.adapter:
                if self.adapter_type == 'nlp':
                    layer_list.append(w2v2_adapter_nlp(transformer_encoder=model.encoder.transformer.layers[i]))
                else:
                    layer_list.append(w2v2_adapter_conv(transformer_encoder=model.encoder.transformer.layers[i]))
            else:
                # fine_tune enoder in case we donot use adapters
                for p in model.encoder.transformer.layers[i].parameters(): p.requires_grad = True
                layer_list.append(model.encoder.transformer.layers[i])

        self.TRANSFORMER = nn.Sequential(*layer_list)

        # linear classifier
        self.classifier = nn.Sequential(
            nn.Linear(768, 2)
        )

    def forward(self, x):

        features, _ = self.FEATURE_EXTRACTOR(x, None)
        projections = self.FEATURE_PROJECTOR(features)
        output_tokens = self.TRANSFORMER(projections)
        logits = self.classifier(output_tokens)

        return torch.mean(logits, 1)

# model = W2V2_Model(num_encoders=4, adapter=True, adapter_type='nlp')
