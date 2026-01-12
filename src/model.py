import math

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

try:
    from .torch_cfc import Cfc
except ImportError:
    from torch_cfc import Cfc


class ECAModule(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class BasicBlockWithECA(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = ECAModule(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18WithECA(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super().__init__()

        if pretrained:
            resnet18 = models.resnet18(pretrained=True)
        else:
            resnet18 = models.resnet18(pretrained=False)

        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool

        self.inplanes = 64
        self.layer1 = self._make_layer_with_eca(BasicBlockWithECA, 64, 2, pretrained_layer=resnet18.layer1)
        self.layer2 = self._make_layer_with_eca(
            BasicBlockWithECA, 128, 2, stride=2, pretrained_layer=resnet18.layer2
        )
        self.layer3 = self._make_layer_with_eca(
            BasicBlockWithECA, 256, 2, stride=2, pretrained_layer=resnet18.layer3
        )
        self.layer4 = self._make_layer_with_eca(
            BasicBlockWithECA, 512, 2, stride=2, pretrained_layer=resnet18.layer4
        )

        self.avgpool = resnet18.avgpool
        self.fc = nn.Linear(512 * BasicBlockWithECA.expansion, num_classes)

        if pretrained and hasattr(resnet18, "fc"):
            if resnet18.fc.weight.shape == self.fc.weight.shape:
                self.fc.weight.data.copy_(resnet18.fc.weight.data)
                self.fc.bias.data.copy_(resnet18.fc.bias.data)

    def _make_layer_with_eca(self, block, planes, blocks, stride=1, pretrained_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        if pretrained_layer is not None:
            self._copy_pretrained_weights(layers[0], pretrained_layer[0])
            if (
                downsample is not None
                and hasattr(pretrained_layer[0], "downsample")
                and pretrained_layer[0].downsample is not None
            ):
                downsample[0].weight.data.copy_(pretrained_layer[0].downsample[0].weight.data)
                downsample[1].weight.data.copy_(pretrained_layer[0].downsample[1].weight.data)
                downsample[1].bias.data.copy_(pretrained_layer[0].downsample[1].bias.data)

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            if pretrained_layer is not None and i < len(pretrained_layer):
                self._copy_pretrained_weights(layers[i], pretrained_layer[i])

        return nn.Sequential(*layers)

    def _copy_pretrained_weights(self, eca_block, pretrained_block):
        eca_block.conv1.weight.data.copy_(pretrained_block.conv1.weight.data)
        eca_block.bn1.weight.data.copy_(pretrained_block.bn1.weight.data)
        eca_block.bn1.bias.data.copy_(pretrained_block.bn1.bias.data)
        eca_block.bn1.running_mean.data.copy_(pretrained_block.bn1.running_mean.data)
        eca_block.bn1.running_var.data.copy_(pretrained_block.bn1.running_var.data)

        eca_block.conv2.weight.data.copy_(pretrained_block.conv2.weight.data)
        eca_block.bn2.weight.data.copy_(pretrained_block.bn2.weight.data)
        eca_block.bn2.bias.data.copy_(pretrained_block.bn2.bias.data)
        eca_block.bn2.running_mean.data.copy_(pretrained_block.bn2.running_mean.data)
        eca_block.bn2.running_var.data.copy_(pretrained_block.bn2.running_var.data)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class TemporalAttention(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, num_stages=3):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_stages = num_stages

        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1),
        )

        self.global_context_net = nn.Sequential(
            nn.Linear(feature_dim * num_stages, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_stages),
        )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        self.position_encoding = nn.Parameter(torch.randn(num_stages, feature_dim) * 0.1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, stage_features, mask=None):
        assert isinstance(stage_features, list), f"stage_features应该是list类型，实际类型: {type(stage_features)}"
        assert len(stage_features) == self.num_stages, (
            f"应该有{self.num_stages}个阶段特征，实际数量: {len(stage_features)}"
        )

        batch_size = stage_features[0].size(0)
        feature_dim = stage_features[0].size(1)

        for i, stage_feat in enumerate(stage_features):
            assert stage_feat.dim() == 2, f"阶段{i + 1}特征应该是2维张量，实际维度: {stage_feat.dim()}"
            assert stage_feat.size(0) == batch_size, (
                f"阶段{i + 1}批次大小不匹配，期望: {batch_size}, 实际: {stage_feat.size(0)}"
            )
            assert stage_feat.size(1) == feature_dim, (
                f"阶段{i + 1}特征维度不匹配，期望: {feature_dim}, 实际: {stage_feat.size(1)}"
            )

            if torch.isnan(stage_feat).any() or torch.isinf(stage_feat).any():
                print(f"警告: 阶段{i + 1}特征包含NaN或Inf值")
                stage_feat = torch.nan_to_num(stage_feat, nan=0.0, posinf=1e6, neginf=-1e6)
                stage_features[i] = stage_feat

        enhanced_features = []
        for i, stage_feat in enumerate(stage_features):
            pos_encoded_feat = stage_feat + self.position_encoding[i].unsqueeze(0)
            enhanced_features.append(pos_encoded_feat)

        stage_sequence = torch.stack(enhanced_features, dim=1)

        attended_sequence, _ = self.self_attention(stage_sequence, stage_sequence, stage_sequence)

        local_attention_scores = []
        for i in range(self.num_stages):
            attended_feat = attended_sequence[:, i, :]
            attended_feat_norm = F.normalize(attended_feat, p=2, dim=1)
            score = self.attention_net(attended_feat_norm)
            score = torch.clamp(score, min=-8.0, max=8.0)
            local_attention_scores.append(score)

        local_attention_scores = torch.cat(local_attention_scores, dim=1)

        global_context = torch.cat(stage_features, dim=1)
        global_adjustment = self.global_context_net(global_context)

        final_attention_scores = local_attention_scores + global_adjustment

        if torch.isnan(final_attention_scores).any() or torch.isinf(final_attention_scores).any():
            print("警告: 注意力分数包含NaN或Inf值，使用均匀权重")
            attention_weights = torch.ones(batch_size, self.num_stages, device=final_attention_scores.device) / self.num_stages
            attention_entropy = torch.tensor(math.log(self.num_stages), device=final_attention_scores.device)
        else:
            scores_max = torch.max(final_attention_scores, dim=1, keepdim=True)[0]
            scores_stable = final_attention_scores - scores_max
            attention_weights = F.softmax(scores_stable, dim=1)

            log_weights = torch.log(attention_weights + 1e-8)
            attention_entropy = -torch.sum(attention_weights * log_weights, dim=1).mean()

        if torch.isnan(attention_weights).any() or torch.isinf(attention_weights).any():
            print("警告: 最终注意力权重包含NaN或Inf值，使用均匀权重")
            attention_weights = torch.ones(batch_size, self.num_stages, device=attention_weights.device) / self.num_stages
            attention_entropy = torch.tensor(math.log(self.num_stages), device=attention_weights.device)

        weight_sum = torch.sum(attention_weights, dim=1, keepdim=True)
        weight_sum = torch.clamp(weight_sum, min=1e-8)
        attention_weights = attention_weights / weight_sum

        weighted_features = torch.zeros_like(stage_features[0])
        for i, stage_feat in enumerate(stage_features):
            weight = attention_weights[:, i : i + 1]
            weighted_features += weight * stage_feat

        if torch.isnan(weighted_features).any() or torch.isinf(weighted_features).any():
            print("警告: 加权特征包含NaN或Inf值")
            weighted_features = torch.nan_to_num(weighted_features, nan=0.0, posinf=1e6, neginf=-1e6)

        assert weighted_features.shape == (batch_size, feature_dim), (
            f"加权特征形状错误，期望: ({batch_size}, {feature_dim}), 实际: {weighted_features.shape}"
        )
        assert attention_weights.shape == (batch_size, self.num_stages), (
            f"注意力权重形状错误，期望: ({batch_size}, {self.num_stages}), 实际: {attention_weights.shape}"
        )
        assert attention_entropy.dim() == 0, f"注意力熵应该是标量，实际维度: {attention_entropy.dim()}"

        weight_sums = torch.sum(attention_weights, dim=1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6), (
            f"注意力权重和不为1，实际和: {weight_sums}"
        )

        return weighted_features, attention_weights, attention_entropy


class MultiStageVideoClassifier(nn.Module):
    def __init__(
        self,
        num_classes=4,
        hidden_size=256,
        dropout_rate=0.5,
        temporal_aggregation="gru",
        pretrained=True,
        enable_inter_stage_transfer=True,
        enable_cross_stage_modeling=True,
        enable_continuous_dynamics=True,
        enable_hidden_state_transfer=True,
        stage_max_weights=None,
        cfc_mode="cfc",
        cfc_units=64,
        cfc_proj_size=None,
        cfc_return_sequences=True,
        cfc_mixed_memory=False,
        cfc_no_gate=False,
        cfc_minimal=False,
        ltc_ode_unfolds=6,
        ltc_epsilon=1e-8,
        num_stages=3,
        frame_dt=0.04,
        use_label_guidance_for_temporal=False,
    ):
        super().__init__()

        self.num_stages = num_stages

        self.backbone = ResNet18WithECA(num_classes=1000, pretrained=pretrained)
        self.feature_dim = 512
        self.num_classes = num_classes

        self.enable_inter_stage_transfer = enable_inter_stage_transfer
        self.enable_cross_stage_modeling = enable_cross_stage_modeling
        self.enable_continuous_dynamics = enable_continuous_dynamics
        self.enable_hidden_state_transfer = enable_hidden_state_transfer

        self.use_label_guidance_for_temporal = bool(use_label_guidance_for_temporal)
        self.frame_head = nn.Linear(self.feature_dim, self.num_classes)

        self.temporal_aggregation = temporal_aggregation

        self.cfc_mode = cfc_mode
        self.cfc_units = cfc_units
        self.cfc_proj_size = cfc_proj_size
        self.cfc_return_sequences = cfc_return_sequences
        self.cfc_mixed_memory = cfc_mixed_memory
        self.cfc_no_gate = cfc_no_gate
        self.cfc_minimal = cfc_minimal
        self.ltc_ode_unfolds = ltc_ode_unfolds
        self.ltc_epsilon = ltc_epsilon
        self.frame_dt = float(frame_dt)

        if temporal_aggregation == "gru":
            self.gru_hidden_size = hidden_size // 4

            add_dim = self.num_classes if self.use_label_guidance_for_temporal else 0
            if enable_inter_stage_transfer:
                self.stage_grus = nn.ModuleList(
                    [
                        nn.GRU(
                            input_size=(self.feature_dim + add_dim)
                            if i == 0
                            else (self.feature_dim + add_dim + self.gru_hidden_size * 2),
                            hidden_size=self.gru_hidden_size,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.0,
                            bidirectional=True,
                        )
                        for i in range(num_stages)
                    ]
                )

                self.inter_stage_transform = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(self.gru_hidden_size * 2, self.gru_hidden_size * 2),
                            nn.Tanh(),
                            nn.Dropout(dropout_rate * 0.3),
                        )
                        for _ in range(num_stages - 1)
                    ]
                )

                if enable_hidden_state_transfer:
                    self.hidden_state_transform = nn.ModuleList(
                        [
                            nn.Sequential(
                                nn.Linear(self.gru_hidden_size * 2, self.gru_hidden_size * 2),
                                nn.Tanh(),
                                nn.Dropout(dropout_rate * 0.2),
                            )
                            for _ in range(num_stages - 1)
                        ]
                    )
            else:
                self.stage_grus = nn.ModuleList(
                    [
                        nn.GRU(
                            input_size=self.feature_dim + add_dim,
                            hidden_size=self.gru_hidden_size,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.0,
                            bidirectional=True,
                        )
                        for _ in range(num_stages)
                    ]
                )

            stage_feature_dim = self.gru_hidden_size * 2

        elif temporal_aggregation == "lstm":
            self.lstm_hidden_size = hidden_size // 4

            add_dim = self.num_classes if self.use_label_guidance_for_temporal else 0
            if enable_inter_stage_transfer:
                self.stage_lstms = nn.ModuleList(
                    [
                        nn.LSTM(
                            input_size=(self.feature_dim + add_dim)
                            if i == 0
                            else (self.feature_dim + add_dim + self.lstm_hidden_size * 2),
                            hidden_size=self.lstm_hidden_size,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.0,
                            bidirectional=True,
                        )
                        for i in range(num_stages)
                    ]
                )

                self.inter_stage_transform = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(self.lstm_hidden_size * 2, self.lstm_hidden_size * 2),
                            nn.Tanh(),
                            nn.Dropout(dropout_rate * 0.3),
                        )
                        for _ in range(num_stages - 1)
                    ]
                )

                if enable_hidden_state_transfer:
                    self.hidden_state_transform = nn.ModuleList(
                        [
                            nn.Sequential(
                                nn.Linear(self.lstm_hidden_size * 2, self.lstm_hidden_size * 2),
                                nn.Tanh(),
                                nn.Dropout(dropout_rate * 0.2),
                            )
                            for _ in range(num_stages - 1)
                        ]
                    )

                    self.cell_state_transform = nn.ModuleList(
                        [
                            nn.Sequential(
                                nn.Linear(self.lstm_hidden_size * 2, self.lstm_hidden_size * 2),
                                nn.Tanh(),
                                nn.Dropout(dropout_rate * 0.2),
                            )
                            for _ in range(num_stages - 1)
                        ]
                    )
            else:
                self.stage_lstms = nn.ModuleList(
                    [
                        nn.LSTM(
                            input_size=self.feature_dim + add_dim,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.0,
                            bidirectional=True,
                        )
                        for _ in range(num_stages)
                    ]
                )

            stage_feature_dim = self.lstm_hidden_size * 2

        elif temporal_aggregation in ["cfc", "ltc"]:
            add_dim = self.num_classes if self.use_label_guidance_for_temporal else 0
            if enable_inter_stage_transfer:
                self.stage_cfcs = nn.ModuleList(
                    [
                        Cfc(
                            in_features=(self.feature_dim + add_dim + 1)
                            if i == 0
                            else (self.feature_dim + add_dim + 1 + self.cfc_units),
                            hidden_size=self.cfc_units,
                            out_feature=self.cfc_units,
                            hparams={
                                "backbone_activation": "silu",
                                "backbone_units": 64,
                                "backbone_layers": 1,
                                "backbone_dr": 0.1,
                                "no_gate": self.cfc_no_gate,
                                "minimal": self.cfc_minimal,
                            },
                            return_sequences=self.cfc_return_sequences,
                            use_mixed=self.cfc_mixed_memory,
                            use_ltc=(self.cfc_mode == "ltc"),
                        )
                        for i in range(num_stages)
                    ]
                )

                self.inter_stage_transform = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(self.cfc_units, self.cfc_units),
                            nn.Tanh(),
                            nn.Dropout(dropout_rate * 0.3),
                        )
                        for _ in range(num_stages - 1)
                    ]
                )

                if enable_hidden_state_transfer:
                    self.hidden_state_transform = nn.ModuleList(
                        [
                            nn.Sequential(
                                nn.Linear(self.cfc_units, self.cfc_units),
                                nn.Tanh(),
                                nn.Dropout(dropout_rate * 0.2),
                            )
                            for _ in range(num_stages - 1)
                        ]
                    )
            else:
                self.stage_cfcs = nn.ModuleList(
                    [
                        Cfc(
                            in_features=self.feature_dim + add_dim + 1,
                            hidden_size=self.cfc_units,
                            out_feature=self.cfc_units,
                            hparams={
                                "backbone_activation": "silu",
                                "backbone_units": 64,
                                "backbone_layers": 1,
                                "backbone_dr": 0.1,
                                "no_gate": self.cfc_no_gate,
                                "minimal": self.cfc_minimal,
                            },
                            return_sequences=self.cfc_return_sequences,
                            use_mixed=self.cfc_mixed_memory,
                            use_ltc=(self.cfc_mode == "ltc"),
                        )
                        for _ in range(num_stages)
                    ]
                )

            stage_feature_dim = self.cfc_units
        else:
            stage_feature_dim = self.feature_dim

        if enable_cross_stage_modeling:
            self.cross_stage_lstm = nn.LSTM(
                input_size=stage_feature_dim,
                hidden_size=hidden_size // 2,
                num_layers=2,
                batch_first=True,
                dropout=dropout_rate * 0.5,
                bidirectional=True,
            )
            cross_stage_feature_dim = hidden_size
        else:
            cross_stage_feature_dim = 0

        if enable_continuous_dynamics:
            self.continuous_dynamics = nn.Sequential(
                nn.Linear(stage_feature_dim * num_stages, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate * 0.3),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Dropout(dropout_rate * 0.2),
                nn.Linear(hidden_size // 2, stage_feature_dim),
            )
            continuous_feature_dim = stage_feature_dim
        else:
            continuous_feature_dim = 0

        self.temporal_attention = TemporalAttention(stage_feature_dim, hidden_size // 2, num_stages)

        projection_dim = 64
        if enable_cross_stage_modeling:
            self.cross_stage_projection = nn.Sequential(
                nn.Linear(cross_stage_feature_dim, projection_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate * 0.2),
            )
        if enable_continuous_dynamics:
            self.continuous_projection = nn.Sequential(
                nn.Linear(continuous_feature_dim, projection_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate * 0.2),
            )

        fusion_input_dim = stage_feature_dim * num_stages
        if enable_cross_stage_modeling:
            fusion_input_dim += projection_dim
        if enable_continuous_dynamics:
            fusion_input_dim += projection_dim

        self.dynamic_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_size, stage_feature_dim),
        )

        classifier_input_dim = stage_feature_dim * 2

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_input_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.25),
            nn.Linear(hidden_size // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GRU, nn.LSTM)):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.constant_(param.data, 0)

    def _extract_stage_features(self, stage_frames, stage_idx, prev_stage_features=None, prev_hidden_state=None):
        assert isinstance(stage_frames, torch.Tensor), f"stage_frames must be tensor, got {type(stage_frames)}"
        assert stage_frames.dim() == 5, (
            f"stage_frames must be 5D [batch_size, seq_len, C, H, W], got {stage_frames.dim()}D"
        )
        assert 0 <= stage_idx < self.num_stages, (
            f"stage_idx must be between 0 and {self.num_stages - 1}, got {stage_idx}"
        )

        batch_size, seq_len = stage_frames.shape[:2]
        assert batch_size > 0 and seq_len > 0, f"Invalid batch_size={batch_size} or seq_len={seq_len}"

        if prev_stage_features is not None:
            assert isinstance(prev_stage_features, torch.Tensor), (
                f"prev_stage_features must be tensor, got {type(prev_stage_features)}"
            )
            assert prev_stage_features.dim() == 2, (
                f"prev_stage_features must be 2D [batch_size, feature_dim], got {prev_stage_features.dim()}D"
            )
            assert prev_stage_features.size(0) == batch_size, (
                f"Batch size mismatch: {prev_stage_features.size(0)} vs {batch_size}"
            )

        stage_frames = stage_frames.view(-1, *stage_frames.shape[2:])

        features = self.backbone(stage_frames)

        if torch.isnan(features).any() or torch.isinf(features).any():
            print("警告: Backbone特征包含NaN或Inf值，执行nan_to_num清理")
            features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

        feature_dim = features.size(-1)
        features = features.view(batch_size, seq_len, feature_dim)

        assert features.dim() == 3, (
            f"CNN features must be 3D [batch_size, seq_len, feature_dim], got {features.dim()}D"
        )
        assert features.size(0) == batch_size and features.size(1) == seq_len, (
            f"CNN features shape mismatch: {features.shape} vs expected [{batch_size}, {seq_len}, {feature_dim}]"
        )

        if self.use_label_guidance_for_temporal:
            features_2d = features.view(-1, feature_dim)
            frame_logits = self.frame_head(features_2d)
            label_probs = F.softmax(frame_logits, dim=-1)
            label_probs = label_probs.view(batch_size, seq_len, self.num_classes)
            features = torch.cat([features, label_probs], dim=-1)

        if self.enable_inter_stage_transfer and prev_stage_features is not None:
            transformed_prev_features = self.inter_stage_transform[stage_idx - 1](prev_stage_features)
            expanded_prev_features = transformed_prev_features.unsqueeze(1).expand(-1, seq_len, -1)
            features = torch.cat([features, expanded_prev_features], dim=-1)

        initial_hidden = None
        if self.enable_hidden_state_transfer and prev_hidden_state is not None and stage_idx > 0:
            if self.temporal_aggregation == "gru":
                transformed_hidden = self.hidden_state_transform[stage_idx - 1](prev_hidden_state.view(batch_size, -1))
                initial_hidden = transformed_hidden.view(batch_size, 2, -1).transpose(0, 1).contiguous()

            elif self.temporal_aggregation == "lstm":
                prev_h, prev_c = prev_hidden_state
                transformed_h = self.hidden_state_transform[stage_idx - 1](prev_h.view(batch_size, -1))
                transformed_h = transformed_h.view(batch_size, 2, -1).transpose(0, 1).contiguous()
                transformed_c = self.cell_state_transform[stage_idx - 1](prev_c.view(batch_size, -1))
                transformed_c = transformed_c.view(batch_size, 2, -1).transpose(0, 1).contiguous()
                initial_hidden = (transformed_h, transformed_c)

        if self.temporal_aggregation == "gru":
            if initial_hidden is not None:
                gru_output, final_hidden = self.stage_grus[stage_idx](features, initial_hidden)
            else:
                gru_output, final_hidden = self.stage_grus[stage_idx](features)
            temporal_features = gru_output[:, -1, :]
            hidden_state = final_hidden.transpose(0, 1).contiguous().view(batch_size, -1)

        elif self.temporal_aggregation == "lstm":
            if initial_hidden is not None:
                lstm_output, (final_h, final_c) = self.stage_lstms[stage_idx](features, initial_hidden)
            else:
                lstm_output, (final_h, final_c) = self.stage_lstms[stage_idx](features)
            temporal_features = lstm_output[:, -1, :]
            final_h = final_h.transpose(0, 1).contiguous().view(batch_size, -1)
            final_c = final_c.transpose(0, 1).contiguous().view(batch_size, -1)
            hidden_state = (final_h, final_c)

        elif self.temporal_aggregation in ["cfc", "ltc"]:
            seq_len = features.size(1)
            elapsed_dt = torch.full((batch_size, seq_len, 1), self.frame_dt, device=features.device)
            cumulative_time = torch.arange(seq_len, device=features.device).float() * self.frame_dt
            cumulative_time = cumulative_time.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)

            assert cumulative_time.shape == (batch_size, seq_len, 1), (
                f"Cumulative time shape mismatch: {cumulative_time.shape} vs expected [{batch_size}, {seq_len}, 1]"
            )

            features_with_time = torch.cat([features, cumulative_time], dim=-1)

            expected_time_feature_dim = features.size(-1) + 1
            assert features_with_time.size(-1) == expected_time_feature_dim, (
                f"Features with time shape mismatch: {features_with_time.size(-1)} vs expected {expected_time_feature_dim}"
            )

            timespans = elapsed_dt.squeeze(-1)

            assert timespans.shape == (batch_size, seq_len), (
                f"Timespans shape mismatch: {timespans.shape} vs expected [{batch_size}, {seq_len}]"
            )

            with torch.cuda.amp.autocast(enabled=False):
                cfc_output = self.stage_cfcs[stage_idx](features_with_time.float(), timespans=timespans.float())

            if self.cfc_return_sequences:
                assert cfc_output.dim() == 3, (
                    f"CFC output (return_sequences=True) must be 3D, got {cfc_output.dim()}D"
                )
                assert cfc_output.size(0) == batch_size and cfc_output.size(1) == seq_len, (
                    f"CFC output shape mismatch: {cfc_output.shape[:2]} vs expected [{batch_size}, {seq_len}]"
                )
                temporal_features = cfc_output[:, -1, :]
            else:
                assert cfc_output.dim() == 2, (
                    f"CFC output (return_sequences=False) must be 2D, got {cfc_output.dim()}D"
                )
                assert cfc_output.size(0) == batch_size, (
                    f"CFC output batch size mismatch: {cfc_output.size(0)} vs expected {batch_size}"
                )
                temporal_features = cfc_output

            hidden_state = None

        elif self.temporal_aggregation == "mean":
            temporal_features = torch.mean(features, dim=1)
            hidden_state = None
        elif self.temporal_aggregation == "max":
            temporal_features, _ = torch.max(features, dim=1)
            hidden_state = None
        elif self.temporal_aggregation == "last":
            temporal_features = features[:, -1, :]
            hidden_state = None
        else:
            temporal_features = torch.mean(features, dim=1)
            hidden_state = None

        if torch.isnan(temporal_features).any() or torch.isinf(temporal_features).any():
            print("警告: temporal_features包含NaN或Inf，执行nan_to_num清理")
            temporal_features = torch.nan_to_num(temporal_features, nan=0.0, posinf=1e6, neginf=-1e6)
        assert temporal_features.dim() == 2, (
            f"temporal_features must be 2D [batch_size, feature_dim], got {temporal_features.dim()}D"
        )
        assert temporal_features.size(0) == batch_size, (
            f"temporal_features batch size mismatch: {temporal_features.size(0)} vs expected {batch_size}"
        )
        assert not torch.isnan(temporal_features).any(), "temporal_features contains NaN values"
        assert not torch.isinf(temporal_features).any(), "temporal_features contains Inf values"

        if hidden_state is not None:
            if isinstance(hidden_state, tuple):
                assert len(hidden_state) == 2, (
                    f"LSTM hidden_state must be tuple of length 2, got {len(hidden_state)}"
                )
                for i, h in enumerate(hidden_state):
                    assert h.dim() == 2, f"LSTM hidden_state[{i}] must be 2D, got {h.dim()}D"
                    assert h.size(0) == batch_size, (
                        f"LSTM hidden_state[{i}] batch size mismatch: {h.size(0)} vs expected {batch_size}"
                    )
            else:
                assert hidden_state.dim() == 2, f"GRU hidden_state must be 2D, got {hidden_state.dim()}D"
                assert hidden_state.size(0) == batch_size, (
                    f"GRU hidden_state batch size mismatch: {hidden_state.size(0)} vs expected {batch_size}"
                )

        return temporal_features, hidden_state

    def forward(self, x):
        stage_frames_list = list(x)
        num_stages = len(stage_frames_list)

        stage_features = []
        stage_hidden_states = []

        if self.enable_inter_stage_transfer or self.enable_hidden_state_transfer:
            prev_features = None
            prev_hidden = None

            for stage_idx in range(num_stages):
                stage_frames = stage_frames_list[stage_idx]
                current_features, current_hidden = self._extract_stage_features(
                    stage_frames, stage_idx, prev_features, prev_hidden
                )

                stage_features.append(current_features)
                stage_hidden_states.append(current_hidden)

                prev_features = current_features if self.enable_inter_stage_transfer else None
                prev_hidden = current_hidden if self.enable_hidden_state_transfer else None
        else:
            for stage_idx in range(num_stages):
                stage_frames = stage_frames_list[stage_idx]
                current_features, _ = self._extract_stage_features(stage_frames, stage_idx)
                stage_features.append(current_features)

        cross_stage_features = None
        if self.enable_cross_stage_modeling:
            stage_sequence = torch.stack(stage_features, dim=1)
            cross_stage_output, _ = self.cross_stage_lstm(stage_sequence)
            cross_stage_features = cross_stage_output[:, -1, :]

        continuous_features = None
        if self.enable_continuous_dynamics:
            concatenated_for_dynamics = torch.cat(stage_features, dim=1)
            continuous_features = self.continuous_dynamics(concatenated_for_dynamics)

        attention_features, attention_weights, attention_entropy = self.temporal_attention(stage_features)

        fusion_input = torch.cat(stage_features, dim=1)

        if cross_stage_features is not None:
            cross_stage_projected = self.cross_stage_projection(cross_stage_features)
            fusion_input = torch.cat([fusion_input, cross_stage_projected], dim=1)

        if continuous_features is not None:
            continuous_projected = self.continuous_projection(continuous_features)
            fusion_input = torch.cat([fusion_input, continuous_projected], dim=1)

        fusion_features = self.dynamic_fusion(fusion_input)

        final_features = torch.cat([attention_features, fusion_features], dim=1)
        output = self.classifier(final_features)

        return output, attention_weights, attention_entropy


__all__ = [
    "ECAModule",
    "BasicBlockWithECA",
    "ResNet18WithECA",
    "TemporalAttention",
    "MultiStageVideoClassifier",
]

