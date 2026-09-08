"""PFMG gesture generator with explicit periodic and non-periodic branches.

The non-periodic branch predicts body and hand VQ codes. The periodic branch
uses predicted phase parameters to blend motion experts. Both branches receive
the same fused conditions and are added in pose space. Hand generation is
conditioned on the combined body prediction, following the body-to-hand
hierarchy described in the paper.
"""

import math
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.nn.utils import weight_norm

from .audio2face import audio2face
from .motion_vqvae import VQVAE
from .utils.build_vocab import Vocab  # noqa: F401 - needed by legacy vocab.pkl files


BODY_DIMS = 27
HAND_DIMS = 114
POSE_DIMS = BODY_DIMS + HAND_DIMS


class _LegacyVocabUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "Vocab":
            return Vocab
        return super().find_class(module, name)


def _project_path(root_path, path):
    """Resolve a config path relative to the repository root."""
    return Path(str(root_path)) / str(path).lstrip("/")


def _load_checkpoint(model, checkpoint_path, label):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing {label} checkpoint: {checkpoint_path}. "
            "Download the pretrained weights before constructing PFMG."
        )

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("model_state", checkpoint)
    state_dict = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    logger.info(f"Loaded pretrained {label} checkpoint from {checkpoint_path}")


def _freeze(module):
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, inputs):
        return inputs[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        dropout,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.network = nn.Sequential(
            weight_norm(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU()
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.network:
            if isinstance(module, nn.Conv1d):
                module.weight.data.normal_(0, 0.01)
        if isinstance(self.residual, nn.Conv1d):
            self.residual.weight.data.normal_(0, 0.01)

    def forward(self, inputs):
        return self.activation(self.network(inputs) + self.residual(inputs))


class TemporalConvNet(nn.Module):
    def __init__(self, in_channels, channels, kernel_size=2, dropout=0.2):
        super().__init__()
        blocks = []
        for level, out_channels in enumerate(channels):
            block_in = in_channels if level == 0 else channels[level - 1]
            blocks.append(
                TemporalBlock(
                    block_in,
                    out_channels,
                    kernel_size,
                    dilation=2**level,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*blocks)

    def forward(self, inputs):
        return self.network(inputs)


class TextEncoderTCN(nn.Module):
    def __init__(
        self,
        args,
        n_words,
        embedding_size=300,
        pretrained_embedding=None,
        kernel_size=2,
        dropout=0.3,
        embedding_dropout=0.1,
    ):
        super().__init__()
        if pretrained_embedding is not None:
            if pretrained_embedding.shape != (n_words, embedding_size):
                raise ValueError(
                    "Unexpected word embedding shape: "
                    f"{pretrained_embedding.shape}, expected {(n_words, embedding_size)}"
                )
            self.embedding = nn.Embedding.from_pretrained(
                torch.as_tensor(pretrained_embedding, dtype=torch.float32),
                freeze=args.freeze_wordembed,
            )
        else:
            self.embedding = nn.Embedding(n_words, embedding_size)

        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.tcn = TemporalConvNet(
            embedding_size,
            [args.hidden_size] * args.n_layer,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.output = nn.Linear(args.hidden_size, args.word_f)
        nn.init.zeros_(self.output.bias)
        nn.init.normal_(self.output.weight, 0, 0.01)

    def forward(self, token_ids):
        embedded = self.embedding_dropout(self.embedding(token_ids))
        encoded = self.tcn(embedded.transpose(1, 2)).transpose(1, 2)
        return self.output(encoded).contiguous(), None


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        first_padding,
        downsample=False,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=first_padding,
            bias=True,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.LeakyReLU(inplace=True)
        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=first_padding,
                    bias=True,
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, inputs):
        residual = inputs if self.downsample is None else self.downsample(inputs)
        hidden = self.act1(self.bn1(self.conv1(inputs)))
        hidden = self.bn2(self.conv2(hidden))
        return self.act2(hidden + residual)


class ExpertLinear(nn.Module):
    """A linear layer whose expert outputs are blended per frame."""

    def __init__(self, experts, in_features, out_features):
        super().__init__()
        self.experts = experts
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(experts, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(experts, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        for expert_weight in self.weight:
            nn.init.kaiming_uniform_(expert_weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs, expert_weights):
        if inputs.shape[:-1] != expert_weights.shape[:-1]:
            raise ValueError(
                "Expert input and gating weights must share batch/time dimensions: "
                f"{inputs.shape} vs. {expert_weights.shape}"
            )
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} expert features, got {inputs.shape[-1]}"
            )
        if expert_weights.shape[-1] != self.experts:
            raise ValueError(
                f"Expected {self.experts} expert weights, got {expert_weights.shape[-1]}"
            )

        outputs = torch.einsum("bti,eio->bteo", inputs, self.weight)
        outputs = outputs + self.bias.view(1, 1, self.experts, self.out_features)
        return torch.sum(outputs * expert_weights.unsqueeze(-1), dim=2)


class PhaseConditionedMoE(nn.Module):
    """Weight-blended experts gated by phase shifts, not by the time axis."""

    def __init__(
        self,
        feature_dim,
        phase_dim,
        output_dim,
        experts=5,
        gating_hidden=64,
        expert_hidden=1024,
        dropout=0.3,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.phase_dim = phase_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.gating = nn.Sequential(
            nn.Linear(phase_dim, gating_hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(gating_hidden, gating_hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(gating_hidden, experts),
        )
        self.expert1 = ExpertLinear(experts, feature_dim, expert_hidden)
        self.expert2 = ExpertLinear(experts, expert_hidden, expert_hidden)
        self.expert3 = ExpertLinear(experts, expert_hidden, output_dim)

    def forward(self, features, phase_shift):
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected periodic feature dim {self.feature_dim}, "
                f"got {features.shape[-1]}"
            )
        if phase_shift.shape[-1] != self.phase_dim:
            raise ValueError(
                f"Expected phase dim {self.phase_dim}, got {phase_shift.shape[-1]}"
            )
        if features.shape[:2] != phase_shift.shape[:2]:
            raise ValueError(
                "Periodic features and phase shifts must share batch/time dimensions"
            )

        expert_weights = F.softmax(self.gating(phase_shift), dim=-1)
        hidden = F.dropout(features, self.dropout, training=self.training)
        hidden = F.elu(self.expert1(hidden, expert_weights))
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        hidden = F.elu(self.expert2(hidden, expert_weights))
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        return self.expert3(hidden, expert_weights), expert_weights


class PfMG_VQ_PAE(nn.Module):
    """Hierarchical PFMG variant with VQ non-periodic generators."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pose_dims = int(args.pose_dims)
        self.pose_length = int(args.pose_length)
        self.seed_dim = self.pose_dims + 1
        self.audio_f = int(args.audio_f)
        self.facial_f = int(args.facial_f)
        self.word_f = int(args.word_f)
        self.speaker_f = int(args.speaker_f)
        self.emotion_f = int(args.emotion_f)
        self.hidden_size = int(args.hidden_size)
        self.n_layer = int(args.n_layer)

        phase_channels = float(args.embedding_channels)
        if not phase_channels.is_integer() or phase_channels <= 0:
            raise ValueError("embedding_channels must be a positive integer")
        self.phase_channels = int(phase_channels)
        self.phase_parameter_dim = 4 * self.phase_channels
        self.periodic_descriptor_dim = 3 * self.phase_channels

        if self.pose_dims != POSE_DIMS:
            raise ValueError(
                f"PFMG uses 27 body + 114 hand dimensions; got pose_dims={self.pose_dims}"
            )
        self.latent_steps = self.pose_length // 4
        if self.latent_steps * 4 + 2 != self.pose_length:
            raise ValueError(
                "The released VQ decoder requires pose_length = 4 * latent_steps + 2; "
                f"got pose_length={self.pose_length}"
            )

        self.audio2face = audio2face(args)
        audio_encoder = self.audio2face.audio_encoder
        content_projection = getattr(audio_encoder, "audio_feature_map_cont", None)
        emotion_projection = getattr(audio_encoder, "audio_feature_map_emo2", None)
        if content_projection is None or emotion_projection is None:
            raise ValueError("Unsupported Audio2Face encoder: output dimensions are unavailable")
        self.raw_audio_dim = (
            content_projection.out_features + emotion_projection.out_features
        )
        if self.audio_f != self.raw_audio_dim:
            raise ValueError(
                "audio_f must match the concatenated Audio2Face content/emotion features: "
                f"expected {self.raw_audio_dim}, got {self.audio_f}"
            )
        if self.emotion_f != 8:
            raise ValueError(
                "The released Audio2Face decoder expects emotion_f=8; "
                f"got {self.emotion_f}"
            )

        self.facial_encoder = nn.Sequential(
            BasicBlock(args.facial_dims, self.facial_f // 2, 7, 3, downsample=True),
            BasicBlock(self.facial_f // 2, self.facial_f // 2, 3, 1, downsample=True),
            BasicBlock(self.facial_f // 2, self.facial_f // 2, 3, 1),
            BasicBlock(self.facial_f // 2, self.facial_f, 3, 1, downsample=True),
        )

        self.text_encoder = None
        if self.word_f:
            train_dir = _project_path(args.root_path, args.train_data_path)
            vocab_path = train_dir.parent / "vocab.pkl"
            if not vocab_path.is_file():
                raise FileNotFoundError(f"Missing vocabulary: {vocab_path}")
            with vocab_path.open("rb") as vocab_file:
                language_model = _LegacyVocabUnpickler(vocab_file).load()
            self.text_encoder = TextEncoderTCN(
                args,
                args.word_index_num,
                args.word_dims,
                pretrained_embedding=language_model.word_embedding_weights,
                dropout=args.dropout_prob,
            )

        self.speaker_embedding = None
        if self.speaker_f:
            self.speaker_embedding = nn.Sequential(
                nn.Embedding(args.speaker_dims, self.speaker_f),
                nn.Linear(self.speaker_f, self.speaker_f),
                nn.LeakyReLU(inplace=True),
            )

        self.emotion_embedding = nn.Sequential(
            nn.Embedding(args.emotion_dims, self.emotion_f),
            nn.Linear(self.emotion_f, self.emotion_f),
        )
        self.emotion_embedding_tail = nn.Sequential(
            nn.Conv1d(self.emotion_f, 8, 9, padding=4),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(8, 16, 9, padding=4),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(16, 16, 9, padding=4),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(16, self.emotion_f, 9, padding=4),
            nn.BatchNorm1d(self.emotion_f),
            nn.LeakyReLU(0.3, inplace=True),
        )

        self.audio_fusion_input_dim = (
            self.raw_audio_dim + self.word_f + self.emotion_f + self.speaker_f
        )
        self.audio_fusion = nn.Sequential(
            nn.Linear(self.audio_fusion_input_dim, self.hidden_size // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_size // 2, self.audio_f),
            nn.LeakyReLU(inplace=True),
        )
        self.facial_fusion_input_dim = (
            self.facial_f
            + self.audio_f
            + self.word_f
            + self.emotion_f
            + self.speaker_f
        )
        self.facial_fusion = nn.Sequential(
            nn.Linear(self.facial_fusion_input_dim, self.hidden_size // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_size // 2, self.facial_f),
            nn.LeakyReLU(inplace=True),
        )

        self.condition_dim = (
            self.speaker_f
            + self.emotion_f
            + self.word_f
            + self.audio_f
            + self.facial_f
        )
        self.feature_fusion_lstm = nn.LSTM(
            self.condition_dim,
            self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.feature_fusion_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_size, self.condition_dim),
        )
        self.generator_context_dim = self.seed_dim + self.condition_dim

        self.body_vqvae = VQVAE(args, BODY_DIMS)
        self.hand_vqvae = VQVAE(args, HAND_DIMS)

        lstm_dropout = args.dropout_prob if self.n_layer > 1 else 0.0
        self.body_lstm = nn.LSTM(
            self.generator_context_dim,
            self.hidden_size,
            num_layers=self.n_layer,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.body_code_head = nn.Linear(
            self.hidden_size, self.body_vqvae.num_embeddings
        )

        self.hand_context_dim = self.generator_context_dim + BODY_DIMS
        self.hand_lstm = nn.LSTM(
            self.hand_context_dim,
            self.hidden_size,
            num_layers=self.n_layer,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.hand_code_head = nn.Linear(
            self.hidden_size, self.hand_vqvae.num_embeddings
        )

        self.phase_lstm = nn.LSTM(
            self.generator_context_dim,
            self.hidden_size,
            num_layers=self.n_layer,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.phase_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_size // 2, self.phase_parameter_dim),
        )

        moe_experts = int(getattr(args, "moe_experts", 5))
        moe_hidden = int(getattr(args, "moe_hidden_size", 1024))
        moe_dropout = float(getattr(args, "moe_dropout", 0.3))
        self.body_periodic = PhaseConditionedMoE(
            self.generator_context_dim + self.periodic_descriptor_dim,
            self.phase_channels,
            BODY_DIMS,
            experts=moe_experts,
            expert_hidden=moe_hidden,
            dropout=moe_dropout,
        )
        self.hand_periodic = PhaseConditionedMoE(
            self.hand_context_dim + self.periodic_descriptor_dim,
            self.phase_channels,
            HAND_DIMS,
            experts=moe_experts,
            expert_hidden=moe_hidden,
            dropout=moe_dropout,
        )

        self.vq_temperature = float(getattr(args, "vq_temperature", 1.0))
        if self.vq_temperature <= 0:
            raise ValueError("vq_temperature must be positive")

        if getattr(args, "load_pretrained", True):
            weights_dir = _project_path(args.root_path, args.train_data_path).parent / "weights"
            _load_checkpoint(self.audio2face, weights_dir / "face.bin", "face")
            _load_checkpoint(self.body_vqvae, weights_dir / "b_vqvae.bin", "body VQ-VAE")
            _load_checkpoint(self.hand_vqvae, weights_dir / "h_vqvae.bin", "hand VQ-VAE")

        _freeze(self.audio2face)
        _freeze(self.body_vqvae)
        _freeze(self.hand_vqvae)

    def train(self, mode=True):
        super().train(mode)
        self.audio2face.eval()
        self.body_vqvae.eval()
        self.hand_vqvae.eval()
        return self

    @staticmethod
    def _align_time(features, time_steps, name):
        if features.ndim != 3:
            raise ValueError(f"{name} must have shape [B, T, C], got {features.shape}")
        if features.shape[1] == time_steps:
            return features
        if features.shape[1] <= 0:
            raise ValueError(f"{name} has an empty time dimension")
        return F.interpolate(
            features.transpose(1, 2),
            size=time_steps,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    @staticmethod
    def _normalise_labels(labels, batch_size, time_steps, name, constant=False):
        if labels is None:
            raise ValueError(f"{name} is required by the current configuration")
        while labels.ndim > 2 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)
        if labels.ndim == 1:
            if labels.shape[0] == batch_size:
                labels = labels.unsqueeze(1)
            elif batch_size == 1:
                labels = labels.unsqueeze(0)
        if labels.ndim != 2 or labels.shape[0] != batch_size:
            raise ValueError(
                f"{name} must have shape [B], [B, 1], or [B, T], got {labels.shape}"
            )
        labels = labels.long()
        if constant:
            return labels[:, :1]
        if labels.shape[1] == time_steps:
            return labels
        if labels.shape[1] == 1:
            return labels.expand(-1, time_steps)
        indices = torch.linspace(
            0,
            labels.shape[1] - 1,
            time_steps,
            device=labels.device,
        ).round().long()
        return labels.index_select(1, indices)

    @staticmethod
    def _sum_bidirectional(output, hidden_size):
        if output.shape[-1] != 2 * hidden_size:
            raise RuntimeError(
                f"Expected bidirectional LSTM dim {2 * hidden_size}, got {output.shape[-1]}"
            )
        return output[..., :hidden_size] + output[..., hidden_size:]

    def _encode_conditions(self, pre_seq, in_audio, in_text, in_id, in_emo):
        batch_size, time_steps, _ = pre_seq.shape
        if in_audio is None or in_audio.ndim != 2 or in_audio.shape[0] != batch_size:
            shape = None if in_audio is None else tuple(in_audio.shape)
            raise ValueError(f"in_audio must have shape [B, samples], got {shape}")

        speaker_labels = None
        if self.speaker_f:
            speaker_labels = self._normalise_labels(
                in_id, batch_size, time_steps, "in_id", constant=True
            )
        emotion_labels = self._normalise_labels(
            in_emo, batch_size, time_steps, "in_emo"
        )
        token_ids = None
        if self.word_f:
            token_ids = self._normalise_labels(
                in_text, batch_size, time_steps, "in_text"
            )

        audio_encoder = self.audio2face.audio_encoder
        if hasattr(audio_encoder, "device"):
            audio_encoder.device = in_audio.device
        with torch.no_grad():
            predicted_face, content_audio, emotion_audio = self.audio2face(
                in_audio=in_audio,
                in_text=token_ids,
                in_id=speaker_labels,
                in_emo=emotion_labels,
            )

        predicted_face = self._align_time(predicted_face, time_steps, "predicted face")
        content_audio = self._align_time(content_audio, time_steps, "content audio")
        emotion_audio = self._align_time(emotion_audio, time_steps, "emotion audio")
        if predicted_face.shape[-1] != self.args.facial_dims:
            raise RuntimeError(
                f"Audio2Face produced {predicted_face.shape[-1]} face dims; "
                f"expected {self.args.facial_dims}"
            )

        raw_audio = torch.cat((content_audio, emotion_audio), dim=-1)
        if raw_audio.shape[-1] != self.raw_audio_dim:
            raise RuntimeError(
                f"Audio2Face produced {raw_audio.shape[-1]} audio features; "
                f"expected {self.raw_audio_dim}"
            )

        speaker_features = None
        if self.speaker_embedding is not None:
            speaker_features = self.speaker_embedding(speaker_labels[:, 0])
            speaker_features = speaker_features.unsqueeze(1).expand(-1, time_steps, -1)

        emotion_features = self.emotion_embedding(emotion_labels)
        emotion_features = self.emotion_embedding_tail(
            emotion_features.transpose(1, 2)
        ).transpose(1, 2)

        text_features = None
        if self.text_encoder is not None:
            text_features, _ = self.text_encoder(token_ids)
            text_features = self._align_time(text_features, time_steps, "text features")

        audio_parts = [raw_audio]
        if text_features is not None:
            audio_parts.append(text_features)
        audio_parts.append(emotion_features)
        if speaker_features is not None:
            audio_parts.append(speaker_features)
        audio_fusion_input = torch.cat(audio_parts, dim=-1)
        if audio_fusion_input.shape[-1] != self.audio_fusion_input_dim:
            raise RuntimeError(
                f"Audio fusion dim mismatch: expected {self.audio_fusion_input_dim}, "
                f"got {audio_fusion_input.shape[-1]}"
            )
        audio_features = self.audio_fusion(audio_fusion_input)

        face_features = self.facial_encoder(predicted_face.transpose(1, 2)).transpose(1, 2)
        face_features = self._align_time(face_features, time_steps, "face features")
        face_parts = [face_features, audio_features]
        if text_features is not None:
            face_parts.append(text_features)
        face_parts.append(emotion_features)
        if speaker_features is not None:
            face_parts.append(speaker_features)
        facial_fusion_input = torch.cat(face_parts, dim=-1)
        if facial_fusion_input.shape[-1] != self.facial_fusion_input_dim:
            raise RuntimeError(
                f"Face fusion dim mismatch: expected {self.facial_fusion_input_dim}, "
                f"got {facial_fusion_input.shape[-1]}"
            )
        face_features = self.facial_fusion(facial_fusion_input)

        condition_parts = []
        if speaker_features is not None:
            condition_parts.append(speaker_features)
        condition_parts.append(emotion_features)
        if text_features is not None:
            condition_parts.append(text_features)
        condition_parts.extend((audio_features, face_features))
        conditions = torch.cat(condition_parts, dim=-1)
        if conditions.shape[-1] != self.condition_dim:
            raise RuntimeError(
                f"Condition dim mismatch: expected {self.condition_dim}, "
                f"got {conditions.shape[-1]}"
            )

        fused_conditions, _ = self.feature_fusion_lstm(conditions)
        fused_conditions = self.feature_fusion_head(fused_conditions)
        return fused_conditions, speaker_features

    def _decode_vq(self, context, recurrent, code_head, vqvae, name):
        hidden, _ = recurrent(context)
        hidden = self._sum_bidirectional(hidden, self.hidden_size)
        frame_logits = code_head(hidden)
        code_logits = F.adaptive_avg_pool1d(
            frame_logits.transpose(1, 2), self.latent_steps
        ).transpose(1, 2)

        if self.training:
            probabilities = F.softmax(code_logits / self.vq_temperature, dim=-1)
            indices = probabilities.argmax(dim=-1)
            hard_codes = F.one_hot(
                indices, num_classes=vqvae.num_embeddings
            ).to(probabilities.dtype)
            assignments = hard_codes + probabilities - probabilities.detach()
            quantized = torch.matmul(assignments, vqvae.vq_layer.embeddings)
        else:
            indices = code_logits.argmax(dim=-1)
            quantized = F.embedding(indices, vqvae.vq_layer.embeddings)

        decoded, _ = vqvae.decoder(quantized.transpose(1, 2).contiguous())
        decoded = decoded.transpose(1, 2).contiguous()
        expected_shape = (context.shape[0], context.shape[1], vqvae.in_dim)
        if tuple(decoded.shape) != expected_shape:
            raise RuntimeError(
                f"{name} VQ decoder produced {tuple(decoded.shape)}, "
                f"expected {expected_shape}"
            )
        return decoded

    @staticmethod
    def _merge_body_and_hands(body, hands):
        if body.shape[-1] != BODY_DIMS or hands.shape[-1] != HAND_DIMS:
            raise ValueError(
                f"Expected body/hand dims {BODY_DIMS}/{HAND_DIMS}, "
                f"got {body.shape[-1]}/{hands.shape[-1]}"
            )
        return torch.cat(
            (
                body[..., :18],
                hands[..., :57],
                body[..., 18:],
                hands[..., 57:],
            ),
            dim=-1,
        )

    def forward(
        self,
        pre_seq,
        in_audio=None,
        in_facial=None,
        in_pae=None,
        in_text=None,
        in_id=None,
        in_emo=None,
        in_pose=None,
    ):
        del in_facial, in_pose  # Kept in the signature for trainer compatibility.
        expected_seed_shape = (self.pose_length, self.seed_dim)
        if pre_seq.ndim != 3 or tuple(pre_seq.shape[1:]) != expected_seed_shape:
            raise ValueError(
                "pre_seq must have shape "
                f"[B, {self.pose_length}, {self.seed_dim}], got {tuple(pre_seq.shape)}"
            )
        if in_pae is not None:
            expected_pae_shape = (
                pre_seq.shape[0],
                self.pose_length,
                self.phase_parameter_dim,
            )
            if tuple(in_pae.shape) != expected_pae_shape:
                raise ValueError(
                    f"in_pae must have shape {expected_pae_shape}, got {tuple(in_pae.shape)}"
                )

        for recurrent in (
            self.feature_fusion_lstm,
            self.body_lstm,
            self.hand_lstm,
            self.phase_lstm,
        ):
            recurrent.flatten_parameters()

        fused_conditions, speaker_features = self._encode_conditions(
            pre_seq, in_audio, in_text, in_id, in_emo
        )
        generator_context = torch.cat((pre_seq, fused_conditions), dim=-1)
        if generator_context.shape[-1] != self.generator_context_dim:
            raise RuntimeError(
                f"Generator context dim mismatch: expected {self.generator_context_dim}, "
                f"got {generator_context.shape[-1]}"
            )

        phase_hidden, _ = self.phase_lstm(generator_context)
        phase_hidden = self._sum_bidirectional(phase_hidden, self.hidden_size)
        predicted_pae = self.phase_head(phase_hidden)
        phase_shift, frequency, amplitude, offset = predicted_pae.split(
            self.phase_channels, dim=-1
        )
        periodic_descriptors = torch.cat((frequency, amplitude, offset), dim=-1)

        # Body branches are parallel: neither branch consumes the other's output.
        body_non_periodic = self._decode_vq(
            generator_context,
            self.body_lstm,
            self.body_code_head,
            self.body_vqvae,
            "body",
        )
        body_periodic, _ = self.body_periodic(
            torch.cat((generator_context, periodic_descriptors), dim=-1),
            phase_shift,
        )
        body = body_non_periodic + body_periodic

        # Both hand branches share the same body-conditioned context.
        hand_context = torch.cat((generator_context, body), dim=-1)
        hand_non_periodic = self._decode_vq(
            hand_context,
            self.hand_lstm,
            self.hand_code_head,
            self.hand_vqvae,
            "hand",
        )
        hand_periodic, _ = self.hand_periodic(
            torch.cat((hand_context, periodic_descriptors), dim=-1),
            phase_shift,
        )
        hands = hand_non_periodic + hand_periodic

        motion = self._merge_body_and_hands(body, hands)
        if motion.shape[-1] != self.pose_dims:
            raise RuntimeError(
                f"Final pose dim mismatch: expected {self.pose_dims}, got {motion.shape[-1]}"
            )
        return motion, predicted_pae, speaker_features


class ConvDiscriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_size = 64
        self.pre_conv = nn.Sequential(
            nn.Conv1d(args.pose_dims, 16, 3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(16, 8, 3),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(8, 8, 3),
        )
        self.recurrent = nn.LSTM(
            8,
            hidden_size=self.hidden_size,
            num_layers=4,
            bidirectional=True,
            dropout=0.3,
            batch_first=True,
        )
        self.frame_output = nn.Linear(self.hidden_size, 1)
        discriminator_steps = int(args.pose_length) - 6
        if discriminator_steps <= 0:
            raise ValueError("pose_length must be greater than 6")
        self.sequence_output = nn.Linear(discriminator_steps, 1)

    def forward(self, poses):
        self.recurrent.flatten_parameters()
        features = self.pre_conv(poses.transpose(1, 2)).transpose(1, 2)
        output, _ = self.recurrent(features)
        output = output[..., : self.hidden_size] + output[..., self.hidden_size :]
        output = self.frame_output(output).squeeze(-1)
        return torch.sigmoid(self.sequence_output(output))
