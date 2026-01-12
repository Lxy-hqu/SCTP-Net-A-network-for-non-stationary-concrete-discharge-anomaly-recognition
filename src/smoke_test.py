import argparse

import torch

from config import CONFIG
from model import MultiStageVideoClassifier


def build_model(cfg, num_stages, pretrained):
    mcfg = cfg["model"]
    return MultiStageVideoClassifier(
        num_classes=mcfg["num_classes"],
        hidden_size=mcfg["hidden_size"],
        dropout_rate=mcfg["dropout_rate"],
        temporal_aggregation=mcfg["temporal_aggregation"],
        pretrained=pretrained,
        enable_inter_stage_transfer=mcfg.get("enable_inter_stage_transfer", True),
        enable_cross_stage_modeling=mcfg.get("enable_cross_stage_modeling", True),
        enable_continuous_dynamics=mcfg.get("enable_continuous_dynamics", True),
        enable_hidden_state_transfer=mcfg.get("enable_hidden_state_transfer", True),
        cfc_mode=mcfg.get("cfc_mode", "cfc"),
        cfc_units=mcfg.get("cfc_units", 64),
        cfc_proj_size=mcfg.get("cfc_proj_size", None),
        cfc_return_sequences=mcfg.get("cfc_return_sequences", True),
        cfc_mixed_memory=mcfg.get("cfc_mixed_memory", False),
        cfc_no_gate=mcfg.get("cfc_no_gate", False),
        cfc_minimal=mcfg.get("cfc_minimal", False),
        ltc_ode_unfolds=mcfg.get("ltc_ode_unfolds", 6),
        ltc_epsilon=mcfg.get("ltc_epsilon", 1e-8),
        frame_dt=mcfg.get("frame_dt", 0.04),
        num_stages=num_stages,
        use_label_guidance_for_temporal=mcfg.get("use_label_guidance_for_temporal", False),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-stages", type=int, default=int(CONFIG["data"].get("num_stages", 3)))
    parser.add_argument("--seq-len", type=int, default=int(CONFIG["data"].get("min_frames_per_stage", 8)))
    parser.add_argument("--height", type=int, default=int(CONFIG["data"]["image_size"][0]))
    parser.add_argument("--width", type=int, default=int(CONFIG["data"]["image_size"][1]))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)

    model = build_model(CONFIG, num_stages=args.num_stages, pretrained=args.pretrained).to(device)
    model.eval()

    x = tuple(
        torch.randn(args.batch_size, args.seq_len, 3, args.height, args.width, device=device)
        for _ in range(args.num_stages)
    )

    with torch.no_grad():
        logits, attention_weights, attention_entropy = model(x)

    print("logits.shape:", tuple(logits.shape))
    print("attention_weights.shape:", tuple(attention_weights.shape))
    print("attention_entropy.shape:", tuple(attention_entropy.shape))
    print("attention_weights.sum(dim=1) (first 4):", attention_weights.sum(dim=1)[:4].detach().cpu().tolist())


if __name__ == "__main__":
    main()

