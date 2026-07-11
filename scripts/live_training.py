from __future__ import annotations

"""Live Matplotlib dashboard for a Modal router training run.

Local run:
    python scripts/live_training.py --progress /tmp/progress.jsonl

Remote Modal volume run:
    python scripts/live_training.py --run-name router-h100-... \
        --modal-volume routeur-artifacts
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def _sync_modal(volume: str, run_name: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    remote = f"/runs/{run_name}/progress.jsonl"
    subprocess.run(
        ["modal", "volume", "get", volume, remote, str(destination), "--force"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Live Matplotlib dashboard for router training.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--progress", type=Path, help="Local progress.jsonl path.")
    source.add_argument("--run-name", help="Modal run name under /runs/<run-name>.")
    parser.add_argument("--modal-volume", default="routeur-artifacts")
    parser.add_argument("--compare-run", action="append", default=[], help="Optional baseline run to overlay.")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/routeur-live"))
    parser.add_argument("--refresh", type=float, default=5.0, help="Refresh period in seconds.")
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    progress_path = args.progress or args.cache_dir / f"{args.run_name}.progress.jsonl"
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    figure.canvas.manager.set_window_title("Routeur — live training")

    def update(_frame: int) -> None:
        if args.run_name:
            _sync_modal(args.modal_volume, args.run_name, progress_path)
        events = _read_events(progress_path)
        epochs = [event for event in events if event.get("status") == "epoch"]
        completed = next((event for event in reversed(events) if event.get("status") == "completed"), None)
        latest = completed or (events[-1] if events else {})
        x = [int(event.get("epoch", index + 1)) for index, event in enumerate(epochs)]
        batches = [event for event in events if event.get("status") == "batch"]
        batch_x = [
            float(event.get("epoch", 1)) - 1.0
            + float(event.get("batch", 0)) / max(1.0, float(event.get("batches", 1)))
            for event in batches
        ]

        for axis in axes.flat:
            axis.clear()
            axis.grid(alpha=0.25)

        def legend_if_needed(axis: Any, **kwargs: Any) -> None:
            handles, labels = axis.get_legend_handles_labels()
            if handles and labels:
                axis.legend(**kwargs)

        def plot_metric(axis: Any, key: str, label: str, *, business: bool = False) -> None:
            values: list[float | None] = []
            for event in epochs:
                source_value = event.get("business", {}) if business else event
                values.append(_number(source_value.get(key)))
            valid = [(epoch, value) for epoch, value in zip(x, values, strict=True) if value is not None]
            if valid:
                axis.plot([item[0] for item in valid], [item[1] for item in valid], marker="o", label=label)

        plot_metric(axes[0, 0], "train_loss", "train")
        plot_metric(axes[0, 0], "val_loss", "validation")
        batch_losses = [_number(event.get("train_loss")) for event in batches]
        valid_batch_losses = [
            (position, value)
            for position, value in zip(batch_x, batch_losses, strict=True)
            if value is not None
        ]
        if valid_batch_losses:
            axes[0, 0].plot(
                [item[0] for item in valid_batch_losses],
                [item[1] for item in valid_batch_losses],
                alpha=0.45,
                linewidth=1.0,
                label="train (batches)",
            )
        axes[0, 0].set_title("Pertes")
        axes[0, 0].set_xlabel("Époque")
        legend_if_needed(axes[0, 0], loc="best")

        for key, label in (
            ("level_macro_f1", "F1 niveau"),
            ("task_accuracy", "task accuracy"),
            ("task_macro_f1", "task F1"),
            ("preference_accuracy", "Arena preferences"),
        ):
            plot_metric(axes[0, 1], key, label)
        for key, label in (("risk_accuracy", "risk accuracy"), ("capability_micro_f1", "capability F1")):
            plot_metric(axes[0, 1], key, label)
        axes[0, 1].set_title("Classification quality")
        axes[0, 1].set_ylim(0, 1)
        legend_if_needed(axes[0, 1], loc="best")

        for key, label in (("underroute_rate", "under-routing"), ("severe_underroute_rate", "severe under-routing"), ("overroute_rate", "over-routing")):
            plot_metric(axes[0, 2], key, label, business=True)
        axes[0, 2].set_title("Risque de routage")
        axes[0, 2].set_ylim(0, 1)
        legend_if_needed(axes[0, 2], loc="best")

        for key, label in (("savings_vs_always_level_5", "savings"), ("cost_ratio_vs_oracle", "cost/oracle ratio")):
            plot_metric(axes[1, 0], key, label, business=True)
        axes[1, 0].set_title("Cost")
        legend_if_needed(axes[1, 0], loc="best")

        plot_metric(axes[1, 1], "learning_rate", "learning rate")
        axes[1, 1].set_title("Learning rate")
        axes[1, 1].set_xlabel("Époque")
        axes[1, 1].set_yscale("log")
        legend_if_needed(axes[1, 1], loc="best")

        for compare_run in args.compare_run:
            compare_path = args.cache_dir / f"{compare_run}.progress.jsonl"
            _sync_modal(args.modal_volume, compare_run, compare_path)
            compare_events = [event for event in _read_events(compare_path) if event.get("status") == "epoch"]
            compare_x = [int(event.get("epoch", index + 1)) for index, event in enumerate(compare_events)]

            def overlay(axis: Any, key: str, *, business: bool = False) -> None:
                values = [
                    _number((event.get("business", {}) if business else event).get(key))
                    for event in compare_events
                ]
                valid = [(epoch, value) for epoch, value in zip(compare_x, values, strict=True) if value is not None]
                if valid:
                    axis.plot(
                        [item[0] for item in valid],
                        [item[1] for item in valid],
                        linestyle="--",
                        alpha=0.75,
                        label=f"{compare_run}: {key}",
                    )

            overlay(axes[0, 0], "val_loss")
            overlay(axes[0, 1], "level_macro_f1")
            overlay(axes[0, 1], "task_accuracy")
            overlay(axes[0, 1], "preference_accuracy")
            overlay(axes[0, 2], "severe_underroute_rate", business=True)
            overlay(axes[1, 0], "savings_vs_always_level_5", business=True)
        if args.compare_run:
            for axis in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]):
                legend_if_needed(axis, loc="best", fontsize=8)

        axes[1, 2].axis("off")
        metric_event = completed or (epochs[-1] if epochs else latest)
        business = metric_event.get("business", {})
        details = metric_event.get("metrics") if isinstance(metric_event.get("metrics"), dict) else metric_event
        lines = [
            f"Statut : {latest.get('status', 'en attente')}",
            f"GPU : {latest.get('gpu', '?')}",
            f"Époque : {latest.get('epoch', 0)} / {latest.get('epochs', '?')}",
            f"Batch : {latest.get('batch', '-')} / {latest.get('batches', '-')}",
            f"Temps : {float(latest.get('elapsed_seconds', 0.0)) / 60:.1f} min",
        ]
        if business:
            weakest_levels = sorted(
                (details.get("level_f1_by_class") or {}).items(), key=lambda item: float(item[1])
            )[:2]
            weakest_tasks = sorted(
                (details.get("task_f1_by_class") or {}).items(), key=lambda item: float(item[1])
            )[:3]
            lines.extend([
                f"F1 niveau : {float(metric_event.get('level_macro_f1', 0.0)):.3f}",
                f"Task accuracy: {float(metric_event.get('task_accuracy', 0.0)):.3f}",
                f"Task F1: {float(metric_event.get('task_macro_f1', 0.0)):.3f}",
                f"Risk accuracy: {float(metric_event.get('risk_accuracy', 0.0)):.3f}",
                f"Capability F1: {float(metric_event.get('capability_micro_f1', 0.0)):.3f}",
                f"Arena preferences: {float(metric_event.get('preference_accuracy', 0.0)):.3f}",
                f"Benchmark externe : {float((details.get('external_router_benchmark') or {}).get('accuracy', metric_event.get('external_router_accuracy', 0.0))):.3f}",
                f"ECE niveau : {float(details.get('level_ece', 0.0)):.3f}",
                f"Throughput: {float(metric_event.get('inference_examples_per_second', 0.0)):.1f} examples/s",
                f"Sous-routage : {float(business.get('underroute_rate', 0.0)):.3f}",
                f"Severe under-routing: {float(business.get('severe_underroute_rate', 0.0)):.3f}",
                f"Sur-routage : {float(business.get('overroute_rate', 0.0)):.3f}",
                f"Économie vs niveau 5 : {float(business.get('savings_vs_always_level_5', 0.0)):.1%}",
                "Niveaux faibles : " + ", ".join(f"{name}={float(score):.2f}" for name, score in weakest_levels),
                "Weakest tasks: " + ", ".join(f"{name}={float(score):.2f}" for name, score in weakest_tasks),
            ])
        axes[1, 2].text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=12)
        figure.suptitle(f"Routeur — {args.run_name or progress_path.name}", fontsize=16)
        figure.tight_layout(rect=[0, 0, 1, 0.95])
        figure.canvas.draw_idle()

    animation = FuncAnimation(figure, update, interval=max(1.0, args.refresh) * 1000, cache_frame_data=False)
    # Keep a strong reference for Matplotlib backends that otherwise garbage
    # collect the animation before the first refresh.
    figure._routeur_animation = animation  # type: ignore[attr-defined]
    update(0)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
