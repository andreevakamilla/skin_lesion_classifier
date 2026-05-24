import matplotlib.pyplot as plt
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:8082")
client = mlflow.tracking.MlflowClient()

runs = client.search_runs(experiment_ids=["1"], order_by=["start_time DESC"])
run_id = runs[0].info.run_id

metrics = ["val/balanced_accuracy", "val/loss", "val/mel_recall"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, metric in zip(axes, metrics):
    history = client.get_metric_history(run_id, metric)
    steps = [m.step for m in history]
    values = [m.value for m in history]
    ax.plot(steps, values)
    ax.set_title(metric)
    ax.set_xlabel("Epoch")
    ax.grid(True)

plt.tight_layout()
plt.savefig("plots/training_metrics.png", dpi=150)
print("Сохранено: plots/training_metrics.png")
