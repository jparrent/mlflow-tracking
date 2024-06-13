import matplotlib.pyplot as plt
from pathlib import Path

class Plotting:
    @staticmethod
    def plot_curve(results, model_run_name, plots_directory: Path, metric: str):
        """
        Plot the specified metric curve.
        """
        epochs = len(results['validation_0'][metric])
        x_axis = range(0, epochs)

        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0'][metric], label='Test', color='orange')
        ax.plot(x_axis, results['validation_1'][metric], label='Train', color='blue')
        ax.legend()
        plt.ylabel(metric)
        plt.xlabel('epochs')
        plt.savefig(plots_directory / f"{model_run_name}_{metric}_curve.png")
