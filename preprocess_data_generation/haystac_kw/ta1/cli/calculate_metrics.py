import click
from haystac_kw.ta1.eval import Evaluator, MetricsComparison
import pandas as pd
import os


@click.group()
def cli():
    pass


@cli.command()
@click.argument('ref_dataset', help='Path to file of metrics computed on a reference dataset')
@click.argument('dataset', help='Path to file of metrics computed on dataset')
@click.argument('output_folder', help='Folder to save metrics plots')
def plot_comparison(ref_dataset, dataset, output_folder):
    """Generate metrics plots for computed metrics against a
    reference set of computed metrics.

    Parameters
    ----------
    ref_dataset : str
        Path to file of metrics computed on a reference dataset      
    dataset : str
        Path to file of metrics computed on dataset   
    output_folder : str
        Folder to save metrics plots
    """
    dataset = Evaluator().from_pickle(dataset)
    ref_dataset = Evaluator().from_pickle(ref_dataset)
    comparsion = MetricsComparison(dataset, ref_dataset)
    comparsion.plot_metrics(output_folder)


@cli.command()
@click.argument('input_file', help='CSV file containing metrics computed stop points.')
@click.argument('output_file', help='Path to location to save computed metrics as a pickle file.')
@click.option('--reference_dataset', default=None, help='Reference dataset to use (use the same histogram bins)')
@click.option('--num_bins', default=30, help='Number of bins to use in the metrics histograms.')
def run_evaluation(input_file, output_file, reference_dataset, num_bins):
    """Compute TA-1 metrics on a collection of stop points computed for
    a dataset of ULLT trajectories.

    Parameters
    ----------
    input_file : str
        CSV file containing metrics computed stop points.
    output_file : str
        Path to location to save computed metrics as a pickle file.
    reference_dataset : str
        Reference dataset to use (use the same histogram bins) (defaults to None)
    num_bins : int
        Number of bins to use in the metrics histograms. (defaults to 30)
    """
    evaluator = Evaluator(initialize_hist=reference_dataset is None,
                          num_bins=num_bins)
    if reference_dataset:
        ref_evaluator = Evaluator().from_pickle(reference_dataset)
        evaluator.intialize_histograms(ref_evaluator.metrics_histograms)
    df = pd.read_csv(input_file)
    evaluator.run_metrics(df)
    if len(os.path.dirname(output_file)) > 0:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    evaluator.to_pickle(output_file)


if __name__ == "__main__":

    cli()
