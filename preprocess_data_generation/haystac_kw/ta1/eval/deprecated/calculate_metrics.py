import click
import pickle
from haystac_eval import Evaluator, MetricsComparison
import pandas as pd
import os


@click.group()
def cli():
    pass


@cli.command()
@click.argument('ref_dataset')
@click.argument('dataset')
@click.argument('output_folder')
def plot_comparison(ref_dataset, dataset, output_folder):

    dataset = Evaluator().from_pickle(dataset)
    ref_dataset = Evaluator().from_pickle(ref_dataset)
    comparsion = MetricsComparison(dataset, ref_dataset)
    comparsion.plot_metrics(output_folder)


@cli.command()
@click.argument('input_file')
@click.argument('output_file')
@click.option('--reference_dataset', default=None)
@click.option('--num_bins', default=30)
def run_evaluation(input_file, output_file, reference_dataset, num_bins):

    evaluator = Evaluator(initialize_hist=reference_dataset is None,
                          num_bins=num_bins)
    if reference_dataset:
        ref_evaluator = Evaluator().from_pickle(reference_dataset)
        evaluator.intialize_histograms(ref_evaluator.metrics_histograms)
    df = pd.read_csv(input_file)
    evaluator.run_metrics(df)
    evaluator_results = evaluator.to_dict()
    print(evaluator_results)
    if len(os.path.dirname(output_file)) > 0:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    evaluator.to_pickle(output_file)


if __name__ == "__main__":

    cli()
