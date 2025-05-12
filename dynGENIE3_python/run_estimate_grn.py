#!/usr/bin/env python
# coding: utf-8

# dynGENIE3

# IO
from pathlib import Path

# Data manipulation
import pandas as pd
import numpy as np

# handy / other
import re, os, sys, time, pickle, yaml
from datetime import datetime
from collections import defaultdict

# local library
sys.path.append("../tools/dynGENIE3/dynGENIE3_python/")
import dynGENIE3

class Config:

    def __init__(self, config_file):
        self.config_file = config_file
        
        with open(config_file, "r", encoding="utf-8") as opt_file:
            options = yaml.safe_load(opt_file)
        for k in options:
            setattr(self.__class__, k, options[k])

    def get_output_file_names(self):
        paths = {}
        for treatment in self.inputs:       
            treatment_config = self.inputs[treatment]
            folder = Path(treatment_config['input_folder'])
            output_folder = Path(treatment_config['output_folder'])
            out_file_path_basis = f"dynGENI3_results_{self.config_file.stem}_{treatment}"
    
            paths[treatment] = output_folder, out_file_path_basis
    
        return paths

def run_dynGENIE3(ts_data, timepoints, gene_names, regulators, decay_rates, ss_data):

    result = []
    stability_score = None
    prediction_score = None
    nedges = None

    try:
        (VIM, alphas, prediction_score, stability_score,
         treeEstimators) = dynGENIE3.dynGENIE3(
             ts_data,
             timepoints,
             gene_names=gene_names,
             regulators=regulators,
             alpha=decay_rates,
             SS_data=ss_data,
             nthreads=config.nthreads,
             tree_method=config.tree_method,
             ntrees=config.ntrees,
             compute_quality_scores=config.compute_quality_scores,
             # ntop=config.ntop
         )

        result = (VIM, alphas, prediction_score, stability_score,
                  treeEstimators)

    except Exception as err:
        print("FAILED!", err)

    return result, stability_score, prediction_score

def save_results(result, output_folder, out_file_path_basis, gene_names, regulators):

    with open(output_folder / f"{out_file_path_basis}.pickle", 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    edges = pd.DataFrame(dynGENIE3.get_link_list(
        result[0], gene_names=gene_names, regulators=regulators),
                         columns=["source", "target", "weight"])

    edges = edges[edges["weight"] > 0]
    edges.to_csv(output_folder / f"{out_file_path_basis}_edges.tsv",
                 sep='\t', index=False)

    return len(edges)

def prepare_timeseries_data(df, shuffle=False):
    '''
    Prepare the time series data for dynGENIE3
    Aggregate replicates - optionally shuffle replicates

    df: pandas DataFrame, shape (n_genes, n_timepoints * n_replicates)
    '''

    df = df.T

    if shuffle:
        # per timepoint, randomly reassign the replicate column number
        def reindex_replicates(df):
            idx0 = list(df.index.get_level_values(0))
            np.random.shuffle(idx0)
            return df.reindex(idx0, axis=0, level=0)

        df = df.groupby("timepoint").apply(reindex_replicates).reset_index(0, drop=True)

    ts_data = []
    timepoints = []

    for rep, subdf in df.groupby("replicate"):
        experiment_timepoints = list(subdf.index.get_level_values(1))
        print('\t', rep, len(experiment_timepoints),
              subdf.shape[0] == len(experiment_timepoints))
        ts_data.append(subdf.values)
        timepoints.append(experiment_timepoints)

    return ts_data, timepoints

if __name__ == '__main__':
    today = datetime.today().strftime('%Y-%m-%d')
    print("Today is: ", today)

    config_file = Path(sys.argv[1])
    print("Using config file: ", config_file)
        
    config = Config(config_file)

    with open(config_file.stem + "_logging.txt", "a", buffering=1) as log:
        log.write(f"\nBEGINNING OF RUN on {today}\n")

        for num_t, treatment in enumerate(config.inputs):
            print(f"Running {treatment} ({num_t} of {len(config.inputs)})")
    
            out_file_path_basis = f"dynGENI3_results_{config_file.stem}_{treatment}"
    
            treatment_config = config.inputs[treatment]
            folder = Path(treatment_config['input_folder'])
            output_folder = Path(treatment_config['output_folder'])
    
            output_folder.mkdir(parents=True, exist_ok=True)
    
            ts_df = pd.read_csv(folder / treatment_config['ts_file'],
                                sep='\t',
                                header=[0, 1],
                                index_col=0)
   
            if config.use_subset:
                gene_subset = list(
                    set(
                        pd.read_csv(folder / treatment_config['gene_subset_file'],
                                    sep="\t",
                                    header=None)[0].values))
                ts_df = ts_df.loc[gene_subset]
                subset_flag = 'list'
            else:
                subset_flag = 'all'
    
            if config.use_std_filter:
                ts_df = ts_df[ts_df.std(
                    axis=1) > treatment_config['gene_subset_std']]
                subset_flag += f'-std{treatment_config["gene_subset_std"]}'
            else:
                subset_flag += '-none'
    
            # Smaller subset to run for testing code
            if config.TESTING:
                subset = [
                    "AT1G07640",
                    "AT1G18330",
                    "AT2G17230",
                    "AT1G25580",
                    "AT1G32870",
                    "AT2G01760",
                    "AT2G25000",
                    "AT2G31230",
                    "AT5G08330",
                    "AT5G22570",
                    "AT5G61890",
                    "AT5G66320",
                    "AT1G06160",
                    "AT1G30210",
                    "AT1G31320",
                    "AT1G56170",
                    "AT1G68360",
                ]
                ts_df = ts_df.loc[subset]
    
            # gene list
            gene_names = list(ts_df.index)
            with open(output_folder / f"{out_file_path_basis}_genes.txt",
                      "w") as out:
                for gene in gene_names:
                    out.write(f"{gene}\n")
    
            # steady state data
            if config.use_ss_data:
                ss_df = pd.read_csv(folder / treatment_config['ss_file'],
                                    sep='\t',
                                    header=[0],
                                    index_col=0)
    
                # subset data to only included in TS
                ss_df = ss_df.loc[ts_df.index]
                ss_data = ss_df.T.values
                ss_flag = 'y'
    
            else:
                ss_data = None
                ss_flag = 'none'
    
            # regulators
            tfs = set(
                pd.read_csv(folder / treatment_config['tfs_file'],
                            sep="\t",
                            header=None)[0].values)
            regulators = list(tfs.intersection(ts_df.index))
    
            # mRNA decay rates
            if config.use_decays:
                decay_df = pd.read_csv(folder / treatment_config['decays'],
                                       sep="\t",
                                       index_col=0)
                decay_df = decay_df.loc[ts_df.index]
                decay_rates = list(decay_df.values)
                decay_flag = 'input'
            else:
                decay_rates = 'from_data'
                decay_flag = 'from_data'
    
            if config.shuffle_replicates:
                print(f"Shuffling replicates {config.n_shuffle} times.")
                for i in range(config.n_shuffle):
    
                    stability_score = False
                    prediction_score = False
                    nedges = False
                    out_file_path_basis_i = f"{out_file_path_basis}_shuffle_{i}"
    
                    print(f"Running dynGENIE3 for {config_file} - {treatment}: shuffle #{i}")
    
                    ts_data, timepoints = prepare_timeseries_data(ts_df, shuffle=True)
                    result, stability_score, prediction_score = run_dynGENIE3(ts_data=ts_data,
                                          timepoints=timepoints,
                                          gene_names=gene_names,
                                          regulators=regulators,
                                          decay_rates=decay_rates,
                                          ss_data=ss_data
                                          )
    
                    if result:
                        nedges = save_results(result, output_folder, out_file_path_basis_i, gene_names, regulators)
                    else:
                        nedges = 0
    
                    log.write("\t".join([
                        str(x) for x in [
                            out_file_path_basis_i, treatment, config.use_subset,
                            config.use_std_filter, treatment_config['gene_subset_std'],
                            config.use_ss_data, config.use_decays, config.ntrees,
                            len(gene_names), len(regulators),
                            stability_score, prediction_score,
                            nedges
                        ]
                    ]) + "\n")
    
    
            else:
                print(f"Running dynGENIE3 for {config_file} - {treatment}.")
                ts_data, timepoints = prepare_timeseries_data(ts_df)
                result, stability_score, prediction_score = run_dynGENIE3(ts_data=ts_data,
                                      timepoints=timepoints,
                                      gene_names=gene_names,
                                      regulators=regulators,
                                      decay_rates=decay_rates,
                                      ss_data=ss_data
                                      )
    
                if result:
                    nedges = save_results(result, output_folder, out_file_path_basis, gene_names, regulators)
                else:
                    nedges = 0
                
                log.write("\t".join([
                    str(x) for x in [
                        out_file_path_basis, treatment, config.use_subset,
                        config.use_std_filter, treatment_config['gene_subset_std'],
                        config.use_ss_data, config.use_decays, config.ntrees,
                        len(gene_names), len(regulators),
                        stability_score, prediction_score,
                        nedges
                    ]
                ]) + "\n")
    
            print(f"Done {treatment}!\n---------------------\n")
    
        log.write(f"END OF RUN\n")
    print("Ran GRN inference for all treatments")
