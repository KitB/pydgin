#!/usr/bin/env python
# -*- coding: utf-8 -*-
import multiprocessing
import os

import progress.bar

import creole_model_numpy as model


class ModelRunner(object):
    def __init__(self, learning_rate_coordinated, learning_rate_uncoordinated, epochs):
        self.lr_c = learning_rate_coordinated
        self.lr_u = learning_rate_uncoordinated
        self.epochs_path = epochs

    def __call__(self, element):
        i, populations_path = element
        m = model.Model(populations_path, self.epochs_path, self.lr_c, self.lr_u)
        m.run()
        return i, m


def run_dir(dir_path, epochs_path, learning_rate_coordinated, learning_rate_uncoordinated):
    pool = multiprocessing.Pool(processes=8)
    run_model = ModelRunner(learning_rate_coordinated, learning_rate_uncoordinated, epochs_path)
    pops = sorted(os.listdir(dir_path))

    # Progress bar stuff
    bar = progress.bar.IncrementalBar('Running models')
    bar.max = len(pops)

    pops = [os.path.join(dir_path, pop) for pop in pops]

    models = pool.imap_unordered(run_model, enumerate(pops))
    out = [model for model in bar.iter(models)]  # Consume the iterator (meaning wait for all of the processes to complete)

    return [m for i, m in sorted(out)]  # Sort by the first element (their index) and return the second


def main():
    args = model.get_args()
    models = run_dir(args.population, args.time, args.lc, args.lu)
    print [m.winner for m in models]


if __name__ == '__main__':
    main()
