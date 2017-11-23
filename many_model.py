#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import progress.bar

import creole_model_numpy as model


class ModelRunner(object):
    def __init__(self, learning_rate_coordinated, learning_rate_uncoordinated):
        self.lr_c = learning_rate_coordinated
        self.lr_u = learning_rate_uncoordinated

    def __call__(self, populations_path, epochs_path):
        m = model.Model(populations_path, epochs_path, self.lr_c, self.lr_u)
        m.run()
        return m


def run_dir(dir_path, epochs_path, learning_rate_coordinated, learning_rate_uncoordinated):
    run_model = ModelRunner(learning_rate_coordinated, learning_rate_uncoordinated)
    pops = sorted(os.listdir(dir_path))
    bar = progress.bar.IncrementalBar('Running models')
    models = [run_model(os.path.join(dir_path, pop), epochs_path) for pop in bar.iter(pops)]
    return models


def main():
    args = model.get_args()
    models = run_dir(args.population, args.time, args.lc, args.lu)
    print [m.winner for m in models]


if __name__ == '__main__':
    main()
