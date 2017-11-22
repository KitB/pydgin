#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import collections


def get_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("feature_set", type=str)
    ap.add_argument("original_demographics", type=str)
    ap.add_argument('-o', '--output-directory', type=str, default='outputs')
    return ap.parse_args()


def main():
    args = get_args()

    with open(args.feature_set, 'r') as feature_set_file:
        feature_set_reader = csv.reader(feature_set_file, delimiter=' ')
        feature_indiceses = []
        for row in feature_set_reader:
            feature_indices = collections.defaultdict(list)
            for i, value in enumerate(row):
                feature_indices[value].append(i)
            feature_indiceses.append(feature_indices)

    with open(args.original_demographics, 'r') as original_demographics_file:
        original_demographics_reader = csv.reader(original_demographics_file, delimiter='\t')
        outputs = []
        for feature_indices in feature_indiceses:
            output = []
            for row in original_demographics_reader:
                output.append([sum(int(row[index]) for index in feature)
                               for feature in feature_indices.values()])
            outputs.append(output)

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    output_template = os.path.join(args.output_directory, 'feature_set_{i}_population.txt')

    for i, output in enumerate(outputs):
        with open(output_template.format(i=i), 'wb') as output_file:
            writer = csv.writer(output_file)
            writer.writerows(output)


if __name__ == '__main__':
    main()
