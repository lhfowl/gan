#!/usr/bin/python3.7
"""Prints interesting scalars from tf eval log files.

Usage:
chmod ugo+x ./tensorflow_gan/examples/print_tf_log.py
gsutil ls gs://ilya-test-eu/experiments/imagenet128_expt/eval_eval/* | ./tensorflow_gan/examples/print_tf_log.py
"""

import fileinput

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_tensorflow_log(path, scores={}, event_tags={}):

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 0,
        'images': 0,
        'scalars': 1000,
        'histograms': 0
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    print('Event tags are')
    print(event_acc.Tags())

    fids =   event_acc.Scalars('eval/fid')
    iss = event_acc.Scalars('eval/incscore')

    for r in fids:
        scores[r.step] = [r.value]

    for r in iss:
        if r.step in scores:
            scores[r.step].append(r.value)
        else:
            scores[r.step] = [-1,r.value]
            
    return scores

def print_tensorflow_log(scores):

    steps = sorted(scores.keys())

    print('step,fid,IS')
    for step in steps:
        print('{:06d},{:.2f},{:.2f}'.format(step, scores[step][0], scores[step][1]))

if __name__ == '__main__':
    scores = {}
    with fileinput.input() as f_input:
        for i, line in enumerate(f_input):
            if i == 0:
                print('Skipping first line: {}',format(line.strip()))
            else:
                print('Getting scores for: {}',format(line.strip()))
                scores = get_tensorflow_log(line.strip(), scores)
    print('Printing scores:')
    print_tensorflow_log(scores)
