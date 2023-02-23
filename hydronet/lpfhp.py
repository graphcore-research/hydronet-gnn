# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Longest-pack-first histogram-packing."""
import time
from collections import defaultdict

import numpy as np
from tabulate import tabulate


def add_pack(pack, count, tmp, final, limit, offset, max_sequence_length=512):
    """Filter out packs that reached maximum length or number of components."""
    # sanity checks
    assert max_sequence_length - sum(pack) == offset, "Incorrect offset."
    assert offset >= 0, "Too small offset."
    assert offset < max_sequence_length, "Too large offset."
    if len(pack) == limit or offset == 0:
        final[offset].append((count, pack))
    else:
        tmp[offset].append((count, pack))


def pack_using_lpfhp(
    histogram, max_sequence_length, max_sequences_per_pack, distribute=True
):
    """Longest-pack-first histogram-packing algorithm."""
    reversed_histogram = np.flip(histogram)
    # Initialize main strategy data dictionary.
    # The key indicates how many tokens are left for full length.
    # The value is a list of tuples, consisting of counts and respective packs.
    # A pack is a (sorted) list of sequence length values that is concatenated.
    tmp_strategies_per_length = defaultdict(list)
    strategies_per_length = defaultdict(list)
    if max_sequences_per_pack == "max":
        max_sequences_per_pack = max_sequence_length
    # Index i indicates here, how much space is left, due to reversed histogram
    for i in range(max_sequence_length):
        n_sequences_to_bin = reversed_histogram[i]
        length_to_bin = max_sequence_length - i
        offset = 0  # smallest possible offset for perfect fit
        while n_sequences_to_bin > 0:
            if (length_to_bin + offset) in tmp_strategies_per_length:
                # extract worst pack that will get modified
                n_sequences_to_pack, pack = tmp_strategies_per_length[
                    length_to_bin + offset
                ].pop()
                # calculate how often the current sequence maximally fits in
                repeat = min(
                    1 + offset // length_to_bin, max_sequences_per_pack - len(pack)
                )
                # correct dependent on count
                while n_sequences_to_bin // repeat == 0:
                    repeat -= 1
                if not distribute:
                    repeat = 1
                new_pack = pack + [length_to_bin] * repeat
                count = min(n_sequences_to_pack, n_sequences_to_bin // repeat)
                if n_sequences_to_pack > count:
                    # old pack gets reduced
                    n_sequences_to_pack -= count
                    tmp_strategies_per_length[length_to_bin + offset].append(
                        (n_sequences_to_pack, pack)
                    )
                    n_sequences_to_bin -= count * repeat
                else:
                    n_sequences_to_bin -= n_sequences_to_pack * repeat
                add_pack(
                    new_pack,
                    count,
                    tmp_strategies_per_length,
                    strategies_per_length,
                    max_sequences_per_pack,
                    offset - (repeat - 1) * length_to_bin,
                    max_sequence_length,
                )
                # clean up to speed up main key search
                if not tmp_strategies_per_length[length_to_bin + offset]:
                    tmp_strategies_per_length.pop(length_to_bin + offset)
                # reset offset in case best fit changed
                offset = 0
            else:
                offset += 1
            # Does not fit anywhere. Create new pack.
            if offset >= max_sequence_length - length_to_bin + 1:
                # similar repetition  but no dependence on pack.
                repeat = min(
                    max_sequence_length // length_to_bin, max_sequences_per_pack
                )
                while n_sequences_to_bin // repeat == 0:
                    repeat -= 1
                if not distribute:
                    repeat = 1
                add_pack(
                    [length_to_bin] * repeat,
                    n_sequences_to_bin // repeat,
                    tmp_strategies_per_length,
                    strategies_per_length,
                    max_sequences_per_pack,
                    max_sequence_length - length_to_bin * repeat,
                    max_sequence_length,
                )
                n_sequences_to_bin -= n_sequences_to_bin // repeat * repeat
    # merge all strategies
    for key in tmp_strategies_per_length:
        strategies_per_length[key].extend(tmp_strategies_per_length[key])
    # flatten strategies dictionary
    strategy_set = []
    strategy_repeat_count = []
    for key in strategies_per_length:
        for count, pack in strategies_per_length[key]:
            pack.reverse()
            strategy_set.append(pack)
            strategy_repeat_count.append(count)
    return strategy_set, np.array(strategy_repeat_count)


"""Max depth analysis of longest-pack-first histogram-packing."""


def evaluate_lpfhp(histogram, max_sequence_length):
    """Evaluate shortest-pack-first histogram-packing algorithm."""
    stats_data = [
        [
            "pack. depth",
            "# strat. used",
            "# packs",
            "# tokens",
            "# padding tok.",
            "efficiency (%)",
            "pack.factor",
            "time",
        ]
    ]
    for max_sequences_per_pack in [1, 2, 3, 4, 8, 16, "max"]:
        start = time.time()
        strategy_set, strategy_repeat_count = pack_using_lpfhp(
            histogram, max_sequence_length, max_sequences_per_pack
        )
        duration = time.time() - start

        # Performance Evaluation of packing approach
        n_strategies = int(len(strategy_set))
        packs = int(sum(strategy_repeat_count))
        sequences = sum(
            [
                count * len(pack)
                for count, pack in zip(strategy_repeat_count, strategy_set)
            ]
        )
        total_tokens = int(max_sequence_length * packs)
        empty_tokens = int(
            sum(
                [
                    count * (max_sequence_length - sum(pack))
                    for count, pack in zip(strategy_repeat_count, strategy_set)
                ]
            )
        )
        token_efficiency = 100 - empty_tokens / total_tokens * 100
        if max_sequences_per_pack == "max":
            m_length = max([len(pack) for pack in strategy_set])
            max_sequences_per_pack = "max ({})".format(m_length)
        stats_data.append(
            [
                max_sequences_per_pack,
                n_strategies,
                packs,
                total_tokens,
                empty_tokens,
                token_efficiency,
                sequences / packs,
                duration,
            ]
        )
    print(tabulate(stats_data, headers="firstrow", floatfmt=".3f"))


def evaluate_sequence_length_lpfhp(histogram, sequence_range=range(29, 228)):
    """Evaluate shortest-pack-first histogram-packing algorithm."""
    stats_data = [
        [
            "pack. depth",
            "# strat. used",
            "# packs",
            "# padding tok.",
            "efficiency (%)",
            "pack.factor",
            "time",
            "max length",
        ]
    ]
    curve_x = []
    curve_y = []
    for max_sequence_length in sequence_range:
        max_sequences_per_pack = "max"
        start = time.time()
        scaled_histogram = [i for i in histogram]
        scaled_histogram.extend([0] * (max_sequence_length - len(histogram)))
        strategy_set, strategy_repeat_count = pack_using_lpfhp(
            scaled_histogram,
            max_sequence_length,
            max_sequences_per_pack=max_sequences_per_pack,
        )
        duration = time.time() - start

        # Performance Evaluation of packing approach
        n_strategies = int(len(strategy_set))
        packs = int(sum(strategy_repeat_count))
        sequences = sum(
            [
                count * len(pack)
                for count, pack in zip(strategy_repeat_count, strategy_set)
            ]
        )
        total_tokens = int(max_sequence_length * packs)
        empty_tokens = int(
            sum(
                [
                    count * (max_sequence_length - sum(pack))
                    for count, pack in zip(strategy_repeat_count, strategy_set)
                ]
            )
        )
        token_efficiency = 100 - empty_tokens / total_tokens * 100
        m_length = max([len(pack) for pack in strategy_set])
        max_sequences_per_pack = "max ({})".format(m_length)
        stats_data.append(
            [
                max_sequences_per_pack,
                n_strategies,
                packs,
                empty_tokens,
                token_efficiency,
                sequences / packs,
                duration,
                max_sequence_length,
            ]
        )
        curve_x.append(max_sequence_length)
        curve_y.append(token_efficiency)
    data_rows = stats_data[1:]
    data_rows = sorted(data_rows, key=lambda r: r[4], reverse=True)
    print(tabulate(data_rows[0:8], headers=stats_data[0], floatfmt=".3f"))
    # print(curve_x)
    # print(curve_y)
    return np.array(curve_x), np.array(curve_y)
