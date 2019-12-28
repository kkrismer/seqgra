"""
MIT - CSAIL - Gifford Lab - seqgra

Shuffles DNA sequence while preserving dinucleotide frequencies

Adapted from Peter Clote, Oct 2003
(https://github.com/wassermanlab/BiasAway/blob/master/altschulEriksonDinuclShuffle.py)

@author: Peter Clote
@author: Konstantin Krismer
"""
from __future__ import annotations

import sys
import string
import random


class DinucleotideShuffle:

    @staticmethod
    def compute_count_and_lists(s):
        # WARNING: Use of function count(s,"UU") returns 1 on word UUU
        # since it apparently counts only nonoverlapping words UU
        # For this reason, we work with the indices.

        # Initialize lists and mono- and dinucleotide dictionaries
        mono_dict = {}  # List is a dictionary of lists
        mono_dict["A"] = []
        mono_dict["C"] = []
        mono_dict["G"] = []
        mono_dict["T"] = []
        nucl_list = ["A", "C", "G", "T"]
        s = s.upper()
        s = s.replace("T", "T")
        nucl_cnt = {}  # empty dictionary
        dinucl_cnt = {}  # empty dictionary
        for x in nucl_list:
            nucl_cnt[x] = 0
            dinucl_cnt[x] = {}
            for y in nucl_list:
                dinucl_cnt[x][y] = 0

        # Compute count and lists
        nucl_cnt[s[0]] = 1
        nucl_total = 1
        dinucl_total = 0
        for i in range(len(s) - 1):
            x = s[i]
            y = s[i + 1]
            mono_dict[x].append(y)
            nucl_cnt[y] += 1
            nucl_total += 1
            dinucl_cnt[x][y] += 1
            dinucl_total += 1
        assert (nucl_total == len(s))
        assert (dinucl_total == len(s) - 1)
        return nucl_cnt, dinucl_cnt, mono_dict

    @staticmethod
    def choose_edge(x, dinucl_cnt):
        num_in_list = 0
        for y in ["A", "C", "G", "T"]:
            num_in_list += dinucl_cnt[x][y]
        z = random.random()
        denom = dinucl_cnt[x]["A"] + dinucl_cnt[x]["C"] + \
            dinucl_cnt[x]["G"] + dinucl_cnt[x]["T"]
        numerator = dinucl_cnt[x]["A"]
        if z < float(numerator) / float(denom):
            dinucl_cnt[x]["A"] -= 1
            return "A"
        numerator += dinucl_cnt[x]["C"]
        if z < float(numerator) / float(denom):
            dinucl_cnt[x]["C"] -= 1
            return "C"
        numerator += dinucl_cnt[x]["G"]
        if z < float(numerator) / float(denom):
            dinucl_cnt[x]["G"] -= 1
            return "G"
        dinucl_cnt[x]["T"] -= 1
        return "T"

    @staticmethod
    def connected_to_last(edge_list, nucl_list, last_ch) -> bool:
        D = {}
        for x in nucl_list:
            D[x] = 0
        for edge in edge_list:
            a = edge[0]
            b = edge[1]
            if b == last_ch:
                D[a] = 1
        for _ in range(2):
            for edge in edge_list:
                a = edge[0]
                b = edge[1]
                if D[b] == 1:
                    D[a] = 1
        for x in nucl_list:
            if x != last_ch and D[x] == 0:
                return False
        return True

    @staticmethod
    def eulerian(s):
        _, dinucl_cnt, _ = DinucleotideShuffle.compute_count_and_lists(s)
        # compute nucleotides appearing in s
        nucl_list = []
        for x in ["A", "C", "G", "T"]:
            if x in s:
                nucl_list.append(x)
        # compute num_in_list[x] = number of dinucleotides beginning with x
        num_in_list = {}
        for x in nucl_list:
            num_in_list[x] = 0
            for y in nucl_list:
                num_in_list[x] += dinucl_cnt[x][y]
        # create dinucleotide shuffle L
        last_ch = s[-1]
        edge_list = []
        for x in nucl_list:
            if x != last_ch:
                edge_list.append(
                    [x, DinucleotideShuffle.choose_edge(x, dinucl_cnt)])
        ok = DinucleotideShuffle.connected_to_last(edge_list, nucl_list, 
                                                   last_ch)
        return ok, edge_list, nucl_list, last_ch

    @staticmethod
    def shuffle_edge_list(x):
        n = len(x)
        barrier = n
        for _ in range(n - 1):
            z = int(random.random() * barrier)
            tmp = x[z]
            x[z] = x[barrier - 1]
            x[barrier - 1] = tmp
            barrier -= 1
        return x

    @staticmethod
    def shuffle(s: str) -> str:
        ok = False
        while not ok:
            ok, edge_list, nucl_list, _ = DinucleotideShuffle.eulerian(s)
        _, _, mono_dict = DinucleotideShuffle.compute_count_and_lists(s)

        # remove last edges from each vertex list, shuffle, then add back
        # the removed edges at end of vertex lists.
        for [x, y] in edge_list:
            mono_dict[x].remove(y)
        for x in nucl_list:
            DinucleotideShuffle.shuffle_edge_list(mono_dict[x])
        for [x, y] in edge_list:
            mono_dict[x].append(y)

        # construct the eulerian path
        path = [s[0]]
        prev_ch = s[0]
        for _ in range(len(s) - 2):
            ch = mono_dict[prev_ch][0]
            path.append(ch)
            del mono_dict[prev_ch][0]
            prev_ch = ch
        path.append(s[-1])
        t = "".join(path)
        return t
