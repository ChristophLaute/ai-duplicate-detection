# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from vsc.index import PairMatches, VideoFeature, VideoIndex
from vsc.metrics import CandidatePair


class ScoreAggregation(ABC):
    @abstractmethod
    def aggregate(self, match):
        pass

    def score(self, match):
        score = self.aggregate(match)
        return CandidatePair(query_id=match.query_id, ref_id=match.ref_id, score=score)


class MaxScoreAggregation(ScoreAggregation):
    def aggregate(self, match):
        return np.max([m.score for m in match.matches])


class CandidateGeneration:
    def __init__(self, references, aggregation):
        self.aggregation = aggregation
        dim = references[0].dimensions()
        self.index = VideoIndex(dim)
        self.index.add(references)

    def query(self, queries, global_k):
        matches = self.index.search(queries, global_k=global_k)
        candidates = [self.aggregation.score(match) for match in matches]
        candidates = sorted(candidates, key=lambda match: match.score, reverse=True)
        return candidates
