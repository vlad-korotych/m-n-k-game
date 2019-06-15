from NumRowsFeatures import NumRowsFeatures, Feature
from base import GameState, Mark
from typing import List, NamedTuple, Union, Any
import numpy as np

class CartFeature(NamedTuple):
    feature1: Union[Feature, str]
    feature2: Union[Feature, str]

class CartNumRowsFeatures(NumRowsFeatures):
    def __init__(self, row: int):
        super().__init__(row)
        self._cart_feature_names: List[Union[Feature, str, CartFeature]] = [n for n in self._feature_names] # mypy...
        for n1 in self._feature_names:
            if n1 == 'alone_alone':
                continue
            for n2 in self._feature_names:
                if n2 == 'alone_alone':
                    continue
                self._cart_feature_names.append(CartFeature(n1, n2))


    def feature_names(self) -> List[Any]:
        return self._cart_feature_names

    def features_count(self) -> int:
        a = ((((self.row - 1) * 2) - 1) * 2) + 1
        return a + a * a + 1

    def get_features(self, state: GameState)  -> List[int]:
        base_features = super().get_features(state)
        features = base_features.copy()

        for i, n1 in enumerate(self._feature_names):
            if n1 == 'alone_alone':
                continue
            for j, n2 in enumerate(self._feature_names):
                if n2 == 'alone_alone':
                    continue
                np.append(features, base_features[i] * base_features[j])

        return features

if __name__ == "__main__":
    f = CartNumRowsFeatures(5)
    print(f._feature_names)
    print(f._cart_feature_names)
    print(f.features_count())
    print(len(f._cart_feature_names))