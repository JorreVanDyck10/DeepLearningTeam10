"""Complete, fold-local preprocessing for the provided 5,000-row course dataset."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ['cap_diameter', 'stem_height', 'stem_width']


def preprocessing(X, aws=False, scale=False):
    numeric = [('impute', SimpleImputer(strategy='median'))]
    if scale: numeric.append(('scale', StandardScaler()))
    return ColumnTransformer([
        ('num', Pipeline(numeric), NUMERIC),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='missing')
            if aws else SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore'))]), [c for c in X if c not in NUMERIC])])


class ZeroStemMeasurements(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        result = X.copy()
        result[['stem_height', 'stem_width']] = result[['stem_height', 'stem_width']].replace(0, np.nan)
        return result


class ProximitySporeImputer(TransformerMixin, BaseEstimator):
    """Learn the notebook's unsupervised forest imputation using training rows only.

    Similarities need only the reference rows with known colours, avoiding a full
    n-by-n matrix. Leaf-match counts equal the notebook's one-hot dot product.
    """
    def __init__(self, n_estimators=200, k=30, random_state=42, n_jobs=2):
        self.n_estimators = n_estimators
        self.k = k
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.preprocessor_ = preprocessing(X)
        encoded = self.preprocessor_.fit_transform(X)
        dense = encoded.toarray() if hasattr(encoded, 'toarray') else np.asarray(encoded)
        rng = np.random.default_rng(self.random_state)
        synthetic = np.column_stack([rng.permutation(dense[:, j]) for j in range(dense.shape[1])])
        self.forest_ = RandomForestClassifier(n_estimators=self.n_estimators, min_samples_leaf=2,
            max_features='sqrt', random_state=self.random_state, n_jobs=self.n_jobs)
        self.forest_.fit(np.vstack([dense, synthetic]), np.r_[np.ones(len(X)), np.zeros(len(X))])
        known = X.spore_print_color.notna().to_numpy()
        self.reference_count_ = len(X)
        self.known_indices_ = np.flatnonzero(known)
        self.full_reference_colors_ = X.spore_print_color.to_numpy()
        self.reference_leaves_ = self.forest_.apply(encoded)[known]
        self.reference_colors_ = X.spore_print_color.to_numpy()[known]
        modes = X.spore_print_color.mode()
        self.fallback_ = modes.iloc[0] if len(modes) else 'missing'
        return self

    def transform(self, X):
        result = X.copy()
        missing = np.flatnonzero(result.spore_print_color.isna().to_numpy())
        if not len(missing): return result
        if not len(self.reference_colors_):
            result.iloc[missing, result.columns.get_loc('spore_print_color')] = self.fallback_
            return result
        leaves = self.forest_.apply(self.preprocessor_.transform(X.iloc[missing]))
        for position, leaf in zip(missing, leaves):
            matches = (self.reference_leaves_ == leaf).sum(axis=1)
            similarities = np.full(self.reference_count_, -1, dtype=float)
            similarities[self.known_indices_] = matches / self.n_estimators
            nearest = np.argsort(similarities)[-min(self.k, len(matches)):]
            votes = {}
            for colour, weight in zip(self.full_reference_colors_[nearest], similarities[nearest]):
                votes[colour] = votes.get(colour, 0) + float(weight)
            value = max(votes, key=votes.get) if votes and max(votes.values()) > 0 else self.fallback_
            result.iloc[position, result.columns.get_loc('spore_print_color')] = value
        return result
