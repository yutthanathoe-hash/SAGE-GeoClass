"""Optional XGBoost sklearn-compatible adapter for SAGE-GeoClass 2.5.5.4.3.4.
Encodes arbitrary integer class DNs to 0..K-1 without changing public labels.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

class SageXGBoostClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_estimators=500, max_depth=6, learning_rate=0.05,
                 subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
                 random_state=42, n_jobs=1):
        self.n_estimators=n_estimators
        self.max_depth=max_depth
        self.learning_rate=learning_rate
        self.subsample=subsample
        self.colsample_bytree=colsample_bytree
        self.reg_lambda=reg_lambda
        self.random_state=random_state
        self.n_jobs=n_jobs

    def fit(self, X, y, sample_weight=None):
        try:
            from xgboost import XGBClassifier
        except (ImportError, OSError) as exc:
            raise ImportError('XGBoost is not available in the QGIS Python environment; install a compatible xgboost build: '+str(exc)) from exc
        self._label_encoder=LabelEncoder().fit(np.asarray(y))
        self.classes_=self._label_encoder.classes_
        encoded=self._label_encoder.transform(y)
        if len(self.classes_)<2:
            raise ValueError('XGBoost requires at least two training classes')
        kwargs=dict(n_estimators=self.n_estimators,max_depth=self.max_depth,
                    learning_rate=self.learning_rate,subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,reg_lambda=self.reg_lambda,
                    random_state=self.random_state,n_jobs=self.n_jobs,
                    tree_method='hist', objective='multi:softprob' if len(self.classes_)>2 else 'binary:logistic',
                    eval_metric='mlogloss' if len(self.classes_)>2 else 'logloss')
        self._xgb_model=XGBClassifier(**kwargs)
        self._xgb_model.fit(X,encoded,sample_weight=sample_weight)
        self.n_features_in_=self._xgb_model.n_features_in_
        return self

    def predict(self,X):
        return self._label_encoder.inverse_transform(np.asarray(self._xgb_model.predict(X),dtype=int))

    def predict_proba(self,X):
        return self._xgb_model.predict_proba(X)

    @property
    def feature_importances_(self):
        return self._xgb_model.feature_importances_
