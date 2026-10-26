# -*- coding: utf-8 -*-
"""
SAGE-GeoClass V2.5.5.4.3.5 — General-Purpose Geospatial Classification (Fixed Syntax & Parenthesis)
QGIS Processing Tool for supervised Land Use / Land Cover classification.

Designed for QGIS 4.2.2 / Python 3.12; isolated TensorFlow runtime
"""

import os
import csv
import json
import math
import tempfile
import traceback
import re
import time
import sys
import subprocess
import shutil
import threading
import queue

import numpy as np
from osgeo import gdal, ogr

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterMultipleLayers,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterString,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterFolderDestination,
    QgsVectorLayer,
    QgsField,
    QgsFeature,
    QgsVectorFileWriter,
    QgsCoordinateTransformContext,
)


class SAGEGeoClassV252(QgsProcessingAlgorithm):

    RASTERS = "RASTERS"
    FEATURE_NAMES = "FEATURE_NAMES"
    FEATURE_SCHEMA_JSON = "FEATURE_SCHEMA_JSON"
    OUTPUT_FEATURE_MANIFEST = "OUTPUT_FEATURE_MANIFEST"
    SPECTRAL_BAND_INDICES = "SPECTRAL_BAND_INDICES"
    CLASS_LEGEND_JSON = "CLASS_LEGEND_JSON"
    CLASS_SCHEMA_JSON = "CLASS_SCHEMA_JSON"
    OUTPUT_CLASS_MANIFEST = "OUTPUT_CLASS_MANIFEST"
    OUTPUT_CLASS_QC = "OUTPUT_CLASS_QC"
    MIN_CLASS_SAMPLES = "MIN_CLASS_SAMPLES"
    TRAINING = "TRAINING"
    CLASS_FIELD = "CLASS_FIELD"
    VALIDATION_METHOD = "VALIDATION_METHOD"
    TEST_SIZE = "TEST_SIZE"
    BLOCK_SIZE = "BLOCK_SIZE"
    MODEL_MODE = "MODEL_MODE"
    USE_SPATIAL_CNN = "USE_SPATIAL_CNN"
    PATCH_SIZE = "PATCH_SIZE"
    USE_TEMPORAL_CNN = "USE_TEMPORAL_CNN"
    TEMPORAL_STEPS = "TEMPORAL_STEPS"
    N_TREES = "N_TREES"
    MAX_DEPTH = "MAX_DEPTH"
    DL_EPOCHS = "DL_EPOCHS"
    DL_BATCH = "DL_BATCH"
    ML_INFERENCE_BATCH_SIZE = "ML_INFERENCE_BATCH_SIZE"
    OUTPUT_INFERENCE_BENCHMARK = "OUTPUT_INFERENCE_BENCHMARK"
    ML_ADAPTIVE_BATCH = "ML_ADAPTIVE_BATCH"
    ML_MEMORY_RESERVE_MB = "ML_MEMORY_RESERVE_MB"
    RUN_CONTROLLED_BENCHMARK = "RUN_CONTROLLED_BENCHMARK"
    OUTPUT_CONTROLLED_BENCHMARK = "OUTPUT_CONTROLLED_BENCHMARK"
    BALANCE = "BALANCE"
    STANDARDIZE = "STANDARDIZE"
    PERMUTATION_IMPORTANCE = "PERMUTATION_IMPORTANCE"
    PERM_REPEATS = "PERM_REPEATS"
    APPLY_FILTER = "APPLY_FILTER"
    FILTER_METHOD = "FILTER_METHOD"
    MAJORITY_SIZE = "MAJORITY_SIZE"
    MIN_PATCH_PIXELS = "MIN_PATCH_PIXELS"
    CONF_THRESHOLD = "CONF_THRESHOLD"
    IO_BLOCK_SIZE = "IO_BLOCK_SIZE"
    MAX_TRAIN_PIXELS = "MAX_TRAIN_PIXELS"
    AUTO_ALIGN = "AUTO_ALIGN"
    ALIGN_RESAMPLING = "ALIGN_RESAMPLING"
    
    OUTPUT_PROFILING = "OUTPUT_PROFILING"
    OUTPUT_LEAKAGE_QC = "OUTPUT_LEAKAGE_QC"

    OUTPUT_GRID_QC = "OUTPUT_GRID_QC"
    RUN_SPATIAL_CV = "RUN_SPATIAL_CV"
    CV_FOLDS = "CV_FOLDS"
    TUNE_HYPERPARAMS = "TUNE_HYPERPARAMS"
    TUNE_ITER = "TUNE_ITER"
    RANDOM_SEED = "RANDOM_SEED"
    OUTPUT_CV = "OUTPUT_CV"
    OUTPUT_CV_PER_CLASS = "OUTPUT_CV_PER_CLASS"
    OUTPUT_TRAIN_QC = "OUTPUT_TRAIN_QC"
    OUTPUT_FEATURE_QC = "OUTPUT_FEATURE_QC"
    OUTPUT_FLOOD_METRICS = "OUTPUT_FLOOD_METRICS"
    FLOOD_OBJECTIVE = "FLOOD_OBJECTIVE"
    OUTPUT_FLOOD_MODEL_COMPARISON = "OUTPUT_FLOOD_MODEL_COMPARISON"
    MODEL_OBJECTIVE = "MODEL_OBJECTIVE"
    TARGET_CLASS = "TARGET_CLASS"
    MIN_TARGET_PRECISION = "MIN_TARGET_PRECISION"
    MIN_TARGET_RECALL = "MIN_TARGET_RECALL"
    OUTPUT_TARGET_MODEL_COMPARISON = "OUTPUT_TARGET_MODEL_COMPARISON"

    OUTPUT_RAW_CLASS = "OUTPUT_RAW_CLASS"
    OUTPUT_CLASS = "OUTPUT_CLASS"
    OUTPUT_CONF = "OUTPUT_CONF"
    OUTPUT_ENTROPY = "OUTPUT_ENTROPY"
    OUTPUT_MARGIN = "OUTPUT_MARGIN"
    OUTPUT_RELIABILITY = "OUTPUT_RELIABILITY"
    OUTPUT_RELIABILITY_SUMMARY = "OUTPUT_RELIABILITY_SUMMARY"
    OUTPUT_METRICS = "OUTPUT_METRICS"
    OUTPUT_CM = "OUTPUT_CM"
    OUTPUT_CLASSES = "OUTPUT_CLASSES"
    OUTPUT_MODEL_COMPARISON = "OUTPUT_MODEL_COMPARISON"
    OUTPUT_IMPORTANCE = "OUTPUT_IMPORTANCE"
    FEATURE_SELECTION_TOP_K = "FEATURE_SELECTION_TOP_K"
    RUN_FEATURE_ABLATION = "RUN_FEATURE_ABLATION"
    OUTPUT_FEATURE_ABLATION = "OUTPUT_FEATURE_ABLATION"
    OUTPUT_SPLIT = "OUTPUT_SPLIT"
    OUTPUT_AREA_STATS = "OUTPUT_AREA_STATS"
    OUTPUT_CLASS_ACCURACY = "OUTPUT_CLASS_ACCURACY"
    OUTPUT_MODEL_DIR = "OUTPUT_MODEL_DIR"
    OUTPUT_DIRECTORY = "OUTPUT_DIRECTORY"
    RUN_XAI = "RUN_XAI"
    OUTPUT_OOD = "OUTPUT_OOD"
    OUTPUT_TRUST = "OUTPUT_TRUST"
    OUTPUT_SHAP = "OUTPUT_SHAP"
    OUTPUT_SPATIAL_IMPORTANCE = "OUTPUT_SPATIAL_IMPORTANCE"
    OUTPUT_XAI_AUDIT = "OUTPUT_XAI_AUDIT"
    OUTPUT_SHAP_SAMPLES = "OUTPUT_SHAP_SAMPLES"
    OUTPUT_SHAP_CLASSWISE = "OUTPUT_SHAP_CLASSWISE"
    OUTPUT_SHAP_BEESWARM = "OUTPUT_SHAP_BEESWARM"
    OUTPUT_SHAP_BAR = "OUTPUT_SHAP_BAR"
    SHAP_FEATURE_VALUE_MODE = "SHAP_FEATURE_VALUE_MODE"
    SHAP_SAMPLE_COUNT = "SHAP_SAMPLE_COUNT"
    SHAP_TARGET_CLASS = "SHAP_TARGET_CLASS"

    HOLDOUT_TEST_SIZE = "HOLDOUT_TEST_SIZE"
    OUTPUT_TEST_METRICS = "OUTPUT_TEST_METRICS"
    OUTPUT_TEST_CM = "OUTPUT_TEST_CM"
    OUTPUT_TEST_PER_CLASS = "OUTPUT_TEST_PER_CLASS"
    OUTPUT_RASTER_QA = "OUTPUT_RASTER_QA"
    OUTPUT_SPECTRAL_QA = "OUTPUT_SPECTRAL_QA"
    OUTPUT_CLASS_LEGEND = "OUTPUT_CLASS_LEGEND"

    VALIDATION_NAMES = [
        "Feature/Polygon Group Split (recommended)",
        "Spatial Block Split",
        "Random Pixel Split (legacy / optimistic)"
    ]

    FILTER_NAMES = [
        "None",
        "Majority / Mode",
        "Sieve / Minimum Mapping Unit",
        "Confidence Threshold",
        "Majority + Sieve",
        "Confidence + Majority + Sieve"
    ]

    MODEL_NAMES = [
        "Auto Compare 12 Models (ML9 + DL3; eligible models only)",
        "Random Forest",
        "Extra Trees",
        "HistGradientBoosting",
        "SVM (RBF)",
        "MLP Neural Network",
        "Spatial 2D-CNN (TensorFlow)",
        "Temporal CNN (TensorFlow)",
        "Deep Neural Network (TensorFlow DNN)",
        "XGBoost (optional dependency)",
        "LightGBM (optional dependency)",
        "CatBoost (optional dependency)",
        "Logistic Regression"
    ]

    def tr(self, s):
        return QCoreApplication.translate("SAGEGeoClassV251", s)

    def createInstance(self):
        return SAGEGeoClassV252()

    def name(self):
        return "sage_geoclass_v2_5_5"

    def displayName(self):
        return self.tr("SAGE-GeoClass V2.5.5.4.3.5 — General-Purpose Geospatial Classification")

    def group(self):
        return self.tr("SAGE GeoClass")

    def groupId(self):
        return "sage_geoclass"

    def shortHelpString(self):
        return self.tr(
            "Advanced spatial-temporal supervised land-use classification. "
            "Includes Spatial 2D-CNN with halo block inference, Temporal CNN resampling, "
            "spatial leakage buffer checks, and runtime profiling."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterMultipleLayers(
            self.RASTERS,
            self.tr("Input raster bands / indices (or temporal sequence)"),
            layerType=QgsProcessing.TypeRaster
        ))

        self.addParameter(QgsProcessingParameterString(
            self.FEATURE_NAMES,
            self.tr("Feature names (comma separated; blank = raster layer names)"),
            defaultValue="",
            optional=True
        ))

        self.addParameter(QgsProcessingParameterString(
            self.FEATURE_SCHEMA_JSON,
            self.tr('Optional Feature Registry JSON (ordered features with name/role/kind/unit)'),
            defaultValue='', optional=True
        ))

        self.addParameter(QgsProcessingParameterString(
            self.SPECTRAL_BAND_INDICES,
            self.tr("Spectral roles: global 1-based feature indices for Blue,Green,Red,NIR (blank = no spectral indices)"),
            defaultValue="", optional=True
        ))
        self.addParameter(QgsProcessingParameterString(
            self.CLASS_LEGEND_JSON,
            self.tr('Optional class legend JSON: {"1":{"name":"Forest","color":"#228B22"}}'),
            defaultValue="", optional=True
        ))

        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_FEATURE_MANIFEST,
            self.tr('Universal Feature Registry manifest JSON'),
            fileFilter='JSON files (*.json)'
        ))

        self.addParameter(QgsProcessingParameterString(
            self.CLASS_SCHEMA_JSON,
            self.tr('Optional universal class schema JSON (original values, names, colors)'),
            defaultValue='', optional=True
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_CLASS_SAMPLES, self.tr('QC warning threshold: minimum training pixels per class'),
            QgsProcessingParameterNumber.Integer, defaultValue=20, minValue=1, maxValue=100000
        ))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_CLASS_MANIFEST, self.tr('Universal class schema manifest JSON'),
            fileFilter='JSON files (*.json)'
        ))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_CLASS_QC, self.tr('Universal class sample QC CSV'),
            fileFilter='CSV files (*.csv)'
        ))

        self.addParameter(QgsProcessingParameterVectorLayer(
            self.TRAINING,
            self.tr("Training samples (Point or Polygon)"),
            types=[QgsProcessing.TypeVectorPoint, QgsProcessing.TypeVectorPolygon]
        ))

        self.addParameter(QgsProcessingParameterField(
            self.CLASS_FIELD,
            self.tr("Class field (integer or text)"),
            parentLayerParameterName=self.TRAINING,
            type=QgsProcessingParameterField.Any
        ))

        self.addParameter(QgsProcessingParameterEnum(
            self.VALIDATION_METHOD,
            self.tr("Validation strategy"),
            options=self.VALIDATION_NAMES,
            defaultValue=0
        ))

        self.addParameter(QgsProcessingParameterNumber(
            self.TEST_SIZE,
            self.tr("Validation fraction"),
            QgsProcessingParameterNumber.Double,
            defaultValue=0.20,
            minValue=0.10,
            maxValue=0.50
        ))

        self.addParameter(QgsProcessingParameterNumber(self.HOLDOUT_TEST_SIZE, self.tr("Independent test fraction (held out before validation)"), QgsProcessingParameterNumber.Double, defaultValue=0.20, minValue=0.10, maxValue=0.40))
        self.addParameter(QgsProcessingParameterNumber(
            self.BLOCK_SIZE,
            self.tr("Spatial block size in map units (used for Spatial Block Split)"),
            QgsProcessingParameterNumber.Double,
            defaultValue=1000.0,
            minValue=1.0
        ))

        self.addParameter(QgsProcessingParameterEnum(
            self.MODEL_MODE,
            self.tr("Model Selection"),
            options=self.MODEL_NAMES,
            defaultValue=0
        ))

        self.addParameter(QgsProcessingParameterBoolean(
            self.USE_SPATIAL_CNN,
            self.tr("[V2.4.4] Enable Spatial 2D-CNN context patches"),
            defaultValue=False
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.PATCH_SIZE,
            self.tr("[V2.4.4] Spatial CNN patch size (e.g. 15 for 15x15)"),
            QgsProcessingParameterNumber.Integer,
            defaultValue=15,
            minValue=3,
            maxValue=65
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            self.USE_TEMPORAL_CNN,
            self.tr("[V2.4.4] Enable Temporal CNN for multi-date raster stacks"),
            defaultValue=False
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.TEMPORAL_STEPS,
            self.tr("[V2.4.4] Number of temporal timestamps / dates"),
            QgsProcessingParameterNumber.Integer,
            defaultValue=1,
            minValue=1,
            maxValue=50
        ))

        self.addParameter(QgsProcessingParameterNumber(
            self.N_TREES,
            self.tr("Number of trees (RF / Extra Trees)"),
            QgsProcessingParameterNumber.Integer,
            defaultValue=500,
            minValue=50,
            maxValue=5000
        ))

        self.addParameter(QgsProcessingParameterNumber(
            self.MAX_DEPTH,
            self.tr("Maximum tree depth (0 = unlimited)"),
            QgsProcessingParameterNumber.Integer,
            defaultValue=0,
            minValue=0,
            maxValue=200
        ))

        self.addParameter(QgsProcessingParameterNumber(
            self.DL_EPOCHS,
            self.tr("DNN / CNN epochs"),
            QgsProcessingParameterNumber.Integer,
            defaultValue=50,
            minValue=5,
            maxValue=1000
        ))

        self.addParameter(QgsProcessingParameterNumber(
            self.DL_BATCH,
            self.tr("DNN / CNN batch size"),
            QgsProcessingParameterNumber.Integer,
            defaultValue=256,
            minValue=8,
            maxValue=8192
        ))

        self.addParameter(QgsProcessingParameterNumber(
            self.ML_INFERENCE_BATCH_SIZE,
            self.tr("ML raster probability batch size (256 baseline; 4096 diagnostic candidate; 1024 default)"),
            QgsProcessingParameterNumber.Integer,
            defaultValue=1024, minValue=256, maxValue=8192
        ))

        self.addParameter(QgsProcessingParameterBoolean(
            self.ML_ADAPTIVE_BATCH, self.tr("Adaptive ML inference batch (memory-aware; conservative)"), defaultValue=True
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.ML_MEMORY_RESERVE_MB, self.tr("Minimum available RAM reserve (MB)"),
            QgsProcessingParameterNumber.Integer, defaultValue=2048, minValue=256, maxValue=65536
        ))

        self.addParameter(QgsProcessingParameterBoolean(
            self.BALANCE,
            self.tr("Balance classes where supported"),
            defaultValue=True
        ))

        self.addParameter(QgsProcessingParameterBoolean(
            self.STANDARDIZE,
            self.tr("Standardize features for SVM / Neural Networks / CNN"),
            defaultValue=True
        ))

        self.addParameter(QgsProcessingParameterBoolean(
            self.PERMUTATION_IMPORTANCE,
            self.tr("Calculate permutation feature importance on validation data"),
            defaultValue=True
        ))

        self.addParameter(QgsProcessingParameterNumber(
            self.PERM_REPEATS,
            self.tr("Permutation importance repeats"),
            QgsProcessingParameterNumber.Integer,
            defaultValue=10,
            minValue=3,
            maxValue=100
        ))

        self.addParameter(QgsProcessingParameterNumber(
            self.FEATURE_SELECTION_TOP_K,
            self.tr("[V2.5.5] Select top K features for pixel-based ML (0 = all; CNN/DNN diagnostic only)"),
            QgsProcessingParameterNumber.Integer, defaultValue=0, minValue=0, maxValue=999
        ))

        self.addParameter(QgsProcessingParameterBoolean(
            self.RUN_FEATURE_ABLATION,
            self.tr("[V2.5.5] Run train-only spatial ML feature ablation (extra training cost; CNN/DNN skipped)"),
            defaultValue=False
        ))

        self.addParameter(QgsProcessingParameterBoolean(
            self.APPLY_FILTER,
            self.tr("Apply post-classification filter"),
            defaultValue=True
        ))

        self.addParameter(QgsProcessingParameterEnum(
            self.FILTER_METHOD,
            self.tr("Post-classification filter method"),
            options=self.FILTER_NAMES,
            defaultValue=4
        ))

        self.addParameter(QgsProcessingParameterEnum(
            self.MAJORITY_SIZE,
            self.tr("Majority filter window"),
            options=["3 x 3", "5 x 5", "7 x 7"],
            defaultValue=0
        ))

        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_PATCH_PIXELS,
            self.tr("Minimum patch size for sieve (pixels)"),
            QgsProcessingParameterNumber.Integer,
            defaultValue=8,
            minValue=2,
            maxValue=100000
        ))

        self.addParameter(QgsProcessingParameterNumber(
            self.CONF_THRESHOLD,
            self.tr("Confidence threshold (pixels below become 0/Uncertain)"),
            QgsProcessingParameterNumber.Double,
            defaultValue=0.60,
            minValue=0.0,
            maxValue=1.0
        ))

        self.addParameter(QgsProcessingParameterEnum(
            self.IO_BLOCK_SIZE, self.tr("Memory-safe raster block size"),
            options=["256", "512", "1024"], defaultValue=0
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.MAX_TRAIN_PIXELS, self.tr("Maximum training pixels (0 = no cap)"),
            QgsProcessingParameterNumber.Integer, defaultValue=250000,
            minValue=0, maxValue=5000000
        ))

        self.addParameter(QgsProcessingParameterBoolean(
            self.AUTO_ALIGN, self.tr("Auto-align mismatched rasters to first raster (master grid)"),
            defaultValue=True
        ))
        self.addParameter(QgsProcessingParameterEnum(
            self.ALIGN_RESAMPLING, self.tr("Alignment resampling"),
            options=["Bilinear (continuous; recommended)", "Nearest neighbour", "Cubic"],
            defaultValue=0
        ))

        self.addParameter(QgsProcessingParameterEnum(
            self.FLOOD_OBJECTIVE, self.tr("Legacy flood DN=1 objective (compatibility only)"),
            options=["Flood F1 (recommended)", "Flood IoU", "Flood Recall (precision >= 0.50)",
                     "Flood Precision (recall >= 0.50)", "Macro-F1 (legacy V2.4.1)"],
            defaultValue=0
        ))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_FLOOD_MODEL_COMPARISON,
            self.tr("Flood-focused model comparison CSV"), fileFilter="CSV files (*.csv)"
        ))

        self.addParameter(QgsProcessingParameterEnum(
            self.MODEL_OBJECTIVE, self.tr("General model selection objective"),
            options=["Macro-F1 (recommended for multi-class)", "Weighted-F1",
                     "Balanced Accuracy", "Target-class F1", "Target-class IoU",
                     "Target-class Recall (minimum precision)",
                     "Target-class Precision (minimum recall)"], defaultValue=0
        ))
        self.addParameter(QgsProcessingParameterString(
            self.TARGET_CLASS,
            self.tr("Target class: original training label or output DN (blank = no target)"),
            defaultValue="", optional=True
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_TARGET_PRECISION, self.tr("Minimum target precision (0–1)"),
            QgsProcessingParameterNumber.Double, defaultValue=0.50, minValue=0.0, maxValue=1.0
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_TARGET_RECALL, self.tr("Minimum target recall (0–1)"),
            QgsProcessingParameterNumber.Double, defaultValue=0.50, minValue=0.0, maxValue=1.0
        ))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_TARGET_MODEL_COMPARISON,
            self.tr("General per-class model comparison CSV"), fileFilter="CSV files (*.csv)"
        ))

        self.addParameter(QgsProcessingParameterBoolean(
            self.RUN_SPATIAL_CV, self.tr("Run spatial GroupKFold CV on selected model (ML / CNN; training partition only)"), defaultValue=True
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.CV_FOLDS, self.tr("Requested group K-folds"),
            QgsProcessingParameterNumber.Integer, defaultValue=5, minValue=2, maxValue=20
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            self.TUNE_HYPERPARAMS, self.tr("Tune selected ML model (group-aware randomized search)"), defaultValue=False
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.TUNE_ITER, self.tr("Hyperparameter search iterations"),
            QgsProcessingParameterNumber.Integer, defaultValue=8, minValue=2, maxValue=50
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.RANDOM_SEED, self.tr("Random seed for research CV and tuning"),
            QgsProcessingParameterNumber.Integer, defaultValue=42, minValue=0, maxValue=999999
        ))
        
        self.addParameter(QgsProcessingParameterBoolean(
            self.RUN_CONTROLLED_BENCHMARK,
            self.tr("Run controlled paired ML inference benchmark (optional; additional runtime)"),
            defaultValue=False
        ))
        file_destinations = [
            (self.OUTPUT_CV, "Group K-fold validation CSV"),
            (self.OUTPUT_CV_PER_CLASS, "Group K-fold per-class metrics CSV"),
            (self.OUTPUT_TRAIN_QC, "Training class/group QC CSV"),
            (self.OUTPUT_PROFILING, "[V2.5.5.4.3.5] Runtime profiling & execution breakdown CSV"),
            (self.OUTPUT_INFERENCE_BENCHMARK, "[V2.5.5.4.3.5] ML inference benchmark CSV"),
            (self.OUTPUT_CONTROLLED_BENCHMARK, "[V2.5.5.4.3.5] Paired diagnostic benchmark CSV"),
            (self.OUTPUT_LEAKAGE_QC, "[V2.4.4] Spatial leakage & patch overlap QC CSV"),
            (self.OUTPUT_FLOOD_METRICS, "Flood class (DN=1) metrics CSV")
        ]
        for key, label in file_destinations:
            self.addParameter(QgsProcessingParameterFileDestination(
                key, self.tr(label), fileFilter="CSV files (*.csv)"
            ))

        self.addParameter(QgsProcessingParameterBoolean(
            self.RUN_XAI, self.tr("Enable integrated XAI / OOD / Reliability (RF, ET, HistGB, SVM, MLP, XGBoost, SpatialCNN, TemporalCNN, DNN)"), defaultValue=False))
        for _k, _label in ((self.OUTPUT_OOD, "OOD distance raster"),
                           (self.OUTPUT_TRUST, "Heuristic reliability raster (0-1)")):
            self.addParameter(QgsProcessingParameterRasterDestination(_k, self.tr(_label), optional=True))
        for _k, _label in ((self.OUTPUT_SHAP, "SHAP mean absolute importance CSV"),
                           (self.OUTPUT_SPATIAL_IMPORTANCE, "Spatial group permutation importance CSV"),
                           (self.OUTPUT_XAI_AUDIT, "XAI audit status CSV")):
            self.addParameter(QgsProcessingParameterFileDestination(
                _k, self.tr(_label), fileFilter="CSV files (*.csv)", optional=True))

        self.addParameter(QgsProcessingParameterNumber(
            self.SHAP_SAMPLE_COUNT,
            self.tr("SHAP foreground sample count (16-500; 100 recommended; increases XAI runtime)"),
            QgsProcessingParameterNumber.Integer, defaultValue=100, minValue=16, maxValue=500))
        self.addParameter(QgsProcessingParameterString(
            self.SHAP_TARGET_CLASS,
            self.tr("SHAP beeswarm target class ID (blank = most impactful class) [optional]"),
            defaultValue="", optional=True))
        self.addParameter(QgsProcessingParameterEnum(
            self.SHAP_FEATURE_VALUE_MODE,
            self.tr("CNN SHAP beeswarm color: patch mean (recommended), center, median, or std"),
            options=["PATCH_MEAN", "CENTER_PIXEL", "PATCH_MEDIAN", "PATCH_STD"],
            defaultValue=0))
        for _key, _label in ((self.OUTPUT_SHAP_SAMPLES, "SHAP signed sample-level long CSV"),
                             (self.OUTPUT_SHAP_CLASSWISE, "SHAP per-class importance CSV")):
            self.addParameter(QgsProcessingParameterFileDestination(
                _key, self.tr(_label), fileFilter="CSV files (*.csv)", optional=True, defaultValue="TEMPORARY_OUTPUT"))
        for _key, _label in ((self.OUTPUT_SHAP_BEESWARM, "SHAP beeswarm plot PNG"),
                             (self.OUTPUT_SHAP_BAR, "SHAP global bar plot PNG")):
            self.addParameter(QgsProcessingParameterFileDestination(
                _key, self.tr(_label), fileFilter="PNG images (*.png)", optional=True, defaultValue="TEMPORARY_OUTPUT"))

        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT_RAW_CLASS, self.tr("Raw classified land-use raster")
        ))
        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT_CLASS, self.tr("Filtered/final classified land-use raster")
        ))
        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT_CONF, self.tr("Classification confidence raster (0-1)")
        ))
        for _key, _label in ((self.OUTPUT_ENTROPY, 'Normalized predictive entropy raster (0-1)'),
                              (self.OUTPUT_MARGIN, 'Top-2 probability margin raster (0-1)')):
            self.addParameter(QgsProcessingParameterRasterDestination(_key, self.tr(_label)))
        for _key, _label in ((self.OUTPUT_RELIABILITY, 'Independent-test reliability bins CSV'),
                              (self.OUTPUT_RELIABILITY_SUMMARY, 'Independent-test reliability summary CSV')):
            self.addParameter(QgsProcessingParameterFileDestination(_key, self.tr(_label), fileFilter='CSV files (*.csv)'))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_METRICS, self.tr("Best model validation metrics CSV"),
            fileFilter="CSV files (*.csv)"
        ))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_CM, self.tr("Best model confusion matrix CSV"),
            fileFilter="CSV files (*.csv)"
        ))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_CLASSES, self.tr("Class mapping CSV"),
            fileFilter="CSV files (*.csv)"
        ))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_MODEL_COMPARISON, self.tr("Model comparison CSV"),
            fileFilter="CSV files (*.csv)"
        ))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_IMPORTANCE, self.tr("Feature importance CSV"),
            fileFilter="CSV files (*.csv)"
        ))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_FEATURE_ABLATION, self.tr("[V2.5.5] Spatial feature ablation audit CSV"),
            fileFilter="CSV files (*.csv)"
        ))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_SPLIT, self.tr("Train/validation split summary CSV"),
            fileFilter="CSV files (*.csv)"
        ))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_AREA_STATS, self.tr("Area statistics by class CSV"),
            fileFilter="CSV files (*.csv)"
        ))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_CLASS_ACCURACY, self.tr("Per-class User's / Producer's Accuracy CSV"),
            fileFilter="CSV files (*.csv)"
        ))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_GRID_QC, self.tr("Raster grid QC CSV"),
            fileFilter="CSV files (*.csv)"
        ))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_FEATURE_QC, self.tr("Multiband feature QC CSV"),
            fileFilter="CSV files (*.csv)"
        ))
        for key, label in [(self.OUTPUT_TEST_METRICS,"Independent test metrics CSV"),(self.OUTPUT_TEST_CM,"Independent test confusion matrix CSV"),(self.OUTPUT_TEST_PER_CLASS,"Independent test per-class accuracy CSV")]:
            self.addParameter(QgsProcessingParameterFileDestination(key, self.tr(label), fileFilter="CSV files (*.csv)"))
        for key, title in (
            (self.OUTPUT_RASTER_QA, "Raster integrity and held-out inference QA CSV"),
            (self.OUTPUT_SPECTRAL_QA, "Spectral consistency by class CSV"),
            (self.OUTPUT_CLASS_LEGEND, "Thai LULC class legend CSV"),
        ):
            self.addParameter(QgsProcessingParameterFileDestination(key, self.tr(title), fileFilter="CSV files (*.csv)"))
        self.addParameter(QgsProcessingParameterFolderDestination(
            self.OUTPUT_DIRECTORY, self.tr("Save ALL results to this directory (optional; overrides individual output destinations)"), optional=True
        ))
        self.addParameter(QgsProcessingParameterFolderDestination(
            self.OUTPUT_MODEL_DIR, self.tr("Saved model folder")
        ))

    def _class_map(self, layer, class_field):
        values = []
        for f in layer.getFeatures():
            val = f[class_field]
            sval = "" if val is None else str(val).strip()
            if sval.upper() not in ("", "NULL", "NONE", "NAN", "<NULL>"):
                values.append(sval)
        labels = sorted(set(values), key=lambda x: (0, int(x)) if re.fullmatch(r"[+-]?\d+", x) else (1, x.casefold()))
        if len(labels) < 2:
            raise QgsProcessingException("At least 2 classes are required.")
        return {label: i + 1 for i, label in enumerate(labels)}

    def _make_numeric_training(self, layer, class_field, class_map):
        if layer.geometryType() == 0:
            geom_name = "Point"
        else:
            geom_name = "Polygon"
            
        crs_part = layer.crs().authid()
        mem = QgsVectorLayer(f"{geom_name}?crs={crs_part}", "training_v244", "memory")
        if not mem.isValid():
            raise QgsProcessingException("Could not create temporary training layer.")

        dp = mem.dataProvider()
        dp.addAttributes([
            QgsField("__class_id", QVariant.Int),
            QgsField("__group_id", QVariant.Int)
        ])
        mem.updateFields()

        feats = []
        gid = 1
        for f in layer.getFeatures():
            val = f[class_field]
            if val is None or str(val).strip() == "":
                continue
            label = str(val)
            if label not in class_map:
                continue
            nf = QgsFeature(mem.fields())
            nf.setGeometry(f.geometry())
            nf.setAttributes([int(class_map[label]), int(gid)])
            feats.append(nf)
            gid += 1

        if not feats:
            raise QgsProcessingException("No usable training features found.")

        dp.addFeatures(feats)
        mem.updateExtents()

        tmp = os.path.join(tempfile.gettempdir(), "landuse_v244_training.gpkg")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

        opts = QgsVectorFileWriter.SaveVectorOptions()
        opts.driverName = "GPKG"
        opts.layerName = "training"
        opts.fileEncoding = "UTF-8"
        err, msg, _, _ = QgsVectorFileWriter.writeAsVectorFormatV3(
            mem, tmp, QgsCoordinateTransformContext(), opts
        )
        if err != QgsVectorFileWriter.NoError:
            raise QgsProcessingException("Temporary vector creation failed: " + str(msg))
        return tmp

    def _rasterize_attribute(self, vector_path, ref, attribute, dtype=gdal.GDT_Int32):
        mem = gdal.GetDriverByName("MEM").Create(
            "", ref.RasterXSize, ref.RasterYSize, 1, dtype
        )
        mem.SetGeoTransform(ref.GetGeoTransform())
        mem.SetProjection(ref.GetProjection())
        b = mem.GetRasterBand(1)
        b.Fill(0)
        b.SetNoDataValue(0)

        vds = ogr.Open(vector_path)
        if vds is None:
            raise QgsProcessingException("Cannot open temporary training vector.")
        lyr = vds.GetLayer(0)

        err = gdal.RasterizeLayer(
            mem, [1], lyr,
            options=[f"ATTRIBUTE={attribute}", "ALL_TOUCHED=TRUE"]
        )
        if err != 0:
            raise QgsProcessingException("Rasterization failed for " + attribute)
        return mem

    def _extract_samples_spatial_temporal(self, raster_dss, class_ds, group_ds, ref,
                                           validation_method, block_size, feedback,
                                           use_spatial_cnn, patch_size, use_temporal_cnn, temporal_steps,
                                           io_block_size=512, max_train_pixels=250000):
        bands = [(ds.GetRasterBand(i)) for ds in raster_dss for i in range(1,ds.RasterCount+1)]
        W,H = ref.RasterXSize,ref.RasterYSize
        rad = patch_size//2 if use_spatial_cnn else 0
        Xs,Ps,Ys,Gs,Rows,Cols = [],[],[],[],[],[]
        rng = np.random.RandomState(42)
        # Sample per block; halo permits valid samples at block boundaries.
        for yo in range(0,H,io_block_size):
            for xo in range(0,W,io_block_size):
                if feedback.isCanceled(): raise QgsProcessingException("Canceled")
                nr,nc=min(io_block_size,H-yo),min(io_block_size,W-xo)
                labels=class_ds.GetRasterBand(1).ReadAsArray(xo,yo,nc,nr)
                rr,cc=np.where(labels>0)
                if not len(rr): continue
                groups=group_ds.GetRasterBand(1).ReadAsArray(xo,yo,nc,nr)
                x0,y0=max(0,xo-rad),max(0,yo-rad)
                x1,y1=min(W,xo+nc+rad),min(H,yo+nr+rad)
                arrays=[];valid=np.ones((y1-y0,x1-x0),bool)
                for b in bands:
                    a=b.ReadAsArray(x0,y0,x1-x0,y1-y0).astype(np.float32)
                    nd=b.GetNoDataValue();valid &= np.isfinite(a)
                    if nd is not None: valid &= (a!=nd)
                    arrays.append(a)
                cube=np.stack(arrays,axis=-1)
                for r,c in zip(rr,cc):
                    lr,lc=yo+r-y0,xo+c-x0
                    if not valid[lr,lc]:continue
                    if use_spatial_cnn:
                        if lr<rad or lc<rad or lr+rad>=len(valid) or lc+rad>=valid.shape[1]:continue
                        if not valid[lr-rad:lr+rad+1,lc-rad:lc+rad+1].all():continue
                        Ps.append(cube[lr-rad:lr+rad+1,lc-rad:lc+rad+1,:].copy())
                    Xs.append(cube[lr,lc,:].copy());Ys.append(int(labels[r,c]));Gs.append(int(groups[r,c]));Rows.append(yo+r);Cols.append(xo+c)
        if len(Ys)<20: raise QgsProcessingException("Insufficient valid training pixels")
        X=np.asarray(Xs,np.float32);y=np.asarray(Ys,np.int32);fg=np.asarray(Gs,np.int64)
        rows=np.asarray(Rows,np.int64);cols=np.asarray(Cols,np.int64)
        patches=np.asarray(Ps,np.float32) if use_spatial_cnn else None
        if max_train_pixels>0 and len(y)>max_train_pixels:
            keep=[]
            for c in np.unique(y):
                idx=np.where(y==c)[0];n=max(1,round(max_train_pixels*len(idx)/len(y)))
                keep.extend(rng.choice(idx,min(n,len(idx)),replace=False))
            keep=np.asarray(keep,dtype=int)
            if len(keep)>max_train_pixels:keep=rng.choice(keep,max_train_pixels,replace=False)
            X,y,fg,rows,cols=X[keep],y[keep],fg[keep],rows[keep],cols[keep]
            if patches is not None:patches=patches[keep]
        if validation_method==0: groups=fg
        elif validation_method==1:
            gt=ref.GetGeoTransform()
            xx=gt[0]+(cols+.5)*gt[1]+(rows+.5)*gt[2]
            yy=gt[3]+(cols+.5)*gt[4]+(rows+.5)*gt[5]
            bx=np.floor((xx-xx.min())/block_size).astype(np.int64)
            by=np.floor((yy-yy.min())/block_size).astype(np.int64)
            groups=by*(int(bx.max())+1)+bx+1
        else:groups=np.arange(len(y),dtype=np.int64)
        if use_temporal_cnn and temporal_steps>1:
            if X.shape[1]%temporal_steps:raise QgsProcessingException("Temporal bands must be divisible by TEMPORAL_STEPS")
            X=X.reshape(len(X),temporal_steps,X.shape[1]//temporal_steps)
        return X,patches,y,groups,rows,cols

    def _split(self, X, y, groups, method, test_size):
        from sklearn.model_selection import train_test_split, GroupShuffleSplit

        if method == 2:
            tr, va = train_test_split(
                np.arange(len(y)),
                test_size=test_size,
                random_state=42,
                stratify=y
            )
            return tr, va

        unique_groups = np.unique(groups)
        if len(unique_groups) < 3:
            raise QgsProcessingException(
                f"Only {len(unique_groups)} independent validation groups exist. "
                "Create more training polygons/features or reduce spatial block size."
            )

        all_classes = set(np.unique(y).tolist())
        for seed in [42, 7, 17, 23, 101, 2026, 99, 1234]:
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=seed
            )
            tr, va = next(splitter.split(X, y, groups))
            train_classes = set(np.unique(y[tr]).tolist())
            val_classes = set(np.unique(y[va]).tolist())
            if train_classes == all_classes and len(val_classes) >= 2:
                return tr, va

        raise QgsProcessingException(
            "Could not create a valid group-based split containing all classes "
            "in training data. Add more independent polygons/groups per class."
        )

    def _purge_patch_overlap(self, tr, other, rows, cols, radius):
        if radius<=0:return tr,0
        from scipy.spatial import cKDTree
        # Chebyshev metric: patches share pixels when center distance <= 2*radius.
        tree=cKDTree(np.column_stack((rows[other],cols[other])))
        near=tree.query(np.column_stack((rows[tr],cols[tr])),k=1,p=np.inf)[0]
        keep=near>2*radius
        return tr[keep],int((~keep).sum())

    def _check_spatial_leakage(self, groups, tr, va, te, rows, cols, patch_size, spatial, path, feedback):
        sets=[set(groups[z].tolist()) for z in (tr,va,te)]
        records={"train":len(tr),"validation":len(va),"test":len(te),
                 "train_val_group_overlap":len(sets[0]&sets[1]),
                 "train_test_group_overlap":len(sets[0]&sets[2]),
                 "val_test_group_overlap":len(sets[1]&sets[2]),
                 "patch_radius_pixels":patch_size//2 if spatial else 0,
                 "random_pixel_split_warning": "Random pixels are not independent" if len(sets[0]&sets[1]) else ""}
        self._write_csv_dict(path,records)
        if any(records[k] for k in ("train_val_group_overlap","train_test_group_overlap","val_test_group_overlap")):
            feedback.pushWarning("Group overlap detected; use group/spatial split for scientific reporting")

    def _make_ml_model(self, key, n_trees, max_depth, balance, standardize):
        from sklearn.ensemble import (
            RandomForestClassifier,
            ExtraTreesClassifier,
            HistGradientBoostingClassifier
        )
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        depth = None if max_depth <= 0 else max_depth
        cw = "balanced" if balance else None

        if key == "XGBoost":
            import importlib.util
            helper_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sage_xgboost_v255434.py')
            if not os.path.isfile(helper_path):
                raise QgsProcessingException('Missing XGBoost helper: '+helper_path)
            import sys
            module_name = 'sage_xgboost_v255434'
            mod = sys.modules.get(module_name)
            if mod is None or not hasattr(mod, 'SageXGBoostClassifier'):
                spec = importlib.util.spec_from_file_location(module_name, helper_path)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = mod  # Required for joblib/pickle class resolution
                try:
                    spec.loader.exec_module(mod)
                except Exception:
                    sys.modules.pop(module_name, None)
                    raise
            return mod.SageXGBoostClassifier(n_estimators=n_trees,
                        max_depth=6 if max_depth<=0 else max_depth,
                        random_state=42,n_jobs=1)
        if key == "LightGBM":
            try:
                from lightgbm import LGBMClassifier
            except ImportError as e:
                raise QgsProcessingException('LightGBM dependency missing. Install lightgbm into QGIS Python: '+str(e))
            return LGBMClassifier(n_estimators=n_trees, learning_rate=0.05,
                max_depth=-1 if depth is None else depth, num_leaves=31,
                random_state=42, n_jobs=1, verbosity=-1)
        if key == "CatBoost":
            try:
                from catboost import CatBoostClassifier
            except ImportError as e:
                raise QgsProcessingException('CatBoost dependency missing. Install catboost into QGIS Python: '+str(e))
            return CatBoostClassifier(iterations=n_trees, depth=6 if depth is None else min(depth,16),
                learning_rate=0.05, loss_function='MultiClass', verbose=False,
                random_seed=42, thread_count=1, allow_writing_files=False)
        if key == "LogReg":
            from sklearn.linear_model import LogisticRegression
            base = LogisticRegression(max_iter=2000, class_weight=cw, random_state=42)
            return Pipeline([('scaler',StandardScaler()),('model',base)]) if standardize else base
        if key == "RF":
            return RandomForestClassifier(
                n_estimators=n_trees, max_depth=depth, class_weight=cw, n_jobs=-1, random_state=42
            )
        if key == "ExtraTrees":
            return ExtraTreesClassifier(
                n_estimators=n_trees, max_depth=depth, class_weight=cw, n_jobs=-1, random_state=42
            )
        if key == "HistGB":
            return HistGradientBoostingClassifier(
                max_depth=depth, learning_rate=0.08, max_iter=300, random_state=42
            )
        if key == "SVM":
            base = SVC(C=10.0, gamma="scale", kernel="rbf", probability=True, class_weight=cw, random_state=42)
            return Pipeline([("scaler", StandardScaler()), ("model", base)]) if standardize else base
        if key == "MLP":
            base = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64), activation="relu", solver="adam",
                alpha=1e-4, batch_size=256, learning_rate_init=1e-3, max_iter=300,
                early_stopping=True, validation_fraction=0.15, random_state=42
            )
            return Pipeline([("scaler", StandardScaler()), ("model", base)]) if standardize else base
        raise QgsProcessingException("Unknown ML model key: " + key)

    def _tf_worker_path(self):
        script_file = globals().get("__file__", "")
        candidates = [os.path.dirname(os.path.abspath(script_file))] if script_file else []
        candidates += [os.path.join(os.path.expanduser("~"),"AppData","Roaming","QGIS","QGIS4","profiles","default","processing","scripts"), os.getcwd()]
        path = next((os.path.join(folder,"LandUse_TF_Worker_V2_4_4_6.py") for folder in candidates if os.path.isfile(os.path.join(folder,"LandUse_TF_Worker_V2_4_4_6.py"))), "")
        if not os.path.isfile(path):
            raise QgsProcessingException("Missing TensorFlow worker next to Processing script: " + path)
        return path

    def _run_tf_worker(self, action, payload):
        """Run h5py-first worker in fresh process; QGIS/GDAL DLLs never enter worker."""
        worker = self._tf_worker_path()
        with tempfile.TemporaryDirectory(prefix="landuse_tf_v2444_") as folder:
            request = os.path.join(folder, "request.npz")
            response = os.path.join(folder, "response.npz")
            np.savez_compressed(request, **payload)
            # OSGeo4W/QGIS modifies DLL search PATH; keep Python executable and
            # packages, but remove GDAL/OSGeo4W paths to reduce HDF5 DLL collision.
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            cmd = [sys.executable, worker, action, "--request", request, "--response", response]
            completed = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, errors="replace")
            if completed.returncode != 0:
                raise QgsProcessingException("Isolated TensorFlow worker failed (exit %s):\n%s\n%s" %
                                             (completed.returncode, completed.stdout[-4000:], completed.stderr[-10000:]))
            if not os.path.isfile(response):
                raise QgsProcessingException("TensorFlow worker produced no response")
            with np.load(response, allow_pickle=False) as result:
                prob = result["probabilities"].copy()
            if not np.isfinite(prob).all():
                raise QgsProcessingException("TensorFlow worker produced non-finite probabilities")
            return prob

    class _IsolatedTFModel:
        """One long-lived worker per selected DL model; reuse loaded TensorFlow."""
        def __init__(self, owner, model_path, batch):
            self.owner, self.model_path, self.batch = owner, model_path, batch
            self._process = None
            self._folder = None
            self._stderr_file = None
            self._responses = queue.Queue()
            self._reader = None
            self._requests = 0

        def _start(self):
            if self._process is not None:
                if self._process.poll() is not None:
                    raise QgsProcessingException("TensorFlow persistent worker exited unexpectedly: " + self._error_tail())
                return
            self._folder = tempfile.TemporaryDirectory(prefix="landuse_tf_persistent_")
            log_path = os.path.join(self._folder.name, "worker_stderr.log")
            self._stderr_file = open(log_path, "w", encoding="utf-8", errors="replace")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["TF_CPP_MIN_LOG_LEVEL"] = "2"
            env["TF_NUM_INTEROP_THREADS"] = "2"
            env["TF_NUM_INTRAOP_THREADS"] = "4"
            env["OMP_NUM_THREADS"] = "4"
            cmd = [sys.executable, self.owner._tf_worker_path(), "serve", "--model", self.model_path,
                   "--batch", str(self.batch)]
            self._process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                              stderr=self._stderr_file, text=True, bufsize=1,
                                              encoding="utf-8", errors="replace", env=env)
            def reader():
                try:
                    for line in self._process.stdout:
                        self._responses.put(line.rstrip("\r\n"))
                finally:
                    self._responses.put(None)
            self._reader = threading.Thread(target=reader, daemon=True)
            self._reader.start()
            self._await_reply(timeout=240)  # READY after TensorFlow and model load

        def _error_tail(self):
            try:
                if self._stderr_file: self._stderr_file.flush()
                with open(os.path.join(self._folder.name, "worker_stderr.log"),
                          "r", encoding="utf-8", errors="replace") as f:
                    return f.read()[-5000:]
            except Exception:
                return "(worker stderr unavailable)"

        def _await_reply(self, timeout=600):
            try:
                line = self._responses.get(timeout=timeout)
            except queue.Empty:
                self.close()
                raise QgsProcessingException("TensorFlow worker timeout after %ds; try smaller IO blocks / patch size" % timeout)
            if line is None:
                raise QgsProcessingException("TensorFlow worker stopped: " + self._error_tail())
            try:
                msg = json.loads(line)
            except Exception:
                raise QgsProcessingException("Invalid worker protocol response: " + line[:500])
            if not msg.get("ok"):
                raise QgsProcessingException("TensorFlow worker error: " + str(msg.get("error")) + "\n" + self._error_tail())
            return msg

        def predict(self, x, verbose=0):
            """Reliable single-flight disk protocol with bounded retries and diagnostics."""
            self._start()
            import shutil
            self._requests += 1
            # Each transaction has a distinct immutable pathname. Never replace a
            # destination that a Windows mmap/AV scanner might still hold open.
            transaction = self._requests
            stem = "batch_%08d" % transaction
            request = os.path.join(self._folder.name, stem + "_request.npy")
            staged = os.path.join(self._folder.name, stem + "_request.staging")
            response = os.path.join(self._folder.name, stem + "_response.npy")
            import shutil
            array = np.ascontiguousarray(x, dtype=np.float32)
            expected_bytes = int(array.nbytes) + 1024
            free_bytes = shutil.disk_usage(self._folder.name).free
            if free_bytes < max(expected_bytes * 4, 128 * 1024 * 1024):
                raise QgsProcessingException(
                    "Insufficient disk space for TensorFlow IPC: free %.1f MiB, request %.1f MiB, directory %s" %
                    (free_bytes / 1048576, expected_bytes / 1048576, self._folder.name))
            try:
                for attempt in range(3):
                    try:
                        # Unique final filename avoids WinError 5 from replacing
                        # a previous memory-mapped request on Windows.
                        with open(staged, "wb") as out:
                            np.save(out, array, allow_pickle=False)
                            out.flush()
                            os.fsync(out.fileno())
                        actual = os.path.getsize(staged)
                        if actual < array.nbytes:
                            raise OSError("Incomplete IPC request: %d of %d bytes" % (actual, array.nbytes))
                        if os.path.exists(request):
                            raise OSError("IPC transaction ID collision: " + request)
                        os.rename(staged, request)
                        break
                    except OSError as exc:
                        if attempt == 2:
                            raise QgsProcessingException(
                                "Cannot publish TensorFlow IPC request (batch %d, free %.1f MiB): %s" %
                                (transaction, shutil.disk_usage(self._folder.name).free / 1048576, exc)) from exc
                        time.sleep(0.4 * (attempt + 1))
                try:
                    self._process.stdin.write(json.dumps({"command": "predict", "request": request,
                                                           "response": response, "batch_id": transaction}) + "\n")
                    self._process.stdin.flush()
                except (BrokenPipeError, OSError) as exc:
                    raise QgsProcessingException("TensorFlow worker IPC pipe failed: %s\n%s" %
                                                 (exc, self._error_tail())) from exc
                reply = self._await_reply(timeout=600)
                if reply.get("batch_id") != transaction:
                    raise QgsProcessingException("TensorFlow IPC response ID mismatch for batch %d: %s" %
                                                 (transaction, reply))
                if not os.path.isfile(response):
                    raise QgsProcessingException("TensorFlow worker did not write response for batch %d" % transaction)
                with open(response, "rb") as inp:
                    prob = np.load(inp, allow_pickle=False).copy()
                if prob.ndim != 2 or prob.shape[0] != len(array) or not np.isfinite(prob).all():
                    raise QgsProcessingException("Invalid probabilities returned from persistent TensorFlow worker")
                return prob
            finally:
                for path in (staged, request, response):
                    try:
                        os.remove(path)
                    except OSError:
                        # Windows security scanners can briefly hold file handles;
                        # never reuse these paths, so deferred cleanup is safe.
                        pass

        def close(self):
            proc = self._process
            self._process = None
            if proc is not None:
                if proc.poll() is None:
                    try:
                        proc.stdin.write('{"command":"shutdown"}\n')
                        proc.stdin.flush()
                        proc.wait(timeout=8)
                    except Exception:
                        proc.terminate()
                        try: proc.wait(timeout=5)
                        except Exception: proc.kill()
                for stream in (proc.stdin, proc.stdout):
                    try: stream.close()
                    except Exception: pass
            if self._stderr_file:
                self._stderr_file.close()
                self._stderr_file = None
            if self._folder:
                self._folder.cleanup()
                self._folder = None

    def _train_tf_isolated(self, kind, Xtr, ytr, Xva, yva, n_classes, epochs, batch, model_dir, feedback):
        filename = {"SpatialCNN":"spatial_2d_cnn.keras", "TemporalCNN":"temporal_cnn.keras",
                    "DNN":"landuse_dnn.keras"}[kind]
        model_path = os.path.join(model_dir, filename)
        feedback.pushInfo("Isolated TensorFlow worker: " + kind + " (h5py imported before TensorFlow)")
        prob = self._run_tf_worker("train", {
            "xtr": np.asarray(Xtr, dtype=np.float32), "xva": np.asarray(Xva, dtype=np.float32),
            "ytr": np.asarray(ytr-1, dtype=np.int32), "yva": np.asarray(yva-1, dtype=np.int32),
            "n_classes": np.asarray(n_classes, dtype=np.int32),
            "epochs": np.asarray(epochs, dtype=np.int32), "batch": np.asarray(batch, dtype=np.int32),
            "seed": np.asarray(42, dtype=np.int32), "kind": np.asarray(kind),
            "model_path": np.asarray(model_path)
        })
        if prob.shape != (len(yva), n_classes):
            raise QgsProcessingException("Worker validation output shape mismatch: " + str(prob.shape))
        model = self._IsolatedTFModel(self, model_path, batch)
        self._tf_models.append(model)
        self._active_tf_model = model
        return model, np.argmax(prob,axis=1).astype(np.int32)+1

    def _train_cnn_spatial(self, Xtr_patch, ytr, Xva_patch, yva, n_classes, epochs, batch, model_dir, feedback):
        return self._train_tf_isolated("SpatialCNN",Xtr_patch,ytr,Xva_patch,yva,n_classes,epochs,batch,model_dir,feedback)

    def _train_cnn_temporal(self, Xtr, ytr, Xva, yva, n_classes, epochs, batch, model_dir, feedback):
        return self._train_tf_isolated("TemporalCNN",Xtr,ytr,Xva,yva,n_classes,epochs,batch,model_dir,feedback)

    def _train_dnn(self, Xtr, ytr, Xva, yva, n_classes, epochs, batch, standardize, model_dir, feedback):
        from sklearn.preprocessing import StandardScaler
        import joblib
        scaler = None
        if standardize:
            scaler = StandardScaler()
            Xtr, Xva = scaler.fit_transform(Xtr), scaler.transform(Xva)
            joblib.dump(scaler,os.path.join(model_dir,"dnn_scaler.pkl"))
        model,pred = self._train_tf_isolated("DNN",Xtr,ytr,Xva,yva,n_classes,epochs,batch,model_dir,feedback)
        return model,scaler,pred

    def _metrics(self, y_true, y_pred):
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score,
            precision_score, recall_score, f1_score, cohen_kappa_score
        )
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
            "n_validation": int(len(y_true))
        }

    def _write_csv_dict(self, path, d):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in d.items():
                w.writerow([k, v])

    def _write_cm(self, path, y, pred, class_ids):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y, pred, labels=class_ids)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["true\\pred"] + class_ids)
            for cid, row in zip(class_ids, cm):
                w.writerow([cid] + row.tolist())

    def _new_raster(self, path, ref, dtype, nodata):
        ds = gdal.GetDriverByName("GTiff").Create(
            path, ref.RasterXSize, ref.RasterYSize, 1, dtype,
            options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"]
        )
        ds.SetGeoTransform(ref.GetGeoTransform())
        ds.SetProjection(ref.GetProjection())
        b = ds.GetRasterBand(1)
        b.SetNoDataValue(nodata)
        b.Fill(nodata)
        return ds

    def _majority_filter_raster(self, in_path, out_path, size, io_block_size=512):
        try:
            from scipy.ndimage import generic_filter
        except ImportError:
            raise QgsProcessingException("Majority filter requires scipy.")
        src = gdal.Open(in_path)
        sb = src.GetRasterBand(1)
        rad = size // 2
        dst = self._new_raster(out_path, src, gdal.GDT_UInt16, 0)
        ob = dst.GetRasterBand(1)
        W, H = src.RasterXSize, src.RasterYSize
        def mode(v):
            v = v.astype(np.int64)
            v = v[v > 0]
            return 0 if len(v) == 0 else int(np.bincount(v).argmax())
        for yo in range(0, H, io_block_size):
            nr = min(io_block_size, H - yo)
            for xo in range(0, W, io_block_size):
                nc = min(io_block_size, W - xo)
                x0 = max(0, xo - rad)
                y0 = max(0, yo - rad)
                x1 = min(W, xo + nc + rad)
                y1 = min(H, yo + nr + rad)
                a = sb.ReadAsArray(x0, y0, x1 - x0, y1 - y0).astype(np.uint16)
                f = generic_filter(a, mode, size=size, mode="nearest")
                cx, cy = xo - x0, yo - y0
                ob.WriteArray(f[cy:cy + nr, cx:cx + nc].astype(np.uint16), xo, yo)
        ob.FlushCache()
        dst.FlushCache()

    def _sieve_raster(self, in_path, out_path, min_pixels):
        src = gdal.Open(in_path, gdal.GA_ReadOnly)
        band = src.GetRasterBand(1)
        dst = gdal.GetDriverByName("GTiff").Create(
            out_path, src.RasterXSize, src.RasterYSize, 1, gdal.GDT_UInt16,
            options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"]
        )
        dst.SetGeoTransform(src.GetGeoTransform())
        dst.SetProjection(src.GetProjection())
        outb = dst.GetRasterBand(1)
        outb.SetNoDataValue(0)
        gdal.SieveFilter(srcBand=band, maskBand=None, dstBand=outb, threshold=int(min_pixels), connectedness=8)
        outb.FlushCache()

    def _copy_raster(self, src_path, dst_path):
        src = gdal.Open(src_path, gdal.GA_ReadOnly)
        out = gdal.GetDriverByName("GTiff").CreateCopy(dst_path, src, strict=0, options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"])
        out.FlushCache()
        out = None
        src = None

    def _apply_post_filter(self, raw_path, conf_path, final_path,
                           apply_filter, method, majority_size, min_patch_pixels,
                           conf_threshold, feedback, io_block_size=512):
        if (not apply_filter) or method == 0:
            self._copy_raster(raw_path, final_path)
            return
        w1 = os.path.join(tempfile.gettempdir(), "lulc_v244_f1.tif")
        w2 = os.path.join(tempfile.gettempdir(), "lulc_v244_f2.tif")
        for p in (w1, w2):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        current = raw_path
        if method in (3, 5):
            src = gdal.Open(current)
            cds = gdal.Open(conf_path)
            out = self._new_raster(w1, src, gdal.GDT_UInt16, 0)
            sb, cb, ob = src.GetRasterBand(1), cds.GetRasterBand(1), out.GetRasterBand(1)
            W, H = src.RasterXSize, src.RasterYSize
            for yo in range(0, H, io_block_size):
                nr = min(io_block_size, H - yo)
                for xo in range(0, W, io_block_size):
                    nc = min(io_block_size, W - xo)
                    a = sb.ReadAsArray(xo, yo, nc, nr).astype(np.uint16)
                    c = cb.ReadAsArray(xo, yo, nc, nr).astype(np.float32)
                    a[(c >= 0) & (c < conf_threshold)] = 0
                    ob.WriteArray(a, xo, yo)
            ob.FlushCache()
            current = w1
        if method in (1, 4, 5):
            target = w2 if current == w1 else w1
            self._majority_filter_raster(current, target, majority_size, io_block_size)
            current = target
        if method in (2, 4, 5):
            target = w2 if current == w1 else w1
            self._sieve_raster(current, target, min_patch_pixels)
            current = target
        self._copy_raster(current, final_path)

    def _write_per_class_accuracy(self, path, y_true, y_pred, class_ids, inv):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred, labels=class_ids)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["class_id", "class_label", "reference_total", "classified_total", "correct", "producer_accuracy", "user_accuracy"])
            for i, cid in enumerate(class_ids):
                correct = int(cm[i, i])
                ref_total = int(cm[i, :].sum())
                cls_total = int(cm[:, i].sum())
                w.writerow([
                    cid, inv[cid], ref_total, cls_total, correct,
                    correct / ref_total if ref_total else 0.0,
                    correct / cls_total if cls_total else 0.0
                ])

    def _write_area_stats(self, path, raster_path, class_ids, inv, io_block_size=512):
        ds = gdal.Open(raster_path)
        b = ds.GetRasterBand(1)
        gt = ds.GetGeoTransform()
        pa = abs(gt[1] * gt[5] - gt[2] * gt[4])
        counts = {0: 0, **{int(c): 0 for c in class_ids}}
        W, H = ds.RasterXSize, ds.RasterYSize
        for yo in range(0, H, io_block_size):
            nr = min(io_block_size, H - yo)
            for xo in range(0, W, io_block_size):
                nc = min(io_block_size, W - xo)
                vals, nums = np.unique(b.ReadAsArray(xo, yo, nc, nr), return_counts=True)
                for v, n in zip(vals, nums):
                    counts[int(v)] = counts.get(int(v), 0) + int(n)
        total = sum(counts.get(int(c), 0) for c in class_ids)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["class_id", "class_label", "pixel_count", "percent_classified", "area_map_units2", "area_ha"])
            for c in class_ids:
                n = counts.get(int(c), 0)
                area = n * pa
                w.writerow([c, inv[c], n, 100 * n / total if total else 0, area, area / 10000])

    def _write_class_legend(self, path, class_ids):
        labels = {1:("ป่าไม้","#228B22"), 2:("เกษตรมีพืช","#B6D957"),
                  3:("เกษตรเป็นดินเปิดโล่ง","#D5AF70"), 4:("ยางพารา","#087E62"),
                  5:("ชุมชนเมือง","#D83A46"), 6:("แหล่งน้ำ","#2B7FFF"),
                  7:("เหมือง","#8E65A8")}
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w=csv.writer(f); w.writerow(["class_id","thai_label","hex_color"])
            for cid in class_ids:
                name,color=self._sage_class_legend.get(int(cid), (str(cid),"#808080"))
                w.writerow([cid,name,color])

    def _write_spectral_qa(self, path, X, y, tr, te, class_ids, band_count, feedback):
        # Reflectance statistics are observational, not an automatic label validator.
        # Explicit four-band B,G,R,NIR order is used only for the derived indices.
        with open(path,"w",newline="",encoding="utf-8-sig") as f:
            w=csv.writer(f)
            w.writerow(["partition","class_id","n","band_1_blue_mean","band_2_green_mean",
                        "band_3_red_mean","band_4_nir_mean","ndvi_mean","ndwi_green_nir_mean"])
            for partition,ids in (("train",tr),("independent_test",te)):
                for cid in class_ids:
                    a=np.asarray(X[ids][y[ids]==cid],dtype=np.float64)
                    if a.ndim != 2 or not len(a) or self._sage_spectral_roles is None:
                        w.writerow([partition,cid,len(a),"","","","","",""])
                        continue
                    a=a[:,self._sage_spectral_roles]
                    means=np.mean(a,axis=0)
                    ndvi=(a[:,3]-a[:,2])/(a[:,3]+a[:,2]+1e-12)
                    ndwi=(a[:,1]-a[:,3])/(a[:,1]+a[:,3]+1e-12)
                    w.writerow([partition,cid,len(a),*map(float,means),
                                float(np.mean(ndvi)),float(np.mean(ndwi))])
        if self._sage_spectral_roles is None:
            feedback.pushWarning("Spectral roles not configured; index columns left blank. Set global B,G,R,NIR indices to enable.")

    def _append_predicted_spectral_qa(self, path, raw_path, predictor_bands, class_ids, feedback):
        if self._sage_spectral_roles is None:
            return
        ds=gdal.Open(raw_path,gdal.GA_ReadOnly)
        rb=ds.GetRasterBand(1)
        rng=np.random.RandomState(42)
        collected={int(c):[] for c in class_ids}
        # Uniform tile sample across the whole scene; capped for reproducibility.
        for yo in range(0,ds.RasterYSize,256):
            for xo in range(0,ds.RasterXSize,256):
                h=min(256,ds.RasterYSize-yo);w=min(256,ds.RasterXSize-xo)
                labels=rb.ReadAsArray(xo,yo,w,h)
                candidates=rng.choice(h*w,size=min(16,h*w),replace=False)
                rr,cc=np.divmod(candidates,w)
                chosen=[(int(r),int(c),int(labels[r,c])) for r,c in zip(rr,cc)
                        if int(labels[r,c]) in collected]
                if not chosen:continue
                spectral=[predictor_bands[i][0].GetRasterBand(predictor_bands[i][1]).ReadAsArray(xo,yo,w,h) for i in self._sage_spectral_roles]
                for r,c,cid in chosen:
                    v=np.asarray([a[r,c] for a in spectral],dtype=float)
                    if np.isfinite(v).all():collected[cid].append(v)
        with open(path,"a",newline="",encoding="utf-8-sig") as f:
            w=csv.writer(f)
            for cid in class_ids:
                a=np.asarray(collected[int(cid)],dtype=float).reshape(-1,4)
                if len(a):
                    means=a.mean(axis=0)
                    ndvi=(a[:,3]-a[:,2])/(a[:,3]+a[:,2]+1e-12)
                    ndwi=(a[:,1]-a[:,3])/(a[:,1]+a[:,3]+1e-12)
                    w.writerow(["predicted_raw_map_sample",cid,len(a),*map(float,means),
                                float(np.mean(ndvi)),float(np.mean(ndwi))])
                else:w.writerow(["predicted_raw_map_sample",cid,0,"","","","","",""])
        ds=None
        feedback.pushInfo("Spectral QA appended: predicted raw map sample by class")

    def _write_raster_integrity(self, path, raw_path, final_path, conf_path, ref,
                                class_ids, rows, cols, te, test_pred, feedback):
        # Reopen after explicit GDAL close: prevents stale GeoTIFF caches.
        raw=gdal.Open(raw_path,gdal.GA_ReadOnly)
        final=gdal.Open(final_path,gdal.GA_ReadOnly)
        confidence=gdal.Open(conf_path,gdal.GA_ReadOnly)
        if any(ds is None for ds in (raw,final,confidence)):
            raise QgsProcessingException("Raster QA: output raster cannot be reopened")
        expected=(ref.RasterXSize,ref.RasterYSize)
        if any((ds.RasterXSize,ds.RasterYSize)!=expected for ds in (raw,final,confidence)):
            raise QgsProcessingException("Raster QA: raster dimensions mismatch")
        counts_raw={int(c):0 for c in class_ids};counts_final={int(c):0 for c in class_ids}
        raw_nodata=0;final_nodata=0;confidence_invalid=0
        step=512
        for yo in range(0,expected[1],step):
            for xo in range(0,expected[0],step):
                ww=min(step,expected[0]-xo);hh=min(step,expected[1]-yo)
                a=raw.GetRasterBand(1).ReadAsArray(xo,yo,ww,hh)
                b=final.GetRasterBand(1).ReadAsArray(xo,yo,ww,hh)
                c=confidence.GetRasterBand(1).ReadAsArray(xo,yo,ww,hh)
                raw_nodata+=int(np.count_nonzero(a==0))
                final_nodata+=int(np.count_nonzero(b==0))
                confidence_invalid+=int(np.count_nonzero(~np.isfinite(c) | (c < 0) | (c > 1)))
                for cid in class_ids:
                    counts_raw[int(cid)]+=int(np.count_nonzero(a==cid))
                    counts_final[int(cid)]+=int(np.count_nonzero(b==cid))
        # Verify held-out model predictions against exact pixel positions on production map.
        match=0;compared=0;raw_band=raw.GetRasterBand(1)
        for idx,pred in zip(te,test_pred):
            rr=int(rows[idx]);cc=int(cols[idx]);value=int(raw_band.ReadAsArray(cc,rr,1,1)[0,0])
            if value!=0:
                compared+=1;match+=int(value==int(pred))
        agreement=match/compared if compared else float('nan')
        entries=[("raster_width",expected[0]),("raster_height",expected[1]),
                 ("raw_valid_pixels",sum(counts_raw.values())),
                 ("final_valid_pixels",sum(counts_final.values())),
                 ("raw_nodata_pixels",raw_nodata),("final_nodata_pixels",final_nodata),
                 ("confidence_invalid_pixels",confidence_invalid),
                 ("heldout_pixels_compared",compared),
                 ("heldout_raw_raster_prediction_agreement",agreement)]
        for cid in class_ids:
            entries.extend((("raw_class_%s_pixels"%cid,counts_raw[int(cid)]),
                            ("final_class_%s_pixels"%cid,counts_final[int(cid)])))
        with open(path,"w",newline="",encoding="utf-8-sig") as f:
            w=csv.writer(f);w.writerow(["metric","value"]);w.writerows(entries)
        raw=None;final=None;confidence=None
        if sum(counts_raw.values())==0 or sum(counts_final.values())==0:
            raise QgsProcessingException("Raster QA FAILED: no classified pixels; inspect raster inference and post-filter")
        if compared==0 or agreement<0.999:
            raise QgsProcessingException("Raster QA FAILED: direct held-out predictions differ from production raster (agreement=%s); check patch center/halo/band order"%agreement)
        feedback.pushInfo("Raster QA PASS: valid raw=%d, final=%d, held-out inference agreement=%.5f"%
                          (sum(counts_raw.values()),sum(counts_final.values()),agreement))

    def _v23_same_grid(self, a, b, tol=1e-7):
        return (a.RasterXSize == b.RasterXSize and a.RasterYSize == b.RasterYSize and
                a.GetProjection() == b.GetProjection() and
                np.allclose(np.asarray(a.GetGeoTransform()), np.asarray(b.GetGeoTransform()), rtol=0, atol=tol))

    def _v23_align_inputs(self, paths, auto_align, resampling, qc_path, feedback):
        master = gdal.Open(paths[0], gdal.GA_ReadOnly)
        gt = master.GetGeoTransform()
        W, H = master.RasterXSize, master.RasterYSize
        xmin, ymax = gt[0], gt[3]
        xmax, ymin = gt[0] + W * gt[1] + H * gt[2], gt[3] + W * gt[4] + H * gt[5]
        work = [paths[0]]
        rows = []
        tmp = tempfile.mkdtemp(prefix="lulc_v244_align_")
        for i, path in enumerate(paths):
            ds = gdal.Open(path, gdal.GA_ReadOnly)
            same = self._v23_same_grid(master, ds)
            action = "MASTER" if i == 0 else ("KEEP" if same else "MISMATCH")
            wp = path
            if i > 0 and not same:
                if not auto_align:
                    raise QgsProcessingException(f"Raster #{i+1} grid differs from master. Enable Auto-align.")
                wp = os.path.join(tmp, f"aligned_{i+1}.tif")
                alg = ["bilinear", "near", "cubic"][resampling]
                out = gdal.Warp(wp, path, format="GTiff", dstSRS=master.GetProjection(),
                    outputBounds=(min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax)),
                    width=W, height=H, resampleAlg=alg, creationOptions=["TILED=YES", "COMPRESS=LZW"])
                out.SetGeoTransform(master.GetGeoTransform())
                out.SetProjection(master.GetProjection())
                out = None
                action = "AUTO_ALIGNED"
                work.append(wp)
            elif i > 0:
                work.append(path)
            g = ds.GetGeoTransform()
            rows.append([i + 1, path, ds.RasterXSize, ds.RasterYSize, g[0], g[3], action, wp])
            ds = None
        with open(qc_path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['raster_no', 'source', 'width', 'height', 'origin_x', 'origin_y', 'action', 'working_raster'])
            w.writerows(rows)
        return work

    def processAlgorithm(self, parameters, context, feedback):
        self._active_tf_model = None
        self._tf_models = []
        profile_timers = {}
        t_start_total = time.time()

        try:
            # Directory mode: rewrite destination parameters before QGIS resolves outputs.
            # Keep all existing output names and return paths; do not touch scientific core.
            _directory = self.parameterAsString(parameters, self.OUTPUT_DIRECTORY, context).strip()
            if _directory and _directory != 'TEMPORARY_OUTPUT':
                _directory = os.path.abspath(os.path.expanduser(_directory))
                os.makedirs(_directory, exist_ok=True)
                parameters = dict(parameters)
                _raster_keys = {self.OUTPUT_RAW_CLASS,self.OUTPUT_CLASS,self.OUTPUT_CONF,
                    self.OUTPUT_ENTROPY,self.OUTPUT_MARGIN,self.OUTPUT_OOD,self.OUTPUT_TRUST}
                _json_keys = {self.OUTPUT_FEATURE_MANIFEST,self.OUTPUT_CLASS_MANIFEST}
                _png_keys = {self.OUTPUT_SHAP_BEESWARM,self.OUTPUT_SHAP_BAR}
                for _output_key in tuple(parameters):
                    if not _output_key.startswith('OUTPUT_') or _output_key in (self.OUTPUT_DIRECTORY,self.OUTPUT_MODEL_DIR):
                        continue
                    _extension = ('.tif' if _output_key in _raster_keys else
                        '.json' if _output_key in _json_keys else
                        '.png' if _output_key in _png_keys else '.csv')
                    parameters[_output_key] = os.path.join(_directory,_output_key+_extension)
                parameters[self.OUTPUT_MODEL_DIR] = os.path.join(_directory,'models')
                feedback.pushInfo('Unified OUTPUT_DIRECTORY: '+_directory)
            rasters = self.parameterAsLayerList(parameters, self.RASTERS, context)
            feature_names_text = self.parameterAsString(parameters, self.FEATURE_NAMES, context)
            training = self.parameterAsVectorLayer(parameters, self.TRAINING, context)
            class_field = self.parameterAsString(parameters, self.CLASS_FIELD, context)
            validation_method = self.parameterAsEnum(parameters, self.VALIDATION_METHOD, context)
            test_size = self.parameterAsDouble(parameters, self.TEST_SIZE, context)
            holdout_size = self.parameterAsDouble(parameters, self.HOLDOUT_TEST_SIZE, context)
            block_size = self.parameterAsDouble(parameters, self.BLOCK_SIZE, context)
            model_mode = self.parameterAsEnum(parameters, self.MODEL_MODE, context)
            
            use_spatial_cnn = self.parameterAsBoolean(parameters, self.USE_SPATIAL_CNN, context)
            patch_size = self.parameterAsInt(parameters, self.PATCH_SIZE, context)
            use_temporal_cnn = self.parameterAsBoolean(parameters, self.USE_TEMPORAL_CNN, context)
            temporal_steps = self.parameterAsInt(parameters, self.TEMPORAL_STEPS, context)

            n_trees = self.parameterAsInt(parameters, self.N_TREES, context)
            max_depth = self.parameterAsInt(parameters, self.MAX_DEPTH, context)
            epochs = self.parameterAsInt(parameters, self.DL_EPOCHS, context)
            batch = self.parameterAsInt(parameters, self.DL_BATCH, context)
            ml_infer_batch = self.parameterAsInt(parameters, self.ML_INFERENCE_BATCH_SIZE, context)
            ml_adaptive = self.parameterAsBoolean(parameters, self.ML_ADAPTIVE_BATCH, context)
            ml_memory_reserve_mb = self.parameterAsInt(parameters, self.ML_MEMORY_RESERVE_MB, context)
            balance = self.parameterAsBoolean(parameters, self.BALANCE, context)
            standardize = self.parameterAsBoolean(parameters, self.STANDARDIZE, context)
            do_perm = self.parameterAsBoolean(parameters, self.PERMUTATION_IMPORTANCE, context)
            perm_repeats = self.parameterAsInt(parameters, self.PERM_REPEATS, context)
            feature_top_k = self.parameterAsInt(parameters, self.FEATURE_SELECTION_TOP_K, context)
            run_feature_ablation = self.parameterAsBoolean(parameters, self.RUN_FEATURE_ABLATION, context)
            apply_filter = self.parameterAsBoolean(parameters, self.APPLY_FILTER, context)
            filter_method = self.parameterAsEnum(parameters, self.FILTER_METHOD, context)
            majority_size_idx = self.parameterAsEnum(parameters, self.MAJORITY_SIZE, context)
            min_patch_pixels = self.parameterAsInt(parameters, self.MIN_PATCH_PIXELS, context)
            conf_threshold = self.parameterAsDouble(parameters, self.CONF_THRESHOLD, context)
            io_block_size = [256, 512, 1024][self.parameterAsEnum(parameters, self.IO_BLOCK_SIZE, context)]
            max_train_pixels = self.parameterAsInt(parameters, self.MAX_TRAIN_PIXELS, context)
            auto_align = self.parameterAsBoolean(parameters, self.AUTO_ALIGN, context)
            align_resampling = self.parameterAsEnum(parameters, self.ALIGN_RESAMPLING, context)

            out_raw_class = self.parameterAsOutputLayer(parameters, self.OUTPUT_RAW_CLASS, context)
            out_class = self.parameterAsOutputLayer(parameters, self.OUTPUT_CLASS, context)
            out_conf = self.parameterAsOutputLayer(parameters, self.OUTPUT_CONF, context)
            _run_xai = self.parameterAsBoolean(parameters, self.RUN_XAI, context)
            _shap_sample_count = self.parameterAsInt(parameters, self.SHAP_SAMPLE_COUNT, context)
            _shap_target_class = self.parameterAsString(parameters, self.SHAP_TARGET_CLASS, context).strip()
            _shap_color_mode = ["PATCH_MEAN", "CENTER_PIXEL", "PATCH_MEDIAN", "PATCH_STD"][
                self.parameterAsEnum(parameters, self.SHAP_FEATURE_VALUE_MODE, context)]
            _xai_paths = {key: self.parameterAsOutputLayer(parameters,key,context) if key in (self.OUTPUT_OOD,self.OUTPUT_TRUST)
                          else self.parameterAsFileOutput(parameters,key,context)
                          for key in (self.OUTPUT_OOD,self.OUTPUT_TRUST,self.OUTPUT_SHAP,self.OUTPUT_SPATIAL_IMPORTANCE,self.OUTPUT_XAI_AUDIT,
                                      self.OUTPUT_SHAP_SAMPLES,self.OUTPUT_SHAP_CLASSWISE,
                                      self.OUTPUT_SHAP_BEESWARM,self.OUTPUT_SHAP_BAR)}
            out_entropy = self.parameterAsOutputLayer(parameters, self.OUTPUT_ENTROPY, context)
            out_margin = self.parameterAsOutputLayer(parameters, self.OUTPUT_MARGIN, context)
            out_reliability = self.parameterAsFileOutput(parameters, self.OUTPUT_RELIABILITY, context)
            out_reliability_summary = self.parameterAsFileOutput(parameters, self.OUTPUT_RELIABILITY_SUMMARY, context)
            # QGIS Processing scripts are not necessarily on sys.path.
            # Load the bundled uncertainty module from the script directory.
            import importlib.util as _unc_importlib
            _unc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sage_uncertainty_v255.py')
            if not os.path.isfile(_unc_path):
                raise QgsProcessingException(
                    'Missing sage_uncertainty_v255.py beside SAGE_GeoClass_V2_5_5_3.py: ' + _unc_path
                )
            _unc_spec = _unc_importlib.spec_from_file_location('sage_uncertainty_v255', _unc_path)
            _unc = _unc_importlib.module_from_spec(_unc_spec)
            _unc_spec.loader.exec_module(_unc)
            out_metrics = self.parameterAsFileOutput(parameters, self.OUTPUT_METRICS, context)
            out_cm = self.parameterAsFileOutput(parameters, self.OUTPUT_CM, context)
            out_classes = self.parameterAsFileOutput(parameters, self.OUTPUT_CLASSES, context)
            out_compare = self.parameterAsFileOutput(parameters, self.OUTPUT_MODEL_COMPARISON, context)
            out_importance = self.parameterAsFileOutput(parameters, self.OUTPUT_IMPORTANCE, context)
            out_feature_ablation = self.parameterAsFileOutput(parameters, self.OUTPUT_FEATURE_ABLATION, context)
            out_split = self.parameterAsFileOutput(parameters, self.OUTPUT_SPLIT, context)
            out_area_stats = self.parameterAsFileOutput(parameters, self.OUTPUT_AREA_STATS, context)
            out_class_accuracy = self.parameterAsFileOutput(parameters, self.OUTPUT_CLASS_ACCURACY, context)
            out_grid_qc = self.parameterAsFileOutput(parameters, self.OUTPUT_GRID_QC, context)
            out_cv = self.parameterAsFileOutput(parameters, self.OUTPUT_CV, context)
            out_cv_per_class = self.parameterAsFileOutput(parameters, self.OUTPUT_CV_PER_CLASS, context)
            out_train_qc = self.parameterAsFileOutput(parameters, self.OUTPUT_TRAIN_QC, context)
            out_profiling = self.parameterAsFileOutput(parameters, self.OUTPUT_PROFILING, context)
            out_inference_benchmark = self.parameterAsFileOutput(parameters, self.OUTPUT_INFERENCE_BENCHMARK, context)
            out_controlled_benchmark = self.parameterAsFileOutput(parameters, self.OUTPUT_CONTROLLED_BENCHMARK, context)
            run_controlled_benchmark = self.parameterAsBoolean(parameters, self.RUN_CONTROLLED_BENCHMARK, context)
            out_leakage_qc = self.parameterAsFileOutput(parameters, self.OUTPUT_LEAKAGE_QC, context)
            out_flood_metrics = self.parameterAsFileOutput(parameters, self.OUTPUT_FLOOD_METRICS, context)
            out_flood_compare = self.parameterAsFileOutput(parameters, self.OUTPUT_FLOOD_MODEL_COMPARISON, context)
            out_target_compare = self.parameterAsFileOutput(parameters, self.OUTPUT_TARGET_MODEL_COMPARISON, context)

            run_cv = self.parameterAsBoolean(parameters, self.RUN_SPATIAL_CV, context)
            requested_folds = self.parameterAsInt(parameters, self.CV_FOLDS, context)
            tune_hyperparams = self.parameterAsBoolean(parameters, self.TUNE_HYPERPARAMS, context)
            tune_iter = self.parameterAsInt(parameters, self.TUNE_ITER, context)
            random_seed = self.parameterAsInt(parameters, self.RANDOM_SEED, context)
            
            flood_objective = self.parameterAsEnum(parameters, self.FLOOD_OBJECTIVE, context)
            objective = self.parameterAsEnum(parameters, self.MODEL_OBJECTIVE, context)
            target_label = self.parameterAsString(parameters, self.TARGET_CLASS, context).strip()
            min_target_precision = self.parameterAsDouble(parameters, self.MIN_TARGET_PRECISION, context)
            min_target_recall = self.parameterAsDouble(parameters, self.MIN_TARGET_RECALL, context)
            model_dir = self.parameterAsString(parameters, self.OUTPUT_MODEL_DIR, context)
            out_test_metrics=self.parameterAsFileOutput(parameters,self.OUTPUT_TEST_METRICS,context)
            out_test_cm=self.parameterAsFileOutput(parameters,self.OUTPUT_TEST_CM,context)
            out_test_per_class=self.parameterAsFileOutput(parameters,self.OUTPUT_TEST_PER_CLASS,context)
            out_raster_qa=self.parameterAsFileOutput(parameters,self.OUTPUT_RASTER_QA,context)
            out_spectral_qa=self.parameterAsFileOutput(parameters,self.OUTPUT_SPECTRAL_QA,context)
            out_class_legend=self.parameterAsFileOutput(parameters,self.OUTPUT_CLASS_LEGEND,context)

            if tune_hyperparams:
                feedback.pushWarning("TUNE_HYPERPARAMS requested but not implemented; no tuning is performed.")
            if not rasters:
                raise QgsProcessingException("At least one raster feature is required.")
            if not model_dir or model_dir == "TEMPORARY_OUTPUT":
                model_dir = os.path.join(tempfile.gettempdir(), "LandUse_V2444_Models")
            os.makedirs(model_dir, exist_ok=True)
            for value in parameters.values():
                if isinstance(value,str) and value.lower().endswith((".csv",".tif")):
                    os.makedirs(os.path.dirname(os.path.abspath(value)),exist_ok=True)

            feedback.pushInfo("=== SAGE-GeoClass V2.5.5.4.3.5 — General-Purpose Geospatial Classification ===")

            t0 = time.time()
            source_paths = [r.source() for r in rasters]
            working_paths = self._v23_align_inputs(source_paths, auto_align, align_resampling, out_grid_qc, feedback)
            raster_dss = [gdal.Open(p, gdal.GA_ReadOnly) for p in working_paths]
            ref = raster_dss[0]
            profile_timers["raster_alignment_and_qc"] = time.time() - t0

            # V2.5.1: validate semantic feature metadata without changing the feature array.
            # QGIS Processing Script Editor may execute scripts without __file__ and
            # without adding the script directory to sys.path. Resolve explicitly.
            import importlib.util
            from qgis.core import QgsApplication
            registry_filename = 'sage_feature_registry_v251.py'
            script_file = globals().get('__file__', '')
            search_dirs = []
            if script_file:
                search_dirs.append(os.path.dirname(os.path.abspath(script_file)))
            search_dirs.append(os.path.join(QgsApplication.qgisSettingsDirPath(), 'processing', 'scripts'))
            search_dirs.append(os.path.join(os.path.expanduser('~'), 'AppData', 'Roaming', 'QGIS', 'QGIS4', 'profiles', 'default', 'processing', 'scripts'))
            search_dirs.extend(p for p in sys.path if p)
            search_dirs.append(os.getcwd())
            registry_path = next((os.path.join(d, registry_filename) for d in dict.fromkeys(search_dirs)
                                  if os.path.isfile(os.path.join(d, registry_filename))), None)
            if registry_path is None:
                raise QgsProcessingException('Feature Registry file not found: ' + registry_filename +
                    '\nSearched directories:\n' + '\n'.join(dict.fromkeys(search_dirs)))
            try:
                spec = importlib.util.spec_from_file_location('sage_feature_registry_v251', registry_path)
                if spec is None or spec.loader is None:
                    raise ImportError('Could not create module spec')
                registry_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(registry_module)
                build_registry = registry_module.build_registry
                feedback.pushInfo('Feature Registry loaded: ' + registry_path)
            except Exception as exc:
                raise QgsProcessingException('Feature Registry import failed at ' + registry_path +
                    ': ' + repr(exc)) from exc
            sources = []
            for ds, original in zip(raster_dss, source_paths):
                sources.append({'path':original,'bands':[
                    {'nodata':ds.GetRasterBand(b).GetNoDataValue(),
                     'description':ds.GetRasterBand(b).GetDescription()}
                    for b in range(1,ds.RasterCount+1)]})
            try:
                manifest=build_registry(sources,self.parameterAsString(parameters,self.FEATURE_SCHEMA_JSON,context))
            except (ValueError,TypeError,KeyError) as exc:
                raise QgsProcessingException('Feature Registry invalid: '+str(exc)) from exc
            out_feature_manifest=self.parameterAsFileOutput(parameters,self.OUTPUT_FEATURE_MANIFEST,context)
            os.makedirs(os.path.dirname(os.path.abspath(out_feature_manifest)),exist_ok=True)
            with open(out_feature_manifest,'w',encoding='utf-8') as fh:
                json.dump(manifest,fh,ensure_ascii=False,indent=2)
            feedback.pushInfo('SAGE Feature Registry: %d features, SHA256=%s' %
                              (len(manifest['features']),manifest['schema_sha256'][:16]))
            if manifest['spectral_indices']:
                feedback.pushInfo('Registry spectral roles B,G,R,NIR: '+str(manifest['spectral_indices']))

            # General-purpose feature stack: preserve source order and every band.
            # Spectral indices are optional diagnostics, never additional model inputs.
            spec_text = self.parameterAsString(parameters, self.SPECTRAL_BAND_INDICES, context).strip()
            self._sage_spectral_roles = None
            if not spec_text and manifest['spectral_indices']:
                spec_text=','.join(map(str,manifest['spectral_indices']))
            if spec_text:
                try:
                    roles = [int(v.strip()) - 1 for v in spec_text.split(",")]
                except ValueError:
                    raise QgsProcessingException("Spectral roles must be four comma-separated 1-based integers")
                if len(roles) != 4 or len(set(roles)) != 4 or min(roles) < 0 or max(roles) >= sum(ds.RasterCount for ds in raster_dss):
                    raise QgsProcessingException("Spectral roles must reference four distinct valid global feature indices")
                self._sage_spectral_roles = roles
                feedback.pushInfo("SAGE spectral roles B,G,R,NIR (1-based): " + spec_text)
            if (self.parameterAsString(parameters,self.CLASS_SCHEMA_JSON,context).strip() and
                self.parameterAsString(parameters,self.CLASS_LEGEND_JSON,context).strip()):
                raise QgsProcessingException('Specify CLASS_SCHEMA_JSON or legacy CLASS_LEGEND_JSON, not both')
            legend_text = self.parameterAsString(parameters, self.CLASS_LEGEND_JSON, context).strip()
            self._sage_class_legend = {}
            if legend_text:
                try:
                    parsed = json.loads(legend_text)
                    if not isinstance(parsed, dict): raise ValueError("Expected JSON object")
                    for k,v in parsed.items():
                        if not isinstance(v, dict): raise ValueError("Each class must map to an object")
                        color = str(v.get("color", "#808080"))
                        if not re.fullmatch(r"#[0-9a-fA-F]{6}", color): raise ValueError("Invalid hex color")
                        self._sage_class_legend[int(k)] = (str(v.get("name",k)), color)
                except (ValueError, TypeError, KeyError) as e:
                    raise QgsProcessingException("Invalid CLASS_LEGEND_JSON: " + str(e))
            legacy_legend = dict(self._sage_class_legend)
            predictor_bands = [(ds, bi) for ds in raster_dss for bi in range(1, ds.RasterCount + 1)]
            feature_names = [f"{layer.name()}_Band{bi}" for layer, ds in zip(rasters, raster_dss) for bi in range(1, ds.RasterCount + 1)]
            
            out_feature_qc = self.parameterAsFileOutput(parameters, self.OUTPUT_FEATURE_QC, context)
            with open(out_feature_qc, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["feature", "raster", "band", "nodata"])
                for name, (ds, bi) in zip(feature_names, predictor_bands):
                    w.writerow([name, ds.GetDescription(), bi, ds.GetRasterBand(bi).GetNoDataValue()])

            # V2.5.5: strict, domain-agnostic class mapping and metadata.
            import importlib.util
            class_filename = 'sage_class_schema_v252.py'
            class_path = next((os.path.join(d, class_filename) for d in dict.fromkeys(search_dirs)
                               if os.path.isfile(os.path.join(d, class_filename))), None)
            if class_path is None:
                raise QgsProcessingException('Class Schema module not found: ' + class_filename +
                    '\nSearched directories:\n' + '\n'.join(dict.fromkeys(search_dirs)))
            try:
                class_spec = importlib.util.spec_from_file_location('sage_class_schema_v252', class_path)
                class_mod = importlib.util.module_from_spec(class_spec)
                class_spec.loader.exec_module(class_mod)
                observed_values = [f[class_field] for f in training.getFeatures()]
                class_schema = class_mod.build_class_schema(
                    observed_values, self.parameterAsString(parameters,self.CLASS_SCHEMA_JSON,context),
                    self.parameterAsInt(parameters,self.MIN_CLASS_SAMPLES,context))
            except Exception as exc:
                raise QgsProcessingException('Class Schema invalid: ' + str(exc)) from exc
            out_class_manifest = self.parameterAsFileOutput(parameters,self.OUTPUT_CLASS_MANIFEST,context)
            out_class_qc = self.parameterAsFileOutput(parameters,self.OUTPUT_CLASS_QC,context)
            os.makedirs(os.path.dirname(os.path.abspath(out_class_manifest)),exist_ok=True)
            with open(out_class_manifest,'w',encoding='utf-8') as fh:
                json.dump(class_schema,fh,ensure_ascii=False,indent=2)
            self._sage_class_legend = {e['dn']:(e['name'],e['color']) for e in class_schema['classes']}
            if legend_text:
                self._sage_class_legend.update(legacy_legend)
            feedback.pushInfo('Universal Class Schema: %d classes, SHA256=%s' %
                              (class_schema['class_count'],class_schema['schema_sha256'][:16]))
            cmap = {e['value']:e['dn'] for e in class_schema['classes']}
            inv = {v: k for k, v in cmap.items()}
            class_ids = sorted(inv.keys())
            if class_ids != list(range(1,len(class_ids)+1)):
                raise QgsProcessingException("Class IDs must be contiguous 1..K")
            
            target_dn = cmap.get(target_label)
            if target_dn is None and target_label.startswith("DN:") and target_label[3:].isdigit():
                target_dn = int(target_label[3:])

            with open(out_classes, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["class_id", "class_label", "display_name", "color", "description"])
                for cid in class_ids:
                    e=class_schema['classes'][cid-1]
                    w.writerow([cid, inv[cid],e['name'],e['color'],e['description']])

            t0 = time.time()
            temp_vec = self._make_numeric_training(training, class_field, cmap)
            class_ds = self._rasterize_attribute(temp_vec, ref, "__class_id")
            group_ds = self._rasterize_attribute(temp_vec, ref, "__group_id")

            X, X_patch, y, groups, sample_rows, sample_cols = self._extract_samples_spatial_temporal(
                raster_dss, class_ds, group_ds, ref,
                validation_method, block_size, feedback,
                use_spatial_cnn, patch_size, use_temporal_cnn, temporal_steps,
                io_block_size, max_train_pixels
            )
            profile_timers["sample_extraction"] = time.time() - t0

            class_qc_rows = class_mod.class_qc(class_schema, y, groups,
                self.parameterAsInt(parameters,self.MIN_CLASS_SAMPLES,context))
            with open(out_class_qc, 'w', newline='', encoding='utf-8-sig') as f:
                w=csv.DictWriter(f, fieldnames=['dn','value','name','sample_pixels','independent_groups','status','warnings'])
                w.writeheader(); w.writerows(class_qc_rows)
            for row in class_qc_rows:
                if row['status'] != 'PASS':
                    feedback.pushWarning('Class QC %s (%s): %s' % (row['value'],row['name'],row['warnings']))

            with open(out_train_qc, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["class_id", "class_label", "pixels", "independent_groups"])
                for cid in class_ids:
                    mask = y == cid
                    w.writerow([cid, inv[cid], int(mask.sum()), len(np.unique(groups[mask]))])

            # Spatially independent three-way split. Validation/test are preserved;
            # purge only training centers, then verify ALL sets remain usable.
            from sklearn.model_selection import GroupShuffleSplit, train_test_split
            all_labels = set(np.unique(y).tolist())
            chosen = None
            best_min = -1
            for seed in (42,7,17,23,101,2026,99,1234,987,73,991,2027,333,51,79,117):
                try:
                    if validation_method == 2:
                        idx=np.arange(len(y))
                        trval,te=train_test_split(idx,test_size=holdout_size,random_state=seed,stratify=y)
                        tr,va=train_test_split(trval,test_size=test_size,random_state=seed+1,stratify=y[trval])
                    else:
                        outer=GroupShuffleSplit(n_splits=1,test_size=holdout_size,random_state=seed)
                        trval,te=next(outer.split(X,y,groups))
                        inner=GroupShuffleSplit(n_splits=1,test_size=test_size,random_state=seed+1)
                        il,iv=next(inner.split(X[trval],y[trval],groups[trval]))
                        tr,va=trval[il],trval[iv]
                    initial_train=len(tr)
                    if use_spatial_cnn:
                        tr,p1=self._purge_patch_overlap(tr,np.concatenate((va,te)),sample_rows,sample_cols,patch_size//2)
                        va,p2=self._purge_patch_overlap(va,te,sample_rows,sample_cols,patch_size//2)
                    else:p1=p2=0
                    if min(len(tr),len(va),len(te))<2:continue
                    if set(np.unique(y[tr]).tolist()) != all_labels:continue
                    if len(tr)<max(10,len(all_labels)*2):continue
                    score=min(len(tr),len(va),len(te))
                    if score>best_min:
                        chosen=(tr.copy(),va.copy(),te.copy(),p1,p2,seed,initial_train)
                        best_min=score
                except (ValueError,StopIteration):
                    continue
            if chosen is None:
                raise QgsProcessingException(
                    "Scientific validation FAILED: no independent train/validation/test split "
                    "survived patch overlap purging. Add geographically separated polygons "
                    "for every class, reduce PATCH_SIZE, or use a larger study area. "
                    "Do not report CNN accuracy with empty validation.")
            tr,va,te,p1,p2,chosen_seed,initial_train=chosen
            feedback.pushInfo(f"Independent split seed={chosen_seed}: train={len(tr)}, validation={len(va)}, test={len(te)}")
            feedback.pushInfo(f"Patch overlap removed: train={p1}/{initial_train}, validation={p2}; all sets nonempty")
            if validation_method==2:
                feedback.pushWarning("Random pixel split is not spatially independent; use polygon/group split for research")
            Xtr,Xva=X[tr],X[va];ytr,yva=y[tr],y[va]
            Xtr_patch=X_patch[tr] if X_patch is not None else None
            Xva_patch=X_patch[va] if X_patch is not None else None
            self._check_spatial_leakage(groups,tr,va,te,sample_rows,sample_cols,patch_size,use_spatial_cnn,out_leakage_qc,feedback)
            with open(out_split,"w",newline="",encoding="utf-8-sig") as f:
                w=csv.writer(f);w.writerow(["split","samples"])
                for label,ids in (("train",tr),("validation",va),("test",te)):w.writerow([label,len(ids)])

            self._write_class_legend(out_class_legend, class_ids)
            self._write_spectral_qa(out_spectral_qa, X, y, tr, te, class_ids, len(predictor_bands), feedback)
            if len(predictor_bands)==4:
                feedback.pushInfo("Spectral QA assumes raster bands 1=Blue, 2=Green, 3=Red, 4=NIR; check OUTPUT_SPECTRAL_QA")

            t0 = time.time()
            candidates = [
                ("RF", "Random Forest"),
                ("ExtraTrees", "Extra Trees"),
                ("HistGB", "HistGradientBoosting"),
                ("SVM", "SVM (RBF)"),
                ("MLP", "MLP Neural Network"),
                ("XGBoost", "XGBoost (gradient-boosted trees)"),
                ("LightGBM", "LightGBM"),
                ("CatBoost", "CatBoost"),
                ("LogReg", "Logistic Regression")
            ]
            if use_spatial_cnn:
                candidates.append(("SpatialCNN", "Spatial 2D-CNN"))
            if use_temporal_cnn and temporal_steps>1:
                candidates.append(("TemporalCNN", "Temporal CNN"))
            candidates.append(("DNN", "Deep Neural Network (TensorFlow)"))

            if model_mode == 6 and not use_spatial_cnn: raise QgsProcessingException("Enable Spatial CNN for this model")
            if model_mode == 7 and (not use_temporal_cnn or temporal_steps<2): raise QgsProcessingException("Enable Temporal CNN and >=2 dates")
            if use_temporal_cnn and temporal_steps>1 and X.ndim!=3:raise QgsProcessingException("Invalid temporal tensor")
            if model_mode > 0:
                mode_map = {1: "RF", 2: "ExtraTrees", 3: "HistGB", 4: "SVM", 5: "MLP", 6: "SpatialCNN", 7: "TemporalCNN", 8: "DNN", 9: "XGBoost", 10: "LightGBM", 11: "CatBoost", 12: "LogReg"}
                target_key = mode_map.get(model_mode)
                candidates = [c for c in candidates if c[0] == target_key]

            if X.ndim==3:
                # ML and DNN use flattened spectral-temporal vectors, CNN uses 3D tensor.
                Xtr_ml=Xtr.reshape(len(Xtr),-1);Xva_ml=Xva.reshape(len(Xva),-1)
            else:Xtr_ml,Xva_ml=Xtr,Xva
            # Load shared V2.5.5 selection/diagnostics module without modifying the worker.
            import importlib.util as _mf_importlib
            _mf_path = os.path.join(os.path.dirname(class_mod.__file__), 'sage_model_feature_v254.py')
            if not os.path.isfile(_mf_path):
                raise QgsProcessingException('Missing sage_model_feature_v254.py: ' + _mf_path)
            _mf_spec = _mf_importlib.spec_from_file_location('sage_model_feature_v254', _mf_path)
            _mf = _mf_importlib.module_from_spec(_mf_spec)
            _mf_spec.loader.exec_module(_mf)
            self._sage_feature_top_k = feature_top_k if 0 < feature_top_k < Xtr_ml.shape[1] else 0
            self._sage_feature_seed = random_seed
            if feature_top_k > 0 and self._sage_feature_top_k == 0:
                feedback.pushWarning('Feature top K >= available features; retaining all features')
            if self._sage_feature_top_k:
                feedback.pushInfo('ML feature selection: train-only embedded ExtraTrees top %d / %d; refitted within CV folds' %
                                  (self._sage_feature_top_k, Xtr_ml.shape[1]))
                if use_spatial_cnn or use_temporal_cnn:
                    feedback.pushWarning('CNN/DNN use full features; Top K selection applies to pixel-based ML only')
            comparison = []
            best_model, best_scaler, best_pred, best_metrics = None, None, None, None
            best_name, best_key, best_score = None, None, -1e99

            for key, display in candidates:
                if feedback.isCanceled():
                    break
                feedback.pushInfo(f"Training and Evaluating: {display}...")
                try:
                    scaler = None
                    if key == "SpatialCNN":
                        model, pred = self._train_cnn_spatial(Xtr_patch, ytr, Xva_patch, yva, len(class_ids), epochs, batch, model_dir, feedback)
                    elif key == "TemporalCNN":
                        model, pred = self._train_cnn_temporal(Xtr, ytr, Xva, yva, len(class_ids), epochs, batch, model_dir, feedback)
                    elif key == "DNN":
                        model, scaler, pred = self._train_dnn(Xtr_ml, ytr, Xva_ml, yva, len(class_ids), epochs, batch, standardize, model_dir, feedback)
                    else:
                        model = self._make_ml_model(key, n_trees, max_depth, balance, standardize)
                        if self._sage_feature_top_k:
                            model = _mf.wrap_ml_selector(model, self._sage_feature_top_k, random_seed)
                        model.fit(Xtr_ml, ytr)
                        pred = model.predict(Xva_ml)

                    m = self._metrics(yva, pred)
                    comparison.append({"model": display, "status": "OK", **m})
                    
                    from sklearn.metrics import precision_recall_fscore_support, jaccard_score
                    target_idx=target_dn if target_dn is not None else 1
                    tpv=precision_recall_fscore_support(yva,pred,labels=[target_idx],zero_division=0)
                    precision,recall,tf1=[float(z[0]) for z in tpv[:3]]
                    tiou=float(jaccard_score(yva==target_idx,pred==target_idx,zero_division=0))
                    scores=[m["macro_f1"],m["weighted_f1"],m["balanced_accuracy"],tf1,tiou,recall if precision>=min_target_precision else -1,precision if recall>=min_target_recall else -1]
                    score=scores[objective]
                    comparison[-1].update({"objective_score":score,"target_precision":precision,"target_recall":recall,"target_f1":tf1,"target_iou":tiou})
                    if score > best_score:
                        best_score = score
                        best_key = key
                        best_name = display
                        best_model = model
                        best_scaler = scaler
                        best_pred = pred
                        best_metrics = m
                except Exception as e:
                    comparison.append({"model": display, "status": f"FAILED: {str(e)}", "macro_f1": 0})
                    feedback.reportError(f"{display} failed: {str(e)}")

            profile_timers["model_training_and_comparison"] = time.time() - t0
            if best_model is None:
                raise QgsProcessingException("All candidate models failed training.")

            # V2.5.5: spatial GroupKFold only on the original training partition.
            # Validation is used for model selection, independent test remains untouched.
            # CV must never overwrite production TensorFlow model artifacts.
            if run_cv:
                import importlib.util as _cv_importlib
                _cv_path = os.path.join(os.path.dirname(class_mod.__file__), 'sage_spatial_cv_v253.py')
                if not os.path.isfile(_cv_path):
                    raise QgsProcessingException('Missing sage_spatial_cv_v253.py alongside sage_class_schema_v252.py: ' + _cv_path)
                _cv_spec = _cv_importlib.spec_from_file_location('sage_spatial_cv_v253', _cv_path)
                _cv_module = _cv_importlib.module_from_spec(_cv_spec)
                _cv_spec.loader.exec_module(_cv_module)
                run_spatial_cv = _cv_module.run_spatial_cv
                cv_rows, cv_class_rows, cv_summary = run_spatial_cv(
                    owner=self, key=best_key, display=best_name, X=X, X_patch=X_patch,
                    y=y, groups=groups, rows=sample_rows, cols=sample_cols,
                    train_indices=tr, n_classes=len(class_ids), class_ids=class_ids,
                    n_folds=requested_folds, patch_size=patch_size,
                    epochs=epochs, batch=batch, model_dir=model_dir,
                    n_trees=n_trees, max_depth=max_depth, balance=balance,
                    standardize=standardize, feedback=feedback, random_seed=random_seed)
                import csv as _cv_csv
                with open(out_cv,"w",newline="",encoding="utf-8-sig") as _f:
                    _fields=['model','fold','status','n_train','n_validation','n_train_groups',
                             'n_validation_groups','purged_train','accuracy','macro_f1',
                             'balanced_accuracy','patch_overlap_before','patch_overlap_after','group_overlap_after','missing_train_classes','missing_validation_classes','reason']
                    _w=_cv_csv.DictWriter(_f,fieldnames=_fields,extrasaction='ignore');_w.writeheader();_w.writerows(cv_rows)
                    for _metric in ('accuracy','macro_f1','balanced_accuracy'):
                        _w.writerow({'model':best_name,'fold':'MEAN','status':cv_summary['status'],
                                     _metric:cv_summary.get(_metric+'_mean',''),'reason':cv_summary.get('reason','')})
                        _w.writerow({'model':best_name,'fold':'STD','status':cv_summary['status'],
                                     _metric:cv_summary.get(_metric+'_std','')})
                with open(out_cv_per_class,"w",newline="",encoding="utf-8-sig") as _f:
                    _fields=['model','fold','class_id','support','precision','recall','f1','status']
                    _w=_cv_csv.DictWriter(_f,fieldnames=_fields,extrasaction='ignore');_w.writeheader();_w.writerows(cv_class_rows)
                if cv_summary['status'] != 'PASS':
                    feedback.pushWarning('Spatial CV SCIENTIFIC INTEGRITY: '+cv_summary['reason'])
                feedback.pushInfo('Spatial CV: %s; completed %s/%s folds; accuracy mean=%s std=%s; macro F1 mean=%s std=%s; balanced accuracy mean=%s std=%s' % (
                    cv_summary['status'],cv_summary['completed'],requested_folds,
                    cv_summary.get('accuracy_mean'),cv_summary.get('accuracy_std'),
                    cv_summary.get('macro_f1_mean'),cv_summary.get('macro_f1_std'),
                    cv_summary.get('balanced_accuracy_mean'),cv_summary.get('balanced_accuracy_std')))
            else:
                for _path in (out_cv,out_cv_per_class):
                    with open(_path,'w',newline='',encoding='utf-8-sig') as _f:
                        _w=csv.writer(_f);_w.writerow(['status','reason']);_w.writerow(['NOT_RUN','RUN_SPATIAL_CV disabled'])

            # Optional spatial ablation is an independent diagnostic on TRAIN only.
            # CNN/DNN deliberately skipped: cannot claim ablation without retraining.
            import importlib.util as _audit_importlib
            _audit_path = os.path.join(os.path.dirname(class_mod.__file__), 'sage_feature_audit_v2541.py')
            if run_feature_ablation and best_key not in ('SpatialCNN','TemporalCNN','DNN'):
                try:
                    _audit_spec = _audit_importlib.spec_from_file_location('sage_feature_audit_v2541',_audit_path)
                    _audit_mod = _audit_importlib.module_from_spec(_audit_spec)
                    _audit_spec.loader.exec_module(_audit_mod)
                    _audit_rows,_audit_info = _audit_mod.audit_ml(
                        self,best_key,X,y,groups,tr,class_ids,requested_folds,
                        feature_names,n_trees,max_depth,balance,standardize,
                        seed=random_seed,top_k=feature_top_k,feedback=feedback)
                    _audit_mod.write_audit(out_feature_ablation,_audit_rows)
                    feedback.pushInfo('V2.5.5 spatial ML ablation: %d rows; TRAIN-only' % len(_audit_rows))
                except Exception as _audit_error:
                    feedback.pushWarning('V2.5.5 ablation failed: '+str(_audit_error))
                    with open(out_feature_ablation,'w',newline='',encoding='utf-8-sig') as _af:
                        _aw=csv.writer(_af);_aw.writerow(['status','reason']);_aw.writerow(['FAILED',str(_audit_error)])
            else:
                _reason = ('CNN/DNN ablation requires per-fold retraining; not implemented' if run_feature_ablation
                           else 'RUN_FEATURE_ABLATION disabled')
                with open(out_feature_ablation,'w',newline='',encoding='utf-8-sig') as _af:
                    _aw=csv.writer(_af);_aw.writerow(['status','reason']);_aw.writerow(['NOT_RUN',_reason])
                if run_feature_ablation: feedback.pushWarning(_reason)

            # Diagnostic importance on the model-selection validation partition only.
            # Never use independent test labels to select features or models.
            if do_perm:
                try:
                    if best_key == 'SpatialCNN':
                        _importance_x = Xva_patch
                        _importance_names = feature_names
                    elif best_key == 'TemporalCNN':
                        _importance_x = Xva
                        _importance_names = [f'Channel_{i+1}' for i in range(Xva.shape[-1])]
                    else:
                        _importance_x = Xva_ml
                        _importance_names = feature_names if len(feature_names) == Xva_ml.shape[1] else [f'Feature_{i+1}' for i in range(Xva_ml.shape[1])]
                    _importance_rows = _mf.permutation_feature_importance(
                        best_model, best_key, _importance_x, yva, _importance_names,
                        len(class_ids), repeats=perm_repeats, seed=random_seed,
                        scaler=best_scaler, max_samples=2000)
                    _mf.write_importance(out_importance, _importance_rows)
                    feedback.pushInfo('V2.5.5 permutation importance: %d features, %d repeats, validation-only (diagnostic)' %
                                      (len(_importance_rows), perm_repeats))
                except Exception as _imp_err:
                    feedback.pushWarning('Feature importance failed: ' + str(_imp_err))
                    with open(out_importance,'w',newline='',encoding='utf-8-sig') as _imp_f:
                        _imp_w=csv.writer(_imp_f);_imp_w.writerow(['status','reason']);_imp_w.writerow(['FAILED',str(_imp_err)])
            else:
                with open(out_importance,'w',newline='',encoding='utf-8-sig') as _imp_f:
                    _imp_w=csv.writer(_imp_f);_imp_w.writerow(['status','reason']);_imp_w.writerow(['NOT_RUN','PERMUTATION_IMPORTANCE disabled'])

            feedback.pushInfo(f"Best Model Selected: {best_name} (objective score: {best_score:.4f})")
            if best_key not in ("SpatialCNN","TemporalCNN","DNN"):
                import joblib
                joblib.dump(best_model,os.path.join(model_dir,"selected_ml_model.joblib"))

            # Integrated optional diagnostics; never modify fitted model or original outputs.
            _xai_mod = None
            _xai_ref = None
            if _run_xai:
                import importlib.util as _xi
                _xp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sage_xai_universal_v25542.py')
                if not os.path.isfile(_xp):
                    raise QgsProcessingException('Missing sage_xai_universal_v25542.py')
                _xs = _xi.spec_from_file_location('sage_xai_universal_v25542', _xp)
                _xai_mod = _xi.module_from_spec(_xs); _xs.loader.exec_module(_xai_mod)
                # Preserve the established input-space OOD definition and TRAIN-only
                # reference for ML; DL additionally uses center-pixel / time-band
                # spectral vectors so the raster writer has exactly the same schema.
                _xai_ref = _xai_mod.fit_ood_reference(Xtr_ml, seed=random_seed)
                feedback.pushInfo('Universal XAI input-space OOD reference: TRAIN partition; model=%s' % best_key)
                _names = feature_names if len(feature_names)==Xva_ml.shape[1] else ['Feature_%d'%(i+1) for i in range(Xva_ml.shape[1])]
                _is_dl_xai = best_key in ('SpatialCNN','TemporalCNN','DNN')
                if _is_dl_xai:
                    _dlxp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sage_xai_dl_v25543.py')
                    if not os.path.isfile(_dlxp):
                        raise QgsProcessingException('Missing sage_xai_dl_v25543.py')
                    _dls = _xi.spec_from_file_location('sage_xai_dl_v25543', _dlxp)
                    _dl_xai = _xi.module_from_spec(_dls); _dls.loader.exec_module(_dl_xai)
                    _dl_train = Xtr_patch if best_key=='SpatialCNN' else (Xtr if best_key=='TemporalCNN' else Xtr_ml)
                    _dl_val = Xva_patch if best_key=='SpatialCNN' else (Xva if best_key=='TemporalCNN' else Xva_ml)
                _samples_path = _xai_paths.get(self.OUTPUT_SHAP_SAMPLES)
                _classwise_path = _xai_paths.get(self.OUTPUT_SHAP_CLASSWISE)
                _beeswarm_path = _xai_paths.get(self.OUTPUT_SHAP_BEESWARM)
                _bar_path = _xai_paths.get(self.OUTPUT_SHAP_BAR)
                for _key, _action in ((self.OUTPUT_SHAP,'shap'),(self.OUTPUT_SPATIAL_IMPORTANCE,'spatial')):
                    _path = _xai_paths.get(_key)
                    if not _path: continue
                    try:
                        if _is_dl_xai:
                            if _action=='shap':
                                _dl_xai.grouped_kernel_shap(best_model,_dl_train,_dl_val,_names,_path,
                                                            best_key,scaler=best_scaler,seed=random_seed,
                                                            sample_output=_samples_path, classwise_output=_classwise_path,
                                                            color_mode=_shap_color_mode,cap=_shap_sample_count)
                            else:
                                _dl_xai.grouped_spatial_importance(best_model,_dl_train,_dl_val,yva,groups[va],
                                                                  _names,_path,best_key,scaler=best_scaler,seed=random_seed)
                        elif _action=='shap':
                            _xai_mod.shap_summary(best_model,Xva_ml,_names,_path,model_key=best_key,seed=random_seed,
                                                  sample_output=_samples_path,classwise_output=_classwise_path,cap=_shap_sample_count)
                        else:
                            _xai_mod.spatial_group_importance(best_model,Xva_ml,yva,groups[va],_names,_path,seed=random_seed)
                    except Exception as _xe:
                        feedback.pushWarning('Optional %s diagnostic unavailable: %s'%(_action,_xe))
                        with open(_path,'w',newline='',encoding='utf-8-sig') as _f:
                            _w=csv.writer(_f);_w.writerow(['status','reason']);_w.writerow(['FAILED',str(_xe)])

                if _beeswarm_path or _bar_path:
                    try:
                        _viz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 'sage_shap_visualization_v255432.py')
                        _vzs = _xi.spec_from_file_location('sage_shap_visualization_v255432', _viz_path)
                        _vzm = _xi.module_from_spec(_vzs); _vzs.loader.exec_module(_vzm)
                        _vzm.make_plots(_samples_path, _xai_paths.get(self.OUTPUT_SHAP),
                                        _beeswarm_path, _bar_path,
                                        model_name=best_key, color_mode=_shap_color_mode, target_class=_shap_target_class)
                    except Exception as _plot_error:
                        feedback.pushWarning('Optional SHAP visualization unavailable: '+str(_plot_error))

            # Independent test is evaluated once after model selection.
            if best_key=="SpatialCNN":test_x=X_patch[te]
            elif best_key=="TemporalCNN":test_x=X[te]
            else:test_x=X[te].reshape(len(te),-1)
            if best_key in ("SpatialCNN","TemporalCNN","DNN"):
                if best_key=="DNN" and best_scaler is not None:test_x=best_scaler.transform(test_x)
                test_pred=np.argmax(best_model.predict(test_x,verbose=0),axis=1)+1
            else:
                # One probability pass is sufficient for test labels and reliability.
                # Constrain parallel RF probability inference inside the QGIS process.
                _test_jobs = getattr(best_model, 'n_jobs', None)
                if _test_jobs is not None: best_model.n_jobs = 1
                try:
                    _test_prob_raw = best_model.predict_proba(test_x) if hasattr(best_model, 'predict_proba') else None
                    test_pred = (np.asarray(best_model.classes_)[np.argmax(_test_prob_raw, axis=1)]
                                 if _test_prob_raw is not None else best_model.predict(test_x))
                finally:
                    if _test_jobs is not None: best_model.n_jobs = _test_jobs
            # Independent test is never used to tune or select the model.
            try:
                if best_key in ('SpatialCNN','TemporalCNN','DNN'):
                    test_prob = best_model.predict(test_x, verbose=0)
                elif hasattr(best_model, 'predict_proba'):
                    test_prob = _test_prob_raw
                    model_classes = getattr(best_model, 'classes_', None)
                    if model_classes is not None:
                        _ordered = np.zeros((len(test_prob), len(class_ids)), dtype=np.float64)
                        for _j, _c in enumerate(model_classes):
                            _ordered[:, int(_c)-1] = test_prob[:, _j]
                        test_prob = _ordered
                else:
                    test_prob = None
                if test_prob is not None:
                    _bins, _summary = _unc.reliability_rows(test_prob, y[te], class_ids)
                    _unc.write_rows(out_reliability, _bins)
                    _unc.write_rows(out_reliability_summary, [_summary])
                    feedback.pushInfo('V2.5.5 reliability audit: ECE=%.4f, Brier=%.4f (independent test; no tuning)' %
                                      (_summary['ece_10bin'], _summary['multiclass_brier']))
                else:
                    _unc.write_rows(out_reliability, [{'status':'NOT_AVAILABLE'}])
                    _unc.write_rows(out_reliability_summary, [{'status':'NOT_AVAILABLE'}])
            except Exception as _rel_err:
                feedback.pushWarning('Reliability audit failed: '+str(_rel_err))
                _unc.write_rows(out_reliability, [{'status':'FAILED','reason':str(_rel_err)}])
                _unc.write_rows(out_reliability_summary, [{'status':'FAILED','reason':str(_rel_err)}])
            test_metrics=self._metrics(y[te],test_pred)
            test_metrics = {("n_test" if k == "n_validation" else k): v for k,v in test_metrics.items()}
            self._write_csv_dict(out_test_metrics,{"selected_model":best_name,"objective_score_validation":best_score,**test_metrics})
            self._write_cm(out_test_cm,y[te],test_pred,class_ids)
            self._write_per_class_accuracy(out_test_per_class,y[te],test_pred,class_ids,inv)

            t0 = time.time()
            feedback.pushInfo("Generating production classified raster with Halo/Padding and Shape Alignment...")
            cls_ds = self._new_raster(out_raw_class, ref, gdal.GDT_UInt16, 0)
            cnf_ds = self._new_raster(out_conf, ref, gdal.GDT_Float32, -9999.0)
            ent_ds = self._new_raster(out_entropy, ref, gdal.GDT_Float32, -9999.0)
            mar_ds = self._new_raster(out_margin, ref, gdal.GDT_Float32, -9999.0)
            cb, pb = cls_ds.GetRasterBand(1), cnf_ds.GetRasterBand(1)
            eb, mb = ent_ds.GetRasterBand(1), mar_ds.GetRasterBand(1)
            
            width, height = ref.RasterXSize, ref.RasterYSize
            if use_spatial_cnn:
                half_p = patch_size // 2
            else:
                half_p = 0

            if best_key == "SpatialCNN" and io_block_size > 256:
                feedback.pushWarning("Spatial CNN inference block capped at 256 to prevent large patch-view memory allocations")
                io_block_size = 256
            if best_key in ("SpatialCNN", "TemporalCNN", "DNN"):
                self._active_tf_model = best_model
                feedback.pushInfo("Persistent TensorFlow inference worker: load model once; reuse for all batches")
            # QGIS-safe single-pass RF inference; preserve the fitted model and training configuration.
            _infer_jobs = getattr(best_model, 'n_jobs', None) if best_key not in ('SpatialCNN','TemporalCNN','DNN') else None
            if _infer_jobs is not None:
                best_model.n_jobs = 1
                feedback.pushInfo('V2.5.5.4.3.5: serial ML predict_proba for QGIS stability; training parallelism unchanged')
            _dl_predict_seconds = 0.0
            _dl_batches = 0
            _dl_pixels = 0
            _infer_predict_seconds = 0.0
            _infer_memory_seconds = 0.0
            _infer_batch_preparation_seconds = 0.0
            _infer_memory_checks = 0
            _infer_memory_interval = 32
            _infer_memory_last_available = None
            _infer_batches = 0
            _infer_pixels = 0
            _infer_peak_rss = 0
            _infer_memory_source = 'unavailable'
            # Processing script directory may not be importable; load sibling explicitly.
            import importlib.util as _mem_importlib
            _mem_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sage_inference_memory_v25536.py')
            _mem_spec = _mem_importlib.spec_from_file_location('sage_inference_memory_v25536', _mem_path)
            if _mem_spec is None or _mem_spec.loader is None:
                raise QgsProcessingException('Cannot load sage_inference_memory_v25536.py')
            _mem_module = _mem_importlib.module_from_spec(_mem_spec)
            _mem_spec.loader.exec_module(_mem_module)
            MemoryMonitor, adaptive_batch = _mem_module.MemoryMonitor, _mem_module.adaptive_batch
            _sage_memory = MemoryMonitor()
            _safety_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sage_adaptive_safety_v25537.py')
            _safety_spec = _mem_importlib.spec_from_file_location('sage_adaptive_safety_v25537', _safety_path)
            if _safety_spec is None or _safety_spec.loader is None:
                raise QgsProcessingException('Cannot load sage_adaptive_safety_v25537.py')
            _batch_safety = _mem_importlib.module_from_spec(_safety_spec)
            _safety_spec.loader.exec_module(_batch_safety)
            _infer_batch_size = ml_infer_batch
            _infer_min_batch = ml_infer_batch
            _infer_max_batch = ml_infer_batch
            _infer_downshifts = 0
            _infer_safety_low_reserve_checks = 0
            _infer_safety_unknown_available_checks = 0
            _infer_safety_invalid_available_checks = 0
            _infer_safety_min_headroom_bytes = None
            _infer_monitor_samples = 0
            _infer_peak_rss = 0
            _infer_peak_rss_source = 'unavailable'
            _infer_memory_guard = 'adaptive' if ml_adaptive else 'fixed'
            if ml_adaptive and not _sage_memory.available_supported:
                _infer_memory_guard = 'unavailable_fixed_batch'
                feedback.pushWarning('Memory availability unavailable; adaptive mode disabled, fixed batch retained')
            feedback.pushInfo('V2.5.5.4.3.5 ML inference batch=%d adaptive=%s reserve=%d MB; DL batch independent' % (
                _infer_batch_size, ml_adaptive, ml_memory_reserve_mb))
            # Optional paired benchmark: same fitted model, identical sampled raster pixels,
            # ABBA-style counterbalanced order, serial predict_proba, no retraining.
            # Never modifies the production prediction, the model, or training split.
            if run_controlled_benchmark and best_key not in ('SpatialCNN','TemporalCNN','DNN'):
                import importlib.util as _diag_importlib
                _diag_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sage_controlled_benchmark_v25532.py')
                _diag_spec = _diag_importlib.spec_from_file_location('sage_controlled_benchmark_v25532', _diag_path)
                if _diag_spec is None or _diag_spec.loader is None:
                    raise QgsProcessingException('Missing sage_controlled_benchmark_v25532.py')
                _diag_module = _diag_importlib.module_from_spec(_diag_spec)
                _diag_spec.loader.exec_module(_diag_module)
                _diag_sample = None
                _diag_limit = 8192
                for _dy in range(0, height, min(io_block_size, 256)):
                    _dr = min(256, height - _dy)
                    for _dx in range(0, width, min(io_block_size, 256)):
                        _dc = min(256, width - _dx)
                        _arrays = []
                        _valid = np.ones((_dr, _dc), dtype=bool)
                        for _ds, _bi in predictor_bands:
                            _band = _ds.GetRasterBand(_bi)
                            _a = _band.ReadAsArray(_dx, _dy, _dc, _dr).astype(np.float32)
                            _nd = _band.GetNoDataValue()
                            _valid &= np.isfinite(_a)
                            if _nd is not None: _valid &= (_a != _nd)
                            _arrays.append(_a)
                        _features = np.stack(_arrays, axis=-1)[_valid]
                        if len(_features):
                            _diag_sample = _features[:_diag_limit].copy()
                            break
                    if _diag_sample is not None: break
                if _diag_sample is None:
                    feedback.pushWarning('Controlled benchmark skipped: no valid raster pixels')
                    _diag_module.write_skipped(out_controlled_benchmark, 'no_valid_pixels')
                else:
                    _diag_rows = _diag_module.run_paired_benchmark(
                        best_model, _diag_sample, _sage_memory,
                        ml_memory_reserve_mb * 1024 * 1024,
                        feedback=feedback)
                    _diag_module.write_results(out_controlled_benchmark, _diag_rows)
                    feedback.pushInfo('Controlled paired benchmark completed: %d records' % len(_diag_rows))
            else:
                import csv as _diag_csv
                with open(out_controlled_benchmark, 'w', newline='', encoding='utf-8-sig') as _diag_f:
                    _diag_w = _diag_csv.writer(_diag_f)
                    _diag_w.writerow(['status','reason'])
                    _diag_w.writerow(['NOT_RUN', 'disabled' if not run_controlled_benchmark else 'SKIPPED_NON_ML_MODEL'])
            # Blockwise batch inference with true halo; only the selected model determines tensor shape.
            from numpy.lib.stride_tricks import sliding_window_view
            n_block_rows = (height + io_block_size - 1) // io_block_size
            for yoff in range(0,height,io_block_size):
                if feedback.isCanceled():raise QgsProcessingException("Canceled")
                feedback.setProgress(min(95.0, 65.0 + 30.0 * (yoff // io_block_size) / max(1,n_block_rows)))
                feedback.pushInfo("Raster inference: row %d/%d (%.1f%%)" % (min(yoff+io_block_size,height),height,100.0*min(yoff+io_block_size,height)/max(1,height)))
                rows=min(io_block_size,height-yoff)
                for xoff in range(0,width,io_block_size):
                    cols=min(io_block_size,width-xoff)
                    rad=patch_size//2 if best_key=="SpatialCNN" else 0
                    x0,y0=max(0,xoff-rad),max(0,yoff-rad)
                    x1,y1=min(width,xoff+cols+rad),min(height,yoff+rows+rad)
                    arrs=[];valid=np.ones((y1-y0,x1-x0),bool)
                    for ds,bi in predictor_bands:
                        b=ds.GetRasterBand(bi);a=b.ReadAsArray(x0,y0,x1-x0,y1-y0).astype(np.float32)
                        nd=b.GetNoDataValue();valid &= np.isfinite(a)
                        if nd is not None:valid &= a!=nd
                        arrs.append(a)
                    cube=np.stack(arrs,axis=-1)
                    oc=np.zeros((rows,cols),np.uint16);op=np.full((rows,cols),-9999.,np.float32)
                    oe=np.full((rows,cols),-9999.,np.float32);om=np.full((rows,cols),-9999.,np.float32)
                    core=valid[yoff-y0:yoff-y0+rows,xoff-x0:xoff-x0+cols].copy()
                    if best_key=="SpatialCNN":
                        # Pad outer raster boundaries with edge values, but invalidate patches touching nodata.
                        top=max(0,rad-yoff);left=max(0,rad-xoff)
                        bottom=max(0,yoff+rows+rad-height);right=max(0,xoff+cols+rad-width)
                        padded=np.pad(cube,((top,bottom),(left,right),(0,0)),mode="edge")
                        vpad=np.pad(valid,((top,bottom),(left,right)),mode="edge")
                        # Windows indexed by the core pixel centers.
                        windows=sliding_window_view(padded,(patch_size,patch_size),axis=(0,1))
                        vw=sliding_window_view(vpad,(patch_size,patch_size))
                        sr=yoff-y0+top-rad;sc=xoff-x0+left-rad
                        patch_view=windows[sr:sr+rows,sc:sc+cols].transpose(0,1,3,4,2)
                        core &= vw[sr:sr+rows,sc:sc+cols].all(axis=(-1,-2))
                        feats=patch_view[core]
                    else:
                        feats=cube[yoff-y0:yoff-y0+rows,xoff-x0:xoff-x0+cols][core]
                        if best_key=="TemporalCNN":feats=feats.reshape(len(feats),temporal_steps,-1)
                    if len(feats):
                        preds=[];confs=[];entropies=[];margins=[]
                        _step = min(len(feats), 2048) if best_key in ("SpatialCNN","TemporalCNN","DNN") else _infer_batch_size
                        i = 0
                        while i < len(feats):
                            # DL monitoring only; keep worker/inference behavior unchanged.
                            if best_key in ('SpatialCNN', 'TemporalCNN', 'DNN') and _dl_batches % 32 == 0:
                                _m_start = time.perf_counter()
                                _snapshot = _sage_memory.sample()
                                _infer_memory_seconds += time.perf_counter() - _m_start
                                _infer_memory_checks += 1
                                if _snapshot['rss_bytes'] is not None:
                                    _infer_peak_rss = max(_infer_peak_rss, _snapshot['rss_bytes'])
                                    _infer_monitor_samples += 1
                                    _infer_peak_rss_source = _sage_memory.rss_source
                            if best_key not in ('SpatialCNN','TemporalCNN','DNN'):
                                # Sample once per block and then every 32 batches. Keep
                                # monitoring OUTSIDE the measured model prediction timer.
                                if _infer_batches % _infer_memory_interval == 0:
                                    _m_start = time.perf_counter()
                                    _snapshot = _sage_memory.sample()
                                    _infer_memory_seconds += time.perf_counter() - _m_start
                                    _infer_memory_checks += 1
                                    if _snapshot['rss_bytes'] is not None:
                                        _infer_peak_rss = max(_infer_peak_rss, _snapshot['rss_bytes'])
                                        _infer_monitor_samples += 1
                                        _infer_peak_rss_source = _sage_memory.rss_source
                                    _infer_memory_last_available = _snapshot['available_bytes']
                                    # Audit only: never change prediction/model/spatial-CV decisions.
                                    _safe = _batch_safety.audit_available(
                                        _infer_memory_last_available, ml_memory_reserve_mb * 1048576)
                                    if _safe['status'] == 'below_reserve':
                                        _infer_safety_low_reserve_checks += 1
                                    elif _safe['status'] == 'unavailable':
                                        _infer_safety_unknown_available_checks += 1
                                    elif _safe['status'] == 'invalid':
                                        _infer_safety_invalid_available_checks += 1
                                    if _safe['headroom_bytes'] is not None:
                                        _infer_safety_min_headroom_bytes = (
                                            _safe['headroom_bytes'] if _infer_safety_min_headroom_bytes is None
                                            else min(_infer_safety_min_headroom_bytes, _safe['headroom_bytes']))
                                    if ml_adaptive and _sage_memory.available_supported:
                                        _new_step = adaptive_batch(_step, ml_infer_batch,
                                            _infer_memory_last_available,
                                            ml_memory_reserve_mb * 1024 * 1024, feats.shape[-1])
                                        if _new_step < _step: _infer_downshifts += 1
                                        _step = _new_step
                                _infer_min_batch = min(_infer_min_batch, _step)
                                _infer_max_batch = max(_infer_max_batch, _step)
                            _prep_start = time.perf_counter()
                            part=feats[i:i+_step]
                            i += len(part)
                            if best_key in ("SpatialCNN","TemporalCNN","DNN"):
                                if best_key=="DNN" and best_scaler is not None:part=best_scaler.transform(part)
                                _dl_start = time.perf_counter()
                                prob=best_model.predict(part,verbose=0)
                                _dl_predict_seconds += time.perf_counter() - _dl_start
                                _dl_batches += 1
                                _dl_pixels += len(part)
                                pred=np.argmax(prob,axis=1)+1
                            else:
                                part=part.reshape(len(part),-1)
                                _infer_batch_preparation_seconds += time.perf_counter() - _prep_start
                                _p_start = time.perf_counter()
                                prob=best_model.predict_proba(part) if hasattr(best_model,"predict_proba") else None
                                if prob is not None:
                                    pred=np.asarray(best_model.classes_)[np.argmax(prob,axis=1)]
                                else:
                                    pred=best_model.predict(part)
                                _infer_predict_seconds += time.perf_counter() - _p_start
                                _infer_batches += 1
                                _infer_pixels += len(part)

                            preds.append(np.asarray(pred,np.uint16))
                            if prob is not None:
                                if best_key not in ('SpatialCNN','TemporalCNN','DNN'):
                                    _aligned = np.zeros((len(prob), len(class_ids)), dtype=np.float64)
                                    for _j, _c in enumerate(best_model.classes_):
                                        _aligned[:, int(_c)-1] = prob[:, _j]
                                    prob = _aligned
                                _confidence, _entropy, _margin = _unc.uncertainty(prob, len(class_ids))
                            else:
                                _confidence = np.full(len(pred), -9999., np.float32)
                                _entropy = np.full(len(pred), -9999., np.float32)
                                _margin = np.full(len(pred), -9999., np.float32)
                            confs.append(_confidence);entropies.append(_entropy);margins.append(_margin)
                        oc[core]=np.concatenate(preds);op[core]=np.concatenate(confs).astype(np.float32)
                        oe[core]=np.concatenate(entropies);om[core]=np.concatenate(margins)
                    cb.WriteArray(oc,xoff,yoff);pb.WriteArray(op,xoff,yoff)
                    eb.WriteArray(oe,xoff,yoff);mb.WriteArray(om,xoff,yoff)
            cb.FlushCache()
            pb.FlushCache()
            eb.FlushCache();mb.FlushCache()
            # Explicitly close datasets before CreateCopy/post-filter and area stats.
            cls_ds.FlushCache();cnf_ds.FlushCache();ent_ds.FlushCache();mar_ds.FlushCache()
            cb=None;pb=None;eb=None;mb=None;cls_ds=None;cnf_ds=None;ent_ds=None;mar_ds=None
            if _xai_mod is not None and _xai_ref is not None:
                _ood_path = _xai_paths.get(self.OUTPUT_OOD)
                _trust_path = _xai_paths.get(self.OUTPUT_TRUST)
                if _ood_path and _trust_path:
                    _drv = gdal.GetDriverByName('GTiff')
                    _ood_ds = _drv.Create(_ood_path,width,height,1,gdal.GDT_Float32,options=['TILED=YES','COMPRESS=LZW'])
                    _trust_ds = _drv.Create(_trust_path,width,height,1,gdal.GDT_Float32,options=['TILED=YES','COMPRESS=LZW'])
                    for _ds in (_ood_ds,_trust_ds):
                        _ds.SetGeoTransform(predictor_bands[0][0].GetGeoTransform());_ds.SetProjection(predictor_bands[0][0].GetProjection())
                        _ds.GetRasterBand(1).SetNoDataValue(-9999.)
                    _confidence_ds = gdal.Open(out_conf,gdal.GA_ReadOnly)
                    if _confidence_ds is None:
                        raise QgsProcessingException('XAI QA FAIL: cannot reopen confidence raster')
                    _xai_expected_valid = 0
                    _xai_ood_written = 0
                    _xai_trust_written = 0
                    _center,_scale,_threshold = _xai_ref
                    for _y in range(0,height,256):
                        if feedback.isCanceled():raise QgsProcessingException('Canceled during XAI')
                        _rows=min(256,height-_y)
                        for _x in range(0,width,256):
                            _cols=min(256,width-_x)
                            _arrays=[];_mask=np.ones((_rows,_cols),bool)
                            for _pds,_bi in predictor_bands:
                                _b=_pds.GetRasterBand(_bi)
                                _a=_b.ReadAsArray(_x,_y,_cols,_rows).astype(np.float32)
                                _mask &= np.isfinite(_a)
                                _nd=_b.GetNoDataValue()
                                if _nd is not None:_mask &= _a!=_nd
                                _arrays.append(_a)
                            _conf=_confidence_ds.GetRasterBand(1).ReadAsArray(_x,_y,_cols,_rows)
                            _mask &= np.isfinite(_conf)&(_conf>=0.)
                            _ood=np.full((_rows,_cols),-9999.,np.float32)
                            _trust=np.full((_rows,_cols),-9999.,np.float32)
                            if _mask.any():
                                _features=np.stack(_arrays,axis=-1)[_mask]
                                _scores=_xai_mod.ood_scores(_features,_center,_scale)
                                _ood[_mask]=_scores
                                _trust[_mask]=_xai_mod.reliability(_conf[_mask],_scores,_threshold)
                            _n_valid = int(np.count_nonzero(_mask))
                            _xai_expected_valid += _n_valid
                            _n_ood = int(np.count_nonzero(np.isfinite(_ood) & (_ood >= 0)))
                            _n_trust = int(np.count_nonzero(np.isfinite(_trust) & (_trust >= 0) & (_trust <= 1)))
                            if _n_ood != _n_valid or _n_trust != _n_valid:
                                raise QgsProcessingException('XAI QA FAIL: invalid OOD/Trust values in tile x=%d y=%d expected=%d ood=%d trust=%d' % (_x,_y,_n_valid,_n_ood,_n_trust))
                            _xai_ood_written += _n_ood
                            _xai_trust_written += _n_trust
                            if _ood_ds.GetRasterBand(1).WriteArray(_ood,_x,_y) != 0:
                                raise QgsProcessingException('XAI QA FAIL: OOD raster write failed at x=%d y=%d' % (_x,_y))
                            if _trust_ds.GetRasterBand(1).WriteArray(_trust,_x,_y) != 0:
                                raise QgsProcessingException('XAI QA FAIL: Trust raster write failed at x=%d y=%d' % (_x,_y))
                    if _xai_expected_valid == 0:
                        raise QgsProcessingException('XAI QA FAIL: no valid prediction pixels')
                    _ood_ds.FlushCache(); _trust_ds.FlushCache()
                    _confidence_ds=None;_ood_ds=None;_trust_ds=None
                    # Read back both completed GeoTIFFs, not merely the in-memory arrays.
                    _ood_check = gdal.Open(_ood_path,gdal.GA_ReadOnly)
                    _trust_check = gdal.Open(_trust_path,gdal.GA_ReadOnly)
                    if _ood_check is None or _trust_check is None:
                        raise QgsProcessingException('XAI QA FAIL: cannot reopen OOD/Trust GeoTIFFs')
                    _ood_readback = 0
                    _trust_readback = 0
                    _mask_mismatch = 0
                    for _qy in range(0,height,256):
                        _qr = min(256,height-_qy)
                        _oa = _ood_check.GetRasterBand(1).ReadAsArray(0,_qy,width,_qr)
                        _ta = _trust_check.GetRasterBand(1).ReadAsArray(0,_qy,width,_qr)
                        _ov = np.isfinite(_oa) & (_oa >= 0)
                        _tv = np.isfinite(_ta) & (_ta >= 0) & (_ta <= 1)
                        _ood_readback += int(np.count_nonzero(_ov))
                        _trust_readback += int(np.count_nonzero(_tv))
                        _mask_mismatch += int(np.count_nonzero(_ov != _tv))
                    _ood_check=None;_trust_check=None
                    feedback.pushInfo('Integrated XAI readback QA: expected=%d OOD=%d TRUST=%d mask_mismatch=%d' % (_xai_expected_valid,_ood_readback,_trust_readback,_mask_mismatch))
                    if (_ood_readback != _xai_expected_valid or
                            _trust_readback != _xai_expected_valid or _mask_mismatch):
                        raise QgsProcessingException('XAI QA FAIL: OOD/Trust GeoTIFF readback differs from written valid pixels')
                    feedback.pushInfo('Integrated XAI: OOD and heuristic reliability rasters completed; readback QA PASS')
                else:
                    feedback.pushWarning('XAI rasters require both OOD and reliability output paths')
                _audit=_xai_paths.get(self.OUTPUT_XAI_AUDIT)
                if _audit:
                    with open(_audit,'w',newline='',encoding='utf-8-sig') as _f:
                        _w=csv.writer(_f);_w.writerow(['metric','value']);_w.writerow(['ood_reference','training_partition'])
                        _w.writerow(['ood_distance_threshold_p99',_xai_ref[2]])
                        _w.writerow(['selected_model',best_key])
                        _w.writerow(['xai_version','2.5.5.4.3.4'])
                        _w.writerow(['shap_beeswarm_feature_value_mode',_shap_color_mode if best_key=='SpatialCNN' else 'FEATURE_VALUE'])
                        _w.writerow(['shap_sample_output',_xai_paths.get(self.OUTPUT_SHAP_SAMPLES) or 'NOT_REQUESTED'])
                        _w.writerow(['shap_classwise_output',_xai_paths.get(self.OUTPUT_SHAP_CLASSWISE) or 'NOT_REQUESTED'])
                        _w.writerow(['reliability_definition','heuristic_not_calibrated_probability'])
                        _w.writerow(['ood_space','input_spectral_features_not_embedding'])
                        _w.writerow(['ood_reference_partition','train_only'])
                        _w.writerow(['dl_xai_method','grouped_KernelSHAP_probability' if best_key in ('SpatialCNN','TemporalCNN','DNN') else 'model_appropriate'])
                        _w.writerow(['spatial_importance_method','within_group_channel_permutation' if best_key in ('SpatialCNN','TemporalCNN','DNN') else 'within_group_feature_permutation'])
                        _w.writerow(['xai_feature_groups','spectral_channels_for_spatial_cnn;flattened_time_channels_for_temporal_cnn' if best_key in ('SpatialCNN','TemporalCNN') else 'flat_features'])
            profile_timers['integrated_xai_enabled'] = int(_xai_mod is not None)
            profile_timers["raster_inference"] = time.time() - t0
            profile_timers["dl_prediction_seconds"] = _dl_predict_seconds
            profile_timers["dl_prediction_batches"] = _dl_batches
            profile_timers["dl_prediction_pixels"] = _dl_pixels
            profile_timers["ml_probability_inference"] = _infer_predict_seconds
            profile_timers["ml_memory_monitoring_seconds"] = _infer_memory_seconds
            profile_timers["ml_batch_preparation_seconds"] = _infer_batch_preparation_seconds
            profile_timers["ml_memory_checks"] = _infer_memory_checks
            profile_timers["ml_probability_batches"] = _infer_batches
            profile_timers["ml_probability_pixels"] = _infer_pixels
            profile_timers["ml_probability_batch_size"] = _infer_batch_size
            profile_timers["ml_probability_peak_rss_bytes"] = _infer_peak_rss if _infer_monitor_samples else float("nan")
            profile_timers["ml_probability_batch_downshifts"] = _infer_downshifts
            profile_timers["ml_probability_memory_samples"] = _infer_monitor_samples
            feedback.pushInfo('V2.5.5.4.3.5 Memory telemetry: rss_samples=%d peak_rss_mb=%s source=%s downshifts=%d' % (_infer_monitor_samples, ('%.1f' % (_infer_peak_rss/1048576)) if _infer_monitor_samples else 'unavailable', _infer_peak_rss_source, _infer_downshifts))
            feedback.pushInfo('V2.5.5.4.3.5 Memory diagnostics: available_source=%s rss_supported=%s sample_attempts=%d errors=%s' % (_sage_memory.available_source, _sage_memory.rss_supported, _sage_memory.samples, '; '.join(_sage_memory.errors) if _sage_memory.errors else 'none'))
            if _infer_monitor_samples == 0:
                feedback.pushWarning('Memory RSS unavailable; check Windows API diagnostics and psutil installation; zero samples does NOT mean zero RAM usage')
            if _infer_jobs is not None: best_model.n_jobs = _infer_jobs
            if best_key in ('SpatialCNN','TemporalCNN','DNN'):
                feedback.pushInfo('V2.5.5.4.3.5 DL inference: %d pixels in %d batches; %.2f s model prediction' % (_dl_pixels,_dl_batches,_dl_predict_seconds))
            else:
                feedback.pushInfo('V2.5.5.4.3.5 ML inference: %d pixels in %d batches; %.2f s model prediction' % (_infer_pixels,_infer_batches,_infer_predict_seconds))
            if self._active_tf_model is not None:
                self._active_tf_model.close()
                self._active_tf_model = None
            feedback.pushInfo("Raster inference complete; TensorFlow persistent worker closed")

            t0 = time.time()
            majority_size = [3, 5, 7][majority_size_idx]
            self._apply_post_filter(out_raw_class, out_conf, out_class, apply_filter, filter_method, majority_size, min_patch_pixels, conf_threshold, feedback, io_block_size)
            self._write_per_class_accuracy(out_class_accuracy, yva, best_pred, class_ids, inv)
            self._append_predicted_spectral_qa(out_spectral_qa, out_raw_class,
                                               predictor_bands, class_ids, feedback)
            self._write_area_stats(out_area_stats, out_class, class_ids, inv, io_block_size)
            self._write_raster_integrity(out_raster_qa, out_raw_class, out_class, out_conf,
                                         ref, class_ids, sample_rows, sample_cols, te, test_pred, feedback)
            self._write_csv_dict(out_metrics, {"best_model": best_name, **best_metrics})
            self._write_cm(out_cm, yva, best_pred, class_ids)

            tp = int(np.sum((yva == 1) & (best_pred == 1)))
            fn = int(np.sum((yva == 1) & (best_pred != 1)))
            fp = int(np.sum((yva != 1) & (best_pred == 1)))
            tn = int(np.sum((yva != 1) & (best_pred != 1)))
            f_prec = tp / (tp + fp) if tp + fp else 0.0
            f_rec = tp / (tp + fn) if tp + fn else 0.0
            f_f1 = 2 * f_prec * f_rec / (f_prec + f_rec) if f_prec + f_rec else 0.0
            f_iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
            with open(out_flood_metrics, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["positive_class", "TP", "FP", "FN", "TN", "precision", "recall", "f1", "iou"])
                w.writerow([1, tp, fp, fn, tn, f_prec, f_rec, f_f1, f_iou])

            with open(out_compare, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=["model", "status", "selected", "objective_score", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "cohen_kappa", "n_validation", "target_precision", "target_recall", "target_f1", "target_iou"])
                w.writeheader()
                for row in comparison:
                    w.writerow({k: (int(row.get("model") == best_name) if k == "selected" else row.get(k, "")) for k in w.fieldnames})

            # Explicit status reports for all declared optional research outputs.
            for path,header in ((out_target_compare,["status","reason"]),
                                (out_flood_compare,["status","reason"])):
                with open(path,"w",newline="",encoding="utf-8-sig") as f:
                    w=csv.writer(f);w.writerow(header);w.writerow(["NOT_RUN","Not implemented in V2.5.5; no results fabricated"])
            profile_timers["post_processing"] = time.time() - t0
            profile_timers["total_execution"] = time.time() - t_start_total
            with open(out_profiling, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["processing_stage", "value", "unit"])
                for stage, duration in profile_timers.items():
                    unit = "count" if stage in ("ml_probability_batches", "ml_probability_pixels", "ml_probability_batch_size", "ml_probability_peak_rss_bytes", "ml_probability_batch_downshifts", "ml_probability_memory_samples", "ml_memory_checks", "dl_prediction_batches", "dl_prediction_pixels") else "seconds"
                    # NaN is a legitimate unavailable telemetry marker for DL.
                    # Never coerce it to integer or silently replace it with zero.
                    try:
                        _profile_number = float(duration)
                        if not np.isfinite(_profile_number):
                            _profile_value = "UNAVAILABLE"
                        else:
                            _profile_value = int(_profile_number) if unit == "count" else round(_profile_number, 2)
                    except (TypeError, ValueError, OverflowError):
                        _profile_value = "UNAVAILABLE"
                    w.writerow([stage, _profile_value, unit])

            with open(out_inference_benchmark, 'w', newline='', encoding='utf-8-sig') as _bf:
                _bw = csv.writer(_bf)
                _bw.writerow(['metric', 'value', 'unit', 'notes'])
                for _row in (
                    ('batch_size', _infer_batch_size, 'pixels', 'ML only; deep-learning batches unchanged'),
                    ('dl_prediction_pixels', _dl_pixels, 'pixels', 'DL only'),
                    ('dl_prediction_batches', _dl_batches, 'count', 'DL only'),
                    ('dl_prediction_seconds', round(_dl_predict_seconds, 4), 'seconds', 'DL model.predict wall time'),
                    ('prediction_pixels', _infer_pixels, 'pixels', 'valid ML prediction pixels'),
                    ('prediction_batches', _infer_batches, 'count', 'actual calls to model prediction'),
                    ('prediction_seconds', round(_infer_predict_seconds, 4), 'seconds', 'model predict_proba/predict wall time'),
                    ('throughput', round(_infer_pixels / _infer_predict_seconds, 3) if _infer_predict_seconds else 0, 'pixels_per_second', 'measured'),
                    ('peak_process_rss_sampled', _infer_peak_rss if _infer_monitor_samples else '', 'bytes', _infer_peak_rss_source),
                    ('memory_available_source', _sage_memory.available_source, 'source', 'system available memory'),
                    ('memory_rss_supported', _sage_memory.rss_supported, 'boolean', 'RSS provider initialized'),
                    ('memory_sample_attempts', _sage_memory.samples, 'count', 'monitor sample calls'),
                    ('memory_telemetry_errors', '; '.join(_sage_memory.errors), 'diagnostic', 'errors from psutil and Windows API'),
                    ('adaptive_batch_mode', _infer_memory_guard, 'mode', 'falls back to fixed if RAM unavailable'),
                    ('memory_reserve_mb', ml_memory_reserve_mb, 'MB', 'minimum available RAM threshold'),
                    ('batch_min_used', _infer_min_batch, 'pixels', 'selected ML batch size'),
                    ('batch_max_used', _infer_max_batch, 'pixels', 'selected ML batch size'),
                    ('batch_downshifts', _infer_downshifts, 'count', 'adaptive decreases'),
                    ('memory_samples', _infer_monitor_samples, 'count', 'RSS samples'),
                    ('memory_checks', _infer_memory_checks, 'count', 'available RAM checks'),
                    ('safety_below_reserve_checks', _infer_safety_low_reserve_checks, 'count', 'system available memory <= configured reserve'),
                    ('safety_unavailable_checks', _infer_safety_unknown_available_checks, 'count', 'no system memory sample'),
                    ('safety_invalid_checks', _infer_safety_invalid_available_checks, 'count', 'invalid memory sample'),
                    ('safety_min_headroom_bytes', _infer_safety_min_headroom_bytes if _infer_safety_min_headroom_bytes is not None else '', 'bytes', 'sampled available RAM minus reserve'),
                    ('memory_check_interval', _infer_memory_interval, 'batches', 'sampling interval'),
                    ('memory_monitoring_seconds', round(_infer_memory_seconds, 4), 'seconds', 'RAM sampling only'),
                    ('batch_preparation_seconds', round(_infer_batch_preparation_seconds, 4), 'seconds', 'slicing/reshape overhead'),
                    ('baseline_256_estimated_batches', (_infer_pixels + 255) // 256, 'count', 'theoretical; not a runtime comparison'),
                    ('speedup_vs_baseline', 'NOT_MEASURED', 'ratio', 'requires same-hardware paired 256-batch run'),
                ):
                    _bw.writerow(_row)

            feedback.pushInfo('V2.5.5.4.3.5 Adaptive safety audit: below_reserve=%d unavailable=%d invalid=%d min_headroom_mb=%s' % (
                _infer_safety_low_reserve_checks, _infer_safety_unknown_available_checks,
                _infer_safety_invalid_available_checks,
                ('%.1f' % (_infer_safety_min_headroom_bytes / 1048576)) if _infer_safety_min_headroom_bytes is not None else 'unavailable'))
            if _infer_safety_low_reserve_checks:
                feedback.pushWarning('Available system RAM fell below reserve; minimum batch is 256 and does not guarantee zero OOM risk')
            feedback.pushInfo(f"SAGE-GeoClass V2.5.5.4.3.5 completed successfully in {profile_timers['total_execution']:.2f} seconds.")

            return {
                **({k:v for k,v in _xai_paths.items() if v and _run_xai and _xai_mod is not None}),
                self.OUTPUT_RAW_CLASS: out_raw_class,
                self.OUTPUT_CLASS: out_class,
                self.OUTPUT_CONF: out_conf,
                self.OUTPUT_ENTROPY: out_entropy,
                self.OUTPUT_MARGIN: out_margin,
                self.OUTPUT_RELIABILITY: out_reliability,
                self.OUTPUT_RELIABILITY_SUMMARY: out_reliability_summary,
                self.OUTPUT_METRICS: out_metrics,
                self.OUTPUT_CM: out_cm,
                self.OUTPUT_CLASSES: out_classes,
                self.OUTPUT_CLASS_MANIFEST: out_class_manifest,
                self.OUTPUT_CLASS_QC: out_class_qc,
                self.OUTPUT_MODEL_COMPARISON: out_compare,
                self.OUTPUT_IMPORTANCE: out_importance,
                self.OUTPUT_FEATURE_ABLATION: out_feature_ablation,
                self.OUTPUT_FEATURE_QC: out_feature_qc,
                self.OUTPUT_FEATURE_MANIFEST: out_feature_manifest,
                self.OUTPUT_FLOOD_METRICS: out_flood_metrics,
                self.OUTPUT_FLOOD_MODEL_COMPARISON: out_flood_compare,
                self.OUTPUT_TARGET_MODEL_COMPARISON: out_target_compare,
                self.OUTPUT_SPLIT: out_split,
                self.OUTPUT_AREA_STATS: out_area_stats,
                self.OUTPUT_RASTER_QA: out_raster_qa,
                self.OUTPUT_SPECTRAL_QA: out_spectral_qa,
                self.OUTPUT_CLASS_LEGEND: out_class_legend,
                self.OUTPUT_CLASS_ACCURACY: out_class_accuracy,
                self.OUTPUT_GRID_QC: out_grid_qc,
                self.OUTPUT_CV: out_cv,
                self.OUTPUT_CV_PER_CLASS: out_cv_per_class,
                self.OUTPUT_TRAIN_QC: out_train_qc,
                self.OUTPUT_PROFILING: out_profiling,
                self.OUTPUT_INFERENCE_BENCHMARK: out_inference_benchmark,
                self.OUTPUT_CONTROLLED_BENCHMARK: out_controlled_benchmark,
                self.OUTPUT_LEAKAGE_QC: out_leakage_qc,
                self.OUTPUT_TEST_METRICS:out_test_metrics,
                self.OUTPUT_TEST_CM:out_test_cm,
                self.OUTPUT_TEST_PER_CLASS:out_test_per_class,
                self.OUTPUT_MODEL_DIR: model_dir
            }

        except QgsProcessingException:
            raise
        except Exception as e:
            raise QgsProcessingException(str(e) + "\n\n" + traceback.format_exc())
        finally:
            for model in getattr(self, "_tf_models", []):
                model.close()
            self._tf_models = []
            self._active_tf_model = None