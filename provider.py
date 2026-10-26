import importlib.util
import os
import sys
from qgis.core import QgsProcessingProvider

_SAGE_V1_HELP = '<p><b>Developer:</b> Yutthana Thoenglom<br><b>Affiliation:</b> Maejo University Phrae Campus</p>\n<p>Raster classification using ML/DL, spatial validation, SHAP explainability, and uncertainty assessment.</p>\n<p><b>Inputs:</b> Select predictor rasters and training samples with a class field. Check CRS, resolution and NoData.</p>\n<p><b>Models:</b> Choose a model or compare eligible models. Spatial CNN uses image patches; Temporal CNN needs multiple dates.</p>\n<p><b>Validation:</b> Prefer group-aware splitting and Spatial-CV. Keep the independent test set separate.</p>\n<p><b>Outputs:</b> Save classification, confidence and validation reports to an output directory. Optional SHAP/OOD analyses require additional processing.</p>\n'

class SageGeoClassProvider(QgsProcessingProvider):
    def id(self):
        return 'sage_geoclass'

    def name(self):
        return 'SAGE GeoClass'

    def longName(self):
        return 'SAGE GeoClass — Scientific Classification'

    def loadAlgorithms(self):
        scientific = os.path.join(os.path.dirname(__file__), 'scientific')
        # Original modules use absolute sibling imports; keep paths unchanged.
        if scientific not in sys.path:
            sys.path.insert(0, scientific)
        entry = os.path.join(scientific, 'SAGE_GeoClass_V2_5_5_4_3_6.py')
        spec = importlib.util.spec_from_file_location('sage_geoclass_plugin_entry_v256', entry)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Presentation-only adapter: original scientific modules stay byte-identical.
        class SAGEGeoClassV1(module.SAGEGeoClassV256):
            def displayName(self):
                return 'SAGE GeoClass V1'

            def initAlgorithm(self, config=None):
                super().initAlgorithm(config)
                # Presentation-only: do not modify parameter identifiers or defaults.
                for parameter in self.parameterDefinitions():
                    original = parameter.description()
                    label = __import__('re').sub(r'^\[V\d+(?:\.\d+)+\]\s*', '', original)
                    label = __import__('re').sub(r'\s*\[optional\](?:\s*\[optional\])+', ' [optional]', label, flags=__import__('re').I)
                    label = label.replace('OOD distance raster [optional]', 'Out-of-distribution (OOD) distance raster [optional]')
                    label = label.replace('Heuristic reliability raster (0-1)', 'Model reliability indicator raster (heuristic, 0-1)')
                    parameter.setDescription(label)

            def shortHelpString(self):
                return _SAGE_V1_HELP

            def createInstance(self):
                return SAGEGeoClassV1()

        self.addAlgorithm(SAGEGeoClassV1())
