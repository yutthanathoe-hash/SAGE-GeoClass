"""SAGE-GeoClass 2.5.5.4.3.6 additive guarded QGIS entrypoint.
Frozen V2.5.5.4.3.5 algorithm and modules are used unchanged.
"""
import csv, importlib.util, os
from qgis.core import QgsProcessingException

_DIR=os.path.dirname(os.path.abspath(__file__))
def _load(name):
    path=os.path.join(_DIR,name+'.py')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
_core=_load('SAGE_GeoClass_V2_5_5_4_3_5')
_guard=_load('sage_scientific_guard_v256')

class SAGEGeoClassV256(_core.SAGEGeoClassV252):
    def name(self):return 'sage_geoclass_v2_5_5_4_3_6'
    def displayName(self):return 'SAGE-GeoClass V2.5.5.4.3.6 — Scientific Guard (Frozen Core)'
    def createInstance(self):return SAGEGeoClassV256()
    def processAlgorithm(self,parameters,context,feedback):
        mode=self.parameterAsEnum(parameters,self.MODEL_MODE,context)
        temporal=self.parameterAsInt(parameters,self.TEMPORAL_STEPS,context)
        spatial=self.parameterAsBool(parameters,self.USE_SPATIAL_CNN,context)
        temporal_enabled=self.parameterAsBool(parameters,self.USE_TEMPORAL_CNN,context)
        output=self.parameterAsString(parameters,self.OUTPUT_DIRECTORY,context)
        if not output:raise QgsProcessingException('OUTPUT_DIRECTORY is required for scientific evidence')
        os.makedirs(output,exist_ok=True)
        matrix=_guard.dependency_matrix(temporal,spatial,temporal_enabled)
        _guard.write_csv(os.path.join(output,'OUTPUT_MODEL_PREFLIGHT.csv'),matrix,['model','status','reason','dependencies','python'])
        if mode>0 and matrix[mode-1]['status']!='READY':
            raise QgsProcessingException(matrix[mode-1]['model']+': '+matrix[mode-1]['status']+' — '+matrix[mode-1]['reason'])
        if mode==0:
            for row in matrix:
                if row['status']!='READY':feedback.pushWarning(row['model']+': '+row['status']+' — '+row['reason'])
        if not self.parameterAsString(parameters,self.SPECTRAL_BAND_INDICES,context).strip():
            feedback.pushWarning('Spectral band roles not configured: spectral index QA is provisional; no scientific PASS claimed')
        result=super().processAlgorithm(parameters,context,feedback)
        report=_guard.audit_outputs(output)
        feedback.pushInfo('Scientific guard evidence: '+os.path.join(output,'OUTPUT_SCIENTIFIC_AUDIT.csv'))
        feedback.pushWarning('12-model scientific equivalence NOT certified unless all models share identical CV folds and final test protocol')
        return result
