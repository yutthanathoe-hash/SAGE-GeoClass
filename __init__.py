def classFactory(iface):
    from .plugin import SageGeoClassPlugin
    return SageGeoClassPlugin(iface)
