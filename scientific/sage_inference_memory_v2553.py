"""V2.5.5.3: conservative memory telemetry and batch downshifting; no QGIS dependency."""
import os
import sys


def adaptive_batch(current, maximum, available_bytes, reserve_bytes, feature_count=4):
    """Only downshift on low RAM; never grow during a scene or exceed user's ceiling.

    Reserve threshold is intentionally conservative; it does not guarantee against OOM.
    """
    current = max(256, min(int(current), int(maximum)))
    if available_bytes is None:
        return current
    # Allow breathing room above reserve for model allocations and other processes.
    headroom = int(available_bytes) - int(reserve_bytes)
    if headroom <= 0:
        return 256
    # Conservative proxy: sklearn forest inference may allocate much more than input tensor.
    estimated = current * max(1, int(feature_count)) * 8 * 64
    while current > 256 and estimated > headroom // 4:
        current = max(256, current // 2)
        estimated = current * max(1, int(feature_count)) * 8 * 64
    return current


def candidate_batch(ceiling, available_bytes, reserve_bytes, feature_count=4):
    return adaptive_batch(ceiling, ceiling, available_bytes, reserve_bytes, feature_count)


class MemoryMonitor:
    def __init__(self):
        self.available_source = 'unavailable'
        self.rss_source = 'unavailable'
        self._psutil = None
        self._proc = None
        try:
            import psutil
            self._psutil = psutil
            self._proc = psutil.Process(os.getpid())
            self.available_source = 'psutil.virtual_memory.available'
            self.rss_source = 'psutil.process.memory_info.rss'
        except ImportError:
            pass
        self._windows_process_handle = None
        self._windows_pmc_type = None
        self.available_supported = self._psutil is not None
        if not self.available_supported and sys.platform == 'win32':
            try:
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                               ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                               ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                               ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                               ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
                self._ctypes = ctypes
                self._memstatus = MEMORYSTATUSEX
                self.available_source = 'windows.GlobalMemoryStatusEx'
                self.available_supported = True
                # Native Windows process RSS without a psutil dependency.
                class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                    _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                                ('PeakWorkingSetSize', ctypes.c_size_t),
                                ('WorkingSetSize', ctypes.c_size_t),
                                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                                ('QuotaPagedPoolUsage', ctypes.c_size_t),
                                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                                ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                                ('PagefileUsage', ctypes.c_size_t),
                                ('PeakPagefileUsage', ctypes.c_size_t)]
                self._windows_pmc_type = PROCESS_MEMORY_COUNTERS
                self._windows_process_handle = ctypes.windll.kernel32.GetCurrentProcess()
                self.rss_source = 'windows.GetProcessMemoryInfo.WorkingSetSize'
            except (ImportError, AttributeError, OSError):
                pass

    def sample(self):
        available = None
        rss = None
        if self._psutil is not None:
            try:
                available = int(self._psutil.virtual_memory().available)
                rss = int(self._proc.memory_info().rss)
            except (OSError, AttributeError):
                pass
        elif self.available_supported:
            try:
                status = self._memstatus()
                status.dwLength = self._ctypes.sizeof(status)
                if self._ctypes.windll.kernel32.GlobalMemoryStatusEx(self._ctypes.byref(status)):
                    available = int(status.ullAvailPhys)
                if self._windows_process_handle is not None:
                    counters = self._windows_pmc_type()
                    counters.cb = self._ctypes.sizeof(counters)
                    if self._ctypes.windll.psapi.GetProcessMemoryInfo(
                        self._windows_process_handle, self._ctypes.byref(counters), counters.cb):
                        rss = int(counters.WorkingSetSize)
            except (OSError, AttributeError, ValueError):
                pass
        return {'available_bytes': available, 'rss_bytes': rss}
