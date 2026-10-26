"""SAGE V2.5.5.3.6: Windows-safe process RSS and system-available memory.

No QGIS dependencies. Does not change RF predictions, batch algorithm, or TF worker.
"""
import ctypes
import os
import sys


def adaptive_batch(current, maximum, available_bytes, reserve_bytes, feature_count=4):
    current = max(256, min(int(current), int(maximum)))
    if available_bytes is None:
        return current
    headroom = int(available_bytes) - int(reserve_bytes)
    if headroom <= 0:
        return 256
    estimated = current * max(1, int(feature_count)) * 8 * 64
    while current > 256 and estimated > headroom // 4:
        current = max(256, current // 2)
        estimated = current * max(1, int(feature_count)) * 8 * 64
    return current


class MemoryMonitor:
    """psutil preferred; Windows native APIs independently backstop each metric."""
    def __init__(self):
        self._psutil = None
        self._proc = None
        self._native = None
        self.available_source = 'unavailable'
        self.rss_source = 'unavailable'
        self.available_supported = False
        self.rss_supported = False
        self.errors = []
        self.samples = 0
        self.rss_samples = 0
        self.peak_rss_bytes = 0
        try:
            import psutil
            self._psutil = psutil
            self._proc = psutil.Process(os.getpid())
            self.available_supported = True
            self.rss_supported = True
            self.available_source = 'psutil.virtual_memory.available'
            self.rss_source = 'psutil.Process.memory_info.rss'
        except (ImportError, OSError, AttributeError) as exc:
            self._record('psutil.init', exc)
        if sys.platform == 'win32':
            try:
                self._native = _WindowsMemory()
                self.available_supported = True
                self.rss_supported = True
            except (OSError, AttributeError, TypeError, ValueError) as exc:
                self._record('windows.init', exc)

    def _record(self, where, exc):
        msg = '%s: %s: %s' % (where, type(exc).__name__, str(exc)[:180])
        if msg not in self.errors and len(self.errors) < 12:
            self.errors.append(msg)

    def sample(self):
        self.samples += 1
        available = None
        rss = None
        if self._psutil is not None:
            try:
                available = int(self._psutil.virtual_memory().available)
                self.available_source = 'psutil.virtual_memory.available'
            except (OSError, AttributeError, ValueError, RuntimeError) as exc:
                self._record('psutil.available', exc)
            try:
                rss = int(self._proc.memory_info().rss)
                self.rss_source = 'psutil.Process.memory_info.rss'
            except (OSError, AttributeError, ValueError, RuntimeError) as exc:
                self._record('psutil.rss', exc)
        if self._native is not None:
            if available is None:
                try:
                    available = self._native.available_bytes()
                    self.available_source = 'windows.GlobalMemoryStatusEx'
                except (OSError, AttributeError, ValueError) as exc:
                    self._record('windows.available', exc)
            if rss is None:
                try:
                    rss = self._native.rss_bytes()
                    self.rss_source = 'windows.GetProcessMemoryInfo.WorkingSetSize'
                except (OSError, AttributeError, ValueError) as exc:
                    self._record('windows.rss', exc)
        if rss is not None and rss > 0:
            self.rss_samples += 1
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        else:
            rss = None
            if self.rss_samples == 0:
                self.rss_source = 'unavailable'
        if available is None:
            self.available_source = 'unavailable'
        return {'available_bytes': available, 'rss_bytes': rss}


class _WindowsMemory:
    def __init__(self):
        from ctypes import wintypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [('dwLength', wintypes.DWORD), ('dwMemoryLoad', wintypes.DWORD),
                        ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        self._status_type = MEMORYSTATUSEX
        self._counters_type = PROCESS_MEMORY_COUNTERS
        self._kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self._psapi = ctypes.WinDLL('psapi', use_last_error=True)
        self._kernel.GetCurrentProcess.argtypes = []
        self._kernel.GetCurrentProcess.restype = wintypes.HANDLE
        self._kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
        self._kernel.GlobalMemoryStatusEx.restype = wintypes.BOOL
        self._psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        self._psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        self._handle = self._kernel.GetCurrentProcess()

    def available_bytes(self):
        status = self._status_type()
        status.dwLength = ctypes.sizeof(status)
        if not self._kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(status.ullAvailPhys)

    def rss_bytes(self):
        counters = self._counters_type()
        counters.cb = ctypes.sizeof(counters)
        if not self._psapi.GetProcessMemoryInfo(self._handle, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.WorkingSetSize)
