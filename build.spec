# -*- mode: python ; coding: utf-8 -*-
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# Collect onnxruntime shared libraries and data
onnxruntime_binaries = collect_dynamic_libs('onnxruntime')
onnxruntime_datas = collect_data_files('onnxruntime')

# Exclude MSVC runtime DLLs that PyInstaller may bundle from Python/numpy.libs.
# Bundling these alongside the system-installed CRT causes multiple CRT instances
# to be loaded simultaneously, which corrupts C++ STL state in pybind11 extensions
# like hnswlib and triggers an access violation (0xC0000005) on startup.
# The user's machine must have the Visual C++ Redistributable installed (which
# Windows ships with by default on modern systems).
_MSVC_EXCLUDES = {
    'msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll',
    'vcruntime140.dll', 'vcruntime140_1.dll',
    'api-ms-win-crt-runtime-l1-1-0.dll',
}

def _filter_msvc(binaries):
    result = []
    for entry in binaries:
        # binaries can be 2-tuples (dest, src) or 3-tuples (dest, src, typecode)
        dest = entry[0]
        if dest.lower() not in _MSVC_EXCLUDES:
            result.append(entry)
    return result

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=_filter_msvc(onnxruntime_binaries),
    datas=[
        ('config.json', '.'),
        ('models/imagenet-b2-opti.onnx', 'models'),
        *onnxruntime_datas,
    ],
    hiddenimports=[
        'hnswlib',
        'onnxruntime',
        'onnxruntime.capi._pybind_state',
        'PIL._imaging',
        'PIL.Image',
        'numpy',
        'tqdm',
        'multiprocessing',
        'multiprocessing.pool',
        'multiprocessing.spawn',
        'multiprocessing.forkserver',
        'multiprocessing.reduction',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'onnx',
        'torch',
        'tensorflow',
        'matplotlib',
        'tkinter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='EfficientIR',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    _filter_msvc(a.binaries),
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='EfficientIR',
)
