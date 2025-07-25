# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('D:/PROGRAMS/Anaconda/envs/app_env/Lib/site-packages/mediapipe/modules/pose_landmark/pose_landmark_cpu.binarypb',
     'mediapipe/modules/pose_landmark'),

        ('D:/PROGRAMS/Anaconda/envs/app_env/Lib/site-packages/mediapipe/modules/pose_landmark/pose_landmark_full.tflite',
         'mediapipe/modules/pose_landmark'),

         ('D:/PROGRAMS/Anaconda/envs/app_env/Lib/site-packages/mediapipe/modules/pose_detection/pose_detection.tflite',
         'mediapipe/modules/pose_detection'),

        ('E:/LICENTA/App_pycharm/action_recognition_models', 'action_recognition_models')
        ],
    hiddenimports=['sklearn',
    'scipy._lib.array_api_compat.numpy.fft'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
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
    icon='app_icon/icon.ico'
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)
