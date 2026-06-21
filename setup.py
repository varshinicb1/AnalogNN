from setuptools import setup, find_packages

setup(
    name="open-analog-nn",
    version="0.2.0",
    packages=find_packages(include=[
        'analog_layers*',
        'calibration*',
        'circuit_ir*',
        'spice*',
        'validation*',
        'experiments*',
        'configs*',
        'datasets*',
        'training*',
        'nas*',
        'energy*',
        'reports*',
        'reproduce*',
        'utils*',
    ]),
    include_package_data=True,
)
