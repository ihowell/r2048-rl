from setuptools import setup, find_packages

VERSION = '0.1.0'
setup(
    name='tiles',
    version=VERSION,
    packages=find_packages(),
    install_requires=[
        'scipy',
        'tensorflow>=2.3,<2.4',
        'tensorflow_datasets',
        'tensorflow_probability',
        'dm-sonnet',
        'fire',
        'tqdm',
        'typeguard'
    ],
    extras_require={
        'tests': [
            'pylint',
            'pytest',
            'pytest-cov',
            'pytest-integration',
            'yapf',
        ],
    }
)
