from setuptools import setup, find_packages

VERSION = '0.1.0'
setup(
    name='r2048-rl',
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
        'typeguard',
        # 'git+https://github.com/DrKwint/r2048.git@6fcd24f91959da93dcb8b11055c35653f2fa947c',
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
