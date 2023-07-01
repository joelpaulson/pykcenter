from setuptools import setup

setup(
    name='pykcenter',
    version=0.1,
    packages=['pykcenter'],
    author='Joel Paulson',
    author_email='paulson.82@osu.edu',
    url='',
    license='MIT License',
    long_description='Python implementation of global optimization method for kcenter clustering',
    install_requires=[
        "torch",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib"
    ],
)