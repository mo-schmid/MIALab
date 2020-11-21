import sys
from setuptools import find_packages, setup


if sys.version_info < (3, 7):
    sys.exit("Requires Python 3.7 or higher")

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license_ = f.read()

REQUIRED_PACKAGES = [
    'pymia == 0.3.1',
    'scikit-learn >= 0.23.2',
    'pathos >= 0.2.6',
    'pydensecrf >= 1.0rc3',
    'sphinx >= 3.2.1',
    'sphinx_rtd_theme >= 0.5.0',
]

TEST_PACKAGES = [

]

setup(
    name='MIALab',
    version='0.2.0',
    description='medical image analysis laboratory',
    long_description=readme,
    author='Healthcare Imaging A.I.',
    author_email='mauricio.reyes@med.unibe.ch',
    url='https://github.com/ubern-mia/MIALab',
    license=license_,
    packages=find_packages(exclude=['test', 'docs']),
    install_requires=REQUIRED_PACKAGES,
    tests_require=REQUIRED_PACKAGES + TEST_PACKAGES,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries'
    ],
    keywords=[
        'medical image analysis',
        'machine learning',
        'neuro'
    ]
)
