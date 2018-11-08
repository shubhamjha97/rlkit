from setuptools import setup

setup(name='RLkit',
      version='0.1',
      description='A simple RL library.',
      url='http://github.com/shubhamjha97/RLkit',
      author='Shubham Jha',
      author_email='jha1shubham@gmail.com',
      license='MIT',
      long_description=open('README.md').read(),
      install_requires=[
        'tensorflow==1.11.0',
		    'gym==0.10.8',
		    'ipdb==0.11',
		    'numpy==1.15.4'
    	],
      packages=setuptools.find_packages(),
      classifiers = ('Intended Audience :: Science/Research', 'Natural Language :: English', 'Programming Language :: Python :: 3.6'),
      zip_safe=False)