from setuptools import setup, find_packages

setup(name='RLkit',
      version='0.2.0',
      description='A simple RL library.',
      url='http://github.com/shubhamjha97/RLkit',
      author='Shubham Jha',
      author_email='jha1shubham@gmail.com',
      license='MIT',
      long_description=open('README.md').read(),
      long_description_content_type="text/markdown",
      install_requires=[
        'tensorflow==2.6.4',
		    'gym==0.10.8',
		    'ipdb==0.11',
		    'numpy==1.15.4'
    	],
      packages=find_packages(),
      classifiers = ('Intended Audience :: Science/Research', 'Natural Language :: English', 'Programming Language :: Python :: 3.6'),
      zip_safe=False)