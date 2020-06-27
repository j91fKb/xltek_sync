from setuptools import setup, find_packages

setup(name='xltek_sync',
      version='0.1',
      description='N/A',
      url='https://github.com/j91fKb/xltek_sync.git',
      author='N/A',
      author_email='example@example.com',
      license='MIT',
      package_dir={
          '': 'src',
      },
      packages=find_packages(where='src'),
      zip_safe=False,
      python_requires='>=3.6')
