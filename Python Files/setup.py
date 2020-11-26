from distutils.core import setup

setup(name='number-detector',
      version=open("digit_detector/_version.py").readlines()[-1].split()[-1].strip("\"'"),
      description='SVHN number detector and recognizer',
      author='krakshil',
      author_email='krakshil07@gmail.com',
      packages=['digit_detector'],  
     )

