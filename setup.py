#! /usr/bin/env python

from setuptools import setup

DISTNAME = 'marseille'
DESCRIPTION = ("Mining argument structures with expressive inference "
               "(linear and lstm engines)")
LONG_DESCRIPTION = open('README.md').read()
MAINTAINER = 'Vlad Niculae'
MAINTAINER_EMAIL = 'vlad@vene.ro'
URL = 'https://github.com/vene/marseille'
LICENSE = 'BSD 3-clause'
DOWNLOAD_URL = 'https://github.com/vene/marseille'
VERSION = '0.1.dev0'



if __name__ == "__main__":

    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          include_package_data=True,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=LONG_DESCRIPTION,
          zip_safe=False,  # the package can run out of an .egg file
          packages=['marseille'],
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: Python',
             ]
          )
