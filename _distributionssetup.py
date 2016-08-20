"""
This file is a helper file to compile the C extension _distributions.c which
is used to improve the performance when evaluating PDFs and CDFs.
"""

from distutils.core import setup, Extension
import shutil, sys

sys.argv.append('build') #Add the build command

module1 = Extension('_distributions', sources = ['_distributions.c'])

setup(name = 'Distributions C module',
        version = '1.0',
        description = 'This is a package to calculate pdfs and cdfs.',
        ext_modules = [module1])

#The following lines are OS specific and should be commented out for a release

#shutil.copy('build/lib.macosx-10.5-i386-2.7/_distributions.so', '.')
#shutil.rmtree('build')
#print 'Copied library file. Done.'
