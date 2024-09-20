from setuptools import setup
from setuptools.command.install import install
import subprocess

class PostInstallCommand(install):
    def run(self):
        install.run(self)
        subprocess.call(['python', 'post_install.py'])

setup(
    name='your_package_name',
    version='0.1',
    install_requires=[
        'nltk',
    ],
    cmdclass={
        'install': PostInstallCommand,
    },
)
