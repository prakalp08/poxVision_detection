import setuptools

with open('README.md', 'r', encoding='utf-8') as file:
    long_description = file.read()                        # WE ARE READING EVERYTHING THAT IS THERE IN THE README.md FILE

__version__ = '0.0.0'

REPO_NAME          = 'poxVision_detection'
AUTHOR_USER_NAME   = 'AarizZafar'
SRC_REPO           = 'poxVisionDetection'                 # ALL THE MAJOR PIPELINE WILL BE ADDED OVER HERE
AUTHOR_EMAIL       = 'aariz.zafar01@gmail.com'

setuptools.setup(
    name                        = SRC_REPO,
    version                     = __version__,
    author                      = AUTHOR_USER_NAME,
    author_email                = AUTHOR_EMAIL,
    description                 = 'PYTHON PACKAGE FOR POV VISION CLASSIFICATION WEB APP',
    long_description            = long_description,
    url                         = {f'https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}'},
    package_dir                 = {'' : 'src'},
    packages                    = setuptools.find_packages(where = 'src')
)