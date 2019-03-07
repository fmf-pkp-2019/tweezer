"""
Configuration and constants
"""
from __future__ import absolute_import, print_function, division
import os, warnings, shutil

try:
    from configparser import ConfigParser
except:
    #python 2.7
    from ConfigParser import ConfigParser

DATAPATH = os.path.dirname(__file__)

def read_environ_variable(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return int(default)
        warnings.warn("Environment variable {0:s} was found, but its value is not valid!".format(name))

def get_home_dir():
    """
    Return user home directory
    """
    try:
        path = os.path.expanduser('~')
    except:
        path = ''
    for env_var in ('HOME', 'USERPROFILE', 'TMP'):
        if os.path.isdir(path):
            break
        path = os.environ.get(env_var, '')
    if path:
        return path
    else:
        raise RuntimeError('Please define environment variable $HOME')

#: home directory
HOMEDIR = get_home_dir()

TWEEZER_CONFIG_DIR = os.path.join(HOMEDIR, ".tweezer")
NUMBA_CACHE_DIR = os.path.join(TWEEZER_CONFIG_DIR, "numba_cache")

if not os.path.exists(TWEEZER_CONFIG_DIR):
    try:
        os.makedirs(TWEEZER_CONFIG_DIR)
    except:
        warnings.warn("Could not create folder in user's home directory! Is it writeable?")
       
CONF = os.path.join(TWEEZER_CONFIG_DIR, "tweezer.ini")
CONF_TEMPLATE = os.path.join(DATAPATH, "tweezer.ini")

config = ConfigParser()

if not os.path.exists(CONF):
    try:
        shutil.copy(CONF_TEMPLATE, CONF)
    except:
        warnings.warn("Could not copy config file in user's home directory! Is it writeable?")
        CONF = CONF_TEMPLATE

config.read(CONF)
    
def _readconfig(func, section, name, default):
    try:
        return func(section, name)
    except:
        return default
    
def is_module_installed(name):
    """Checks whether module with name 'name' is istalled or not"""
    try:
        __import__(name)
        return True
    except ImportError:
        return False    
        
NUMBA_INSTALLED = is_module_installed("numba")
MKL_FFT_INSTALLED = is_module_installed("mkl_fft")
SCIPY_INSTALLED = is_module_installed("scipy")

class TweezerConfig(object):
    """Tweezer settings are here. You should use the set_* functions in the
    conf.py module to set these values"""
    def __init__(self):
        self.verbose = _readconfig(config.getint, "default", "verbose",0)
        
    def __getitem__(self, item):
        return self.__dict__[item]
        
    def __repr__(self):
        return repr(self.__dict__)

#: a singleton holding user configuration    
TweezerConfig = TweezerConfig()

def print_config():
    """Prints all compile-time and run-time configurtion parameters and settings."""
    options = {}
    options.update(TweezerConfig.__dict__)
    print(options)

def set_verbose(level):
    """Sets verbose level (0-2) used by compute functions."""
    out = TweezerConfig.verbose
    TweezerConfig.verbose = max(0,int(level))
    return out


    
