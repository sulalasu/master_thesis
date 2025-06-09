# using * is discouraged because of 
# possible naming conflicts/namespace pollution
# from cleaning import *
# from input import *
# from processing import *
# from viz import *
# from utils import *


# Functions have to be accessed like this:
# src.cleaning.my_func()
from . import load
from . import clean
from . import process
from . import viz
from . import utils
from . import config