{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "import seaborn as sns\n",
    "\n",
    "from matplotlib import pylab as plt\n",
    "\n",
    "sns.set_style(\"darkgrid\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "files = {   'difficulty' : \"QDifficulty.csv\",\n",
    "            'overall' : \"QCourseOverall.csv\",\n",
    "            'workload' : \"QWorkload.csv\",\n",
    "            'courses'  : \"courses.csv\"\n",
    "        }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "keys = ['overall', 'difficulty', 'workload']\n",
    "data = {key: pd.read_csv(files[key]) for key in keys}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Using matplotlib backend: Qt4Agg\n"
     ]
    },
    {
     "data": {
      "image/png": "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