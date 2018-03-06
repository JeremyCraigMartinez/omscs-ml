preprocess data
```bash
# create environment
python3.6 -m venv .venv
# activate python virtual environment
source .venv/bin/activate
# install dependencies
pip install -r requirements.txt
python src/helpers/reviews_preprocessing.py
# files are in directory csv/
```

run all RandomSearch

```bash
# first install jython
# create jython environment
virtualenv -p jython .jenv
# activate python virtual environment
source .jenv/bin/activate
# run function optimizations
# modify number after j to how many cores you want to run in parallel
# THIS WILL HAVE A SIGNIFICANT IMPACT ON YOUR CPUs!!!
parallel \
  -j 4 \
  jython src/main.py ::: \
  0 \
  1,45,20,20 \
  1,45,20,10 \
  1,45,10,20 \
  1,45,10,10 \
  1,55,20,20 \
  1,55,20,10 \
  1,55,10,20 \
  1,55,10,10 \
  2 \
  3,0.5 \
  3,0.6 \
  3,0.7 \
  3,0.8 \
  3,0.9

# run fitness function optimizations
jython src/tsp.py
jython src/flipflop.py
jython src/continuouspeaks.py
```
