preprocess data
```bash
# create environment
python3.6 -m venv .venv
# activate python virtual environment
source .venv/bin/activate
# install dependencies
pip install -r requirements.txt
python src/helpers/reviews_preprocessing.py
```

run all RandomSearch

```bash

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
```
