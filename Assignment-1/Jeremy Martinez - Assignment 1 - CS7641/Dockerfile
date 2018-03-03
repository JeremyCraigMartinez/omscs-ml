FROM tensorflow/tensorflow:nightly-py3

RUN mkdir /app
WORKDIR /app

# install dependencies
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# copy in project code
COPY data /app/data
COPY src /app/src
