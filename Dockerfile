FROM tensorflow/tensorflow:2.1.0-py3

# Set the virtualenv path to bin:
#ENV VIRTUAL_ENV=/opt/venv
#RUN python -m venv $VIRTUAL_ENV
#ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set the Workdir:
WORKDIR /usr/src/app

# Install dependencies:
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application:
COPY . .
CMD ["python3", "webstreamer.py"]