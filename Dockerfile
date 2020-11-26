FROM python:3.6

# Set the virtualenv path to bin:
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set the Workdir:
WORKDIR /usr/src/app

# Install dependencies:
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application:
COPY . .
CMD ["python", "webstreamer.py"]