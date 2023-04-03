
# get base image
FROM python:3.10.9

# copy all my content to app folder on image
COPY . /app

# Set the working directory on base image
WORKDIR /app

# install 
RUN pip install -r requirements.txt

# Expose the port with $PORT it will automatically be assigned in cloud
EXPOSE 9696

# Run the app.py (second one) in the dir app
# binding Port to local IP address
CMD gunicorn --workers=4 --bind 0.0.0.0:9696 app:app
# ENTRYPOINT [...runApp(port=80,host='0.0.0.0')] this is the same, or is it?





