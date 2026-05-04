# imitation_learning_
This project runs two OpenAIGym games - MountainCar and Acrobot - for data collection.

This project is able to be deployed on Render using the following commands:
Repository
The repository used for your Web Service.
https://github.com/ebushra/imitation_learning_

Branch
The Git branch to build and deploy.
main

Root Directory - empty

Build Command
Render runs this command to build your app before each deploy.
pip install --upgrade pip && pip install -r website_honors/requirements.txt

Start Command
Render runs this command to start your app with each deploy.
gunicorn website_honors.server.main:app --bind 0.0.0.0:$PORT

A permanent disk is necessary to save and access the data from Render shell. 
Create a mount path (ex. /var/data) and cd into it to see each user's game data.
