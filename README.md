# Meowify
Turn vocals from any youtube song to meows!<br>
The project was developed for [HAMR 2020](https://www.ismir2020.net/hamr/) in 48 hours and received the **Best Hack/Research Direction** award.

![alt text](https://github.com/gulnazaki/meowify/blob/main/index.png?raw=true)

## About
This is an initial release using the Flask microframework. Eventually, I aim to deploy it on [heroku](https://www.heroku.com).

I wouldn't recommend running it without cuda, since ` ddsp.training.metrics.compute_audio_features` will take too long,
especially for long tracks.

DISCLAIMER: Thanks and credits to Magenta for providing [DDSP](https://github.com/magenta/ddsp) and the colab notebooks that I used and modified
for local use and Deezer for providing [Spleeter](https://github.com/deezer/spleeter).

Here is an example for ["Dumb" by Nirvana](https://www.youtube.com/watch?v=8xiwuumLkOQ)

## Installation
```
git clone https://github.com/gulnazaki/meowify.git
cd meowify
pip3 install -r requirements.txt
```
## Running
```
flask run
```
visit [localhost:5000](localhost:5000), enter a youtube link and have fun
