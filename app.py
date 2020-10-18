from flask import Flask, render_template, url_for, request, redirect, session, send_from_directory
from utils import *

app = Flask(__name__)
app.secret_key = "supposed to be a secret"

@app.route('/processing', methods=['GET', 'POST'])
def processing():
	if request.method == 'GET':
		return render_template('waiting.html', img="/static/images/waiting.jpg")

	if request.method == 'POST':
		download(session)
		split_vocals(session)
		meowify(session)
		merge_meows_and_music(session)
		return 'done'

@app.route('/success', methods=['GET'])
def success():
	return render_template("success.html", title=session.get('title'), vocals=session.get('final'), img="/static/images/singing.jpg")

@app.route('/', methods=['POST', 'GET'])
def index():
	
	title = "Meowify"

	if request.method == "POST":
	
		session['requested_url'] = request.form["url"]
		if session.get('requested_url') != "":
			if verify(session.get('requested_url')):
				return render_template("waiting.html", img="/static/images/waiting.jpg")
			else:
				return render_template(
					"index.html", title="That's not a youtube URL, are you kittying me?",
					img="/static/images/angry_kitty.jpg")
		else:
			return render_template(
				"index.html", title="That's an empty URL, are you kittying me?",
				img="/static/images/angrier_kitty.jpg")
	
	else:
		return render_template("index.html", title=title, img="/static/images/starting.png")

if __name__ == "__main__":
	app.run(debug=True)