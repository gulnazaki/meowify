import re
from youtube_dl import YoutubeDL
from spleeter.separator import Separator
import ffmpeg
import os
from ddsp_timbre_transfer import write_to_file

def verify(url):
	p = re.compile(r'^(https?\:\/\/)?(www\.)?(youtube\.com|youtu\.?be)\/.+$')
	if p.match(url):
		return True
	return False

def download(session):
	url = session.get('requested_url')
	with YoutubeDL({'format': 'bestaudio'}) as ydl:
		info = ydl.extract_info(url)
		session['title'] = info['title']
		session['ext'] = info['ext']
		session['filename'] = info['title'] + '-' + info['id']

def split_vocals(session):
	f = session.get('filename')
	e = '.' + session.get('ext')
	if not os.path.exists(f + e):
		ffmpeg.input(f + e).output(f + '.wav').run()

	vocals = os.path.join('split', f, 'vocals.wav')
	acc = os.path.join('split', f, 'accompaniment.wav')
	if not os.path.exists(vocals) or not os.path.exists(acc):
		separator = Separator('spleeter:2stems')
		separator.separate_to_file(f + '.wav', 'split')

	session['vocals'] = vocals
	session['acc'] = acc

def meowify(session):
	f = session.get('filename')
	meows = os.path.join('split', f, 'meows.wav')
	if not os.path.exists(meows):
		write_to_file(session.get('vocals'), 'catophone', meows)

	session['meows'] = meows

def merge_meows_and_music(session):
	f = session.get('filename')
	final = os.path.join('static', f + '-final.wav')
	if not os.path.exist(final):
		ffmpeg.input(session.get('meows')).input(session.get('acc')).filter('amix').output(final).run()

	session['final'] = final