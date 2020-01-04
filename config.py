from configparser import ConfigParser
import os

this_path = os.path.dirname(os.path.abspath(__file__))

def parse():
	parser = ConfigParser()
	parser.read(this_path + '/config.ini')
	return parser

def parse_for_validation():
	parser = ConfigParser()
	parser.read(this_path + '/config_validation.ini')
	return parser
