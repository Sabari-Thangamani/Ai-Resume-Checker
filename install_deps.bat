i@echo off
py -3.11 -m pip install -r requirements.txt
py -3.11 -c "import spacy; spacy.cli.download('en_core_web_sm')"
echo Installation complete.
pause
