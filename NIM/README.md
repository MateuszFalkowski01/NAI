Players alternate removing stones from a single pile. In misère Nim,
the player who takes the last remaining stone loses.
Configuration is read from config.yaml (YAML) where you also pick the play mode. Logging goes to run.log.

https://www.hackerrank.com/challenges/misere-nim-1/problem

Krzysztof Cieślik s27115
Mateusz Falkowski s27426

run instructions:
     Python: Works with Python 3.12.x
     Tested on Linux 6.14.0-33-generic
     Linux/macOS:
        1) Create venv
            python3 -m venv venv && source venv/bin/activate
        2) Install dependencies
            pip install -r requirements.txt
        3) Run
            python src/main.py

     Windows:
        1) Create venv
            py -m venv venv
            .\\venv\\Scripts\\Activate.ps1
        2) Install dependencies
            pip install -r requirements.txt
        3) Run
            python .\\src\\main.py
