Este exemplo utiliza Streamlit para montar a interface do usuário.

Este é o launch.json utilizado para executar o Streamlit:

{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Streamlit",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/.venv/bin/streamlit",
            "args": [
                "run",
                "${workspaceFolder}/app.py"
            ],
            "console": "integratedTerminal",
            "justMyCode": false,
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}${pathSeparator}${env:PYTHONPATH}"
            }
        }
    ]
}

A análise está implementada com 2 métodos:

PANDAS: utiliza principalmente a estrutura de dados Python Pandas para fazer a análise. Ela é mais apropriada para datasets pequenos e análises mais simples, porque geralmente mais lenta.

CHAIN: Utiliza 100% da api ChatGPT. Geralmente é mais rápida e poderosa, e é ais indicada para datasets maiores e análises para complexas. Ela pode ter um custo um pouco maior.


