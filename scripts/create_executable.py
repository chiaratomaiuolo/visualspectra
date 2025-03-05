# scripts/create_executable.py

import PyInstaller.__main__

PyInstaller.__main__.run([
    'main.py',  # Il percorso al file principale del tuo progetto
    '--onefile',           # Crea un singolo file eseguibile
    '--name=VisualizeSpectra',  # Nome dell'eseguibile
    '--distpath=dist',     # Directory di output
])

print("Eseguibile creato in: dist/VisualizeSpectra")