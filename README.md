# Musedetekterings App

En Streamlit app der bruger Azure Custom Vision til at detektere mus og oddere i billeder.

## Installation og opsætning

1. Opret et virtuelt miljø (i PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\activate
```

2. Installer de nødvendige pakker:
```powershell
pip install -r requirements.txt
```

3. Start appen:
```powershell
streamlit run app.py
```

## Funktioner
- Upload enkelte billeder eller flere billeder på én gang
- Detektering af:
  - Mus med stripper
  - Odder
  - Almindelig mus
- Visning af statistik og detaljerede resultater
- Justerbar confidence threshold

## Bemærk
Sørg for at have Python 3.8 eller nyere installeret på din computer. 