.PHONY: api ui dev test clean

api:
	uvicorn backend.main:app --reload --port 8000

ui:
	streamlit run frontend/app.py

dev:
	(uvicorn backend.main:app --reload --port 8000 &) && sleep 2 && streamlit run frontend/app.py

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +
