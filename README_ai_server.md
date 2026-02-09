pip install fastapi uvicorn
python ai_server.py --checkpoint checkpoints/step_8650752.pt --device cuda --temperature 0.3

python -m tensorboard.main --logdir runs --port 6006