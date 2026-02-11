pip install fastapi uvicorn

# Recommandé en prod bot: greedy (moins d'erreurs tactiques)
python ai_server.py --checkpoint checkpoints/step_215744512.pt --device cuda --temperature 0.0 --value-head-hidden 256

# Optionnel pour un bot plus "créatif" (mais plus risqué)
# python ai_server.py --checkpoint checkpoints/step_215744512.pt --device cuda --temperature 0.3 --value-head-hidden 256

python -m tensorboard.main --logdir runs --port 6006

# Evaluate current model with parry analysis
python evaluate_model.py --checkpoint checkpoints/step_215744512.pt \
    --device cuda --num-games 1000 --value-head-hidden 256

# After training with new rewards, compare
python evaluate_model.py \
    --checkpoint-a checkpoints/step_158072832.pt \
    --checkpoint-b checkpoints/step_NEW.pt \
    --device cuda --num-games 500 --value-head-hidden 256
