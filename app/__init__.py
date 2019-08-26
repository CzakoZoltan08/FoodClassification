from flask import Flask
from app.food_classification.food_classification_controller import food_classification_module


app = Flask(__name__)

# Configurations
app.config.from_object('config')

# Register blueprint(s)
app.register_blueprint(food_classification_module)
