from dotenv import load_dotenv 
import os 

load_dotenv()

MAX_STEPS = int(os.getenv("MAX_STEPS"))
GYM_ENV_NAME = os.getenv("GYM_ENV_NAME")