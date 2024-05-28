from dotenv import load_dotenv, find_dotenv
import logging
from transformers.utils import logging as transformers_logging

load_dotenv(find_dotenv())


def create_app():
    """Create a Flask app, gathering all the blueprints.

    Returns:
        Flask.app: the app runtime
    """
    from flask import Flask
    from flask.logging import default_handler
    from clip_search_image_backend.app.routes.image_search import image_search

    # from tods_data_loader_backend.app.routes.mock_service import auth, InterceptRequestMiddleware

    app = Flask(__name__)

    if not app.debug:
        gunicorn_logger = logging.getLogger("gunicorn.error")
        app.logger.handlers = gunicorn_logger.handlers
        app.logger.setLevel(gunicorn_logger.level)
    app.register_blueprint(image_search)
    transformers_logging.set_verbosity_info()
    transformers_logger = transformers_logging.get_logger("transformers")
    transformers_logger.addHandler(default_handler)
    return app
