from app import app
with app.app_context():
    from models import db
    db.create_all()
    print('DB created')
