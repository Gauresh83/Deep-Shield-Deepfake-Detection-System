from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .config import settings

connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(settings.DATABASE_URL, connect_args=connect_args, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    # Import all models so SQLAlchemy sees them before create_all
    from ..models import user, scan, alert  # noqa: F401
    Base.metadata.create_all(bind=engine)
    _seed_demo_user()


def _seed_demo_user():
    """Create the demo@m3id.ai account if it doesn't exist yet."""
    from ..models.user import User
    from ..core.security import hash_password
    db = SessionLocal()
    try:
        exists = db.query(User).filter(User.email == "demo@m3id.ai").first()
        if not exists:
            demo = User(
                email="demo@m3id.ai",
                hashed_password=hash_password("Demo@1234"),
                first_name="Demo",
                last_name="User",
                role="User",
                avatar_initials="DU",
                is_active=True,
            )
            db.add(demo)
            db.commit()
            print("✅  Demo account created: demo@m3id.ai / Demo@1234")
    except Exception as e:
        print(f"⚠️  Demo user seed skipped: {e}")
    finally:
        db.close()
