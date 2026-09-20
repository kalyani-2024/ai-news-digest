from app.database.models import Base
from app.database.connection import engine


def create_tables() -> None:
    Base.metadata.create_all(engine)


if __name__ == "__main__":
    create_tables()
    print("Tables created successfully")
